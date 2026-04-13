# Copyright (c) 2026, Dao AI Lab, Goombalab.

import math
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined


class Mamba3SISO(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        # ----------------------------------------
        # Mamba-3 configs
        rope_fraction=0.5,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        A_floor=1e-4,
        #-------------------------------------------
        # Fused kernel and sharding options
        chunk_size=64, # Recommended: 64 for SISO
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.headdim = headdim
        self.chunk_size = chunk_size
        self.A_floor = A_floor

        self.d_inner = int(self.expand * self.d_model)
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        self.num_bc_heads = ngroups
        
        # RoPE flags
        assert rope_fraction in [0.5, 1.0]
        self.rotary_dim_divisor = int(2/rope_fraction)
        self.split_tensor_size = int(d_state * rope_fraction)
        if self.split_tensor_size % 2 != 0:
            self.split_tensor_size -= 1
        self.num_rope_angles = self.split_tensor_size // 2
        assert self.num_rope_angles > 0

        # Order: [x, B, C, dd_dt, dd_A, trap, angle]
        d_in_proj = self.d_inner + 2 * self.d_state * self.num_bc_heads + 3 * self.nheads + self.num_rope_angles
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=False, **factory_kwargs)

        # dt_bias parameterization        
        _dt = torch.exp(
            torch.rand(self.nheads, device=device, dtype=torch.float32) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        _dt = torch.clamp(_dt, min=dt_init_floor)
        _dt_bias = _dt + torch.log(-torch.expm1(-_dt))
        self.dt_bias = nn.Parameter(_dt_bias, requires_grad=True)
        self.dt_bias._no_weight_decay = True
        
        # B and C biases
        self.B_bias = nn.Parameter(1+torch.zeros((self.nheads, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
        self.C_bias = nn.Parameter(1+torch.zeros((self.nheads, self.d_state), dtype=torch.float32, device=device), requires_grad=True)
                                                       
        # RMS Norm for B and C
        assert RMSNormGated is not None
        self.B_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
        self.C_norm = RMSNormGated(self.d_state, eps=1e-5, **factory_kwargs)
    
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False, **factory_kwargs)

    def forward(self, u):
        """
        u: (batch, seqlen, hidden_dim)
        Returns: same shape as u
        """

        # Apply in_proj
        xBCdtAtrap = self.in_proj(u)
        x, B, C, dd_dt, dd_A, trap, angles = torch.split(
            xBCdtAtrap,
            [
                self.d_inner, 
                self.d_state * self.num_bc_heads,
                self.d_state * self.num_bc_heads,
                self.nheads, self.nheads, self.nheads, 
                self.num_rope_angles
            ],
            dim=-1)

        x = rearrange(x, "b l (h p) -> b l h p", p=self.headdim)
        B = rearrange(B, "b l (g n) -> b l g n", g=self.num_bc_heads)
        C = rearrange(C, "b l (g n) -> b l g n", g=self.num_bc_heads)
        trap = rearrange(trap, "b l h -> b h l")

        # Compute ADT, DT
        _A = -F.softplus(dd_A.to(torch.float32)) # (B, L, N)
        _A = torch.clamp(_A, max=-self.A_floor)            
        DT = F.softplus(dd_dt + self.dt_bias) # (B, L, N)
        ADT = _A * DT
        DT = rearrange(DT, "b l n -> b n l")
        ADT = rearrange(ADT, "b l n -> b n l")

        # Compute angle
        angles = angles.unsqueeze(-2).expand(-1, -1, self.nheads, -1) # (B, L, N, S)

        # Apply RMS Norm on B and C
        B = self.B_norm(B)
        C = self.C_norm(C)
        
        y = mamba3_siso_combined(
            Q=C,
            K=B,
            V=x,
            ADT=ADT,
            DT=DT,
            Trap=trap,
            Q_bias=self.C_bias,
            K_bias=self.B_bias,
            Angles=angles,
            D=self.D,
            Z=None,
            chunk_size=self.chunk_size,
            Input_States=None
        )
        y = rearrange(y, "b l h p -> b l (h p)")
        out = self.out_proj(y.to(x.dtype))
        return out
    
if __name__ == "__main__":
    import sys

    if not torch.cuda.is_available():
        print("CUDA is required for this Mamba3 MIMO smoke test.", file=sys.stderr)
        sys.exit(1)

    d_model, seqlen = 96, 1024

    model = Mamba3SISO(
        d_model=d_model,
        d_state=128
    )
    model.eval()
    device = torch.device("cuda")
    dtype = model.in_proj.weight.dtype
    model.to(device)
    print(model)

    x = torch.randn(1, seqlen, d_model, device=device, dtype=dtype)
    with torch.no_grad():
        y = model(x)
    assert y.shape == x.shape
    print("mamba3 SISO smoke test OK:", tuple(y.shape))
