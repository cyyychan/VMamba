import argparse
import os

from config import get_config
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build vmambav2m3 model and print structure/params."
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__),
            "configs",
            "vssm",
            "vmambav2_tiny_224.yaml",
        ),
        help="Path to config file.",
    )
    parser.add_argument(
        "--opts",
        default=None,
        nargs="+",
        help="Override config options via KEY VALUE pairs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = get_config(args)
    model = build_model(config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("=" * 80)
    print(f"Config: {args.cfg}")
    print(f"Model: {config.MODEL.TYPE}/{config.MODEL.NAME}")
    print("=" * 80)
    print(model)
    print("=" * 80)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
