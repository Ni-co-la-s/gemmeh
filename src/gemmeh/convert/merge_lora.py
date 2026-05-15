"""
Merge LoRA adapter weights into a base Gemmeh checkpoint.

Usage:
uv run -m gemmeh.convert.merge_lora \
    --base_checkpoint checkpoints/pretrain-1B-20B/step_305176_tokens_20000014336.pt \
    --lora_checkpoint checkpoints/finetune-1B-openhermes/best.pt \
    --output_path checkpoints/finetune-1B-openhermes/merged.pt \
    --rank 16 \
    --alpha 32 \
    --targets q k v o gate up down \
    --device cuda
"""

import argparse
import os
from pathlib import Path

import torch

from gemmeh.config.model_config import ModelConfig
from gemmeh.finetune.lora import inject_lora, load_lora_checkpoint, merge_lora_into_base
from gemmeh.model.gemma3 import Gemma3Model


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into a base Gemmeh checkpoint")
    parser.add_argument("--base_checkpoint", type=Path, required=True, help="Path to base model checkpoint (.pt)")
    parser.add_argument("--lora_checkpoint", type=Path, required=True, help="Path to LoRA adapter checkpoint (.pt)")
    parser.add_argument("--output_path", type=Path, required=True, help="Where to save merged checkpoint (.pt)")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["q", "k", "v", "o", "gate", "up", "down"],
        help="LoRA target modules (default: q k v o gate up down)",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank used during finetuning")
    parser.add_argument("--alpha", type=float, default=32.0, help="LoRA alpha used during finetuning")
    parser.add_argument("--device", type=str, default="cuda", help="Device for merge (e.g. cuda or cpu)")
    args = parser.parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load base model on CPU first to avoid double-allocation on GPU.
    print("Loading base model...")
    config = ModelConfig()
    model = Gemma3Model(config)
    ckpt = torch.load(args.base_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict)
    del ckpt, state_dict

    device = torch.device(args.device)
    model = model.to(torch.bfloat16).to(device)
    print("  Base model loaded")

    # Inject LoRA and load adapter weights.
    print("Injecting LoRA...")
    inject_lora(model, rank=args.rank, alpha=args.alpha, targets=args.targets)
    load_lora_checkpoint(model, str(args.lora_checkpoint), device=device)

    # Merge LoRA into base weights.
    print("Merging LoRA into base weights...")
    merge_lora_into_base(model)

    # Save merged checkpoint.
    print(f"Saving to {args.output_path}...")
    torch.save({"model": model.state_dict()}, args.output_path)
    print(f"Done. ({os.path.getsize(args.output_path) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
