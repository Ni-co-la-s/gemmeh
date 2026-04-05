"""
Export model as safetensors file and move it, the config and the tokenizer to a new folder (to be used by llama.cpp and vllm)
It uses the config from gemmeh.config.model_config, so it should match the config used to train the model.

Usage:
uv run -m gemmeh.convert.export_checkpoint \
    --checkpoint checkpoints/pretrain-1B-20B/step_305176_tokens_20000014336.pt \
    --tokenizer data/tokenizers/run_32k_1B/sentencepiece.model \
    --model_name gemmeh-1B
"""

import torch
import json
import argparse
from pathlib import Path
from safetensors.torch import save_file
import shutil

from gemmeh.config.model_config import ModelConfig

config = ModelConfig()


def main():
    parser = argparse.ArgumentParser(description="Export a gemmeh training checkpoint to HuggingFace-compatible format")
    parser.add_argument("--checkpoint", type=Path, help="Path to model.pt training checkpoint")
    parser.add_argument("--tokenizer", type=Path, help="Path to sentencepiece.model file")
    parser.add_argument("--model_name", type=str, help="Model name, output will be saved to models/{model_name}/")
    args = parser.parse_args()

    # Set up output directory
    out_dir = Path("models") / args.model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state_dict = checkpoint["model"]

    # Save weights as safetensors
    safetensors_path = out_dir / "model.safetensors"
    print(f"Saving {len(state_dict)} tensors to {safetensors_path}...")
    save_file(state_dict, safetensors_path)

    # Save config.json
    config_dict = {
        "architectures": ["GemmehForCausalLM"],
        "model_type": "gemmeh",
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "rms_norm_eps": config.rms_norm_eps,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "use_qk_norm": config.use_qk_norm,
        "use_pre_ffw_norm": config.use_pre_ffw_norm,
        "use_post_ffw_norm": config.use_post_ffw_norm,
        "tie_word_embeddings": config.tie_word_embeddings,
    }
    config_path = out_dir / "config.json"
    print(f"Saving config to {config_path}...")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Copy tokenizer
    tokenizer_path = out_dir / "tokenizer.model"
    print(f"Copying tokenizer to {tokenizer_path}...")
    shutil.copy(args.tokenizer, tokenizer_path)

    print(f"\nDone! Model exported to {out_dir}/")


if __name__ == "__main__":
    main()
