"""
Completions server with in-memory RYS layer duplication (Idea from https://dnhkng.github.io/posts/rys/).

Loads the original checkpoint and optionally duplicates layers [i, j]
without writing anything to disk.
"""

import argparse
import gc
import re
from collections import OrderedDict

import sentencepiece as spm
import torch
import uvicorn

from gemmeh.config.model_config import ModelConfig
from gemmeh.utils.completion_api import CompletionModelMixin, create_completion_app
from gemmeh.model.gemma3 import Gemma3Model


class ModelServer(CompletionModelMixin):
    def __init__(
        self,
        checkpoint_path: str,
        tokenizer_path: str,
        number_layers: int,
        device: str = "cuda",
        rys_i: int | None = None,
        rys_j: int | None = None,
    ):
        self.device = torch.device(device)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        print(f"Tokenizer loaded: vocab_size={self.sp.GetPieceSize()}")

        original_layers = number_layers
        if rys_i is not None and rys_j is not None:
            dup_size = rys_j - rys_i + 1
            total_layers = original_layers + dup_size
            print(f"RYS mode: duplicating layers [{rys_i}, {rys_j}] ({dup_size} layers) -> {total_layers} total")
        else:
            total_layers = original_layers
            print(f"Standard mode: {total_layers} layers")

        config = ModelConfig()
        config.num_hidden_layers = total_layers

        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        original_sd = ckpt["model"] if "model" in ckpt else ckpt
        del ckpt
        gc.collect()

        if rys_i is not None and rys_j is not None:
            sd = self._remap_layers(original_sd, original_layers, rys_i, rys_j)
            del original_sd
            gc.collect()
        else:
            sd = original_sd

        model = Gemma3Model(config)

        if self.device.type == "cuda":
            sd = {k: v.to(self.device) for k, v in sd.items()}
            model.load_state_dict(sd, assign=True)
            torch.cuda.empty_cache()
        else:
            model.load_state_dict(sd)
            model.to(self.device)

        del sd
        gc.collect()

        model.eval()
        self.model = model

        total_params = sum(p.numel() for p in model.parameters())
        if self.device.type == "cuda":
            vram = torch.cuda.memory_allocated() / 1e9
            print(f"Model loaded: {total_params:,} params, {vram:.2f} GB VRAM")
        else:
            print(f"Model loaded: {total_params:,} params on {self.device}")

    @staticmethod
    def _remap_layers(old_sd, original_layers: int, i: int, j: int):
        """Remap state dict keys to duplicate layers circuit [i, j]."""
        mapping = []
        for idx in range(0, i):
            mapping.append(idx)
        for idx in range(i, j + 1):
            mapping.append(idx)
        for idx in range(i, j + 1):
            mapping.append(idx)
        for idx in range(j + 1, original_layers):
            mapping.append(idx)

        new_sd = OrderedDict()
        layer_pattern = re.compile(r"^layers\.(\d+)\.(.*)")

        for key, value in old_sd.items():
            if not layer_pattern.match(key):
                new_sd[key] = value

        for new_idx, old_idx in enumerate(mapping):
            prefix_old = f"layers.{old_idx}."
            prefix_new = f"layers.{new_idx}."
            for key, value in old_sd.items():
                if key.startswith(prefix_old):
                    suffix = key[len(prefix_old) :]
                    new_sd[prefix_new + suffix] = value

        return new_sd


server: ModelServer | None = None


def _get_server() -> ModelServer:
    assert server is not None
    return server


app = create_completion_app(
    title="Gemmeh RYS Server",
    model_id="gemmeh",
    get_server=_get_server,
)


def main():
    global server

    parser = argparse.ArgumentParser(description="Gemmeh RYS Completions Server")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--number_layers", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--rys_i", type=int, default=None)
    parser.add_argument("--rys_j", type=int, default=None)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ModelServer(
        args.checkpoint,
        args.tokenizer,
        args.number_layers,
        args.device,
        args.rys_i,
        args.rys_j,
    )
    print(f"\nServer ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
