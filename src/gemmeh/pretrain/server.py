"""
OpenAI-compatible /v1/completions server for the model.
Used as backend to compute the results of the model on benchmarks in gemmeh/pretrain/eval.sh
"""

import argparse

import sentencepiece as spm
import torch
import uvicorn

from gemmeh.config.model_config import ModelConfig
from gemmeh.utils.completion_api import CompletionModelMixin, create_completion_app
from gemmeh.model.gemma3 import Gemma3Model


class ModelServer(CompletionModelMixin):
    def __init__(self, checkpoint_path: str, tokenizer_path: str, device: str = "cuda"):
        self.device = torch.device(device)

        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(tokenizer_path)
        print(f"Tokenizer loaded: vocab_size={self.sp.GetPieceSize()}")

        config = ModelConfig()
        self.model = Gemma3Model(config)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {total_params / 1e6:.1f}M params on {self.device}")


server: ModelServer | None = None


def _get_server() -> ModelServer:
    assert server is not None
    return server


app = create_completion_app(
    title="Gemmeh Completions Server",
    model_id="gemmeh",
    get_server=_get_server,
)


def main():
    global server

    parser = argparse.ArgumentParser(description="Gemmeh Completions Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to sentencepiece.model")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ModelServer(args.checkpoint, args.tokenizer, args.device)
    print(f"\nStarting server on {args.host}:{args.port}")
    print("Tokenizer endpoints: /tokenizer_info, /tokenize, /detokenize")
    print("Completions endpoint: /v1/completions")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
