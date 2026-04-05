"""Shared text generation utilities."""

from contextlib import nullcontext
from typing import Any

import sentencepiece as spm
import torch


@torch.no_grad()
def generate_text_samples(
    model: Any,
    sp: spm.SentencePieceProcessor,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[str]:
    """Generate text completions for fixed prompts (qualitative monitoring)."""
    model.eval()
    results: list[str] = []

    max_pos = model.config.max_position_embeddings

    for prompt in prompts:
        token_ids = sp.encode(prompt, out_type=int)
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            if input_ids.shape[1] > max_pos:
                input_ids = input_ids[:, -max_pos:]

            context = torch.amp.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
            with context:
                logits, _ = model(input_ids)

            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_vals, _ = torch.topk(logits, top_k)
                logits[logits < top_vals[:, [-1]]] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            if next_id.item() == sp.eos_id():
                break

        results.append(sp.decode(input_ids[0].tolist()))

    model.train()
    return results
