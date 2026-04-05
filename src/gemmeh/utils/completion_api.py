"""
Utils for OpenAI compatible /v1/completions server for the model.
"""

import time
import uuid
from contextlib import nullcontext
from typing import Any, Protocol, Union

import sentencepiece as spm
import torch
import torch.nn.functional as F
from fastapi import FastAPI
from pydantic import BaseModel


class CompletionRequest(BaseModel):
    model: str = "gemmeh"
    prompt: Union[str, list[int]] = ""
    max_tokens: int = 64
    temperature: float = 0.0
    top_k: int = 0
    echo: bool = False
    logprobs: int | None = None
    stop: list[str] | None = None
    seed: int | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    logprobs: dict | None = None
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str = "gemmeh"
    choices: list[CompletionChoice]
    usage: UsageInfo


class TokenizeRequest(BaseModel):
    prompt: str
    add_special_tokens: bool = False


class TokenizeResponse(BaseModel):
    tokens: list[int]
    count: int


class DetokenizeRequest(BaseModel):
    tokens: list[int]


class DetokenizeResponse(BaseModel):
    prompt: str


class CompletionBackend(Protocol):
    sp: spm.SentencePieceProcessor

    def get_logprobs(self, token_ids: list[int]) -> torch.Tensor: ...

    def generate(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> tuple[list[int], list[torch.Tensor]]: ...


class CompletionModelMixin:
    """Shared get_logprobs/generate implementation for Gemmeh model servers."""

    device: torch.device
    model: Any
    sp: spm.SentencePieceProcessor

    @torch.no_grad()
    def get_logprobs(self, token_ids: list[int]) -> torch.Tensor:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        context = torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
        with context:
            logits, _ = self.model(input_ids)
        return F.log_softmax(logits[0].float(), dim=-1)

    @torch.no_grad()
    def generate(
        self,
        token_ids: list[int],
        max_new_tokens: int,
        temperature: float = 0.0,
        top_k: int = 0,
    ) -> tuple[list[int], list[torch.Tensor]]:
        generated_ids: list[int] = []
        generated_logprobs: list[torch.Tensor] = []
        current_ids = list(token_ids)

        for _ in range(max_new_tokens):
            ctx = current_ids[-self.model.config.max_position_embeddings :]
            input_t = torch.tensor([ctx], dtype=torch.long, device=self.device)

            context = torch.amp.autocast("cuda", dtype=torch.bfloat16) if self.device.type == "cuda" else nullcontext()
            with context:
                logits, _ = self.model(input_t)

            last_logits = logits[0, -1, :].float()
            lp = F.log_softmax(last_logits, dim=-1)
            generated_logprobs.append(lp.cpu())

            if temperature <= 0 or temperature < 1e-8:
                next_id = last_logits.argmax().item()
            else:
                scaled = last_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(scaled, top_k)
                    scaled[scaled < v[-1]] = float("-inf")
                probs = torch.softmax(scaled, dim=-1)
                next_id = torch.multinomial(probs, 1).item()

            generated_ids.append(next_id)
            current_ids.append(next_id)

            if next_id == self.sp.eos_id():
                break

        return generated_ids, generated_logprobs


def build_logprobs_dict(
    token_ids: list[int],
    log_probs_list,
    sp: spm.SentencePieceProcessor,
    n_top: int = 1,
) -> dict:
    tokens = []
    token_logprobs = []
    top_logprobs = []
    text_offset = []
    running_offset = 0

    for i, tid in enumerate(token_ids):
        piece = sp.IdToPiece(tid)
        tokens.append(piece)
        text_offset.append(running_offset)
        running_offset += len(piece)

        lp_vec = log_probs_list[i]

        if lp_vec is None:
            token_logprobs.append(0.0)
            top_logprobs.append({piece: 0.0})
        else:
            token_logprobs.append(lp_vec[tid].item())
            k = max(n_top, 1)
            topk_vals, topk_ids = torch.topk(lp_vec, min(k, lp_vec.shape[0]))
            entry = {}
            for v, idx in zip(topk_vals.tolist(), topk_ids.tolist()):
                entry[sp.IdToPiece(idx)] = v
            top_logprobs.append(entry)

    return {
        "tokens": tokens,
        "token_logprobs": token_logprobs,
        "top_logprobs": top_logprobs,
        "text_offset": text_offset,
    }


def create_completion_app(
    *,
    title: str,
    model_id: str,
    get_server,
) -> FastAPI:
    """Create a FastAPI app backed by a server implementing CompletionBackend."""
    app = FastAPI(title=title)

    @app.get("/tokenizer_info")
    def tokenizer_info():
        server: CompletionBackend = get_server()
        return {
            "tokenizer_type": "sentencepiece",
            "vocab_size": server.sp.GetPieceSize(),
            "eos_token": server.sp.IdToPiece(server.sp.eos_id()),
            "bos_token": server.sp.IdToPiece(server.sp.bos_id()) if server.sp.bos_id() >= 0 else None,
            "pad_token": None,
        }

    @app.post("/tokenize")
    def tokenize(req: TokenizeRequest):
        server: CompletionBackend = get_server()
        token_ids = server.sp.Encode(req.prompt)
        return TokenizeResponse(tokens=token_ids, count=len(token_ids))

    @app.post("/detokenize")
    def detokenize(req: DetokenizeRequest):
        server: CompletionBackend = get_server()
        text = server.sp.Decode(req.tokens)
        return DetokenizeResponse(prompt=text)

    @app.get("/v1/models")
    def list_models():
        return {
            "object": "list",
            "data": [{"id": model_id, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        server: CompletionBackend = get_server()

        if isinstance(req.prompt, str):
            prompt_ids = server.sp.Encode(req.prompt)
        else:
            prompt_ids = list(req.prompt)

        n_logprobs = req.logprobs if req.logprobs else 1

        if req.echo:
            prompt_lp = server.get_logprobs(prompt_ids)

            per_token_lp: list[torch.Tensor | None] = [None]
            for i in range(1, len(prompt_ids)):
                per_token_lp.append(prompt_lp[i - 1].cpu())

            if req.max_tokens > 0:
                gen_ids, gen_lp_list = server.generate(
                    prompt_ids,
                    max_new_tokens=req.max_tokens,
                    temperature=req.temperature,
                    top_k=req.top_k,
                )
                all_ids = prompt_ids + gen_ids
                per_token_lp.extend(gen_lp_list)
            else:
                all_ids = prompt_ids
                gen_ids = []

            lp_dict = build_logprobs_dict(all_ids, per_token_lp, server.sp, n_logprobs)
            text = server.sp.Decode(all_ids)
            finish = "stop" if (gen_ids and gen_ids[-1] == server.sp.eos_id()) else "length"

            return CompletionResponse(
                id=f"cmpl-{uuid.uuid4().hex[:12]}",
                created=int(time.time()),
                model=model_id,
                choices=[CompletionChoice(text=text, logprobs=lp_dict, finish_reason=finish)],
                usage=UsageInfo(
                    prompt_tokens=len(prompt_ids),
                    completion_tokens=len(gen_ids),
                    total_tokens=len(all_ids),
                ),
            )

        gen_ids, gen_lp_list = server.generate(
            prompt_ids,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )

        text = server.sp.Decode(gen_ids)
        gen_lp_dict: dict | None = None
        if req.logprobs and req.logprobs > 0:
            gen_lp_dict = build_logprobs_dict(gen_ids, gen_lp_list, server.sp, n_logprobs)

        finish = "stop" if (gen_ids and gen_ids[-1] == server.sp.eos_id()) else "length"

        return CompletionResponse(
            id=f"cmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=model_id,
            choices=[CompletionChoice(text=text, logprobs=gen_lp_dict, finish_reason=finish)],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(gen_ids),
                total_tokens=len(prompt_ids) + len(gen_ids),
            ),
        )

    return app
