"""
Training loop for Gemma-3-inspired model.
Configuration of the model in  gemmeh.config.model_config.
Configuration of the training in  gemmeh.config.train_config

- AdamW with cosine LR schedule and linear warmup
- Gradient accumulation to account for smaller batch size
- Mixed precision (bfloat16) with gradient scaling
- Wandb optional logging: loss, perplexity, LR, grad norm, throughput
- Periodic validation and sample generation
- Checkpointing every N tokens + best val loss
- Option to restart from a checkpoint

Usage:
    uv run -m gemmeh.pretrain.train
"""

import os
import time
import math
import torch
import torch.nn as nn
import sentencepiece as spm
from typing import cast

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from gemmeh.config.model_config import ModelConfig
from gemmeh.config.train_config import TrainConfig

from gemmeh.model.gemma3 import Gemma3Model
from gemmeh.pretrain.data import create_dataloader
from gemmeh.utils.generation import generate_text_samples

# Prompts use to test the model outputs
prompts = [
    "Let me tell you a story:",  # Test creativity and coherence of story
    "The capital of France is",  # Basic knowledge
    "In 2023, the population of",  # Bit more complex
]

#  LR Schedule


def get_lr(tokens_seen: int, cfg: TrainConfig) -> float:
    """Cosine schedule with linear warmup, based on tokens seen (not on steps)."""
    # Linear warmup
    if tokens_seen < cfg.warmup_tokens:
        return cfg.learning_rate * (tokens_seen / cfg.warmup_tokens)

    # Cosine decay
    decay_tokens = cfg.max_tokens - cfg.warmup_tokens
    tokens_into_decay = tokens_seen - cfg.warmup_tokens

    if tokens_into_decay >= decay_tokens:
        return cfg.min_lr

    # Cosine from 1.0 to 0.0, scaled between learning_rate and min_lr
    progress = tokens_into_decay / decay_tokens
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * coeff


#  Validation


@torch.no_grad()
def validate(model: Gemma3Model, val_loader, cfg: TrainConfig, device: torch.device) -> float:
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    count = 0

    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.val_batches:
            break
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_loss += loss.item()
        count += 1

    model.train()
    return total_loss / max(count, 1)


#  Checkpointing


def save_checkpoint(
    model: Gemma3Model,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    val_loss: float,
    cfg: TrainConfig,
    is_best: bool = False,
):
    """Save model and optimizer state."""
    run_dir = os.path.join(cfg.checkpoint_dir, cfg.wandb_run_name or "default")
    os.makedirs(run_dir, exist_ok=True)

    maybe_orig = getattr(model, "_orig_mod", None)
    model_to_save: nn.Module = maybe_orig if isinstance(maybe_orig, nn.Module) else model

    state = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "val_loss": val_loss,
        "config": vars(cfg),
    }

    path = os.path.join(run_dir, f"step_{step}_tokens_{tokens_seen}.pt")
    torch.save(state, path)
    print(f"  Saved checkpoint: {path}")

    if is_best:
        best_path = os.path.join(run_dir, "best.pt")
        torch.save(state, best_path)
        print(f"  Saved best checkpoint: {best_path}")


#  Training Loop


def train():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    cfg = TrainConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    model_config = ModelConfig()
    model = Gemma3Model(model_config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e6:.1f}M")

    # Data
    train_loader = create_dataloader(
        cfg.train_bin,
        cfg.seq_len,
        cfg.micro_batch_size,
    )
    val_loader = create_dataloader(
        cfg.val_bin,
        cfg.seq_len,
        cfg.micro_batch_size,
    )

    # Tokenizer (for generation only)
    sp = spm.SentencePieceProcessor()
    sp.load(cfg.tokenizer_path)

    # Optimizer
    # Separate weight decay: don't decay biases, norms, embeddings
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "norm" in name or "embed" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
    )

    print(f"Optimizer groups: {len(decay_params)} decay, {len(no_decay_params)} no-decay")
    print(
        f"Effective batch size: {cfg.effective_batch_size} ({cfg.micro_batch_size} × {cfg.gradient_accumulation_steps})"
    )
    print(f"Tokens per optimizer step: {cfg.tokens_per_step:,}")

    # Resume from checkpoint
    start_step = 0
    start_tokens = 0
    if cfg.resume_from is not None:
        print(f"Resuming from {cfg.resume_from}")
        ckpt = torch.load(cfg.resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        start_tokens = ckpt["tokens_seen"]
        print(f"  Resumed at step {start_step}, tokens {start_tokens / 1e6:.1f}M")

    model = cast(Gemma3Model, torch.compile(model))

    #  Wandb (optional)
    use_wandb = (
        HAS_WANDB
        and bool(cfg.wandb_project and str(cfg.wandb_project).strip())
        and bool(cfg.wandb_run_name and str(cfg.wandb_run_name).strip())
    )

    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            job_type="pretraining",
            name=cfg.wandb_run_name,
            config={
                "model": {k: str(v) if isinstance(v, (list, type)) else v for k, v in vars(model_config).items()},
                "train": {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not callable(v)},
                "total_params": total_params,
            },
        )
        wandb.define_metric("tokens_seen")
        wandb.define_metric("train/*", step_metric="tokens_seen")
        wandb.define_metric("val/*", step_metric="tokens_seen")
        wandb.define_metric("lr", step_metric="tokens_seen")
    else:
        print("W&B disabled: wandb not installed or missing wandb_project/wandb_run_name in TrainConfig.")

    # Training state
    step = start_step
    tokens_seen = start_tokens
    best_val_loss = float("inf")
    last_checkpoint_tokens = tokens_seen
    train_iter = iter(train_loader)

    model.train()

    print(f"\nStarting training for {cfg.max_tokens / 1e9:.0f}B tokens...")
    t0 = time.time()
    running_loss = 0.0
    micro_steps = 0

    while tokens_seen < cfg.max_tokens:
        optimizer.zero_grad(set_to_none=True)

        accum_loss = 0.0
        step_t0 = time.time()

        for micro_step in range(cfg.gradient_accumulation_steps):
            # Get next batch (restart iterator if exhausted)
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y)
                # Scale loss by accumulation steps so gradients average correctly
                scaled_loss = loss / cfg.gradient_accumulation_steps

            scaled_loss.backward()
            accum_loss += loss.item()
            tokens_seen += x.numel()
            micro_steps += 1

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

        # Update learning rate
        lr = get_lr(tokens_seen, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Optimizer step
        optimizer.step()
        step += 1

        avg_loss = accum_loss / cfg.gradient_accumulation_steps
        running_loss += avg_loss
        step_dt = time.time() - step_t0

        # Logging
        if step % cfg.log_interval == 0:
            avg_running = running_loss / cfg.log_interval
            tokens_per_sec = cfg.tokens_per_step / step_dt
            elapsed = time.time() - t0
            pct = tokens_seen / cfg.max_tokens * 100

            print(
                f"step {step:>6d} | tokens {tokens_seen / 1e6:>8.1f}M ({pct:.1f}%) | "
                f"loss {avg_running:.4f} | ppl {math.exp(min(avg_running, 20)):.1f} | "
                f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"{tokens_per_sec / 1000:.1f}k tok/s | {elapsed:.0f}s"
            )

            if use_wandb:
                wandb.log(
                    {
                        "tokens_seen": tokens_seen,
                        "train/loss": avg_running,
                        "train/perplexity": math.exp(min(avg_running, 20)),
                        "lr": lr,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/tokens_per_sec": tokens_per_sec,
                    }
                )

            running_loss = 0.0

        # Validation
        if step % cfg.val_interval == 0:
            val_loss = validate(model, val_loader, cfg, device)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"  ── val loss: {val_loss:.4f} | val ppl: {val_ppl:.1f}")

            if use_wandb:
                wandb.log(
                    {
                        "tokens_seen": tokens_seen,
                        "val/loss": val_loss,
                        "val/perplexity": val_ppl,
                    }
                )

            # Best checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            save_checkpoint(model, optimizer, step, tokens_seen, val_loss, cfg, is_best)

        # Sample generation
        if step % cfg.sample_interval == 0:
            samples = generate_text_samples(model, sp, prompts, device)
            table = wandb.Table(columns=["prompt", "generation"]) if use_wandb else None
            for prompt, text in zip(prompts, samples):
                print(f"  [{prompt}] → {text[:200]}")
                if table is not None:
                    table.add_data(prompt, text)
            if table is not None:
                wandb.log({"tokens_seen": tokens_seen, "samples": table})

        # Periodic checkpoint
        if tokens_seen - last_checkpoint_tokens >= cfg.checkpoint_interval_tokens:
            if step % cfg.val_interval != 0:  # avoid double-saving
                save_checkpoint(model, optimizer, step, tokens_seen, float("nan"), cfg)
            last_checkpoint_tokens = tokens_seen

    # End
    print(f"\nTraining complete. {tokens_seen / 1e9:.2f}B tokens in {(time.time() - t0) / 3600:.1f}h")
    save_checkpoint(model, optimizer, step, tokens_seen, best_val_loss, cfg, is_best=False)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
