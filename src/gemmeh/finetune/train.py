"""
LoRA fine-tuning loop for chat adaptation.

Loads a pretrained Gemma3Model checkpoint, injects LoRA adapters,
and fine-tunes on OpenHermes instruction data.

Configuration of the model in  gemmeh.config.model_config.
Configuration of the training in  gemmeh.config.finetune_config

Key differences from gemmeh.pretrain.train:
  - Loads base weights from a pretrained checkpoint before training
  - Injects LoRA via inject_lora() — base weights stay frozen
  - Loss is masked: only assistant tokens contribute
  - Saves only the LoRA adapter weights (tiny checkpoints)

Usage:
    uv run -m gemmeh.finetune.train

Edit FinetuneConfig in finetune_config.py to change hyperparameters.
"""

import os
import time
import math

import sentencepiece as spm
import torch
import torch.nn.functional as F

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from gemmeh.config.model_config import ModelConfig
from gemmeh.config.finetune_config import FinetuneConfig
from gemmeh.finetune.data import create_chat_dataloaders
from gemmeh.finetune.lora import inject_lora, save_lora_checkpoint, load_lora_checkpoint
from gemmeh.model.gemma3 import Gemma3Model
from gemmeh.utils.generation import generate_text_samples

# Prompts used to test model outputs

prompts = [
    "<start_of_turn>user\nExplain what a transformer neural network is in simple terms.\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWrite a Python function that returns the nth Fibonacci number.\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat is the capital of Australia?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWrite a short poem about a frog.\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nAnswer in exactly one word: What is the capital of Japan?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nAnswer in one sentence: What is gravity?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat is the chemical formula for water?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nHow many planets are in the solar system?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nIf a shirt costs $20 and is 25% off, what is the final price?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nWhat comes next in the sequence: 1, 1, 2, 3, 5, 8, ?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nIf all cats are animals and all animals are living things, are all cats living things?\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nGive me a JSON object with name, age, and city fields.\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nGive me 3 tips to sleep better.\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nGive me a recipe for scrambled eggs.\n<end_of_turn>\n<start_of_turn>model\n",
    "<start_of_turn>user\nHow do I make a bomb?\n<end_of_turn>\n<start_of_turn>model\n",
]

#  LR Schedule


def get_lr(tokens_seen: int, cfg: FinetuneConfig) -> float:
    """Cosine schedule with linear warmup, based on assistant tokens seen."""
    if tokens_seen < cfg.warmup_tokens:
        return cfg.learning_rate * (tokens_seen / cfg.warmup_tokens)

    decay_tokens = cfg.max_tokens - cfg.warmup_tokens
    tokens_into_decay = tokens_seen - cfg.warmup_tokens
    if tokens_into_decay >= decay_tokens:
        return cfg.min_lr

    progress = tokens_into_decay / decay_tokens
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * coeff


#  Validation


@torch.no_grad()
def validate(
    model: Gemma3Model,
    val_loader,
    cfg: FinetuneConfig,
    model_cfg: ModelConfig,
    device: torch.device,
) -> float:
    """Return mean masked loss over val_batches batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i, (x, y, mask) in enumerate(val_loader):
        if i >= cfg.val_batches:
            break
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(x)

        loss = F.cross_entropy(
            logits.reshape(-1, model_cfg.vocab_size),
            y.reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += mask.sum().item()

    model.train()
    return total_loss / max(total_tokens, 1)


#  Training Loop


def finetune():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = FinetuneConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(cfg.tokenizer_path)
    print(f"Tokenizer: vocab={sp.get_piece_size()}, eos={sp.eos_id()}")

    # Data
    train_loader, val_loader, _ = create_chat_dataloaders(
        data_path=cfg.data_path,
        tokenizer=sp,
        max_seq_len=cfg.max_seq_len,
        micro_batch_size=cfg.micro_batch_size,
        val_fraction=cfg.val_fraction,
        shuffle=cfg.shuffle_data,
        seed=cfg.shuffle_seed,
    )

    print(f"Target assistant tokens: {cfg.max_tokens / 1e6:.1f}M")

    # Build model and load pretrained weights
    model_cfg = ModelConfig()
    model = Gemma3Model(model_cfg).to(device)

    print(f"Loading pretrained weights from {cfg.base_checkpoint} ...")
    ckpt = torch.load(cfg.base_checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    del ckpt
    print("  Pretrained weights loaded.")

    # Inject LoRA (freezes base, adds adapters)
    inject_lora(
        model=model,
        rank=cfg.lora_rank,
        alpha=cfg.lora_alpha,
        targets=cfg.lora_targets,
    )
    model = model.to(device).to(torch.bfloat16)

    # Sanity check
    dtypes = {p.dtype for p in model.parameters()}
    print(f"Parameter dtypes: {dtypes}")

    if cfg.use_gradient_checkpointing:
        print("  Gradient checkpointing enabled.")

    # Optimizer with only LoRA params
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    # Optional: resume from a LoRA checkpoint
    start_step = 0
    start_tokens = 0
    best_val_loss = float("inf")

    if cfg.resume_lora_from is not None:
        ckpt_lora = load_lora_checkpoint(model, cfg.resume_lora_from, device, optimizer)
        start_step = ckpt_lora.get("step", 0)
        start_tokens = ckpt_lora.get("tokens_seen", 0)
        best_val_loss = ckpt_lora.get("val_loss", float("inf"))

    #  Wandb (optional)
    run_name = cfg.wandb_run_name or (
        f"lora-r{cfg.lora_rank}-a{int(cfg.lora_alpha)}-{'_'.join(cfg.lora_targets)}-seq{cfg.max_seq_len}"
    )
    use_wandb = (
        HAS_WANDB
        and bool(cfg.wandb_project and str(cfg.wandb_project).strip())
        and bool(run_name and str(run_name).strip())
    )

    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=run_name,
            job_type="finetune",
            config={
                "model": {k: str(v) if isinstance(v, (list, type)) else v for k, v in vars(model_cfg).items()},
                "finetune": {k: v for k, v in vars(cfg).items() if not k.startswith("_") and not callable(v)},
                "total_params": sum(p.numel() for p in model.parameters()),
                "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
        )
        wandb.define_metric("tokens_seen")
        wandb.define_metric("train/*", step_metric="tokens_seen")
        wandb.define_metric("val/*", step_metric="tokens_seen")
        wandb.define_metric("lr", step_metric="tokens_seen")
    else:
        print("W&B disabled: wandb not installed or missing wandb_project/wandb_run_name in FinetuneConfig.")

    # Training
    model.train()
    step = start_step
    tokens_seen = start_tokens
    last_checkpoint_tokens = tokens_seen
    train_iter = iter(train_loader)

    print(f"\nStarting LoRA fine-tuning for {cfg.max_tokens / 1e6:.1f}M assistant tokens...")
    t0 = time.time()
    running_loss_sum = 0.0
    running_tokens = 0

    while tokens_seen < cfg.max_tokens:
        optimizer.zero_grad(set_to_none=True)

        accum_loss_sum = 0.0
        accum_tokens = 0
        successful_micro_steps = 0
        step_t0 = time.time()

        for _ in range(cfg.gradient_accumulation_steps):
            # Get next batch (restart iterator if exhausted)
            try:
                x, y, mask = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y, mask = next(train_iter)

            x, y = x.to(device), y.to(device)

            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(x)

                loss_sum = F.cross_entropy(
                    logits.reshape(-1, model_cfg.vocab_size),
                    y.reshape(-1),
                    ignore_index=-1,
                    reduction="sum",
                )
                n_tokens = mask.sum().item()
                loss = loss_sum / max(n_tokens, 1)

                del logits

                scaled_loss = loss / cfg.gradient_accumulation_steps
                scaled_loss.backward()
                del scaled_loss

            except torch.OutOfMemoryError:
                print(f"  ⚠ OOM on batch shape {x.shape}, skipping")
                torch.cuda.empty_cache()
                continue

            n_assistant_tokens = int(mask.sum().item())
            tokens_seen += n_assistant_tokens
            accum_tokens += n_assistant_tokens
            accum_loss_sum += loss_sum.item()
            successful_micro_steps += 1

            if tokens_seen >= cfg.max_tokens:
                break

        if successful_micro_steps == 0:
            optimizer.zero_grad(set_to_none=True)
            continue

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, cfg.max_grad_norm)

        # Update learning rate
        lr = get_lr(tokens_seen, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Optimizer step
        optimizer.step()
        step += 1

        running_loss_sum += accum_loss_sum
        running_tokens += accum_tokens
        step_dt = time.time() - step_t0

        # Logging
        if step % cfg.log_interval == 0:
            avg_running = running_loss_sum / max(running_tokens, 1)
            tok_per_sec = running_tokens / max(time.time() - t0, 1e-6)
            pct = tokens_seen / cfg.max_tokens * 100
            print(
                f"step {step:>6d} | tokens {tokens_seen / 1e6:>8.1f}M ({pct:.1f}%) | "
                f"loss {avg_running:.4f} | ppl {math.exp(min(avg_running, 20)):.1f} | "
                f"lr {lr:.2e} | grad {grad_norm:.2f} | "
                f"{tok_per_sec / 1000:.1f}k assistant tok/s | {step_dt:.2f}s"
            )
            if use_wandb:
                wandb.log(
                    {
                        "tokens_seen": tokens_seen,
                        "train/loss": avg_running,
                        "train/perplexity": math.exp(min(avg_running, 20)),
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/assistant_tokens_per_sec": tok_per_sec,
                        "lr": lr,
                    }
                )
            running_loss_sum = 0.0
            running_tokens = 0
            t0 = time.time()

        # Validation
        if step % cfg.val_interval == 0:
            val_loss = validate(model, val_loader, cfg, model_cfg, device)
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
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_path = os.path.join(cfg.output_dir, run_name, "best.pt")
                save_lora_checkpoint(model, optimizer, step, tokens_seen, val_loss, best_path)

        # Sample generation
        if step % cfg.sample_interval == 0:
            samples = generate_text_samples(
                model,
                sp,
                prompts,
                device,
                max_new_tokens=cfg.generation_max_new_tokens,
                temperature=cfg.generation_temperature,
                top_k=cfg.generation_top_k,
            )
            table = wandb.Table(columns=["prompt", "response"]) if use_wandb else None
            for prompt, text in zip(prompts, samples):
                short_prompt = prompt.split("\n")[1][:80]
                print(f"  [{short_prompt}] → {text[len(prompt) : len(prompt) + 200]}")
                if table is not None:
                    table.add_data(prompt, text)
            if table is not None:
                wandb.log({"tokens_seen": tokens_seen, "samples": table})

        # Periodic checkpoint
        if tokens_seen - last_checkpoint_tokens >= cfg.checkpoint_interval_tokens:
            if step % cfg.val_interval != 0:  # avoid double-saving
                ckpt_path = os.path.join(cfg.output_dir, run_name, f"step_{step}_tokens_{tokens_seen}.pt")
                save_lora_checkpoint(model, optimizer, step, tokens_seen, float("nan"), ckpt_path)
            last_checkpoint_tokens = tokens_seen

    # Final checkpoint
    final_path = os.path.join(cfg.output_dir, run_name, "final.pt")
    save_lora_checkpoint(model, optimizer, step, tokens_seen, best_val_loss, final_path)
    print(f"\nFine-tuning complete. {tokens_seen / 1e6:.1f}M assistant tokens seen.")
    print(f"Final adapter saved to {final_path}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    finetune()
