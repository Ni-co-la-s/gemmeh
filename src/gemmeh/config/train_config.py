"""Training configuration."""

import dataclasses


@dataclasses.dataclass
class TrainConfig:
    # Data
    train_bin: str = "data/tokenized/train.bin"
    val_bin: str = "data/tokenized/val.bin"
    tokenizer_path: str = "data/tokenizers/run_32k_1B/sentencepiece.model"

    # Model
    seq_len: int = 512

    # Optimization
    micro_batch_size: int = 2  # per-step batch size
    gradient_accumulation_steps: int = 16  # effective batch = micro_batch_size * gradient_accumulation_steps
    max_tokens: int = 2_000_000  # Number of total tokens seen in the run
    learning_rate: float = 3e-4  # peak LR
    min_lr: float = 3e-5  # min LR the schedule drops too
    warmup_tokens: int = 10_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0

    # Logging
    log_interval: int = 10  # log every N steps
    val_interval: int = 10  # validate every N steps
    val_batches: int = 100  # number of val batches to average
    sample_interval: int = 10  # generate samples every N steps
    checkpoint_interval_tokens: int = 500_000  # save every 500M tokens

    # Paths
    checkpoint_dir: str = "checkpoints"
    resume_from: str | None = None

    # Wandb
    wandb_project: str | None = "gemmeh-2024"
    wandb_run_name: str | None = "pretrain-1B-20B"  # auto-generated if None

    @property
    def effective_batch_size(self):
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_step(self):
        """Tokens consumed per optimizer step."""
        return self.effective_batch_size * self.seq_len
