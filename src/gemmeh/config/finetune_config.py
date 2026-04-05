"""
Fine-tuning configuration for LoRA chat adaptation.

base_checkpoint should match the path of the finetuned model
The model config at src/config/model_config.py should match the one that was used for pretraining

LoRA target choices:
    "q"    — query slice of fused qkv_proj
    "v"    — value slice of fused qkv_proj
    "o"    — output projection (o_proj)
    "gate" — gate_proj in GeGLU MLP
    "up"   — up_proj in GeGLU MLP
    "down" — down_proj in GeGLU MLP
"""

import dataclasses


@dataclasses.dataclass
class FinetuneConfig:
    # Paths
    base_checkpoint: str = "checkpoints/pretrain-1B-20B/step_305176_tokens_20000014336.pt"
    data_path: str = "data/openhermes_raw/openhermes.json"
    tokenizer_path: str = "data/tokenizers/run_32k_1B/sentencepiece.model"
    output_dir: str = "checkpoints"

    # LoRA
    lora_rank: int = 16
    lora_alpha: float = 32.0  # scale = alpha / rank = 2.0
    lora_targets: list[str] = dataclasses.field(default_factory=lambda: ["q", "v", "o"])
    # Dropout on LoRA activations
    lora_dropout: float = 0.05

    # max_seq_len caps example length. Sequences are truncated to this.
    max_seq_len: int = 2048

    # Optimization
    micro_batch_size: int = 1  # per-step batch size
    gradient_accumulation_steps: int = 16  # effective batch = micro_batch_size * gradient_accumulation_steps
    max_tokens: int = 200_000_000  # total assistant tokens seen in the run
    learning_rate: float = 2e-4
    min_lr: float = 2e-5  # 10% of peak
    warmup_tokens: int = 100_000
    weight_decay: float = 0.01  # lighter than pretraining
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    use_gradient_checkpointing: bool = False

    # Data
    # Fraction of data reserved for validation (sampled before training)
    val_fraction: float = 0.02
    # If True, shuffle examples before train/val split
    shuffle_data: bool = True
    shuffle_seed: int = 42

    # Logging & checkpointing
    log_interval: int = 10
    val_interval: int = 200
    val_batches: int = 50
    sample_interval: int = 200
    checkpoint_interval_tokens: int = 2_000_000
    resume_lora_from: str | None = None

    # Wandb
    wandb_project: str | None = "gemmeh-2024"
    wandb_run_name: str | None = "finetune-1B-20B"  # auto-generated if None

    # Generation (for quality monitoring)
    generation_max_new_tokens: int = 200
    generation_temperature: float = 0.7
    generation_top_k: int = 50

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.max_seq_len
