"""
LoRA (Low-Rank Adaptation) for Gemma3Model.

Wraps existing nn.Linear layers in-place
Supports targeting individual projection matrices, including Q and V
slices of the fused qkv_proj layer.

Supported target names (passed as a list to inject_lora):
    "q"        — query slice of qkv_proj        [hidden, q_size]
    "k"        — key slice of qkv_proj          [hidden, kv_size]
    "v"        — value slice of qkv_proj         [hidden, kv_size]
    "o"        — o_proj                           [q_size, hidden]
    "gate"     — gate_proj in MLP                 [hidden, intermediate]
    "up"       — up_proj in MLP                   [hidden, intermediate]
    "down"     — down_proj in MLP                 [intermediate, hidden]


Usage:
    model = Gemma3Model(config)
    # load pretrained weights ...
    inject_lora(model, config, targets=["q", "k", "v", "o"], rank=16, alpha=32)
    # only lora params have requires_grad=True
    save_lora_checkpoint(model, optimizer, step, tokens_seen, path)
    load_lora_checkpoint(model, path, device)
"""

import math
from pathlib import Path
from typing import Any, List, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


# Core LoRA modules


class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with a low-rank adapter.

    The forward pass computes:
        y = x @ W.T + (x @ A.T) @ B.T * scale
    where scale = alpha / rank.

    A is initialized with kaiming uniform.
    B is initialized to zero so the adapter starts as identity.
    The base weight W is frozen (requires_grad=False).
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        # Freeze and store the original weight
        self.weight = linear.weight
        self.weight.requires_grad_(False)
        self.bias = linear.bias  # None currently in this model

        # Adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scale
        return base + lora

    def merge_weights(self) -> nn.Linear:
        """Return a plain nn.Linear with the adapter permanently merged."""
        merged_weight = self.weight.to(self.lora_A.device) + (self.lora_B @ self.lora_A) * self.scale
        linear = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=self.bias is not None)
        linear.weight = nn.Parameter(merged_weight)
        if self.bias is not None:
            linear.bias = nn.Parameter(self.bias.clone())
        return linear


class LoRAFusedQKV(nn.Module):
    """
    LoRA adapter for the fused qkv_proj layer.

    The base weight is frozen. Low-rank adapters are added only for the
    requested slices (q, k, v, or any subset).
    """

    def __init__(
        self,
        linear: nn.Linear,
        q_size: int,
        kv_size: int,
        rank: int,
        alpha: float,
        adapt_q: bool = True,
        adapt_k: bool = False,
        adapt_v: bool = True,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.q_size = q_size
        self.kv_size = kv_size
        self.adapt_q = adapt_q
        self.adapt_k = adapt_k
        self.adapt_v = adapt_v

        in_features = linear.in_features

        self.weight = linear.weight
        self.weight.requires_grad_(False)
        self.bias = linear.bias

        if adapt_q:
            self.lora_A_q = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B_q = nn.Parameter(torch.zeros(q_size, rank))
            nn.init.kaiming_uniform_(self.lora_A_q, a=math.sqrt(5))

        if adapt_k:
            self.lora_A_k = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B_k = nn.Parameter(torch.zeros(kv_size, rank))
            nn.init.kaiming_uniform_(self.lora_A_k, a=math.sqrt(5))

        if adapt_v:
            self.lora_A_v = nn.Parameter(torch.empty(rank, in_features))
            self.lora_B_v = nn.Parameter(torch.zeros(kv_size, rank))
            nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)

        if self.adapt_q:
            lora_q = F.linear(F.linear(x, self.lora_A_q), self.lora_B_q) * self.scale
            out[..., : self.q_size] = out[..., : self.q_size] + lora_q

        if self.adapt_k:
            k_start = self.q_size
            lora_k = F.linear(F.linear(x, self.lora_A_k), self.lora_B_k) * self.scale
            out[..., k_start : k_start + self.kv_size] = out[..., k_start : k_start + self.kv_size] + lora_k

        if self.adapt_v:
            v_start = self.q_size + self.kv_size
            lora_v = F.linear(F.linear(x, self.lora_A_v), self.lora_B_v) * self.scale
            out[..., v_start : v_start + self.kv_size] = out[..., v_start : v_start + self.kv_size] + lora_v

        return out


# Injection

VALID_TARGETS = {"q", "k", "v", "o", "gate", "up", "down"}


def _get_model_layers(model: nn.Module) -> nn.ModuleList:
    """Return model.layers as a validated ModuleList."""
    layers = getattr(model, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise TypeError("Expected model.layers to be an nn.ModuleList")
    return layers


def inject_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    targets: List[str],
) -> None:
    """
    Inject LoRA adapters into all transformer layers of model in-place.

    Freezes all base parameters first, then adds trainable adapter
    matrices only for the requested projection targets.

    Args:
        model:   Gemma3Model instance with pretrained weights loaded.
        rank:    LoRA rank (r). Typical values: 8, 16, 32.
        alpha:   LoRA alpha. Scale = alpha / rank. Typically 2× rank.
        targets: List of projection names to adapt. Any subset of:
                 ["q", "k", "v", "o", "gate", "up", "down"]
    """
    unknown = set(targets) - VALID_TARGETS
    if unknown:
        raise ValueError(f"Unknown LoRA targets: {unknown}. Valid: {VALID_TARGETS}")

    attn_targets = set(targets) & {"q", "k", "v", "o"}
    mlp_targets = set(targets) & {"gate", "up", "down"}

    # Freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Inject adapters layer by layer
    for layer_module in _get_model_layers(model):
        layer = cast(Any, layer_module)
        attn = cast(Any, layer.self_attn)
        mlp = cast(Any, layer.mlp)

        # Attention
        if attn_targets & {"q", "k", "v"}:
            adapt_q = "q" in attn_targets
            adapt_k = "k" in attn_targets
            adapt_v = "v" in attn_targets
            attn.qkv_proj = LoRAFusedQKV(
                linear=attn.qkv_proj,
                q_size=attn.q_size,
                kv_size=attn.kv_size,
                rank=rank,
                alpha=alpha,
                adapt_q=adapt_q,
                adapt_k=adapt_k,
                adapt_v=adapt_v,
            )

        if "o" in attn_targets:
            attn.o_proj = LoRALinear(attn.o_proj, rank=rank, alpha=alpha)

        # MLP
        if "gate" in mlp_targets:
            mlp.gate_proj = LoRALinear(mlp.gate_proj, rank=rank, alpha=alpha)

        if "up" in mlp_targets:
            mlp.up_proj = LoRALinear(mlp.up_proj, rank=rank, alpha=alpha)

        if "down" in mlp_targets:
            mlp.down_proj = LoRALinear(mlp.down_proj, rank=rank, alpha=alpha)

    # Report
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA injected | targets={targets} rank={rank} alpha={alpha}")
    print(f"  Trainable: {trainable:,} / {total:,} params ({100 * trainable / total:.2f}%)")


# Checkpoint helpers


def _lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the LoRA adapter parameters from the model."""
    return {k: v for k, v in model.state_dict().items() if "lora_A" in k or "lora_B" in k}


def save_lora_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    tokens_seen: int,
    val_loss: float,
    path: str,
) -> None:
    """
    Save only the LoRA adapter weights + optimizer state.

    The base model weights are not saved, which allows for saving storage and iterating faster.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lora_state = _lora_state_dict(model)
    state: dict[str, Any] = {
        "lora": lora_state,
        "optimizer": optimizer.state_dict(),
        "step": step,
        "tokens_seen": tokens_seen,
        "val_loss": val_loss,
    }
    torch.save(state, path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"  Saved LoRA checkpoint: {path} ({size_mb:.1f} MB, {len(lora_state)} adapter tensors)")


def load_lora_checkpoint(
    model: nn.Module,
    path: str,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    """
    Load LoRA adapter weights into a model.

    inject_lora() must be called before this so the adapter parameter
    names exist in the model's state dict.

    Returns the checkpoint dict (for step/tokens_seen/val_loss).
    """
    ckpt: dict[str, Any] = torch.load(path, map_location=device, weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt["lora"], strict=False)

    # Flag unexpected keys
    lora_unexpected = [k for k in unexpected if "lora" in k]
    if lora_unexpected:
        print(f"  Warning: unexpected LoRA keys: {lora_unexpected}")

    lora_missing = [k for k in missing if "lora" in k]
    if lora_missing:
        raise RuntimeError(f"Missing LoRA keys in checkpoint: {lora_missing}")

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    print(
        f"  Loaded LoRA checkpoint: {path} "
        f"(step={ckpt.get('step', '?')}, "
        f"tokens={ckpt.get('tokens_seen', 0) / 1e6:.1f}M, "
        f"val_loss={ckpt.get('val_loss', float('nan')):.4f})"
    )
    return ckpt


def merge_lora_into_base(model: nn.Module) -> None:
    """
    Permanently merge LoRA adapters into base weights in-place.

    After merging, the model behaves identically to a full fine-tuned
    model with no inference overhead. Useful for deployment or when
    converting back to a standard checkpoint.
    """
    for layer_module in _get_model_layers(model):
        layer = cast(Any, layer_module)
        attn = cast(Any, layer.self_attn)
        mlp = cast(Any, layer.mlp)

        if isinstance(attn.qkv_proj, LoRAFusedQKV):
            lora = attn.qkv_proj
            with torch.no_grad():
                device = lora.lora_B_q.device if lora.adapt_q else lora.lora_B_v.device
                lora.weight.data = lora.weight.data.to(device)
                if lora.adapt_q:
                    delta_q = (lora.lora_B_q @ lora.lora_A_q) * lora.scale
                    lora.weight[: lora.q_size] += delta_q
                if lora.adapt_k:
                    k_start = lora.q_size
                    delta_k = (lora.lora_B_k @ lora.lora_A_k) * lora.scale
                    lora.weight[k_start : k_start + lora.kv_size] += delta_k
                if lora.adapt_v:
                    v_start = lora.q_size + lora.kv_size
                    delta_v = (lora.lora_B_v @ lora.lora_A_v) * lora.scale
                    lora.weight[v_start : v_start + lora.kv_size] += delta_v

            # Replace with a plain Linear using the merged weight
            merged = nn.Linear(lora.weight.shape[1], lora.weight.shape[0], bias=False)
            merged.weight = nn.Parameter(lora.weight)
            attn.qkv_proj = merged

        if isinstance(attn.o_proj, LoRALinear):
            attn.o_proj = attn.o_proj.merge_weights()

        if isinstance(mlp.gate_proj, LoRALinear):
            mlp.gate_proj = mlp.gate_proj.merge_weights()

        if isinstance(mlp.up_proj, LoRALinear):
            mlp.up_proj = mlp.up_proj.merge_weights()

        if isinstance(mlp.down_proj, LoRALinear):
            mlp.down_proj = mlp.down_proj.merge_weights()

    print("LoRA weights merged into base model.")
