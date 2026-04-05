"""
Decoder-only transformer inspired by Gemma 3, adapted for training.
The main difference is that here there is no sliding window-attention (and as
a consequence no differenciation in RoPE).
Reasons being that the context I used for training is a lot smaller (4096 vs 130k)
and that my attempt of implementing it had a throughput 30% lower than using simple
flash attention.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast

from gemmeh.config.model_config import ModelConfig


class RMSNorm(nn.Module):
    """RMSNorm with (1 + weight) scaling."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm * (1.0 + self.weight.float())).type_as(x)


class RotaryEmbedding(nn.Module):
    """Precomputes and applies RoPE."""

    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs)
        # Store cos and sin instead of complex numbers
        self.register_buffer("cos_cached", freqs.cos())  # [max_seq_len, dim//2]
        self.register_buffer("sin_cached", freqs.sin())

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        cos_cached = cast(torch.Tensor, self.cos_cached)
        sin_cached = cast(torch.Tensor, self.sin_cached)
        return cos_cached[:seq_len], sin_cached[:seq_len]


def apply_rotary_emb(
    x: torch.Tensor,  # [batch, n_heads, seq_len, head_dim]
    cos: torch.Tensor,  # [seq_len, head_dim//2]
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings using the real-number formulation."""
    head_dim = x.shape[-1]
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # Reshape cos/sin for broadcasting: [1, 1, seq_len, head_dim//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    out1 = x1 * cos - x2 * sin
    out2 = x2 * cos + x1 * sin
    return torch.cat([out1, out2], dim=-1)


class GemmaMLP(nn.Module):
    """GeGLU MLP: gate_proj and up_proj in parallel, then down_proj."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
        return cast(torch.Tensor, out)


class GemmaAttention(nn.Module):
    """
    Grouped-Query Attention with:
    - Optional QK-norm
    - RoPE
    - Causal masking (Only global attention, local attention not implemented)
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        # Fused QKV projection
        self.qkv_proj = nn.Linear(
            config.hidden_size,
            self.q_size + 2 * self.kv_size,
            bias=False,
        )
        self.o_proj = nn.Linear(self.q_size, config.hidden_size, bias=False)

        self.q_norm: RMSNorm | None
        self.k_norm: RMSNorm | None

        # QK normalization (Gemma 3 style, replaces logit soft-capping)
        if config.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

    def forward(
        self,
        x: torch.Tensor,  # [batch, seq_len, hidden_size]
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project Q, K, V
        qkv = self.qkv_proj(x)  # [batch, seq_len, q_size+2*kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK norm
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Expand KV heads for GQA
        if self.num_kv_heads != self.num_heads:
            k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
            v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Use Flash Attention via SDPA
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=True,
            scale=self.scaling,
        )

        out = out.transpose(1, 2).contiguous().view(B, T, self.q_size)
        return cast(torch.Tensor, self.o_proj(out))


class GemmaDecoderLayer(nn.Module):
    """
    Transformer block with sandwich norm (pre+post norm on both attention and FFN).
    """

    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.self_attn = GemmaAttention(config, layer_idx)
        self.mlp = GemmaMLP(config.hidden_size, config.intermediate_size)

        # Attention norms (sandwich: pre + post)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FFN norms (sandwich: pre + post)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_pre_ffw_norm else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if config.use_post_ffw_norm else None
        )

    def forward(self, x, cos, sin):
        # Attention with sandwich norm
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, cos, sin)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # FFN with sandwich norm
        residual = x
        if self.pre_feedforward_layernorm is not None:
            x = self.pre_feedforward_layernorm(x)
        x = self.mlp(x)
        if self.post_feedforward_layernorm is not None:
            x = self.post_feedforward_layernorm(x)
        x = residual + x
        return x


class Gemma3Model(nn.Module):
    """
    Gemma-3-inspired decoder-only transformer.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding (potentially shared with output projection via weight tying)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

        # Transformer layers
        self.layers = nn.ModuleList([GemmaDecoderLayer(config, i) for i in range(config.num_hidden_layers)])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output projection (optional with weight tying)
        self.lm_head: nn.Linear | None = None
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Embedding scale factor
        self.embed_scale = config.hidden_size**0.5

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """Initialize weights following common practices for transformers."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,  # [batch, seq_len]
        targets: torch.Tensor | None = None,  # [batch, seq_len]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            input_ids: Token IDs [batch, seq_len]
            targets: Target token IDs for loss computation [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: scalar cross-entropy loss (if targets provided)
        """
        B, T = input_ids.shape
        device = input_ids.device

        # Token embeddings scaled by sqrt(hidden_size)
        x = self.embed_tokens(input_ids) * self.embed_scale  # [batch, seq_len, hidden]

        # RoPE frequencies
        cos, sin = self.rotary_emb(T)
        cos = cos.to(device=device, dtype=x.dtype)  # [seq_len, head_dim//2]
        sin = sin.to(device=device, dtype=x.dtype)  # [seq_len, head_dim//2]

        # Transformer layers
        for layer in self.layers:
            x = layer(x, cos, sin)  # [batch, seq_len, hidden]

        # Final norm
        x = self.norm(x)  # [batch, seq_len, hidden]

        # Output projection
        if self.config.tie_word_embeddings:
            logits = F.linear(x, self.embed_tokens.weight)  # [batch, seq_len, vocab_size]
        else:
            assert self.lm_head is not None
            logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config.vocab_size),  # [batch*seq_len, vocab_size]
                targets.reshape(-1),  # [batch*seq_len]
                ignore_index=-1,
            )

        return logits, loss

    def count_parameters(self) -> dict:
        """Count parameters by component for verification."""
        counts = {}

        # Embedding
        counts["embed_tokens"] = sum(p.numel() for p in self.embed_tokens.parameters())

        # lm_head counted if there is no weight tying
        if self.lm_head is not None:
            counts["lm_head"] = sum(p.numel() for p in self.lm_head.parameters())

        # Per-layer breakdown for first layer
        layer0 = self.layers[0]
        layer_params = sum(p.numel() for p in layer0.parameters())
        counts["per_layer"] = layer_params
        counts["all_layers"] = sum(sum(p.numel() for p in layer.parameters()) for layer in self.layers)

        # Final norm
        counts["final_norm"] = sum(p.numel() for p in self.norm.parameters())

        # Total unique parameters (embedding counted once with tying)
        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts
