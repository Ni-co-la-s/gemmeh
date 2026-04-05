"""Model configuration for 1B model inspired by Gemma3 architecture."""

import dataclasses


@dataclasses.dataclass
class ModelConfig:
    # Architecture
    vocab_size: int = 32_768
    hidden_size: int = 1536
    num_hidden_layers: int = 30
    num_attention_heads: int = 8
    num_key_value_heads: int = 1
    head_dim: int = 256
    intermediate_size: int = 6144
    rms_norm_eps: float = 1e-6

    # Context
    max_position_embeddings: int = 4096

    # RoPE
    rope_theta: float = 10_000.0

    # Norms
    use_qk_norm: bool = True
    use_pre_ffw_norm: bool = True
    use_post_ffw_norm: bool = True

    # Embedding
    tie_word_embeddings: bool = True
