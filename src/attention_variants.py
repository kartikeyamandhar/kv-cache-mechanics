"""KV cache tensor-shape and byte-size computation per model config.

Multi-head attention (MHA) stores one key and one value per attention head.
Grouped-query attention (GQA) shares each K/V across a group of query heads, so
only ``num_key_value_heads`` K/V projections are cached. The KV cache size is
therefore proportional to ``num_key_value_heads``, and GQA shrinks the cache by
the factor ``num_attention_heads / num_key_value_heads`` relative to an
otherwise identical MHA model.
"""
from dataclasses import dataclass

from transformers import AutoConfig

from . import config


@dataclass
class ModelDims:
    name: str
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int

    @property
    def attention_type(self) -> str:
        return "MHA" if self.num_key_value_heads == self.num_attention_heads else "GQA"

    @property
    def gqa_reduction_factor(self) -> float:
        """How many query heads share each cached K/V head (1.0 for pure MHA)."""
        return self.num_attention_heads / self.num_key_value_heads


def get_model_dims(model_name: str) -> ModelDims:
    """Extract head/layer dimensions from a model config.

    Handles both the standard names and gpt2's aliases, and infers a missing
    ``num_key_value_heads`` (absent on MHA models) as ``num_attention_heads``.
    """
    cfg = AutoConfig.from_pretrained(model_name)
    n_layers = getattr(cfg, "num_hidden_layers", None) or cfg.n_layer
    n_heads = getattr(cfg, "num_attention_heads", None) or cfg.n_head
    hidden = getattr(cfg, "hidden_size", None) or cfg.n_embd
    head_dim = getattr(cfg, "head_dim", None) or hidden // n_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or n_heads
    return ModelDims(model_name, n_layers, n_heads, n_kv, head_dim)


def kv_bytes_per_token(dims: ModelDims, num_kv_heads: int | None = None,
                       dtype_bytes: int = config.DTYPE_BYTES) -> int:
    """KV cache bytes added per token. Factor 2 accounts for keys and values."""
    kv = dims.num_key_value_heads if num_kv_heads is None else num_kv_heads
    return 2 * dims.num_hidden_layers * kv * dims.head_dim * dtype_bytes


def mha_equivalent_bytes_per_token(dims: ModelDims,
                                   dtype_bytes: int = config.DTYPE_BYTES) -> int:
    """Bytes/token this model *would* cache under full MHA (kv heads = attn heads)."""
    return kv_bytes_per_token(dims, num_kv_heads=dims.num_attention_heads,
                              dtype_bytes=dtype_bytes)
