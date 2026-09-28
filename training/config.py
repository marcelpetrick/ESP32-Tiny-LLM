# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Model hyper-parameters (torch-free, so tools and the web image can use them)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

MLP_TYPES = ("gelu", "swiglu")
POS_TYPES = ("learned", "rope")


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyper-parameters (all stored in the ``.tllm`` header)."""

    vocab_size: int = 1024
    ctx_len: int = 128
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 256
    mlp_type: str = "gelu"
    pos_type: str = "learned"
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.mlp_type not in MLP_TYPES:
            raise ValueError(f"mlp_type must be one of {MLP_TYPES}")
        if self.pos_type not in POS_TYPES:
            raise ValueError(f"pos_type must be one of {POS_TYPES}")
        if self.pos_type == "rope" and self.head_dim % 2:
            raise ValueError("rope needs an even head_dim")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ModelConfig:
        return cls(**json.loads(text))

    def param_count(self) -> int:
        """Exact number of trainable parameters of :class:`TinyLM` with this config."""
        d, ff, L = self.d_model, self.d_ff, self.n_layers  # noqa: N806
        attn = d * d * 2 + d * self.kv_dim * 2
        mlp = d * ff * (3 if self.mlp_type == "swiglu" else 2)
        per_layer = attn + mlp + 2 * d
        pos = self.ctx_len * d if self.pos_type == "learned" else 0
        return self.vocab_size * d + pos + L * per_layer + d
