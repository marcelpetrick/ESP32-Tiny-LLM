# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Decoder-only transformer in PyTorch — the reference the C runtime must match.

Pre-norm blocks with RMSNorm, multi-head attention with optional grouped/multi-query
K/V heads, a GELU (tanh approximation) or SwiGLU MLP, learned or rotary positions, and
tied input/output embeddings. The math mirrors ``runtime/src/transformer.c``; RoPE uses
llama2.c's adjacent-pair convention so its checkpoints convert 1:1.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F  # noqa: N812 - conventional alias
from torch import Tensor, nn

from training.config import MLP_TYPES, POS_TYPES, ModelConfig

__all__ = [
    "MLP_TYPES",
    "POS_TYPES",
    "Block",
    "ModelConfig",
    "RMSNorm",
    "TinyLM",
    "apply_rope",
    "rope_tables",
]


class RMSNorm(nn.Module):
    """Root-mean-square normalisation with a learned gain."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return out


def rope_tables(cfg: ModelConfig, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """cos/sin tables of shape ``(ctx_len, head_dim // 2)``."""
    half = cfg.head_dim // 2
    freqs = 1.0 / (
        cfg.rope_theta ** (torch.arange(0, half, device=device).float() * 2 / cfg.head_dim)
    )
    angles = torch.outer(torch.arange(cfg.ctx_len, device=device).float(), freqs)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate adjacent pairs ``(x[2i], x[2i+1])``; x is ``(B, H, T, head_dim)``."""
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


class Block(nn.Module):
    """One pre-norm decoder block."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.attn_norm = RMSNorm(d, cfg.norm_eps)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, cfg.kv_dim, bias=False)
        self.wv = nn.Linear(d, cfg.kv_dim, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.mlp_norm = RMSNorm(d, cfg.norm_eps)
        self.w1 = nn.Linear(d, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, d, bias=False)
        self.w3 = nn.Linear(d, cfg.d_ff, bias=False) if cfg.mlp_type == "swiglu" else None

    def forward(self, x: Tensor, rope: tuple[Tensor, Tensor] | None) -> Tensor:
        cfg = self.cfg
        b, t, _ = x.shape
        h = self.attn_norm(x)
        q = self.wq(h).view(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.wk(h).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.wv(h).view(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        if rope is not None:
            cos, sin = rope[0][:t], rope[1][:t]
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cfg.n_kv_heads != cfg.n_heads:
            repeat = cfg.n_heads // cfg.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.wo(att.transpose(1, 2).reshape(b, t, cfg.d_model))
        h = self.mlp_norm(x)
        m = (
            F.silu(self.w1(h)) * self.w3(h)
            if self.w3 is not None
            else F.gelu(self.w1(h), approximate="tanh")
        )
        out: Tensor = x + self.w2(m)
        return out


class TinyLM(nn.Module):
    """The full language model."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model) if cfg.pos_type == "learned" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.rope: tuple[Tensor, Tensor] | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """GPT-2-style init: N(0, 0.02), residual projections scaled by depth."""
        for name, param in self.named_parameters():
            if param.dim() < 2:
                continue
            std = 0.02
            if name.endswith(("wo.weight", "w2.weight")):
                std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
            nn.init.normal_(param, mean=0.0, std=std)

    def _rope(self, device: torch.device) -> tuple[Tensor, Tensor] | None:
        if self.cfg.pos_type != "rope":
            return None
        if self.rope is None or self.rope[0].device != device:
            self.rope = rope_tables(self.cfg, device)
        return self.rope

    def forward(self, idx: Tensor) -> Tensor:
        """Logits of shape ``(B, T, vocab)`` for token ids ``idx`` of shape ``(B, T)``."""
        _, t = idx.shape
        if t > self.cfg.ctx_len:
            raise ValueError(f"sequence length {t} exceeds ctx_len {self.cfg.ctx_len}")
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(t, device=idx.device))
        rope = self._rope(idx.device)
        for block in self.blocks:
            x = block(x, rope)
        x = self.final_norm(x)
        logits: Tensor = x @ self.tok_emb.weight.T
        return logits

    @torch.no_grad()
    def generate(
        self, prompt: list[int], max_new: int, stop_ids: tuple[int, ...] = ()
    ) -> list[int]:
        """Greedy decoding (reference for runtime tests); returns only new tokens."""
        ids = list(prompt)
        out: list[int] = []
        device = self.tok_emb.weight.device
        for _ in range(max_new):
            if len(ids) >= self.cfg.ctx_len:
                break
            logits = self(torch.tensor([ids], device=device))[0, -1]
            nxt = int(torch.argmax(logits).item())
            out.append(nxt)
            ids.append(nxt)
            if nxt in stop_ids:
                break
        return out
