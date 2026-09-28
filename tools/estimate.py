# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Bandwidth-roofline estimator behind docs/01-feasibility.md.

Usage: python -m tools.estimate [--bandwidth-mbs 45]

Decode reads every weight once per token, plus the KV cache up to the current position,
so ``tokens/s ≈ bandwidth / bytes read per token`` (docs/01 §3). Numbers are
**estimates**; the benchmark harness replaces them with measurements on hardware.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from training.model import ModelConfig

DEFAULT_BANDWIDTH = 45e6  # bytes/s, derived from published ESP32-S3 results (docs/01 §2)
EFFICIENCY_BAND = (0.5, 0.9)
Q4_BYTES_PER_WEIGHT = 0.5 + 2 / 32  # 4-bit values + one fp16 scale per 32 weights


@dataclass(frozen=True)
class Tier:
    """A named model configuration and its weight precision."""

    name: str
    cfg: ModelConfig
    bytes_per_weight: float = 1.0


TIERS = (
    Tier(
        "S",
        ModelConfig(
            vocab_size=512, ctx_len=32, n_layers=2, d_model=96, n_heads=4, n_kv_heads=4, d_ff=192
        ),
    ),
    Tier("M", ModelConfig(vocab_size=1024, ctx_len=64)),
    Tier("M2", ModelConfig(vocab_size=2048, ctx_len=128)),
    Tier(
        "L",
        ModelConfig(
            vocab_size=2048, ctx_len=128, n_layers=6, d_model=192, n_heads=6, n_kv_heads=6, d_ff=512
        ),
    ),
    Tier(
        "XL",
        ModelConfig(
            vocab_size=4096, ctx_len=256, n_layers=8, d_model=256, n_heads=8, n_kv_heads=8, d_ff=768
        ),
    ),
    Tier(
        "XL-Q4",
        ModelConfig(
            vocab_size=4096, ctx_len=256, n_layers=8, d_model=256, n_heads=8, n_kv_heads=8, d_ff=768
        ),
        Q4_BYTES_PER_WEIGHT,
    ),
)


@dataclass(frozen=True)
class Estimate:
    """Derived per-token costs of a tier."""

    params: int
    weight_bytes: float
    kv_cache_bytes: int
    bytes_per_token: float
    ceiling_tok_s: float
    head_share: float


def matrix_params(cfg: ModelConfig) -> int:
    """Weights read per decoded token: all layer matrices plus the tied output head."""
    d = cfg.d_model
    attn = 2 * d * d + 2 * d * cfg.kv_dim
    mlp = d * cfg.d_ff * (3 if cfg.mlp_type == "swiglu" else 2)
    return cfg.n_layers * (attn + mlp) + cfg.vocab_size * d


def estimate(tier: Tier, bandwidth: float = DEFAULT_BANDWIDTH) -> Estimate:
    """Estimate one tier; KV reads assume the average position ``ctx_len / 2`` (int8 cache)."""
    cfg = tier.cfg
    weight_bytes = matrix_params(cfg) * tier.bytes_per_weight
    kv_cache = cfg.n_layers * 2 * cfg.ctx_len * cfg.kv_dim
    kv_read = cfg.n_layers * 2 * (cfg.ctx_len // 2) * cfg.kv_dim
    per_token = weight_bytes + kv_read
    head = cfg.vocab_size * cfg.d_model * tier.bytes_per_weight
    return Estimate(
        cfg.param_count(),
        weight_bytes,
        kv_cache,
        per_token,
        bandwidth / per_token,
        head / per_token,
    )


def markdown_table(bandwidth: float = DEFAULT_BANDWIDTH) -> str:
    """The tier table as Markdown."""
    low, high = EFFICIENCY_BAND
    lines = [
        "| Tier | Params | Weight bytes | KV cache | Bytes/token | Ceiling tok/s | Expected tok/s | Head share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for tier in TIERS:
        e = estimate(tier, bandwidth)
        lines.append(
            f"| {tier.name} | {e.params / 1e6:.2f} M | {e.weight_bytes / 1e6:.2f} MB | {e.kv_cache_bytes // 1024} KB "
            f"| {e.bytes_per_token / 1e6:.2f} MB | {e.ceiling_tok_s:.1f} "
            f"| {e.ceiling_tok_s * low:.0f}-{e.ceiling_tok_s * high:.0f} "
            f"| {e.head_share:.0%} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bandwidth-mbs", type=float, default=DEFAULT_BANDWIDTH / 1e6)
    args = parser.parse_args(argv)
    print(markdown_table(args.bandwidth_mbs * 1e6))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
