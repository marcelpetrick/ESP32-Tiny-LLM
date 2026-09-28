# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Print and verify a ``.tllm`` model file (header, tokenizer, tensors, CRC).

Usage: python -m tools.inspect_model MODEL.tllm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from training.export import read_tllm


def describe(path: Path) -> str:
    """Human-readable summary of a verified model file."""
    model = read_tllm(path)
    cfg = model.cfg
    lines = [
        f"file        {path} ({path.stat().st_size} bytes)",
        f"model id    {model.model_id.hex()}",
        f"crc32       {model.crc32:08x} (verified)",
        f"weights     {model.weight_dtype}",
        f"arch        {cfg.n_layers} layers x d_model {cfg.d_model}, heads {cfg.n_heads}/{cfg.n_kv_heads} kv, "
        f"d_ff {cfg.d_ff}, mlp {cfg.mlp_type}, pos {cfg.pos_type}",
        f"vocab/ctx   {cfg.vocab_size} / {cfg.ctx_len}",
        f"params      {cfg.param_count()}",
        f"tokenizer   {type(model.tokenizer).__name__}, {len(model.tokenizer.specials)} specials, "
        f"vocab {model.tokenizer.vocab_size}",
        "tensors:",
    ]
    for name, array in model.tensors.items():
        lines.append(
            f"  {name:<14} {'x'.join(map(str, array.shape)):>10}  absmax {float(np.abs(array).max()):.4f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    print(describe(parser.parse_args(argv).model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
