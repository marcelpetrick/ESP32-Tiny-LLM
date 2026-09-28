# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Convert a llama2.c checkpoint (legacy version-0 ``.bin``) + ``tokenizer.bin`` to ``.tllm``.

Usage::

    python -m tools.convert_llama2c models/third_party/stories260K/stories260K.bin \\
        models/third_party/stories260K/tok512.bin --out models/stories260K.tllm [--dtype i8]

The checkpoint layout is the one written by Andrej Karpathy's llama2.c ``export.py``
(MIT licence): 7 int32 config fields, then token embedding, per-layer attention norms,
wq, wk, wv, wo, FFN norms, w1, w2, w3, final norm, RoPE tables (skipped: our runtime
computes them), and an optional classifier when the vocab size is negative (unshared).
The model uses RoPE, SwiGLU and grouped-query attention — all supported by our runtime.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np

from training.export import write_tllm
from training.model import ModelConfig
from training.tokenizer.llama2c import ScoredTokenizer


def read_checkpoint(path: Path) -> tuple[ModelConfig, dict[str, np.ndarray]]:
    """Parse a llama2.c ``.bin`` into our config and tensor names."""
    data = path.read_bytes()
    dim, hidden, n_layers, n_heads, n_kv, vocab, seq_len = struct.unpack_from("<7i", data)
    if vocab < 0:
        raise ValueError("unshared classifier weights are not supported (our head is tied)")
    head = dim // n_heads
    kv_dim = n_kv * head
    floats = np.frombuffer(data, dtype="<f4", offset=28)
    pos = 0

    def take(*shape: int) -> np.ndarray:
        nonlocal pos
        count = int(np.prod(shape))
        out = floats[pos : pos + count].reshape(shape).copy()
        pos += count
        return out

    tensors: dict[str, np.ndarray] = {"tok_emb": take(vocab, dim)}
    rms_att = take(n_layers, dim)
    wq, wk, wv, wo = (
        take(n_layers, dim, dim),
        take(n_layers, kv_dim, dim),
        take(n_layers, kv_dim, dim),
        take(n_layers, dim, dim),
    )
    rms_ffn = take(n_layers, dim)
    w1, w2, w3 = (
        take(n_layers, hidden, dim),
        take(n_layers, dim, hidden),
        take(n_layers, hidden, dim),
    )
    tensors["final_norm"] = take(dim)
    for i in range(n_layers):
        tensors.update(
            {
                f"l{i}.attn_norm": rms_att[i],
                f"l{i}.wq": wq[i],
                f"l{i}.wk": wk[i],
                f"l{i}.wv": wv[i],
                f"l{i}.wo": wo[i],
                f"l{i}.mlp_norm": rms_ffn[i],
                f"l{i}.w1": w1[i],
                f"l{i}.w2": w2[i],
                f"l{i}.w3": w3[i],
            }
        )
    cfg = ModelConfig(
        vocab_size=vocab,
        ctx_len=seq_len,
        n_layers=n_layers,
        d_model=dim,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        d_ff=hidden,
        mlp_type="swiglu",
        pos_type="rope",
    )
    return cfg, tensors


def convert(checkpoint: Path, tokenizer: Path, out: Path, dtype: str = "f32") -> dict[str, object]:
    """Write ``out`` and return its manifest."""
    cfg, tensors = read_checkpoint(checkpoint)
    tok = ScoredTokenizer.from_llama2c(tokenizer, cfg.vocab_size)
    return write_tllm(out, cfg, tensors, tok, dtype)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=("f32", "i8"), default="f32")
    args = parser.parse_args(argv)
    manifest = convert(args.checkpoint, args.tokenizer, args.out, args.dtype)
    print(f"wrote {args.out} ({manifest['file_bytes']} bytes, {manifest['params']} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
