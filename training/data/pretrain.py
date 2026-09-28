# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Language-prior pretraining data from TinyStories (experiment E2, vision §8/§9 stage 1).

Usage::

    python -m training.data.pretrain --text data/cache/TinyStories-valid.txt \\
        --tokenizer data/generated/tokenizer.json --out data/pretrain [--max-tokens 4000000]

TinyStories (Eldan & Li 2023, CDLA-Sharing-1.0) is downloaded into ``data/cache`` and never
committed. Stories are lowercased (the assistant's input is lowercase), tokenized with the
*domain* tokenizer so the pretrained model can be fine-tuned directly, wrapped in
``<bos>``/``<eos>``, and packed into full-length training rows with loss on every token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path

from training.tokenizer import Tokenizer

LICENSE = "CDLA-Sharing-1.0 (roneneldan/TinyStories)"


def stories(text: str) -> list[str]:
    """Split the TinyStories text file into stories."""
    return [s.strip().lower() for s in text.split("<|endoftext|>") if s.strip()]


def pack(token_lists: list[list[int]], row_len: int) -> list[list[int]]:
    """Concatenate token lists and cut into rows of exactly ``row_len`` (last partial row dropped)."""
    flat = [t for tokens in token_lists for t in tokens]
    return [flat[i : i + row_len] for i in range(0, len(flat) - row_len + 1, row_len)]


def build(
    text_path: Path, tokenizer_path: Path, out: Path, ctx: int = 128, max_tokens: int = 4_000_000
) -> dict[str, object]:
    """Write ``train.jsonl``/``val.jsonl`` (prompt_ids empty, loss on all targets)."""
    tok = Tokenizer.load(tokenizer_path)
    bos, eos = tok.special_ids["<bos>"], tok.special_ids["<eos>"]
    items = stories(text_path.read_text(encoding="utf-8"))
    random.Random(0).shuffle(items)
    token_lists: list[list[int]] = []
    total = 0
    for story in items:
        ids = [bos, *tok.encode_text(" " + story), eos]
        token_lists.append(ids)
        total += len(ids)
        if total >= max_tokens:
            break
    rows = pack(token_lists, ctx + 1)
    n_val = max(1, len(rows) // 50)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("val", rows[:n_val]), ("train", rows[n_val:])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in part:
                handle.write(json.dumps({"prompt_ids": [], "target_ids": row}) + "\n")
    shutil.copyfile(tokenizer_path, out / "tokenizer.json")
    manifest: dict[str, object] = {
        "source": str(text_path.name),
        "source_sha256": hashlib.sha256(text_path.read_bytes()).hexdigest(),
        "license": LICENSE,
        "stories": len(token_lists),
        "tokens": total,
        "rows": {"train": len(rows) - n_val, "val": n_val},
        "ctx": ctx,
        "lowercased": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=4_000_000)
    args = parser.parse_args(argv)
    print(
        json.dumps(build(args.text, args.tokenizer, args.out, args.ctx, args.max_tokens), indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
