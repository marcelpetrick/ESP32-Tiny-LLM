# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Build tokenizer + dataset splits with a reproducibility manifest.

Usage::

    python -m training.data.dataset --out data/generated [--scale 1.0] [--ctx 128]

Splits (vision §19, docs/06 P5):

========== =============================================================
train      training frames, light politeness noise
val        same distribution, different seed (early stopping)
test_id    in-distribution test
teacher_test  natural phrasing from the teacher's held-back 20 % of paraphrases
heldout    held-out sentence frames never seen in training
robust     training frames with spelling typos and heavy noise
multiturn  every sample has history and a reference/ellipsis turn
safety     unsupported / out-of-range / interlock / clarification cases
========== =============================================================
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training import project_version
from training.data.dialogue import DialogueGenerator, Sample
from training.data.noise import add_typos
from training.data.textformat import SPECIAL_TOKENS, prompt_text, target_text
from training.tokenizer import Tokenizer, train_bpe

GENERATOR_VERSION = 1
_SPECIAL_SPLIT = re.compile(
    "|".join(re.escape(s) for s in sorted(SPECIAL_TOKENS, key=len, reverse=True))
)


@dataclass(frozen=True)
class SplitSpec:
    """How to generate one split."""

    name: str
    size: int
    seed: int
    heldout: bool = False
    noise: float = 0.15
    typos: int = 0
    force_reference: bool = False
    keep: Callable[[Sample], bool] | None = None
    teacher_part: str | None = None  # "train" or "test" paraphrases, None = templates only
    p_teacher: float = 0.0
    typo_p: float = 0.0


def _is_safety(sample: Sample) -> bool:
    return bool({"safety", "clarify"} & sample.turn.tags)


def default_splits(scale: float = 1.0) -> list[SplitSpec]:
    """The standard split set; ``scale`` multiplies all sizes (tests use tiny scales)."""

    def n(size: int) -> int:
        return max(4, int(size * scale))

    return [
        SplitSpec("train", n(200_000), 1, teacher_part="train", p_teacher=0.5, typo_p=0.1),
        SplitSpec("val", n(4_000), 2, teacher_part="train", p_teacher=0.5, typo_p=0.1),
        SplitSpec("test_id", n(3_000), 3, teacher_part="train", p_teacher=0.5, typo_p=0.1),
        SplitSpec("teacher_test", n(3_000), 8, noise=0.0, teacher_part="test", p_teacher=1.0),
        SplitSpec("heldout", n(3_000), 4, heldout=True, noise=0.0),
        SplitSpec("robust", n(3_000), 5, noise=0.6, typos=1),
        SplitSpec("multiturn", n(2_000), 6, force_reference=True),
        SplitSpec("safety", n(2_000), 7, keep=_is_safety),
    ]


def load_teacher(path: Path | None) -> dict[str, Any]:
    """Teacher paraphrase file (``training.data.teacher``), or an empty bank."""
    if path is None or not path.exists():
        return {"paraphrases": {}, "manifest": None}
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def generate(spec: SplitSpec, teacher: dict[str, Any] | None = None) -> list[Sample]:
    """Generate the samples of one split."""
    bank = {}
    if spec.teacher_part is not None and teacher is not None:
        bank = {key: parts[spec.teacher_part] for key, parts in teacher["paraphrases"].items()}
    gen = DialogueGenerator(
        spec.seed,
        heldout=spec.heldout,
        noise=spec.noise,
        teacher=bank,
        p_teacher=spec.p_teacher,
        typo_p=spec.typo_p,
    )
    typo_rng = random.Random(spec.seed * 7919)
    out: list[Sample] = []
    while len(out) < spec.size:
        sample = gen.sample(force_reference=spec.force_reference)
        if spec.keep is not None and not spec.keep(sample):
            continue
        if spec.typos:
            turn = sample.turn
            noisy = add_typos(turn.user, typo_rng, spec.typos)
            sample = Sample(
                sample.history, sample.state, type(turn)(**{**turn.__dict__, "user": noisy})
            )
        out.append(sample)
    return out


def plain_segments(samples: list[Sample]) -> list[str]:
    """Plain-text segments (special tokens removed) for tokenizer training."""
    segments: list[str] = []
    for sample in samples:
        full = prompt_text(sample) + target_text(sample.turn)
        segments.extend(part for part in _SPECIAL_SPLIT.split(full) if part)
    return segments


def encode_sample(tok: Tokenizer, sample: Sample, ctx: int) -> dict[str, object] | None:
    """Encode with as much history as fits into ``ctx`` tokens (None if nothing fits)."""
    target = tok.encode(target_text(sample.turn))
    for keep in range(len(sample.history), -1, -1):
        prompt_str = prompt_text(sample, history=keep)
        prompt = tok.encode(prompt_str)
        if len(prompt) + len(target) <= ctx:
            return {
                "prompt": prompt_str,
                "target": target_text(sample.turn),
                "prompt_ids": prompt,
                "target_ids": target,
                "intent": sample.turn.intent,
                "tags": sorted(sample.turn.tags),
                "action": sample.turn.action,
                "reply": sample.turn.reply,
                "history_kept": keep,
            }
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    out_dir: Path,
    scale: float = 1.0,
    ctx: int = 128,
    vocab_size: int = 1024,
    teacher_path: Path | None = None,
) -> dict[str, object]:
    """Generate all splits, train the tokenizer on ``train``, write JSONL + manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = default_splits(scale)
    teacher = load_teacher(teacher_path)
    if not teacher["paraphrases"]:
        specs = [spec for spec in specs if spec.name != "teacher_test"]
    samples = {spec.name: generate(spec, teacher) for spec in specs}
    tok = train_bpe(plain_segments(samples["train"]), SPECIAL_TOKENS, vocab_size)
    tok.save(out_dir / "tokenizer.json")
    train_prompts: set[str] = set()
    stats: dict[str, dict[str, int]] = {}
    for spec in specs:
        path = out_dir / f"{spec.name}.jsonl"
        written = dropped = duplicates = 0
        with path.open("w", encoding="utf-8") as handle:
            for sample in samples[spec.name]:
                record = encode_sample(tok, sample, ctx)
                if record is None:
                    dropped += 1
                    continue
                key = str(record["prompt"])
                if spec.name == "train":
                    train_prompts.add(key)
                elif key in train_prompts:
                    duplicates += 1
                    continue
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                written += 1
        stats[spec.name] = {
            "written": written,
            "dropped_too_long": dropped,
            "dropped_in_train": duplicates,
        }
    manifest: dict[str, object] = {
        "project_version": project_version(),
        "generator_version": GENERATOR_VERSION,
        "teacher": "rule-based oracle (training.world) for all labels; wording from "
        + (str(teacher["manifest"]["teacher"]) if teacher["manifest"] else "templates only"),
        "teacher_manifest": teacher["manifest"],
        "teacher_sha256": _sha256(teacher_path)
        if teacher_path is not None and teacher_path.exists()
        else None,
        "ctx": ctx,
        "vocab_size": tok.vocab_size,
        "splits": {
            spec.name: {
                "seed": spec.seed,
                "heldout_frames": spec.heldout,
                "noise": spec.noise,
                "typos": spec.typos,
                "teacher_part": spec.teacher_part,
                "p_teacher": spec.p_teacher,
                "typo_p": spec.typo_p,
                **stats[spec.name],
                "sha256": _sha256(out_dir / f"{spec.name}.jsonl"),
            }
            for spec in specs
        },
        "tokenizer_sha256": _sha256(out_dir / "tokenizer.json"),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--ctx", type=int, default=128)
    parser.add_argument("--vocab", type=int, default=1024)
    parser.add_argument(
        "--teacher", type=Path, default=None, help="paraphrases.json from training.data.teacher"
    )
    args = parser.parse_args(argv)
    manifest = build(args.out, args.scale, args.ctx, args.vocab, args.teacher)
    print(json.dumps(manifest["splits"], indent=2))
    print(f"vocab_size={manifest['vocab_size']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
