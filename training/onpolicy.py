# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""On-policy correction (experiment E5, docs/04-distillation.md): mine the student's mistakes.

Usage::

    python -m training.onpolicy --model models/greenhouse-m-int8.tllm --data data/generated-vocab2048 \\
        --teacher data/teacher/paraphrases.json --out data/onpolicy [--samples 20000] [--seed 101]

The shipped model answers *fresh* dialogues (new seeds, training-side teacher phrasing —
never the held-back test paraphrases) through the C runtime. The rule-based oracle grades
every answer exactly; wrong answers (action, fallback or reply) are kept with the oracle's
correct target. The output dataset mixes these corrections (repeated) with a replay of the
original training data so fine-tuning fixes errors without forgetting (GKD / DAgger idea:
learn from your own mistakes, Agarwal et al. 2024).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

from tools.runtime import Runtime
from training.data import dataset
from training.data.dataset import SplitSpec, encode_sample, load_teacher
from training.eval import MAX_NEW, parse_output
from training.export import read_tllm
from training.tokenizer import Tokenizer


def mine(
    model: Path,
    tokenizer: Tokenizer,
    teacher: dict[str, Any],
    samples: int,
    seed: int,
    ctx: int = 128,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Records the student gets wrong (with the oracle target), plus counts."""
    spec = SplitSpec("mine", samples, seed, teacher_part="train", p_teacher=0.7, typo_p=0.1)
    decoder = read_tllm(model).tokenizer
    wrong: list[dict[str, Any]] = []
    stats = {"graded": 0, "wrong_action": 0, "wrong_fallback": 0, "wrong_reply": 0}
    with Runtime(model) as rt:
        for sample in dataset.generate(spec, teacher):
            record = encode_sample(tokenizer, sample, ctx)
            if record is None:
                continue
            stats["graded"] += 1
            out = rt.generate(list(record["prompt_ids"]), MAX_NEW)  # type: ignore[call-overload]
            pred = parse_output(decoder.decode(out), len(out) >= MAX_NEW)
            expected_fallback = next(
                (
                    m
                    for m in ("clarify", "unsupported")
                    if str(record["reply"]).startswith(f"<{m}>")
                ),
                None,
            )
            reasons = []
            if pred.action != record["action"]:
                reasons.append("wrong_action")
            if pred.fallback != expected_fallback:
                reasons.append("wrong_fallback")
            if not reasons and pred.malformed:
                reasons.append("wrong_reply")
            for reason in reasons:
                stats[reason] += 1
            if reasons:
                wrong.append(record)
    return wrong, stats


def fresh_bank(teacher: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """New teacher lines not already in the committed train *or test* paraphrases (no leakage)."""
    bank: dict[str, dict[str, list[str]]] = {}
    for key, parts in new["paraphrases"].items():
        known = set()
        for part in teacher["paraphrases"].get(key, {}).values():
            known.update(part)
        fresh = sorted({line for part in parts.values() for line in part} - known)
        bank[key] = {"train": fresh, "test": []}
    return {"paraphrases": bank, "manifest": new.get("manifest")}


def build(
    model: Path,
    data: Path,
    teacher_path: Path,
    out: Path,
    samples: int,
    seed: int,
    repeat: int = 4,
    replay: int = 60_000,
    fresh: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a fine-tuning dataset: corrections x repeat + replay of the original training set.

    With ``fresh`` (a new teacher round), mining uses only phrasing the student never saw.
    """
    tokenizer = Tokenizer.load(data / "tokenizer.json")
    teacher = load_teacher(teacher_path)
    bank = fresh_bank(teacher, fresh) if fresh is not None else teacher
    wrong, stats = mine(model, tokenizer, bank, samples, seed)
    rng = random.Random(seed)
    original = (data / "train.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.dumps(r) for r in wrong] * repeat + rng.sample(
        original, min(replay, len(original))
    )
    rng.shuffle(rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    for name in ("val.jsonl", "tokenizer.json"):
        shutil.copyfile(data / name, out / name)
    manifest = {
        "student": str(model),
        "seed": seed,
        "mined_samples": samples,
        **stats,
        "corrections": len(wrong),
        "fresh_teacher_lines": sum(len(p["train"]) for p in bank["paraphrases"].values())
        if fresh
        else 0,
        "repeat": repeat,
        "replay": min(replay, len(original)),
        "base_dataset_manifest": json.loads((data / "manifest.json").read_text()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--replay", type=int, default=60_000)
    parser.add_argument(
        "--fresh-seeds",
        type=int,
        nargs="*",
        default=[],
        help="new teacher round for unseen phrasing",
    )
    parser.add_argument("--cache", type=Path, default=Path("data/cache/teacher"))
    args = parser.parse_args(argv)
    fresh = None
    if args.fresh_seeds:
        from training.data import teacher as teacher_mod  # noqa: PLC0415 - optional, needs Ollama

        fresh = teacher_mod.generate(
            teacher_mod.ollama_ask("http://localhost:11434", teacher_mod.DEFAULT_MODEL),
            24,
            args.cache,
            tuple(args.fresh_seeds),
        )
    manifest = build(
        args.model,
        args.data,
        args.teacher,
        args.out,
        args.samples,
        args.seed,
        args.repeat,
        args.replay,
        fresh,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "base_dataset_manifest"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
