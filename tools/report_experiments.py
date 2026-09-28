# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Compare experiment runs (docs/04 §4) in one Markdown table.

Usage: python -m tools.report_experiments runs/experiments [--out docs/results/experiments-table.md]

Reads ``<name>.json`` files written by ``training.eval --json`` and prints action accuracy
per suite and run, with the best value per suite in bold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUITES = ("test_id", "teacher_test", "heldout", "robust", "multiturn", "safety")
LABELS = {
    "e1-scratch": "E1 scratch, templates only",
    "e2-finetune": "E2 TinyStories prior + templates",
    "e3-teacher": "E3 teacher phrasing (shipped)",
    "e4-ta-kd": "E4 E3 + logit KD from tier L",
}


def table(runs: dict[str, dict[str, dict[str, float]]]) -> str:
    """Markdown table of action accuracy (rows = runs, columns = suites)."""
    best = {
        s: max((r[s]["action_exact"] for r in runs.values() if s in r), default=None)
        for s in SUITES
    }
    lines = ["| Run | " + " | ".join(SUITES) + " |", "|---|" + "---:|" * len(SUITES)]
    for name, result in runs.items():
        cells = []
        for s in SUITES:
            if s not in result:
                cells.append("—")
                continue
            value = result[s]["action_exact"]
            text = f"{100 * value:.1f} %"
            cells.append(f"**{text}**" if value == best[s] else text)
        lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def load(directory: Path) -> dict[str, dict[str, dict[str, float]]]:
    """All evaluation JSON files of a run directory, in E1..E4 order."""
    runs = {}
    for path in sorted(directory.glob("e*.json")):
        if path.name.endswith(".tllm.json"):  # model manifests, not evaluations
            continue
        runs[path.stem] = json.loads(path.read_text())
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    text = table(load(args.directory))
    print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
