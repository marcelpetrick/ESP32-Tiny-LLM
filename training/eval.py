# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Evaluate a ``.tllm`` model **through the C runtime** on the generated suites (vision §19).

Usage::

    python -m training.eval --model models/greenhouse-m.tllm --data data/generated \\
        [--splits test_id heldout robust multiturn safety] [--limit 1000] [--act-int8] \\
        [--json out.json] [--markdown out.md]

Metrics per suite: exact action accuracy, actuator accuracy (ignoring ``diag=``),
fallback accuracy (``<clarify>``/``<unsupported>``), exact reply rate, number grounding
(every number in the reference reply appears in the prediction), malformed-output rate,
hallucinated-action rate (an action where none was expected), proposals rejected by the
firmware validator, and length violations. A deterministic keyword baseline is scored on
the same suites so the transformer has to beat something honest (vision §29).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from tools.runtime import Runtime
from training.export import read_tllm
from training.world import ActionParseError, parse_action, parse_state, validate_action

DEFAULT_SPLITS = ("test_id", "heldout", "robust", "multiturn", "safety")
MAX_NEW = 64
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Prediction:
    """Parsed model output for one sample."""

    reply: str
    action: str | None
    fallback: str | None
    malformed: bool
    hit_limit: bool


def parse_output(text: str, hit_limit: bool) -> Prediction:
    """Split generated text (after ``<A>``) into reply, action, fallback, malformed flag."""
    malformed = "</A>" not in text
    reply_part, _, rest = text.partition("</A>")
    fallback = None
    for marker in ("<clarify>", "<unsupported>"):
        if reply_part.startswith(marker):
            fallback = marker[1:-1]
            reply_part = reply_part[len(marker) :]
    action = None
    rest = rest.replace("<eos>", "")
    if rest:
        if rest.startswith("<ACT>") and "</ACT>" in rest:
            action = rest[len("<ACT>") : rest.index("</ACT>")].strip()
            try:
                parse_action(action)
            except ActionParseError:
                malformed = True
        else:
            malformed = True
    return Prediction(reply_part.strip(), action, fallback, malformed, hit_limit)


def actuators(action: str | None) -> str | None:
    """Action without informational ``diag=`` pairs (None if nothing actuates)."""
    if action is None:
        return None
    pairs = [p for p in action.split() if not p.startswith("diag=")]
    return " ".join(pairs) or None


def expected_fallback(reply: str) -> str | None:
    for marker in ("clarify", "unsupported"):
        if reply.startswith(f"<{marker}>"):
            return marker
    return None


# ----------------------------------------------------------------------------- baseline
_BASELINE_DEVICES = {  # window first: "skylight" contains "light"
    "win": ("window", "skylight", "vent"),
    "fan": ("fan", "ventilator", "blower"),
    "heat": ("heater", "heating", "radiator", "warmer"),
    "pump": ("pump", "irrigation", "sprinkler"),
    "light": ("light", "lamp"),
}


def baseline_action(user: str) -> str | None:
    """Keyword rules over the last user turn only (no state, no history)."""
    text = user.lower()
    device = next(
        (d for d, words in _BASELINE_DEVICES.items() if any(w in text for w in words)), None
    )
    if device is None:
        return None
    if device == "win":
        if "open" in text:
            return "win=1"
        if "close" in text or "shut" in text:
            return "win=0"
        return None
    if " off" in text or "stop" in text:
        return f"{device}=0"
    if " on" in text or "start" in text:
        return {"fan": "fan=1", "light": "light=100"}.get(device, f"{device}=1")
    return None


# ----------------------------------------------------------------------------- scoring
@dataclass
class SuiteResult:
    """Aggregated metrics of one suite."""

    n: int = 0
    counts: Counter[str] = field(default_factory=Counter)
    per_intent: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))

    def rate(self, key: str) -> float:
        return self.counts[key] / self.n if self.n else 0.0

    def summary(self) -> dict[str, Any]:
        keys = (
            "action_exact",
            "actuator_exact",
            "fallback_correct",
            "reply_exact",
            "numbers_grounded",
            "malformed",
            "hallucinated_action",
            "rejected_by_firmware",
            "length_violation",
            "baseline_action_exact",
        )
        return {"n": self.n, **{k: round(self.rate(k), 4) for k in keys}}


def score(record: dict[str, Any], pred: Prediction, result: SuiteResult) -> None:
    """Update ``result`` with one sample."""
    c = result.counts
    result.n += 1
    exp_action = record["action"]
    ok = {
        "action_exact": pred.action == exp_action,
        "actuator_exact": actuators(pred.action) == actuators(exp_action),
        "fallback_correct": pred.fallback == expected_fallback(record["reply"]),
        "malformed": pred.malformed,
        "hallucinated_action": exp_action is None and pred.action is not None,
        "length_violation": pred.hit_limit,
    }
    ok["reply_exact"] = pred.reply == _expected_reply(record["reply"]) and not pred.malformed
    numbers = _NUMBER.findall(record["reply"])
    ok["numbers_grounded"] = all(n in pred.reply for n in numbers)
    state_text = record["prompt"][
        record["prompt"].rindex("<S>") : record["prompt"].rindex("</S>") + 4
    ]
    if pred.action is not None and not pred.malformed:
        verdict = validate_action(parse_state(state_text), parse_action(pred.action))
        ok["rejected_by_firmware"] = not verdict.approved
    user = record["prompt"][record["prompt"].rindex("<U>") + 3 : record["prompt"].rindex("</U>")]
    ok["baseline_action_exact"] = baseline_action(user) == actuators(exp_action)
    for key, value in ok.items():
        if value:
            c[key] += 1
            result.per_intent[record["intent"]][key] += 1
    result.per_intent[record["intent"]]["n"] += 1


def _expected_reply(reply: str) -> str:
    for marker in ("<clarify>", "<unsupported>"):
        if reply.startswith(marker):
            return reply[len(marker) :].strip()
    return reply.strip()


def evaluate(
    model: Path,
    data: Path,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    limit: int | None = None,
    act_int8: bool = False,
) -> dict[str, SuiteResult]:
    """Run every sample of every split through the C runtime."""
    tokenizer = read_tllm(model).tokenizer
    results: dict[str, SuiteResult] = {}
    with Runtime(model, act_int8=act_int8) as rt:
        for split in splits:
            result = SuiteResult()
            lines = (data / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
            for line in lines[:limit]:
                record = json.loads(line)
                out = rt.generate(record["prompt_ids"], MAX_NEW)
                text = tokenizer.decode(out)
                score(
                    record,
                    parse_output(text, len(out) >= MAX_NEW and not text.endswith("<eos>")),
                    result,
                )
            results[split] = result
    return results


def markdown(results: dict[str, SuiteResult], title: str) -> str:
    """Results as a Markdown table (one row per suite)."""
    cols = [
        ("action_exact", "action"),
        ("actuator_exact", "actuators"),
        ("fallback_correct", "fallback"),
        ("reply_exact", "reply exact"),
        ("numbers_grounded", "numbers"),
        ("malformed", "malformed"),
        ("hallucinated_action", "halluc. action"),
        ("rejected_by_firmware", "rejected"),
        ("baseline_action_exact", "baseline"),
    ]
    lines = [f"### {title}", "", "| suite | n | " + " | ".join(h for _, h in cols) + " |"]
    lines.append("|---|---:|" + "---:|" * len(cols))
    for split, res in results.items():
        s = res.summary()
        lines.append(
            f"| {split} | {s['n']} | " + " | ".join(f"{100 * s[k]:.1f} %" for k, _ in cols) + " |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--act-int8", action="store_true")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args(argv)
    results = evaluate(args.model, args.data, tuple(args.splits), args.limit, args.act_int8)
    title = args.title or f"{args.model.name}{' (W8A8)' if args.act_int8 else ''}"
    table = markdown(results, title)
    print(table)
    if args.json:
        payload = {
            split: {
                **res.summary(),
                "per_intent": {k: dict(v) for k, v in sorted(res.per_intent.items())},
            }
            for split, res in results.items()
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n")
    if args.markdown:
        args.markdown.write_text(table)
    return 0


__all__ = [
    "Prediction",
    "SuiteResult",
    "asdict",
    "baseline_action",
    "evaluate",
    "main",
    "parse_output",
]


if __name__ == "__main__":
    raise SystemExit(main())
