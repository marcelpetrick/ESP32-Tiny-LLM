# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Architecture sweep and quality/speed Pareto frontier (vision M8).

Usage::

    python -m tools.sweep --data data/generated --out runs/sweep --steps 2500 \\
        [--python .venv-gpu/bin/python] [--only m-4x128,mqa]
    python -m tools.sweep --report runs/sweep --svg docs/results/pareto.svg --markdown docs/results/sweep.md

Each variant is trained with the same data, tokenizer and step budget, exported as INT8,
and evaluated through the C runtime on the held-out and teacher-paraphrase suites
(generalisation is what separates the variants). Device speed is the bandwidth-roofline
**estimate** of tools/estimate.py until measured on a board. The frontier marks variants
no other variant beats on both quality and estimated tok/s.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.estimate import DEFAULT_BANDWIDTH, Tier, estimate
from training.config import ModelConfig

QUALITY_SPLITS = ("heldout", "teacher_test")


@dataclass(frozen=True)
class Variant:
    """One sweep point: a name and training overrides."""

    name: str
    args: tuple[str, ...]
    n_layers: int = 4
    d_model: int = 128
    n_heads: int = 4
    n_kv_heads: int = 4
    d_ff: int = 256
    mlp: str = "gelu"
    pos: str = "learned"
    vocab: int | None = None  # None: the base dataset's tokenizer; else a retokenised copy

    def data_dir(self, base: Path) -> Path:
        """Dataset of this variant (identical samples, possibly another vocabulary size)."""
        return base if self.vocab is None else base.parent / f"{base.name}-vocab{self.vocab}"

    def config(self, vocab: int, ctx: int = 128) -> ModelConfig:
        return ModelConfig(
            vocab_size=vocab,
            ctx_len=ctx,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_ff=self.d_ff,
            mlp_type=self.mlp,
            pos_type=self.pos,
        )


def _v(name: str, **kw: Any) -> Variant:
    base: dict[str, Any] = {
        "n_layers": 4,
        "d_model": 128,
        "n_heads": 4,
        "n_kv_heads": 4,
        "d_ff": 256,
        "mlp": "gelu",
        "pos": "learned",
    }
    vocab = kw.pop("vocab", None)
    base.update(kw)
    args = (
        "--n-layers", str(base["n_layers"]), "--d-model", str(base["d_model"]), "--n-heads", str(base["n_heads"]),
        "--n-kv-heads", str(base["n_kv_heads"]), "--d-ff", str(base["d_ff"]), "--mlp", str(base["mlp"]),
        "--pos", str(base["pos"]),
    )  # fmt: skip
    return Variant(name, args, vocab=vocab, **base)


VARIANTS = (
    _v("m-4x128"),
    _v("2x128", n_layers=2),
    _v("6x128", n_layers=6),
    _v("4x96", d_model=96, d_ff=192),
    _v("4x160", d_model=160, n_heads=5, n_kv_heads=5, d_ff=320),
    _v("6x96-deep-thin", n_layers=6, d_model=96, d_ff=192),
    _v("8x80-deep-thin", n_layers=8, d_model=80, n_heads=4, n_kv_heads=4, d_ff=160),
    _v("mqa", n_kv_heads=1),
    _v("gqa", n_kv_heads=2),
    _v("swiglu", mlp="swiglu", d_ff=176),
    _v("rope", pos="rope"),
    _v("s-2x96", n_layers=2, d_model=96, d_ff=192),
    _v("vocab-512", vocab=512),
    _v("vocab-2048", vocab=2048),
)


def ensure_data(variant: Variant, base: Path) -> Path:  # pragma: no cover - builds a full dataset
    """Build the retokenised dataset of a vocabulary variant if missing (same seeds and samples)."""
    data = variant.data_dir(base)
    if not (data / "manifest.json").exists():
        teacher = Path("data/teacher/paraphrases.json")
        cmd = [
            sys.executable,
            "-m",
            "training.data.dataset",
            "--out",
            str(data),
            "--vocab",
            str(variant.vocab),
        ]
        subprocess.run([*cmd, "--teacher", str(teacher)], check=True, stdout=subprocess.DEVNULL)
    return data


def run_variant(
    variant: Variant, data: Path, out: Path, steps: int, python: str
) -> dict[str, Any]:  # pragma: no cover - GPU
    """Train, export and evaluate one variant (skips finished steps)."""
    run = out / variant.name
    data = ensure_data(variant, data)
    if not (run / "best.pt").exists():
        cmd = [
            python,
            "-m",
            "training.train",
            "--data",
            str(data),
            "--out",
            str(run),
            "--steps",
            str(steps),
            *variant.args,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    model = out / f"{variant.name}.tllm"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.export",
            "--checkpoint",
            str(run / "best.pt"),
            "--out",
            str(model),
            "--dtype",
            "i8",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    result_json = out / f"{variant.name}.eval.json"
    subprocess.run(
        [sys.executable, "-m", "training.eval", "--model", str(model), "--data", str(data), "--splits", *QUALITY_SPLITS,
         "--limit", "1000", "--json", str(result_json)],
        check=True,
        stdout=subprocess.DEVNULL,
    )  # fmt: skip
    return collect(variant, out, data)


def mean_reply_tokens(data: Path) -> float:
    """Average generated tokens per answer on the held-out suite (tokenizer dependent)."""
    lines = (data / "heldout.jsonl").read_text().splitlines()
    lengths = [len(json.loads(line)["target_ids"]) for line in lines]
    return sum(lengths) / max(1, len(lengths))


def collect(variant: Variant, out: Path, data: Path) -> dict[str, Any]:
    """Merge evaluation results with the size/speed estimate of one variant."""
    evaluation = json.loads((out / f"{variant.name}.eval.json").read_text())
    quality = sum(evaluation[s]["action_exact"] for s in QUALITY_SPLITS) / len(QUALITY_SPLITS)
    vocab = json.loads((data / "manifest.json").read_text())["vocab_size"]
    est = estimate(Tier(variant.name, variant.config(vocab)))
    reply_tokens = mean_reply_tokens(data)
    return {
        "name": variant.name,
        "params": est.params,
        "weight_bytes": est.weight_bytes,
        "tok_s": round(est.ceiling_tok_s, 1),
        "reply_tokens": round(reply_tokens, 1),
        "reply_ms": round(1000.0 * reply_tokens / est.ceiling_tok_s, 1),
        "quality": round(quality, 4),
        **{s: evaluation[s]["action_exact"] for s in QUALITY_SPLITS},
    }


def pareto(points: list[dict[str, Any]]) -> set[str]:
    """Names not dominated in (higher quality, lower estimated reply latency)."""
    front = set()
    for p in points:
        dominated = any(
            q["quality"] >= p["quality"]
            and q["reply_ms"] <= p["reply_ms"]
            and (q["quality"], q["reply_ms"]) != (p["quality"], p["reply_ms"])
            for q in points
        )
        if not dominated:
            front.add(p["name"])
    return front


def svg(points: list[dict[str, Any]], width: int = 720, height: int = 440) -> str:
    """Scatter plot (x = estimated ms per reply, y = quality) with the frontier highlighted."""
    front = pareto(points)
    pad_l, pad_r, pad_t, pad_b = 60, 150, 30, 50
    xs = [float(p["reply_ms"]) for p in points]
    ys = [100.0 * float(p["quality"]) for p in points]
    x0, x1 = 0.0, max(xs) * 1.1
    y0, y1 = max(0.0, min(ys) - 5), min(100.0, max(ys) + 3)

    def sx(x: float) -> float:
        return pad_l + (x - x0) / (x1 - x0) * (width - pad_l - pad_r)

    def sy(y: float) -> float:
        return height - pad_b - (y - y0) / (y1 - y0) * (height - pad_t - pad_b)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="system-ui, sans-serif" font-size="12">',
        "<style>.ax{stroke:#8a958e}.t{fill:#5d6b62}.pt{fill:#9fb5a7}.fr{fill:#2e7d4f}.ln{stroke:#2e7d4f;fill:none;stroke-width:1.5}"
        "@media (prefers-color-scheme: dark){.t{fill:#93a39a}.fr{fill:#5cc98a}.ln{stroke:#5cc98a}}</style>",
        f'<line class="ax" x1="{pad_l}" y1="{height - pad_b}" x2="{width - pad_r}" y2="{height - pad_b}"/>',
        f'<line class="ax" x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height - pad_b}"/>',
        f'<text class="t" x="{(pad_l + width - pad_r) / 2}" y="{height - 12}" text-anchor="middle">'
        f"estimated ESP32-S3 decode time per reply, ms (INT8, {DEFAULT_BANDWIDTH / 1e6:.0f} MB/s PSRAM)</text>",
        f'<text class="t" x="16" y="{(pad_t + height - pad_b) / 2}" text-anchor="middle" '
        f'transform="rotate(-90 16 {(pad_t + height - pad_b) / 2})">action accuracy, unseen phrasing (%)</text>',
    ]
    for i in range(6):
        yv = y0 + (y1 - y0) * i / 5
        parts.append(
            f'<text class="t" x="{pad_l - 8}" y="{sy(yv) + 4:.1f}" text-anchor="end">{yv:.0f}</text>'
        )
        xv = x0 + (x1 - x0) * i / 5
        parts.append(
            f'<text class="t" x="{sx(xv):.1f}" y="{height - pad_b + 18}" text-anchor="middle">{xv:.0f}</text>'
        )
    frontier = sorted((p for p in points if p["name"] in front), key=lambda p: p["reply_ms"])
    if len(frontier) > 1:
        path = " ".join(f"{sx(p['reply_ms']):.1f},{sy(100 * p['quality']):.1f}" for p in frontier)
        parts.append(f'<polyline class="ln" points="{path}"/>')
    boxes: list[tuple[float, float, float, float]] = []  # occupied rectangles (labels + points)
    for p in points:
        x, y = sx(p["reply_ms"]), sy(100 * p["quality"])
        boxes.append((x - 6, y - 6, x + 6, y + 6))

    def free(box: tuple[float, float, float, float]) -> bool:
        return all(box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3] for b in boxes)

    for p in sorted(points, key=lambda q: (q["name"] not in front, q["name"])):
        cls = "fr" if p["name"] in front else "pt"
        cx, cy = sx(p["reply_ms"]), sy(100 * p["quality"])
        parts.append(
            f'<circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="5"><title>{p["name"]}</title></circle>'
        )
        w = 7.0 * len(p["name"])
        # right, above-right, below-right, left, above-left, below-left (text baseline positions)
        candidates = [(cx + 8, cy + 4), (cx + 8, cy - 9), (cx + 8, cy + 17), (cx - 8 - w, cy + 4)]
        candidates += [(cx - 8 - w, cy - 9), (cx - 8 - w, cy + 17)]
        lx, ly = next(
            ((x, y) for x, y in candidates if free((x, y - 11, x + w, y + 2))), candidates[0]
        )
        boxes.append((lx, ly - 11, lx + w, ly + 2))
        parts.append(f'<text class="t" x="{lx:.1f}" y="{ly:.1f}">{p["name"]}</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def markdown(points: list[dict[str, Any]]) -> str:
    front = pareto(points)
    lines = [
        "| Variant | Params | INT8 weights | Est. tok/s | Tokens/reply | Est. ms/reply "
        "| Held-out | Teacher paraphrases | Mean | Pareto |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for p in sorted(points, key=lambda p: -p["quality"]):
        lines.append(
            f"| {p['name']} | {p['params'] / 1e6:.2f} M | {p['weight_bytes'] / 1e6:.2f} MB | {p['tok_s']:.0f} | "
            f"{p['reply_tokens']:.1f} | {p['reply_ms']:.0f} | "
            f"{100 * p['heldout']:.1f} % | {100 * p['teacher_test']:.1f} % | {100 * p['quality']:.1f} % | "
            f"{'●' if p['name'] in front else ''} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=Path("data/generated"))
    parser.add_argument("--out", type=Path, default=Path("runs/sweep"))
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--only", default=None, help="comma-separated variant names")
    parser.add_argument(
        "--report", type=Path, default=None, help="only render from an existing sweep directory"
    )
    parser.add_argument("--svg", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)
    variants = [v for v in VARIANTS if args.only is None or v.name in args.only.split(",")]
    if args.report is not None:
        points = [
            collect(v, args.report, v.data_dir(args.data))
            for v in variants
            if (args.report / f"{v.name}.eval.json").exists()
        ]
    else:  # pragma: no cover - GPU
        args.out.mkdir(parents=True, exist_ok=True)
        points = [run_variant(v, args.data, args.out, args.steps, args.python) for v in variants]
    table = markdown(points)
    print(table)
    if args.markdown:
        args.markdown.write_text(table)
    if args.svg:
        args.svg.write_text(svg(points))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
