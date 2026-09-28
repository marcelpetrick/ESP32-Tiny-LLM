# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Sweep bookkeeping: variants, Pareto frontier, SVG and Markdown rendering."""

import json
from pathlib import Path

import pytest

from tools import sweep


def test_variants_are_valid_configs() -> None:
    names = [v.name for v in sweep.VARIANTS]
    assert len(names) == len(set(names))
    for v in sweep.VARIANTS:
        cfg = v.config(1024)
        assert cfg.param_count() > 0
        assert "--n-layers" in v.args


def test_pareto_frontier() -> None:
    pts = [
        {"name": "a", "quality": 0.9, "tok_s": 10.0},
        {"name": "b", "quality": 0.8, "tok_s": 50.0},
        {"name": "c", "quality": 0.7, "tok_s": 20.0},  # dominated by b
        {"name": "d", "quality": 0.9, "tok_s": 10.0},  # tie with a: both kept
    ]
    assert sweep.pareto(pts) == {"a", "b", "d"}


def test_report_mode_renders(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "manifest.json").write_text(json.dumps({"vocab_size": 1024}))
    runs = tmp_path / "runs"
    runs.mkdir()
    for v, (h, t) in zip(sweep.VARIANTS[:3], [(0.9, 0.8), (0.7, 0.6), (0.95, 0.9)], strict=True):
        (runs / f"{v.name}.eval.json").write_text(
            json.dumps({"heldout": {"action_exact": h}, "teacher_test": {"action_exact": t}})
        )
    svg_path, md_path = tmp_path / "p.svg", tmp_path / "s.md"
    assert (
        sweep.main(
            [
                "--data",
                str(data),
                "--report",
                str(runs),
                "--svg",
                str(svg_path),
                "--markdown",
                str(md_path),
            ]
        )
        == 0
    )
    text = svg_path.read_text()
    assert text.startswith("<svg")
    assert "polyline" in text
    assert "6x128" in md_path.read_text()
    assert "● |" in capsys.readouterr().out
