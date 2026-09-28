# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Estimator (must reproduce docs/01 numbers) and model inspector."""

from pathlib import Path

import pytest

from tools import estimate, inspect_model
from training import export


def _tier(name: str) -> estimate.Tier:
    return next(t for t in estimate.TIERS if t.name == name)


@pytest.mark.parametrize(
    ("name", "params_m", "ceiling"),
    [
        ("M", 0.66, 65.4),
        ("M2", 0.80, 52.8),
        ("L", 2.48, 17.3),
        ("XL", 6.36, 6.6),
        ("XL-Q4", 6.36, 11.1),
    ],
)
def test_estimates_match_feasibility_doc(name: str, params_m: float, ceiling: float) -> None:
    e = estimate.estimate(_tier(name))
    assert e.params / 1e6 == pytest.approx(params_m, abs=0.01)
    assert e.ceiling_tok_s == pytest.approx(ceiling, abs=0.1)


def test_markdown_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    table = estimate.markdown_table()
    assert table.count("\n") == len(estimate.TIERS) + 1
    assert estimate.main(["--bandwidth-mbs", "90"]) == 0
    assert "| M |" in capsys.readouterr().out


def test_inspect_model(tiny_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "m.tllm"
    export.main(["--checkpoint", str(tiny_run / "best.pt"), "--out", str(out)])
    assert inspect_model.main([str(out)]) == 0
    text = capsys.readouterr().out
    assert "crc32" in text
    assert "l0.wq" in text
