# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Experiment comparison table."""

import json
from pathlib import Path

import pytest

from tools import report_experiments as rep


def test_table_marks_best_and_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "e1-scratch.json").write_text(
        json.dumps({"heldout": {"action_exact": 0.7}, "test_id": {"action_exact": 0.99}})
    )
    (tmp_path / "e3-teacher.json").write_text(json.dumps({"heldout": {"action_exact": 0.88}}))
    (tmp_path / "e1-scratch.tllm.json").write_text("{}")  # manifest: ignored
    out = tmp_path / "t.md"
    assert rep.main([str(tmp_path), "--out", str(out)]) == 0
    text = out.read_text()
    assert "| E1 scratch, templates only | **99.0 %** |" in text
    assert "**88.0 %**" in text
    assert "—" in text
    assert "E3 teacher" in capsys.readouterr().out
