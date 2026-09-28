# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""End-to-end evaluation of an exported model through the C runtime."""

import json
from pathlib import Path

import pytest

from training import eval as ev
from training import export


def test_evaluate_cli(
    tiny_run: Path, tiny_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "tiny.tllm"
    export.main(["--checkpoint", str(tiny_run / "best.pt"), "--out", str(model), "--dtype", "i8"])
    out_json, out_md = tmp_path / "r.json", tmp_path / "r.md"
    args = [
        "--model",
        str(model),
        "--data",
        str(tiny_data),
        "--splits",
        "test_id",
        "safety",
        "--limit",
        "3",
    ]
    assert ev.main([*args, "--act-int8", "--json", str(out_json), "--markdown", str(out_md)]) == 0
    assert "(W8A8)" in capsys.readouterr().out
    payload = json.loads(out_json.read_text())
    assert payload["test_id"]["n"] == 3
    assert "per_intent" in payload["safety"]
    assert out_md.read_text().startswith("### tiny.tllm")
