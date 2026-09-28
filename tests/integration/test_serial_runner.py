# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""HIL benchmark runner, exercised through the host runtime (no board needed)."""

import json
from pathlib import Path

import pytest

from tools import serial_runner
from training import export


@pytest.fixture(scope="module")
def model(tiny_run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("hil") / "m.tllm"
    export.main(["--checkpoint", str(tiny_run / "best.pt"), "--out", str(out), "--dtype", "i8"])
    return out


def test_parse_events() -> None:
    text = 'hello\n@@{"event":"ok"}\r\n  @@{"event":"reply","text":"x"}\nnoise @@'
    assert serial_runner.parse_events(text) == [{"event": "ok"}, {"event": "reply", "text": "x"}]


def test_host_dry_run_renders_matrix(
    model: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out, js = tmp_path / "m.md", tmp_path / "m.json"
    assert (
        serial_runner.main(["--host-model", str(model), "--out", str(out), "--json", str(js)]) == 0
    )
    text = out.read_text()
    assert "host dry run" in text
    assert "| Decode |" in text
    assert "| KV cache bytes |" in text
    assert "| diagnose |" in text
    data = json.loads(js.read_text())
    assert data["bandwidth"] is None  # host has no /bandwidth command
    assert "Benchmark matrix" in capsys.readouterr().out


def test_matrix_with_board_events() -> None:
    case = {
        "prompt_tokens": 40,
        "gen_tokens": 20,
        "prefill_ms": 400.0,
        "decode_tok_s": 30.0,
        "first_token_ms": 410.0,
        "mean_token_ms": 33.0,
        "p95_token_ms": 40.0,
        "profile_us": {"head": 6000, "ffn": 12000, "attn_out": 3000},
    }
    results = {
        "boot": {
            "cpu_mhz": 240,
            "flash_bytes": 16 << 20,
            "psram_bytes": 8 << 20,
            "psram_mode": "octal",
            "psram_mhz": 80,
            "idf": "v5.5.1",
        },
        "model": {
            "model_id": "ab",
            "params": 1,
            "bytes": 2,
            "weights": "int8",
            "activations": "fp32",
            "kv_cache": "fp32",
            "vocab_size": 3,
            "ctx_len": 4,
            "n_layers": 5,
            "d_model": 6,
            "n_heads": 7,
            "n_kv_heads": 7,
            "d_ff": 8,
        },
        "memory": {"cold_arena": 9, "hot_arena": 10, "platform": "psram free 1"},
        "bandwidth": {"sram_mbs": 500.0, "psram_mbs": 45.0, "flash_mbs": 20.0},
        "gemv": {"mmacs_a32": 60.0, "mmacs_a8": 120.0},
        "cases": dict.fromkeys(serial_runner.CASES, case),
    }
    text = serial_runner.matrix(results, "devkit", measured=True)
    assert "— measured" in text
    assert "8 MB octal @ 80 MHz" in text
    assert "PSRAM 45.0" in text
    assert "| Prefill | 100.0 tok/s |" in text
    assert "0.100 ms" in text  # head: 6 ms over 60 tokens


def test_git_revision_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError

    monkeypatch.setattr("tools.serial_runner.subprocess.run", boom)
    assert serial_runner.git_revision() == "unknown"
