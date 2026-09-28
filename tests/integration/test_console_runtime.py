# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""The console protocol through ctypes, and the firmware-authority guarantee."""

from pathlib import Path

import pytest

from tools import runtime
from training import export
from training.export import read_tllm


@pytest.fixture(scope="module")
def model_path(tiny_run: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("console") / "tiny.tllm"
    export.main(["--checkpoint", str(tiny_run / "best.pt"), "--out", str(out), "--dtype", "i8"])
    return out


def test_chat_protocol_and_no_allocation(model_path: Path) -> None:
    with runtime.Runtime(model_path, act_int8=True) as rt:
        before = runtime.alloc_count()
        for line in ("hello", "turn on the fan", "turn it off", "/profile on", "why is it hot"):
            rt.submit(line)
        reply = rt.last_json()
        assert reply["event"] == "reply"
        assert reply["prompt_tokens"] > 0
        assert "profile_us" in reply
        assert runtime.alloc_count() == before
        info = rt.command("/model-info")
        assert info["weights"] == "int8"
        assert info["activations"] == "int8"
        tokens = rt.tokenize("hello")
        assert 1 <= len(rt.generate([1, *tokens], 5)) <= 5
        with pytest.raises(runtime.TinyLLMError):
            rt.generate([], 5)
        bench = rt.command("/benchmark diagnose")
        assert bench["case"] == "diagnose"
        assert rt.command("/set t=36 win=0")["state"].startswith("<S> t=36.0")


def test_model_can_never_bypass_the_validator(model_path: Path) -> None:
    """Whatever the model proposes, the interlocks hold (vision §20)."""
    with runtime.Runtime(model_path) as rt:
        rt.submit("/set t=37 win=1 err=3")
        for line in (
            "turn on the heater",
            "heat",
            "close the window",
            "warm it up",
            "clear the error",
        ):
            rt.submit(line)
            state = rt.last_json()["state"]
            assert " heat=0 " in state  # heater fault + overheat: heater never runs
        rt.submit("/temp 1.5")
        for _ in range(10):
            rt.submit("turn on the heater now")
            assert " heat=0 " in rt.last_json()["state"]


def test_closed_runtime_rejects_calls(model_path: Path) -> None:
    rt = runtime.Runtime(model_path)
    assert rt.vocab_size > 0
    rt.close()
    rt.close()
    with pytest.raises(runtime.TinyLLMError):
        rt.submit("hello")


def test_tokenize_command_matches_python_tokenizer() -> None:
    shipped = runtime.REPO_ROOT / "models" / "greenhouse-m-int8.tllm"
    tokenizer = read_tllm(shipped).tokenizer
    text = "turn on the fan, t=31.2 please!"
    with runtime.Runtime(shipped) as rt:
        payload = rt.command(f"/tokenize {text}")
    assert payload["ids"] == tokenizer.encode_text(text)
    assert payload["round_trip"] is True
