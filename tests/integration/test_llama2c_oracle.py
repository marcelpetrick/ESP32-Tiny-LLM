# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""llama2.c as a permanent numerical oracle (vision M1, research item 1).

The vendored upstream run.c (MIT, third_party/llama2c) and our runtime load the same
stories260K checkpoint; greedy continuations must be identical, logits must match the
PyTorch reference, and the scored tokenizer must encode exactly like the Python mirror.
"""

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import convert_llama2c, runtime
from training.export import load_model_from_tllm, read_tllm
from training.tokenizer.llama2c import ScoredTokenizer

ROOT = runtime.REPO_ROOT
STORIES = ROOT / "models" / "third_party" / "stories260K"
PROMPTS = ["Once upon a time", "Lily and Ben went to the park", "The big dog"]


@pytest.fixture(scope="module")
def converted(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("llama2c") / "stories260K.tllm"
    convert_llama2c.convert(STORIES / "stories260K.bin", STORIES / "tok512.bin", out)
    return out


@pytest.fixture(scope="module")
def run_c(tmp_path_factory: pytest.TempPathFactory) -> Path:
    exe = tmp_path_factory.mktemp("oracle") / "run"
    subprocess.run(
        ["cc", "-O2", "-o", str(exe), str(ROOT / "third_party" / "llama2c" / "run.c"), "-lm"],
        check=True,
        capture_output=True,
    )
    return exe


def _upstream(run_c: Path, prompt: str, steps: int) -> str:
    result = subprocess.run(
        [
            str(run_c),
            str(STORIES / "stories260K.bin"),
            "-z",
            str(STORIES / "tok512.bin"),
            "-t",
            "0",
            "-n",
            str(steps),
            "-i",
            prompt,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip("\n")


def _ours(model: Path, prompt: str, max_tokens: int) -> str:
    cli = ROOT / "build" / "runtime-release" / "tinyllm-cli"
    result = subprocess.run(
        [str(cli), str(model), "-c", f"/max-tokens {max_tokens}", "-c", f"/generate {prompt}"],
        check=True,
        capture_output=True,
        text=True,
    )
    after_ok = result.stdout.split('@@{"event":"ok","what":"max tokens set"}\n', 1)[1]
    return after_ok.split("\n@@", 1)[0]


@pytest.mark.parametrize("prompt", PROMPTS)
def test_greedy_generation_matches_upstream_run_c(
    converted: Path, run_c: Path, prompt: str
) -> None:
    upstream = _upstream(run_c, prompt, 200)
    ours = _ours(converted, prompt, 400)
    n = min(len(upstream), len(ours))
    assert n > 300
    assert ours[:n] == upstream[:n]


def test_logits_match_pytorch_reference(converted: Path) -> None:
    tllm = read_tllm(converted)
    model = load_model_from_tllm(tllm)
    tok = tllm.tokenizer
    assert isinstance(tok, ScoredTokenizer)
    ids = [1, *tok.encode(PROMPTS[1])]
    with runtime.Runtime(converted) as rt:
        c_logits = np.array(rt.logits(ids))
        for text in [*PROMPTS, "café ☃!", "  two  spaces", ""]:
            assert rt.tokenize(text) == tok.encode(text), text
    with torch.no_grad():
        ref = model(torch.tensor([ids]))[0, -1].numpy()
    np.testing.assert_allclose(c_logits, ref, rtol=1e-4, atol=1e-4)
