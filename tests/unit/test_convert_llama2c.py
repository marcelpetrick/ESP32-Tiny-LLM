# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Converting llama2.c checkpoints into .tllm."""

import struct
from pathlib import Path

import numpy as np
import pytest

from tools import convert_llama2c
from tools.runtime import REPO_ROOT
from training.export import read_tllm

STORIES = REPO_ROOT / "models" / "third_party" / "stories260K"


@pytest.mark.parametrize("dtype", ["f32", "i8"])
def test_convert_stories260k(
    tmp_path: Path, dtype: str, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "s.tllm"
    args = [
        str(STORIES / "stories260K.bin"),
        str(STORIES / "tok512.bin"),
        "--out",
        str(out),
        "--dtype",
        dtype,
    ]
    assert convert_llama2c.main(args) == 0
    assert "wrote" in capsys.readouterr().out
    model = read_tllm(out)
    cfg = model.cfg
    assert (cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads, cfg.d_ff) == (64, 5, 8, 4, 172)
    assert (cfg.vocab_size, cfg.ctx_len, cfg.mlp_type, cfg.pos_type) == (512, 512, "swiglu", "rope")
    assert model.weight_dtype == dtype


def test_first_tensors_match_checkpoint_bytes() -> None:
    cfg, tensors = convert_llama2c.read_checkpoint(STORIES / "stories260K.bin")
    raw = np.frombuffer(
        (STORIES / "stories260K.bin").read_bytes(), dtype="<f4", offset=28, count=64
    )
    np.testing.assert_array_equal(tensors["tok_emb"][0], raw)
    assert tensors["l4.w3"].shape == (cfg.d_ff, cfg.d_model)


def test_unshared_classifier_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unshared.bin"
    path.write_bytes(struct.pack("<7i", 8, 16, 1, 2, 2, -32, 16) + bytes(4096))
    with pytest.raises(ValueError, match="unshared"):
        convert_llama2c.read_checkpoint(path)
