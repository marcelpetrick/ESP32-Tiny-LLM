# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""INT8 quantisation and the .tllm writer/reader."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from training import export
from training.data.textformat import SPECIAL_TOKENS
from training.model import Block, ModelConfig, TinyLM
from training.quantize import (
    Q4_GROUP,
    dequantize_q4,
    dequantize_rows,
    fake_quantize,
    quantize_q4,
    quantize_rows,
)
from training.tokenizer import Tokenizer, train_bpe


def test_quantize_rows_error_bound() -> None:
    rng = np.random.default_rng(0)
    w = rng.normal(size=(8, 32)).astype(np.float32)
    w[3] = 0.0
    q, s = quantize_rows(w)
    assert q.dtype == np.int8
    assert np.abs(q).max() <= 127
    assert s[3] == 1.0
    err = np.abs(dequantize_rows(q, s) - w)
    assert np.all(err <= s[:, None] / 2 + 1e-7)


def test_fake_quantize_changes_matrices_not_norms() -> None:
    torch.manual_seed(0)
    model = TinyLM(
        ModelConfig(
            vocab_size=40, ctx_len=8, n_layers=1, d_model=16, n_heads=2, n_kv_heads=2, d_ff=32
        )
    )
    fq = fake_quantize(model)
    block_fq, block = fq.blocks[0], model.blocks[0]
    assert isinstance(block_fq, Block)
    assert isinstance(block, Block)
    assert not torch.equal(block_fq.wq.weight, block.wq.weight)
    assert fq.pos_emb is not None
    assert model.pos_emb is not None
    assert torch.equal(fq.pos_emb.weight, model.pos_emb.weight)
    assert torch.equal(fq.final_norm.weight, model.final_norm.weight)


@pytest.fixture
def tokenizer_and_model() -> tuple[Tokenizer, TinyLM]:
    tok = train_bpe(["turn on the fan", "fan=2 t=21.5"] * 5, SPECIAL_TOKENS, 300)
    torch.manual_seed(3)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        ctx_len=16,
        n_layers=2,
        d_model=16,
        n_heads=4,
        n_kv_heads=2,
        d_ff=24,
        mlp_type="swiglu",
        pos_type="rope",
    )
    return tok, TinyLM(cfg).eval()


@pytest.mark.parametrize("dtype", ["f32", "i8"])
def test_write_read_round_trip(
    tmp_path: Path, tokenizer_and_model: tuple[Tokenizer, TinyLM], dtype: str
) -> None:
    tok, model = tokenizer_and_model
    path = tmp_path / "m.tllm"
    manifest = export.write_tllm(
        path, model.cfg, export.state_arrays(model), tok, dtype, b"\x01" * 16
    )
    data = path.read_bytes()
    assert data[:4] == b"TLLM"
    assert len(data) == manifest["file_bytes"]
    assert json.loads(path.with_suffix(".tllm.json").read_text())["weight_dtype"] == dtype
    parsed = export.read_tllm(path)
    assert parsed.cfg == model.cfg
    assert parsed.model_id == b"\x01" * 16
    assert isinstance(parsed.tokenizer, Tokenizer)
    assert parsed.tokenizer.merges == tok.merges
    tensors = manifest["tensors"]
    assert isinstance(tensors, list)
    for entry in tensors:
        assert entry["offset"] % 16 == 0
    rebuilt = export.load_model_from_tllm(parsed)
    idx = torch.tensor([[1, 5, 7, 9]])
    diff = (rebuilt(idx) - model(idx)).abs().max().item()
    assert diff < (1e-6 if dtype == "f32" else 0.05)


def test_reader_rejects_corruption(
    tmp_path: Path, tokenizer_and_model: tuple[Tokenizer, TinyLM]
) -> None:
    tok, model = tokenizer_and_model
    path = tmp_path / "m.tllm"
    export.write_tllm(path, model.cfg, export.state_arrays(model), tok)
    good = path.read_bytes()
    cases = {
        "short": good[:10],
        "magic": b"XXXX" + good[4:],
        "version": good[:4] + struct.pack("<I", 9) + good[8:],
        "payload": good + b"\0",
        "crc": good[:-1] + bytes([good[-1] ^ 0xFF]),
    }
    for expected, blob in cases.items():
        path.write_bytes(blob)
        with pytest.raises(export.TllmFormatError):
            export.read_tllm(path)
        del expected


def test_vocab_mismatch_is_rejected(
    tmp_path: Path, tokenizer_and_model: tuple[Tokenizer, TinyLM]
) -> None:
    tok, _model = tokenizer_and_model
    cfg = ModelConfig(
        vocab_size=999, ctx_len=16, n_layers=1, d_model=16, n_heads=4, n_kv_heads=4, d_ff=16
    )
    with pytest.raises(ValueError, match="tokenizer"):
        export.write_tllm(tmp_path / "x.tllm", cfg, {}, tok)


def test_export_cli_from_checkpoint(
    tiny_run: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "tiny.tllm"
    assert (
        export.main(["--checkpoint", str(tiny_run / "best.pt"), "--out", str(out), "--dtype", "i8"])
        == 0
    )
    assert "wrote" in capsys.readouterr().out
    manifest = json.loads(out.with_suffix(".tllm.json").read_text())
    assert manifest["training"]["best_step"] > 0
    assert export.read_tllm(out).weight_dtype == "i8"


def test_quantize_q4_round_trip_and_bounds() -> None:
    rng = np.random.default_rng(1)
    w = rng.normal(size=(4, 2 * Q4_GROUP)).astype(np.float32)
    w[1, :Q4_GROUP] = 0.0
    packed, scales = quantize_q4(w)
    assert packed.shape == (4, Q4_GROUP)
    assert scales.dtype == np.float16
    err = np.abs(dequantize_q4(packed, scales) - w).reshape(4, 2, Q4_GROUP)
    assert np.all(err <= scales.astype(np.float32)[:, :, None] / 2 + 1e-3)
    with pytest.raises(ValueError, match="multiple"):
        quantize_q4(np.zeros((2, 20), dtype=np.float32))


def test_q4_export_mixes_q4_and_int8(tmp_path: Path) -> None:
    tok = train_bpe(["turn on the fan"] * 5, SPECIAL_TOKENS, 300)
    torch.manual_seed(2)
    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        ctx_len=16,
        n_layers=1,
        d_model=32,
        n_heads=4,
        n_kv_heads=4,
        d_ff=48,
    )
    model = TinyLM(cfg).eval()
    path = tmp_path / "q4.tllm"
    manifest = export.write_tllm(path, cfg, export.state_arrays(model), tok, "q4")
    dtypes = {t["name"]: t["dtype"] for t in manifest["tensors"]}  # type: ignore[attr-defined]
    assert dtypes["l0.wq"] == "q4"
    assert dtypes["l0.w2"] == "i8"  # 48 columns: not a multiple of 32
    assert dtypes["pos_emb"] == "f32"
    parsed = export.read_tllm(path)
    assert parsed.weight_dtype == "q4"
    reference = fake_quantize(model, bits=4)
    idx = torch.tensor([[1, 5, 7, 9]])
    with torch.no_grad():
        diff = (export.load_model_from_tllm(parsed)(idx) - reference(idx)).abs().max().item()
    assert diff < 1e-3
