# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Shared fixtures: a tiny generated dataset and a tiny trained model."""

from pathlib import Path

import pytest

from training.data import dataset
from training.model import ModelConfig
from training.tokenizer import Tokenizer
from training.train import TrainConfig, train

pytest_plugins = ["tests.web_fixtures"]


@pytest.fixture(scope="session")
def tiny_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A very small but complete dataset directory (all splits + tokenizer)."""
    out = tmp_path_factory.mktemp("tiny_data")
    dataset.build(out, scale=0.003, ctx=128, vocab_size=700)
    return out


@pytest.fixture(scope="session")
def tiny_run(tiny_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1-layer model trained for a few CPU steps; returns the run directory."""
    out = tmp_path_factory.mktemp("tiny_run")
    vocab = Tokenizer.load(tiny_data / "tokenizer.json").vocab_size
    cfg = ModelConfig(vocab_size=vocab, n_layers=1, d_model=32, n_heads=2, n_kv_heads=1, d_ff=64)
    train(
        tiny_data,
        out,
        cfg,
        TrainConfig(steps=6, batch_size=8, warmup=2, eval_every=3),
        "cpu",
        log=False,
    )
    return out
