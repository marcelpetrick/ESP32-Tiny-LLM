# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Training loop, schedule, masking and the run manifest."""

import json
from pathlib import Path

import pytest
import torch

from training import train as tr
from training.model import ModelConfig


def test_lr_schedule_warmup_and_decay() -> None:
    cfg = tr.TrainConfig(steps=100, lr=1.0, warmup=10, min_lr_ratio=0.1)
    assert tr.lr_at(0, cfg) == pytest.approx(0.1)
    assert tr.lr_at(9, cfg) == pytest.approx(1.0)
    assert tr.lr_at(99, cfg) == pytest.approx(0.1, abs=1e-3)
    assert tr.lr_at(55, cfg) < tr.lr_at(20, cfg)


def test_masked_loss_ignores_unmasked_positions() -> None:
    logits = torch.zeros(1, 3, 4)
    logits[0, 0, 1] = 10.0
    targets = torch.tensor([[1, 2, 3]])
    only_first = tr.masked_loss(logits, targets, torch.tensor([[1.0, 0.0, 0.0]]))
    assert float(only_first) < 0.01
    assert float(tr.masked_loss(logits, targets, torch.zeros(1, 3))) == 0.0


def test_load_split_masks_targets(tiny_data: Path) -> None:
    x, y, m = tr.load_split(tiny_data / "val.jsonl", 128)
    row = json.loads((tiny_data / "val.jsonl").read_text().splitlines()[0])
    n_prompt, n_target = len(row["prompt_ids"]), len(row["target_ids"])
    assert int(m[0].sum()) == n_target
    assert int(y[0, n_prompt - 1]) == row["target_ids"][0]
    assert x.shape == y.shape == m.shape


def test_tiny_run_manifest(tiny_run: Path) -> None:
    manifest = json.loads((tiny_run / "manifest.json").read_text())
    for key in (
        "git_revision",
        "dataset_manifest_sha256",
        "tokenizer_sha256",
        "model_config",
        "history",
        "train_config",
    ):
        assert key in manifest
    ckpt = torch.load(tiny_run / "best.pt", map_location="cpu", weights_only=False)
    assert ckpt["manifest"]["best_step"] in (3, 6)


def test_early_stopping_and_init_from(tiny_data: Path, tiny_run: Path, tmp_path: Path) -> None:
    ckpt = torch.load(tiny_run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_json(ckpt["config"])
    manifest = tr.train(
        tiny_data,
        tmp_path,
        cfg,
        tr.TrainConfig(steps=40, batch_size=4, lr=0.0, warmup=1, eval_every=1, patience=2),
        "cpu",
        init_from=tiny_run / "best.pt",
        log=False,
    )
    steps_run = manifest["steps_run"]
    assert isinstance(steps_run, int)
    assert steps_run < 40


def test_git_revision_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise OSError("no git")

    monkeypatch.setattr("training.train.subprocess.run", boom)
    assert tr.git_revision() == "unknown"


def test_cli(tiny_data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = tr.main(
        [
            "--data",
            str(tiny_data),
            "--out",
            str(tmp_path),
            "--preset",
            "tiny",
            "--steps",
            "2",
            "--eval-every",
            "1",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--mlp",
            "swiglu",
            "--pos",
            "rope",
            "--n-kv-heads",
            "1",
        ]
    )
    assert code == 0
    assert "best_val_loss" in capsys.readouterr().out


def test_masked_kl_is_zero_for_identical_and_positive_otherwise() -> None:
    a = torch.randn(2, 3, 5)
    mask = torch.ones(2, 3)
    assert float(tr.masked_kl(a, a, mask, 2.0)) == pytest.approx(0.0, abs=1e-6)
    assert float(tr.masked_kl(a, torch.randn(2, 3, 5), mask, 2.0)) > 0.0


def test_logit_distillation_run(tiny_data: Path, tiny_run: Path, tmp_path: Path) -> None:
    ckpt = torch.load(tiny_run / "best.pt", map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_json(ckpt["config"])
    manifest = tr.train(
        tiny_data,
        tmp_path,
        cfg,
        tr.TrainConfig(steps=2, batch_size=4, warmup=1, eval_every=1),
        "cpu",
        log=False,
        teacher_ckpt=tiny_run / "best.pt",
        kd_alpha=0.7,
        kd_temp=3.0,
    )
    assert manifest["distillation"] == {
        "teacher": str(tiny_run / "best.pt"),
        "alpha": 0.7,
        "temperature": 3.0,
        "loss": "CE + a*T^2*KL",
    }
