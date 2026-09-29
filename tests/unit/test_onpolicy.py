# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""On-policy correction (E5): leakage-free teacher bank, mining, dataset assembly."""

import json
from pathlib import Path
from typing import Any

import pytest

from training import onpolicy
from training.tokenizer import Tokenizer


class _FakeRuntime:
    """Answers every prompt with a fixed token list (the C runtime is not needed here)."""

    def __init__(self, _path: Path) -> None:
        self.calls = 0

    def __enter__(self) -> "_FakeRuntime":
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def generate(self, prompt: list[int], _max_new: int) -> list[int]:
        self.calls += 1
        return prompt[:1]


def test_fresh_bank_drops_known_train_and_test_lines() -> None:
    teacher = {"paraphrases": {"on|fan|": {"train": ["a"], "test": ["b"]}}, "manifest": None}
    new = {
        "paraphrases": {
            "on|fan|": {"train": ["a", "c"], "test": ["b", "d"]},
            "off|fan|": {"train": ["e"], "test": []},
        },
        "manifest": {"teacher": "t"},
    }
    bank = onpolicy.fresh_bank(teacher, new)
    assert bank["paraphrases"]["on|fan|"] == {"train": ["c", "d"], "test": []}
    assert bank["paraphrases"]["off|fan|"] == {"train": ["e"], "test": []}
    assert bank["manifest"] == {"teacher": "t"}


@pytest.fixture
def fake_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from training.data import dataset  # noqa: PLC0415

    data = tmp_path / "data"
    dataset.build(data, scale=0.001, vocab_size=400)
    monkeypatch.setattr(onpolicy, "Runtime", _FakeRuntime)

    class _Model:
        tokenizer = Tokenizer.load(data / "tokenizer.json")

    monkeypatch.setattr(onpolicy, "read_tllm", lambda _p: _Model())
    return tmp_path


def test_build_mines_every_wrong_answer(
    fake_tree: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data, out = fake_tree / "data", fake_tree / "out"
    teacher = fake_tree / "teacher.json"
    teacher.write_text(json.dumps({"paraphrases": {}, "manifest": None}))
    code = onpolicy.main(
        [
            "--model", "m.tllm", "--data", str(data), "--teacher", str(teacher),
            "--out", str(out), "--samples", "30", "--repeat", "2", "--replay", "5",
        ]
    )  # fmt: skip
    assert code == 0
    manifest: dict[str, Any] = json.loads((out / "manifest.json").read_text())
    assert manifest["graded"] > 0
    # the fake student answers garbage, so every action or fallback expectation is missed
    assert manifest["corrections"] > 0
    assert manifest["fresh_teacher_lines"] == 0
    rows = (out / "train.jsonl").read_text().splitlines()
    assert len(rows) == 2 * manifest["corrections"] + manifest["replay"]
    assert (out / "tokenizer.json").exists()
    assert '"corrections"' in capsys.readouterr().out


def test_build_with_fresh_round_counts_new_lines(fake_tree: Path) -> None:
    data = fake_tree / "data"
    teacher = fake_tree / "teacher.json"
    teacher.write_text(json.dumps({"paraphrases": {}, "manifest": None}))
    fresh = {"paraphrases": {"on|fan|": {"train": ["x", "y"], "test": ["z"]}}, "manifest": None}
    manifest = onpolicy.build(Path("m.tllm"), data, teacher, fake_tree / "o2", 10, 1, 1, 3, fresh)
    assert manifest["fresh_teacher_lines"] == 3


def test_main_with_fresh_seeds_asks_the_teacher(
    fake_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from training.data import teacher as teacher_mod  # noqa: PLC0415

    seen: dict[str, Any] = {}

    def fake_generate(ask: object, per_key: int, cache: Path, seeds: tuple[int, ...]) -> Any:
        seen["seeds"] = seeds
        return {"paraphrases": {}, "manifest": None}

    monkeypatch.setattr(teacher_mod, "generate", fake_generate)
    monkeypatch.setattr(teacher_mod, "ollama_ask", lambda *_a: None)
    teacher = fake_tree / "teacher.json"
    teacher.write_text(json.dumps({"paraphrases": {}, "manifest": None}))
    onpolicy.main(
        [
            "--model", "m.tllm", "--data", str(fake_tree / "data"), "--teacher", str(teacher),
            "--out", str(fake_tree / "o3"), "--samples", "5", "--fresh-seeds", "3", "4",
        ]
    )  # fmt: skip
    assert seen["seeds"] == (3, 4)
