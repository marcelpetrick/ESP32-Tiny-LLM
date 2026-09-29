# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Text layout, typo noise, and the dataset builder."""

import json
import random
from pathlib import Path

import pytest

from training.data import dataset
from training.data.dialogue import Sample, Turn
from training.data.noise import add_typos, typo
from training.data.textformat import SPECIAL_TOKENS, prompt_text, target_text
from training.tokenizer import train_bpe
from training.world import DeviceState


def _sample() -> Sample:
    hist = (
        Turn("turn on the fan", "setting the fan to level 1.", "fan=1", "on", "fan", "on", 0),
        Turn("hello", "hello. i look after the greenhouse.", None, "greet"),
    )
    final = Turn("switch it on", "<clarify> what should i turn on?", None, "it_on")
    return Sample(hist, DeviceState(), final)


def test_prompt_layout_with_and_without_history() -> None:
    s = _sample()
    full = prompt_text(s)
    assert full.startswith("<bos><U> turn on the fan</U><A> setting the fan to level 1.</A>")
    assert "<ACT> fan=1</ACT><U> hello</U><A> hello." in full
    assert full.endswith("err=0</S><U> switch it on</U><A>")
    assert prompt_text(s, history=0).startswith("<bos><S> t=22.0")
    assert prompt_text(s, history=1).startswith("<bos><U> hello</U>")


def test_target_layout() -> None:
    s = _sample()
    assert target_text(s.turn) == "<clarify> what should i turn on?</A><eos>"
    assert target_text(s.history[0]) == " setting the fan to level 1.</A><ACT> fan=1</ACT><eos>"
    assert SPECIAL_TOKENS[:3] == ("<pad>", "<bos>", "<eos>")


def test_typo_variants() -> None:
    rng = random.Random(0)
    variants = {typo("temperature", rng) for _ in range(50)}
    assert len(variants) > 5
    assert all(v != "temperature" for v in variants if len(v) == len("temperature") + 5)
    assert typo("fan", rng) == "fan"
    assert typo("a1b2c", rng) == "a1b2c"
    noisy = add_typos("please turn on the heater", random.Random(3), 2)
    assert noisy != "please turn on the heater"
    assert len(noisy.split()) == 5


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, object]]:
    out = tmp_path_factory.mktemp("data")
    return out, dataset.build(out, scale=0.002, ctx=128, vocab_size=600)


def test_build_writes_splits_and_manifest(built: tuple[Path, dict[str, object]]) -> None:
    out, manifest = built
    splits = manifest["splits"]
    assert isinstance(splits, dict)
    assert set(splits) == {"train", "val", "test_id", "heldout", "robust", "multiturn", "safety"}
    on_disk = json.loads((out / "manifest.json").read_text())
    assert on_disk["tokenizer_sha256"] == manifest["tokenizer_sha256"]
    for name in splits:
        lines = (out / f"{name}.jsonl").read_text().splitlines()
        assert lines
        record = json.loads(lines[0])
        assert len(record["prompt_ids"]) + len(record["target_ids"]) <= 128
        assert record["prompt"].startswith("<bos>")


def test_safety_split_only_has_safety_or_clarify(built: tuple[Path, dict[str, object]]) -> None:
    out, _ = built
    for line in (out / "safety.jsonl").read_text().splitlines():
        tags = set(json.loads(line)["tags"])
        assert tags & {"safety", "clarify"}


def test_encode_sample_truncates_history_or_gives_up() -> None:
    tok = train_bpe(dataset.plain_segments([_sample()]), SPECIAL_TOKENS, 400)
    record = dataset.encode_sample(tok, _sample(), ctx=70)
    assert record is not None
    assert record["history_kept"] in (0, 1)
    assert dataset.encode_sample(tok, _sample(), ctx=5) is None


def test_main_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = dataset.main(["--out", str(tmp_path), "--scale", "0.001", "--vocab", "500"])
    assert code == 0
    assert "vocab_size=" in capsys.readouterr().out


def test_build_with_teacher_adds_teacher_test_split(tmp_path: Path) -> None:
    bank = {
        "paraphrases": {
            "on|fan|": {"train": ["kindly power the blower"], "test": ["blower on, thanks"]}
        },
        "manifest": {"teacher": "fake"},
    }
    teacher_file = tmp_path / "paraphrases.json"
    teacher_file.write_text(json.dumps(bank))
    manifest = dataset.build(
        tmp_path / "out", scale=0.002, vocab_size=500, teacher_path=teacher_file
    )
    splits = manifest["splits"]
    assert isinstance(splits, dict)
    assert "teacher_test" in splits
    assert splits["train"]["p_teacher"] == 0.5
    assert manifest["teacher_sha256"]
    assert "fake" in str(manifest["teacher"])
    assert dataset.load_teacher(tmp_path / "missing.json")["paraphrases"] == {}


def test_build_reuses_a_given_tokenizer(tmp_path: Path) -> None:
    first = tmp_path / "a"
    dataset.build(first, scale=0.001, vocab_size=450)
    second = tmp_path / "b"
    dataset.main(
        ["--out", str(second), "--scale", "0.001", "--tokenizer", str(first / "tokenizer.json")]
    )
    assert (second / "tokenizer.json").read_text() == (first / "tokenizer.json").read_text()


def test_prompt_layout_with_fact() -> None:
    s = _sample()
    with_fact = Sample(s.history, s.state, s.turn, "the tank holds 20 litres.")
    text = prompt_text(with_fact, history=0)
    assert "</S><F> the tank holds 20 litres.</F><U> switch it on</U><A>" in text


def test_build_with_facts(tmp_path: Path) -> None:
    manifest = dataset.build(tmp_path / "f", scale=0.004, vocab_size=500, facts=True)
    splits = manifest["splits"]
    assert isinstance(splits, dict)
    assert manifest["facts"] is True
    assert "facts_heldout" in splits
    held = [json.loads(line) for line in (tmp_path / "f" / "facts_heldout.jsonl").open()]
    assert held
    assert all("<F>" in r["prompt"] and r["intent"] == "ask_fact" for r in held)
    train = (tmp_path / "f" / "train.jsonl").read_text()
    assert "<F>" in train
    for fact in ("two year warranty", "potting soil with", "insects once a week"):
        assert fact not in train  # held-out facts never reach training
    plain = _plain_tokenizer(tmp_path)
    with pytest.raises(ValueError, match="--facts"):
        dataset.build(tmp_path / "h", scale=0.001, tokenizer_path=plain, facts=True)


def _plain_tokenizer(tmp_path: Path) -> Path:
    path = tmp_path / "plain.json"
    train_bpe(["hello world"], SPECIAL_TOKENS, 300).save(path)
    return path
