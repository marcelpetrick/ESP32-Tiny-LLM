# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Chat-lite small talk: teacher filtering, the bank file, generator and dataset wiring."""

import json
from pathlib import Path

import pytest

from training.data import dataset, smalltalk, teacher
from training.data.dialogue import DialogueGenerator, SmallTalk, render_reply
from training.world import DeviceState


@pytest.mark.parametrize(
    ("text", "ok"),
    [
        ("how are you doing today?", True),
        ("are you a robot", True),
        ("turn on the fan", False),  # a device command
        ("is the heater nice", False),  # device word
        ("hello", False),  # an existing intent frame
        ("is there a warranty", False),  # a fact keyword
        ("what is 2 plus 2", False),  # digits
        ("a b c d e f g h i j k l m n o", False),  # too long
    ],
)
def test_accept_user(text: str, ok: bool) -> None:
    assert smalltalk.accept_user(text) is ok


def test_accept_reply() -> None:
    assert smalltalk.accept_reply("i am a small computer program.")
    assert not smalltalk.accept_reply("ok.")  # too short
    assert not smalltalk.accept_reply("i will turn on the light for you.")  # promises an action
    assert not smalltalk.accept_reply("i am 3 years old.")  # numbers only from the state


def test_render_reply_uses_state_numbers() -> None:
    state = DeviceState(t=31.25, h=70, soil=40)
    assert (
        render_reply("{t} degrees, {h} percent, soil {soil}", state)
        == "31.2 degrees, 70 percent, soil 40"
    )
    assert render_reply("{t}/{h}", DeviceState(t=None, h=None)) == "na/na"


def _fake_ask(prompt: str, seed: int) -> str:
    if prompt.startswith("write 30"):
        return "\n".join(
            [f"- how are you number {w}" for w in ("one", "two", "three")]
            + ["turn on the fan", "1. are you ok"]
        )
    return "i am fine and happy.\nok\n2. i like the plants a lot."


def test_generate_bank_and_load(tmp_path: Path) -> None:
    bank = smalltalk.generate(_fake_ask, "fake", tmp_path / "cache", (1, 2))
    topics = bank["topics"]
    assert set(topics) == {t.name for t in smalltalk.TOPICS}
    how = topics["how_are_you"]
    assert "turn on the fan" not in how["train"] + how["test"]
    assert len(how["test"]) == 1
    assert len(how["train"]) == 3
    assert how["replies"] == ["i am fine and happy.", "i like the plants a lot."]
    assert "{t}" in " ".join(topics["how_is_it"]["replies"])  # grounded: code templates
    assert bank["manifest"]["teacher"] == "fake"
    path = tmp_path / "st.json"
    path.write_text(json.dumps(bank))
    train, test = smalltalk.load(path, "train"), smalltalk.load(path, "test")
    assert train["how_are_you"].replies == tuple(how["replies"])
    assert set(test["how_are_you"].users) == set(how["test"])


def test_main_writes_bank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(teacher, "ollama_ask", lambda *_a: _fake_ask)
    out = tmp_path / "bank.json"
    assert smalltalk.main(["--out", str(out), "--cache", str(tmp_path / "c")]) == 0
    assert "how_are_you" in json.loads(out.read_text())["topics"]
    assert "how_are_you" in capsys.readouterr().out


def test_generator_small_talk_turns() -> None:
    bank = {
        "hi": SmallTalk(("how are you",), ("i am fine.", "all good.")),
        "where": SmallTalk(("how is it",), ("it is {t} degrees.",)),
    }
    gen = DialogueGenerator(3, smalltalk=bank, p_smalltalk=1.0, noise=0.0)
    turns = [gen.sample().turn for _ in range(40)]
    talk = [t for t in turns if t.intent == "smalltalk"]
    assert talk
    for turn in talk:
        assert turn.action is None
        assert turn.reply in turn.alternatives
        assert "smalltalk" in turn.tags
    assert any("degrees" in t.reply and "{t}" not in t.reply for t in talk)


def test_dataset_with_chat(tmp_path: Path) -> None:
    bank = smalltalk.generate(_fake_ask, "fake", None, (1,))
    path = tmp_path / "st.json"
    path.write_text(json.dumps(bank))
    manifest = dataset.build(tmp_path / "d", scale=0.004, vocab_size=500, smalltalk_path=path)
    assert manifest["smalltalk_sha256"]
    rows = [json.loads(line) for line in (tmp_path / "d" / "smalltalk_test.jsonl").open()]
    assert rows
    assert all(r["intent"] == "smalltalk" and r["reply"] in r["allowed"] for r in rows)
    train = (tmp_path / "d" / "train.jsonl").read_text()
    assert '"smalltalk"' in train
    assert '"smalltalk"' not in (tmp_path / "d" / "heldout.jsonl").read_text()
