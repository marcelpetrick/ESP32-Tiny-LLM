# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Teacher paraphrase generation: filtering, number normalisation, caching, generator use."""

import json
from pathlib import Path

import pytest

from training.data import teacher as tm
from training.data.dialogue import DialogueGenerator
from training.world import DeviceState


def spec(key: tm.Key) -> tm.KeySpec:
    return next(s for s in tm.key_specs() if s.key == key)


@pytest.mark.parametrize(
    ("key", "text", "ok"),
    [
        (("on", "fan", ""), "please turn on the blower", True),
        (("on", "fan", ""), "turn off the fan", False),  # meaning flipped
        (("on", "fan", ""), "turn on the fan and the lamp", False),  # other device
        (("on", "fan", ""), "give me a blast of air", False),  # device lost
        (("fan_set", "fan", "2"), "put the fan on level 2", True),
        (("fan_set", "fan", "2"), "put the fan on level 3", False),
        (("fan_set", "fan", "2"), "fan level two please", True),
        (("it_off", "", ""), "switch it off", True),
        (("it_off", "", ""), "switch the fan off", False),
        (("it_off", "", ""), "turn that down please", False),
        (("ood", "", ""), "who won the match", True),
        (("ood", "", ""), "turn on the heater", False),
        (("read_t", "", ""), "how warm is it now", True),
        (("read_t", "", ""), "how are you", False),
        (("status", "", ""), "status at 5", False),  # stray number
        (("on", "fan", ""), "café fan on", False),  # non-ascii
        (("on", "fan", ""), "get the fan going", False),  # held-out frame
        (("on", "fan", ""), " ".join(["fan"] * 15), False),  # too long
    ],
)
def test_accept(key: tm.Key, text: str, ok: bool) -> None:
    assert tm.accept(spec(key), text) is ok


def test_clean_and_numbers() -> None:
    assert tm.clean(' 1. "Turn ON the fan" ') == "turn on the fan"
    assert tm.clean("- open it") == "open it"
    assert tm.digits_for_number_words("set it to one hundred and twenty") == "set it to 120"
    assert tm.digits_for_number_words("thirty-five percent") == "35 percent"


def test_generate_with_fake_teacher(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake(prompt: str, seed: int) -> str:
        calls.append(prompt)
        return "\n".join(
            [
                "1. turn on the fan",
                "switch on the blower",
                "fan on please",
                "turn off the fan",
                "the fan now",
            ]
        )

    data = tm.generate(fake, per_key=5, cache=tmp_path, seeds=(1,))
    assert data["stats"]["accepted"] > 0
    on_fan = data["paraphrases"]["on|fan|"]
    assert on_fan["test"]
    assert on_fan["train"]
    assert "turn off the fan" not in on_fan["train"] + on_fan["test"]
    n_calls = len(calls)
    tm.generate(fake, per_key=5, cache=tmp_path, seeds=(1,))  # cached: no new calls
    assert len(calls) == n_calls
    tm.generate(fake, per_key=5, cache=tmp_path, seeds=(1,), model="another-teacher")
    assert len(calls) == 2 * n_calls  # a different teacher never reuses cached answers
    assert "every line must contain the number 2" in tm.prompt_for(
        spec(("fan_set", "fan", "2")), 5, ["x"]
    )


def test_main_writes_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        tm, "ollama_ask", lambda host, model: lambda prompt, seed: "turn on the fan"
    )
    out = tmp_path / "p.json"
    assert tm.main(["--out", str(out), "--cache", str(tmp_path / "c"), "--seeds", "3"]) == 0
    manifest = json.loads(out.read_text())["manifest"]
    assert manifest["teacher_license"].startswith("Apache-2.0")
    assert manifest["seeds"] == [3]


def test_ollama_ask_posts_chat_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_a: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"message": {"content": "hello"}}).encode()

    seen = {}

    def fake_urlopen(request: object, timeout: int) -> Response:
        seen["body"] = json.loads(request.data)  # type: ignore[attr-defined]
        return Response()

    monkeypatch.setattr("training.data.teacher.urllib.request.urlopen", fake_urlopen)
    assert tm.ollama_ask("http://x", "m")("hi", 5) == "hello"
    assert seen["body"]["think"] is False
    assert seen["body"]["options"]["seed"] == 5


def test_generator_uses_teacher_paraphrases() -> None:
    bank = {"on|fan|": ["kindly power the blower"]}
    gen = DialogueGenerator(3, noise=0.0, teacher=bank, p_teacher=1.0)
    turn = gen.device_turn("on", "fan", "on", DeviceState())
    assert turn.user == "kindly power the blower"
    assert turn.action == "fan=1"
    typo_gen = DialogueGenerator(4, noise=1.0, typo_p=1.0)
    assert typo_gen.decorate("please turn on the heater")[-1] in "?!."
