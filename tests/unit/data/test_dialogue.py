# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Dialogue generator semantics: safety rules, references, determinism, coverage."""

from collections import Counter

import pytest

from training.data import dialogue as dlg
from training.data.dialogue import DialogueGenerator, Turn, device_command
from training.data.lexicon import FRAMES
from training.world import DeviceState, parse_action, validate_action


@pytest.mark.parametrize(
    ("dev", "op", "arg", "state", "reply_start", "action"),
    [
        ("fan", "on", None, DeviceState(), "setting the fan to level 1", "fan=1"),
        ("fan", "on", None, DeviceState(fan=2), "the fan is already at level 2", None),
        ("fan", "set", 7, DeviceState(), "<unsupported> the fan has levels 0 to 3", None),
        ("fan", "up", None, DeviceState(fan=3), "the fan is already at the highest level", None),
        ("fan", "down", None, DeviceState(fan=0), "the fan is already off", None),
        ("fan", "down", None, DeviceState(fan=2), "setting the fan to level 1", "fan=1"),
        ("fan", "max", None, DeviceState(), "setting the fan to level 3", "fan=3"),
        ("light", "on", None, DeviceState(), "setting the light to 100 percent", "light=100"),
        ("light", "on", None, DeviceState(light=40), "the light is already at 40 percent", None),
        ("light", "set", 35, DeviceState(), "<unsupported> the light goes", None),
        (
            "light",
            "up",
            None,
            DeviceState(light=90),
            "setting the light to 100 percent",
            "light=100",
        ),
        ("light", "off", None, DeviceState(light=90), "turning the light off", "light=0"),
        ("heat", "on", None, DeviceState(err=3), "<unsupported> the heater has fault e3", None),
        ("heat", "on", None, DeviceState(t=36.0), "<unsupported> it is 36.0 degrees", None),
        ("heat", "on", None, DeviceState(win=1, err=5), "<unsupported> the window is open", None),
        (
            "heat",
            "on",
            None,
            DeviceState(win=1),
            "closing the window and turning the heater on",
            "win=0 heat=1",
        ),
        ("heat", "off", None, DeviceState(heat=0), "the heater is already off", None),
        (
            "win",
            "on",
            None,
            DeviceState(err=5),
            "<unsupported> the window motor has fault e5",
            None,
        ),
        (
            "win",
            "on",
            None,
            DeviceState(heat=1),
            "turning the heater off and opening the window",
            "heat=0 win=1",
        ),
        ("win", "on", None, DeviceState(win=1), "the window is already open", None),
        ("win", "off", None, DeviceState(), "the window is already closed", None),
        ("pump", "on", None, DeviceState(err=7), "<unsupported> the pump tripped", None),
        ("pump", "on", None, DeviceState(), "starting the pump", "pump=1"),
        ("pump", "on", None, DeviceState(pump=1, pa=1.2), "the pump is already running", None),
        ("pump", "off", None, DeviceState(), "the pump is already off", None),
    ],
)
def test_device_command(
    dev: str, op: str, arg: int | None, state: DeviceState, reply_start: str, action: str | None
) -> None:
    reply, body = device_command(dev, op, arg, state)
    assert reply.startswith(reply_start)
    assert body == action
    if body is not None:
        assert validate_action(state, parse_action(body)).approved


def test_device_value_texts() -> None:
    s = DeviceState(fan=2, heat=1, pump=1, pa=1.3, light=40, win=1)
    assert dlg.device_value_text("fan", s) == "the fan is at level 2."
    assert dlg.device_value_text("heat", s) == "the heater is on."
    assert dlg.device_value_text("pump", s) == "the pump is running and draws 1.3 amps."
    assert dlg.device_value_text("light", s) == "the light is at 40 percent."
    assert dlg.device_value_text("win", s) == "the window is open."
    off = DeviceState()
    texts = [dlg.device_value_text(d, off) for d in dlg.DEVICES]
    assert texts == [
        "the fan is off.",
        "the heater is off.",
        "the pump is off.",
        "the light is off.",
        "the window is closed.",
    ]


def test_describe_ack_and_diag() -> None:
    assert dlg.describe("ack", 3, DeviceState()) == "clearing fault e3"
    assert dlg.describe("diag", "too_hot", DeviceState()) is None
    assert dlg.describe_action("pump=0 diag=pump_dry", DeviceState()) == "stopping the pump"


def test_generation_is_deterministic() -> None:
    a = [DialogueGenerator(42).sample() for _ in range(1)]
    b = [DialogueGenerator(42).sample() for _ in range(1)]
    assert a == b


def _collect(seed: int, count: int, heldout: bool = False) -> list[dlg.Sample]:
    gen = DialogueGenerator(seed, heldout=heldout, noise=0.5)
    return [gen.sample(force_reference=i % 3 == 0) for i in range(count)]


def test_all_intents_occur_and_every_action_is_valid() -> None:
    samples = _collect(5, 6000) + _collect(6, 2000, heldout=True)
    intents = Counter(s.turn.intent for s in samples)
    expected = (set(dlg.SINGLE_INTENTS) | set(dlg.REFERENCE_INTENTS)) - {"what_about"}
    assert expected <= set(intents)
    for s in samples:
        if s.turn.action is not None:
            assert validate_action(s.state, parse_action(s.turn.action)).approved
        assert s.turn.reply
        assert s.turn.reply == s.turn.reply.lower()


def test_heldout_frames_never_appear_in_training_samples() -> None:
    heldout = {
        f[1:].split("{")[0] for frames in FRAMES.values() for f in frames if f.startswith("!")
    }
    heldout = {h for h in heldout if len(h) > 12}
    for s in _collect(8, 3000):
        for h in heldout:
            assert not s.turn.user.startswith(h), (h, s.turn.user)


def test_clarify_without_context() -> None:
    gen = DialogueGenerator(1)
    turn = gen.make_turn("it_off", DeviceState(), None)
    assert turn.reply == "<clarify> what should i turn off?"
    turn = gen.make_turn("it_up", DeviceState(), None)
    assert turn.reply == "<clarify> which device do you mean?"
    turn = gen.make_turn("and_dev", DeviceState(), None)
    assert turn.reply.startswith("<clarify> what should i do with the")


def test_reference_resolution_uses_previous_device() -> None:
    gen = DialogueGenerator(3)
    prev = Turn("turn on the fan", "setting the fan to level 1.", "fan=1", "on", "fan", "on", 0)
    state = DeviceState(fan=1)
    assert gen.make_turn("it_off", state, prev).action == "fan=0"
    assert gen.make_turn("it_up", state, prev).action == "fan=2"
    turn = gen.make_turn("and_dev", state, prev)
    assert turn.intent == "and_dev"
    assert turn.action is None or turn.action.endswith(("=1", "=100"))


def test_correction_reverts_previous_device() -> None:
    gen = DialogueGenerator(9)
    prev = Turn(
        "turn on the light",
        "setting the light to 100 percent.",
        "light=100",
        "on",
        "light",
        "on",
        0,
    )
    turn = gen.make_turn("correct", DeviceState(light=100), prev)
    assert turn.reply.startswith("sorry.")
    assert turn.action is not None
    assert turn.action.startswith("light=0 ")


def test_correction_skips_disallowed_revert() -> None:
    gen = DialogueGenerator(9)
    prev = Turn("turn off the heater", "turning the heater off.", "heat=0", "off", "heat", "off", 1)
    turn = gen.make_turn(
        "correct", DeviceState(t=36.0, heat=0, fan=1, light=50, pump=1, pa=1.2), prev
    )
    assert turn.action is None or "heat=1" not in turn.action


def test_what_about_after_temperature_reads_humidity() -> None:
    gen = DialogueGenerator(2)
    prev = Turn("what is the temperature", "it is 20.0 degrees.", None, "read_t")
    turn = gen.make_turn("what_about", DeviceState(h=61), prev)
    assert turn.reply == "humidity is 61 percent."
    turn = gen.make_turn("what_about", DeviceState(h=None), None)
    assert turn.action == "diag=sensor_offline"


def test_checked_raises_on_invalid_action() -> None:
    gen = DialogueGenerator(1)
    bad = Turn("x", "y", "heat=1", "on")
    with pytest.raises(dlg.GenerationError):
        gen.checked(DeviceState(err=3), bad)


@pytest.mark.parametrize(
    ("intent", "state", "expected"),
    [
        ("complain_hot", DeviceState(t=None), "the temperature sensor is offline."),
        ("complain_humid", DeviceState(h=None), "the humidity sensor is offline."),
        ("complain_humid", DeviceState(h=50), "humidity is 50 percent, which is normal."),
        ("complain_dry", DeviceState(soil=50), "the soil moisture is 50 percent, which is fine."),
        ("complain_cold", DeviceState(t=20.0), "it is 20.0 degrees, which is normal."),
        ("ack", DeviceState(err=5), "clearing fault e5."),
        ("ack", DeviceState(), "there is no active fault."),
        ("read_t", DeviceState(t=None), "the temperature sensor is offline."),
        ("read_pa", DeviceState(pump=1, pa=1.1), "the pump draws 1.1 amps."),
        ("read_pa", DeviceState(), "the pump is off, so it draws no current."),
        ("diagnose", DeviceState(), "i see no problems. everything is normal."),
        ("status", DeviceState(t=33.0), "it is 33.0 degrees, which is too hot."),
    ],
)
def test_fixed_semantics(intent: str, state: DeviceState, expected: str) -> None:
    assert DialogueGenerator(4).make_turn(intent, state, None).reply == expected


def test_diag_dev_normal_and_faulty() -> None:
    gen = DialogueGenerator(4)
    faulty = gen.diag_dev_turn("pump", DeviceState(pump=1, pa=3.0))
    assert faulty.action == "pump=0 diag=pump_blocked"
    normal = gen.diag_dev_turn("light", DeviceState(light=30))
    assert normal.reply == "the light looks normal. the light is at 30 percent."


def test_diagnosis_without_remedy_emits_diag_token() -> None:
    reply, action = dlg.diagnosis_reply("heater_fault", DeviceState(err=3))
    assert action == "diag=heater_fault"
    assert "e3" in reply
    reply, action = dlg.diagnosis_reply("too_humid", DeviceState(h=90, fan=3))
    assert action == "diag=too_humid"
