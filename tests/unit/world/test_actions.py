# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Action grammar, validation rules and state updates."""

import pytest

from training.world import (
    ActionParseError,
    DeviceState,
    apply_action,
    format_action,
    parse_action,
    validate_action,
)


def test_parse_and_format_round_trip() -> None:
    action = parse_action("fan=2 heat=0 diag=too_humid")
    assert action == (("fan", 2), ("heat", 0), ("diag", "too_humid"))
    assert format_action(action) == "fan=2 heat=0 diag=too_humid"


@pytest.mark.parametrize(
    "body",
    [
        "",
        "fan=1 heat=0 pump=0 win=0 light=10",
        "speed=2",
        "fan",
        "fan=1 fan=2",
        "fan=4",
        "fan=-1",
        "light=15",
        "ack=0",
        "diag=unknown",
        "fan=x",
        "fan=\u00b2",
    ],
)
def test_parse_rejects(body: str) -> None:
    with pytest.raises(ActionParseError):
        parse_action(body)


@pytest.mark.parametrize(
    ("state", "body", "reason"),
    [
        (DeviceState(), "fan=2", "ok"),
        (DeviceState(err=3), "heat=1", "fault_heater"),
        (DeviceState(err=3), "heat=0", "ok"),
        (DeviceState(err=5, win=0), "win=1", "fault_window"),
        (DeviceState(err=5, win=1), "win=1", "ok"),
        (DeviceState(err=7), "pump=1", "fault_pump"),
        (DeviceState(err=0), "ack=3", "ack_mismatch"),
        (DeviceState(err=5), "ack=3", "ack_mismatch"),
        (DeviceState(err=5), "ack=5", "ok"),
        (DeviceState(t=36.0), "heat=1", "overheat"),
        (DeviceState(t=None), "heat=1", "ok"),
        (DeviceState(win=1), "heat=1", "heater_window"),
        (DeviceState(heat=1), "win=1", "heater_window"),
        (DeviceState(heat=1, win=1), "fan=1", "ok"),
        (DeviceState(win=1), "win=0 heat=1", "ok"),
        (DeviceState(), "diag=too_hot", "ok"),
    ],
)
def test_validate(state: DeviceState, body: str, reason: str) -> None:
    verdict = validate_action(state, parse_action(body))
    assert verdict.reason == reason
    assert verdict.approved == (reason == "ok")
    assert verdict.message


def test_apply_updates_actuators_and_pump_current() -> None:
    state = apply_action(DeviceState(err=5), parse_action("pump=1 fan=3 ack=5 diag=too_hot"))
    assert (state.pump, state.fan, state.err, state.pa) == (1, 3, 0, 1.2)
    running = DeviceState(pump=1, pa=2.8)
    assert apply_action(running, parse_action("pump=1")).pa == 2.8
    assert apply_action(running, parse_action("pump=0")).pa == 0.0
