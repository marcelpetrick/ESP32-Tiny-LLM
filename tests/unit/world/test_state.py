# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Canonical state rendering and parsing."""

import pytest

from training.world import DeviceState, parse_state, render_state
from training.world.state import StateParseError


def test_render_default_state() -> None:
    assert render_state(DeviceState()) == (
        "<S> t=22.0 h=55 soil=50 fan=0 heat=0 pump=0 light=0 win=0 pa=0.0 vib=0 err=0 </S>"
    )


def test_render_offline_sensor_and_negative_temperature() -> None:
    text = render_state(DeviceState(t=-3.25, h=None))
    assert text.startswith("<S> t=-3.2 h=na ")


@pytest.mark.parametrize(
    "state",
    [DeviceState(), DeviceState(t=None, h=None, fan=3, pa=2.7, err=7), DeviceState(t=35.5)],
)
def test_round_trip(state: DeviceState) -> None:
    assert parse_state(render_state(state)) == state


@pytest.mark.parametrize(
    "text",
    [
        "t=1",
        "<S> t=22.0 h=55 soil=50 fan=0 heat=0 pump=0 light=0 win=0 pa=0.0 vib=0 err=0 </X>",
        "<S> h=22.0 t=55 soil=50 fan=0 heat=0 pump=0 light=0 win=0 pa=0.0 vib=0 err=0 </S>",
        "<S> t=22.0 h=55 soil=na fan=0 heat=0 pump=0 light=0 win=0 pa=0.0 vib=0 err=0 </S>",
        "<S> t 22.0 h=55 soil=5 fan=0 heat=0 pump=0 light=0 win=0 pa=0.0 vib=0 err=0 </S>",
    ],
)
def test_parse_rejects_malformed(text: str) -> None:
    with pytest.raises(StateParseError):
        parse_state(text)


def test_with_changes_is_a_copy() -> None:
    base = DeviceState()
    changed = base.with_changes(fan=2)
    assert changed.fan == 2
    assert base.fan == 0
