# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Diagnosis rules: priorities, explanations, and that every remedy is valid."""

import itertools
import random

import pytest

from training.world import DIAGNOSES, DeviceState, diagnose, parse_action, validate_action


def names(state: DeviceState) -> list[str]:
    return [d.name for d in diagnose(state)]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (DeviceState(), []),
        (DeviceState(pump=1, pa=2.9), ["pump_blocked"]),
        (DeviceState(pump=1, pa=0.1), ["pump_dry"]),
        (DeviceState(err=3), ["heater_fault"]),
        (DeviceState(err=5), ["window_fault"]),
        (DeviceState(err=7), ["pump_fault"]),
        (DeviceState(t=None), ["sensor_offline"]),
        (DeviceState(fan=2, vib=1), ["fan_bearing"]),
        (DeviceState(t=33.0, h=80), ["too_hot", "too_humid"]),
        (DeviceState(t=5.0), ["too_cold"]),
        (DeviceState(soil=10), ["soil_dry"]),
        (DeviceState(soil=90, pump=1, pa=1.2), ["soil_wet"]),
    ],
)
def test_diagnose(state: DeviceState, expected: list[str]) -> None:
    assert names(state) == expected


def test_explanations_mention_values() -> None:
    assert "80 percent and the fan is off" in DIAGNOSES["too_humid"].explain(DeviceState(h=80))
    assert "even with the fan" in DIAGNOSES["too_humid"].explain(DeviceState(h=80, fan=1))
    assert "humidity" in DIAGNOSES["sensor_offline"].explain(DeviceState(h=None))
    assert "temperature" in DIAGNOSES["sensor_offline"].explain(DeviceState(t=None))
    assert "2.9 amps" in DIAGNOSES["pump_blocked"].explain(DeviceState(pump=1, pa=2.9))


def _random_states(count: int) -> list[DeviceState]:
    rng = random.Random(1234)
    states = []
    for _ in range(count):
        states.append(
            DeviceState(
                t=rng.choice([None, round(rng.uniform(-5, 42), 1)]),
                h=rng.choice([None, rng.randint(10, 99)]),
                soil=rng.randint(0, 100),
                fan=rng.randint(0, 3),
                heat=rng.randint(0, 1),
                pump=rng.randint(0, 1),
                light=rng.randrange(0, 101, 10),
                win=rng.randint(0, 1),
                pa=round(rng.uniform(0, 3.5), 1),
                vib=rng.randint(0, 1),
                err=rng.choice([0, 0, 0, 3, 5, 7]),
            )
        )
    return states


def test_every_remedy_parses_and_is_approved() -> None:
    checked = 0
    for state, rule in itertools.product(_random_states(3000), DIAGNOSES.values()):
        if not rule.applies(state):
            continue
        rule.explain(state)
        remedy = rule.remedy(state)
        if remedy is None:
            continue
        # A state that already violates an interlock can't be fixed by an unrelated action.
        if state.heat == 1 and state.win == 1:
            continue
        verdict = validate_action(state, parse_action(remedy))
        assert verdict.approved, (state, rule.name, remedy, verdict)
        checked += 1
    assert checked > 1000
