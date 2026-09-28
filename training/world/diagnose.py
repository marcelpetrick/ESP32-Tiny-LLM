# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Deterministic diagnosis rules over a device state.

Each diagnosis has a fixed priority (lower = more urgent), a plain-English explanation
template, and an optional remedial action the assistant should propose. These rules are
the ground truth the model learns; they are *not* executed on the device.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from training.world.state import DeviceState

HUMID_HIGH = 75
HOT_C = 32.0
COLD_C = 12.0
SOIL_DRY = 25
SOIL_WET = 85
PUMP_BLOCKED_A = 2.5
PUMP_DRY_A = 0.3


@dataclass(frozen=True)
class Diagnosis:
    """One diagnosable condition."""

    name: str
    priority: int
    applies: Callable[[DeviceState], bool]
    explain: Callable[[DeviceState], str]
    remedy: Callable[[DeviceState], str | None]  # action body or None


def _t(s: DeviceState) -> float:
    return s.t if s.t is not None else 0.0


def _h(s: DeviceState) -> int:
    return s.h if s.h is not None else 0


def _cooling(s: DeviceState) -> str | None:
    parts = []
    if s.heat == 1:
        parts.append("heat=0")
    if s.fan < 3 and s.vib == 0:
        parts.append("fan=3")
    if s.win == 0 and s.err != 5:
        parts.append("win=1")
    return " ".join(parts) or None


def _heating(s: DeviceState) -> str | None:
    parts = []
    if s.win == 1 and s.err != 5:
        parts.append("win=0")
    if s.heat == 0 and s.err != 3 and (s.win == 0 or s.err != 5):
        parts.append("heat=1")
    return " ".join(parts) or None


_RULES: tuple[Diagnosis, ...] = (
    Diagnosis(
        "pump_blocked",
        1,
        lambda s: s.pump == 1 and s.pa >= PUMP_BLOCKED_A,
        lambda s: f"the pump draws {s.pa:.1f} amps, far above normal. the pump or pipe is blocked.",
        lambda s: "pump=0 diag=pump_blocked",
    ),
    Diagnosis(
        "pump_dry",
        2,
        lambda s: s.pump == 1 and s.pa <= PUMP_DRY_A,
        lambda s: f"the pump draws only {s.pa:.1f} amps. it is running dry. check the water tank.",
        lambda s: "pump=0 diag=pump_dry",
    ),
    Diagnosis(
        "heater_fault",
        3,
        lambda s: s.err == 3,
        lambda s: "the heater reports fault e3. check the heater before clearing it.",
        lambda s: "diag=heater_fault",
    ),
    Diagnosis(
        "window_fault",
        4,
        lambda s: s.err == 5,
        lambda s: "the window motor reports fault e5. it may be obstructed.",
        lambda s: "diag=window_fault",
    ),
    Diagnosis(
        "pump_fault",
        5,
        lambda s: s.err == 7,
        lambda s: "the pump tripped on overcurrent, fault e7.",
        lambda s: "diag=pump_fault",
    ),
    Diagnosis(
        "sensor_offline",
        6,
        lambda s: s.t is None or s.h is None,
        lambda s: (
            "the climate sensor is offline, so i cannot read "
            + ("temperature" if s.t is None else "humidity")
            + "."
        ),
        lambda s: "diag=sensor_offline",
    ),
    Diagnosis(
        "fan_bearing",
        7,
        lambda s: s.fan > 0 and s.vib == 1,
        lambda s: "the fan vibrates strongly. the bearing may be worn.",
        lambda s: "fan=0 diag=fan_bearing",
    ),
    Diagnosis(
        "too_hot",
        8,
        lambda s: s.t is not None and s.t >= HOT_C,
        lambda s: f"it is {_t(s):.1f} degrees, which is too hot.",
        _cooling,
    ),
    Diagnosis(
        "too_cold",
        9,
        lambda s: s.t is not None and s.t <= COLD_C,
        lambda s: f"it is only {_t(s):.1f} degrees, which is too cold.",
        _heating,
    ),
    Diagnosis(
        "too_humid",
        10,
        lambda s: s.h is not None and s.h >= HUMID_HIGH,
        lambda s: (
            f"humidity is {_h(s)} percent"
            + (" and the fan is off." if s.fan == 0 else " even with the fan running.")
        ),
        lambda s: f"fan={min(3, max(2, s.fan + 1))}" if s.fan < 3 else None,
    ),
    Diagnosis(
        "soil_dry",
        11,
        lambda s: s.soil <= SOIL_DRY and s.pump == 0,
        lambda s: f"the soil moisture is only {s.soil} percent. the plants need water.",
        lambda s: None if s.err == 7 else "pump=1",
    ),
    Diagnosis(
        "soil_wet",
        12,
        lambda s: s.soil >= SOIL_WET,
        lambda s: f"the soil moisture is {s.soil} percent, which is too wet.",
        lambda s: "pump=0" if s.pump == 1 else None,
    ),
)


DIAGNOSES: dict[str, Diagnosis] = {rule.name: rule for rule in _RULES}


def diagnose(state: DeviceState) -> list[Diagnosis]:
    """Return all applicable diagnoses, most urgent first (empty list = all normal)."""
    return sorted((r for r in _RULES if r.applies(state)), key=lambda r: r.priority)
