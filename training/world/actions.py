# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Action grammar, strict parser, firmware-side validation, and state update.

Grammar (between the ``<ACT>`` and ``</ACT>`` markers)::

    action := pair (" " pair){0,3}
    pair   := key "=" value
    key    := fan | heat | pump | light | win | ack | diag

The model only *proposes* an action. :func:`validate_action` is the authority, exactly
like ``tllm_device_validate()`` in the C runtime (vision §10, §20).
"""

from __future__ import annotations

from dataclasses import dataclass

from training.world.diagnose import DIAGNOSES
from training.world.state import (
    ERR_HEATER,
    ERR_NONE,
    ERR_PUMP_OVERCURRENT,
    ERR_WINDOW,
    DeviceState,
)

ACT_OPEN = "<ACT>"
ACT_CLOSE = "</ACT>"
MAX_PAIRS = 4
OVERHEAT_LIMIT_C = 35.0
PUMP_NOMINAL_CURRENT_A = 1.2

# key -> (min, max, step) for integer actuator values
_INT_RANGES: dict[str, tuple[int, int, int]] = {
    "fan": (0, 3, 1),
    "heat": (0, 1, 1),
    "pump": (0, 1, 1),
    "light": (0, 100, 10),
    "win": (0, 1, 1),
    "ack": (1, 99, 1),
}
ACTION_KEYS = (*_INT_RANGES.keys(), "diag")

Action = tuple[tuple[str, int | str], ...]


class ActionParseError(ValueError):
    """Raised when an action body violates the grammar or value ranges."""


def parse_action(body: str) -> Action:
    """Parse an action body such as ``"fan=2 heat=0"`` into ordered key/value pairs."""
    parts = body.split()
    if not parts or len(parts) > MAX_PAIRS:
        raise ActionParseError(f"expected 1..{MAX_PAIRS} pairs, got {len(parts)}")
    pairs: list[tuple[str, int | str]] = []
    seen: set[str] = set()
    for part in parts:
        key, sep, raw = part.partition("=")
        if not sep or key not in ACTION_KEYS:
            raise ActionParseError(f"unknown key in {part!r}")
        if key in seen:
            raise ActionParseError(f"duplicate key {key!r}")
        seen.add(key)
        if key == "diag":
            if raw not in DIAGNOSES:
                raise ActionParseError(f"unknown diagnosis {raw!r}")
            pairs.append((key, raw))
            continue
        if not (raw.isascii() and raw.isdigit()):
            raise ActionParseError(f"value for {key} must be a non-negative integer: {raw!r}")
        value = int(raw)
        low, high, step = _INT_RANGES[key]
        if not low <= value <= high or value % step:
            raise ActionParseError(f"{key}={value} outside {low}..{high} step {step}")
        pairs.append((key, value))
    return tuple(pairs)


def format_action(action: Action) -> str:
    """Inverse of :func:`parse_action`."""
    return " ".join(f"{key}={value}" for key, value in action)


@dataclass(frozen=True)
class Verdict:
    """Outcome of firmware validation."""

    approved: bool
    reason: str  # "ok" or a rejection code shared with the C runtime
    message: str


_REJECTIONS = {
    "fault_heater": "the heater has a fault (E3). it stays off until the fault is cleared.",
    "fault_window": "the window motor has a fault (E5). the window cannot move.",
    "fault_pump": "the pump tripped on overcurrent (E7). clear the fault first.",
    "ack_mismatch": "there is no such active fault to acknowledge.",
    "overheat": "the heater is not allowed above 35 degrees.",
    "heater_window": "the heater may not run while the window is open.",
}


def _reject(reason: str) -> Verdict:
    return Verdict(False, reason, _REJECTIONS[reason])


def validate_action(state: DeviceState, action: Action) -> Verdict:
    """Apply the firmware rules (faults, acknowledgement, interlocks) to a parsed action."""
    values = dict(action)
    if values.get("heat") == 1 and state.err == ERR_HEATER:
        return _reject("fault_heater")
    if "win" in values and values["win"] != state.win and state.err == ERR_WINDOW:
        return _reject("fault_window")
    if values.get("pump") == 1 and state.err == ERR_PUMP_OVERCURRENT:
        return _reject("fault_pump")
    if "ack" in values and (state.err == ERR_NONE or values["ack"] != state.err):
        return _reject("ack_mismatch")
    if values.get("heat") == 1 and state.t is not None and state.t >= OVERHEAT_LIMIT_C:
        return _reject("overheat")
    result = apply_action(state, action)
    if ("heat" in values or "win" in values) and result.heat == 1 and result.win == 1:
        return _reject("heater_window")
    return Verdict(True, "ok", "approved")


def apply_action(state: DeviceState, action: Action) -> DeviceState:
    """Return the state after executing ``action`` (no validation; see validate_action)."""
    changes: dict[str, object] = {}
    for key, value in action:
        if key == "diag":
            continue
        if key == "ack":
            changes["err"] = ERR_NONE
            continue
        changes[key] = value
        if key == "pump":
            changes["pa"] = PUMP_NOMINAL_CURRENT_A if value == 1 else 0.0
            if value == 1 and state.pump == 1:
                changes["pa"] = state.pa
    return state.with_changes(**changes)
