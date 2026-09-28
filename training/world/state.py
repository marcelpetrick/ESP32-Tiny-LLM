# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Greenhouse device state and its canonical, token-efficient text form.

Canonical form (field order is fixed; mirrored by ``runtime/src/device.c``)::

    <S> t=31.2 h=73 soil=41 fan=0 heat=0 pump=0 light=0 win=1 pa=0.0 vib=0 err=0</S>

``t`` and ``h`` may be ``na`` when the sensor is offline.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

STATE_OPEN = "<S>"
STATE_CLOSE = "</S>"

# Error codes latched by the (simulated) firmware.
ERR_NONE = 0
ERR_HEATER = 3
ERR_WINDOW = 5
ERR_PUMP_OVERCURRENT = 7
ERROR_CODES = (ERR_NONE, ERR_HEATER, ERR_WINDOW, ERR_PUMP_OVERCURRENT)


@dataclass(frozen=True)
class DeviceState:
    """Snapshot of the greenhouse controller."""

    t: float | None = 22.0  # air temperature in degrees C, None = sensor offline
    h: int | None = 55  # relative humidity in percent, None = sensor offline
    soil: int = 50  # soil moisture in percent
    fan: int = 0  # fan level 0..3
    heat: int = 0  # heater 0/1
    pump: int = 0  # irrigation pump 0/1
    light: int = 0  # grow light 0..100 percent
    win: int = 0  # roof window 0 closed / 1 open
    pa: float = 0.0  # pump motor current in ampere
    vib: int = 0  # fan vibration 0 normal / 1 high
    err: int = ERR_NONE  # latched fault code

    def with_changes(self, **changes: object) -> DeviceState:
        """Return a copy with the given fields replaced."""
        return replace(self, **changes)  # type: ignore[arg-type]


FIELD_NAMES = tuple(f.name for f in fields(DeviceState))


def _format_value(name: str, value: object) -> str:
    if value is None:
        return "na"
    if name in ("t", "pa"):
        return f"{float(value):.1f}"  # type: ignore[arg-type]
    return str(value)


def render_state(state: DeviceState) -> str:
    """Render ``state`` in canonical form, including the ``<S>``/``</S>`` markers."""
    body = " ".join(f"{name}={_format_value(name, getattr(state, name))}" for name in FIELD_NAMES)
    return f"{STATE_OPEN} {body}{STATE_CLOSE}"


class StateParseError(ValueError):
    """Raised when a state block is not in canonical form."""


def parse_state(text: str) -> DeviceState:
    """Parse a canonical state block produced by :func:`render_state`."""
    if not (text.startswith(STATE_OPEN + " ") and text.endswith(STATE_CLOSE)):
        raise StateParseError(f"not a state block: {text!r}")
    parts = text[len(STATE_OPEN) : -len(STATE_CLOSE)].split()
    if len(parts) != len(FIELD_NAMES):
        raise StateParseError(f"expected {len(FIELD_NAMES)} fields, got {len(parts)}")
    values: dict[str, object] = {}
    for name, part in zip(FIELD_NAMES, parts, strict=True):
        key, sep, raw = part.partition("=")
        if key != name or not sep:
            raise StateParseError(f"expected field {name!r}, got {part!r}")
        try:
            if raw == "na" and name in ("t", "h"):
                values[name] = None
            elif name in ("t", "pa"):
                values[name] = float(raw)
            else:
                values[name] = int(raw)
        except ValueError as exc:
            raise StateParseError(f"bad value for {name}: {raw!r}") from exc
    return DeviceState(**values)  # type: ignore[arg-type]
