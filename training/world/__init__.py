# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""The greenhouse device world: state, action grammar, validation, diagnoses.

This package is the *oracle*: it defines ground truth for dataset generation and
evaluation. ``runtime/src/device.c`` mirrors the state format, action grammar and
validation rules; integration tests assert both agree.
"""

from training.world.actions import (
    ACTION_KEYS,
    Action,
    ActionParseError,
    Verdict,
    apply_action,
    format_action,
    parse_action,
    validate_action,
)
from training.world.diagnose import DIAGNOSES, diagnose
from training.world.state import DeviceState, parse_state, render_state

__all__ = [
    "ACTION_KEYS",
    "DIAGNOSES",
    "Action",
    "ActionParseError",
    "DeviceState",
    "Verdict",
    "apply_action",
    "diagnose",
    "format_action",
    "parse_action",
    "parse_state",
    "render_state",
    "validate_action",
]
