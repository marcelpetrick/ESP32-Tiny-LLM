# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Training side: device world oracle, data generation, tokenizer, model, export."""

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"


def project_version() -> str:
    """Return the project version from the repository ``VERSION`` file."""
    return _VERSION_FILE.read_text(encoding="utf-8").strip()
