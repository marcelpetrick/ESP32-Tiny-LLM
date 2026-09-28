# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Integration fixtures: make sure the C shared library exists."""

import subprocess

import pytest

from tools import runtime


@pytest.fixture(scope="session", autouse=True)
def c_runtime_built() -> None:
    """Build the release runtime once if it is missing (the pipeline builds it first)."""
    if not runtime.library_path().exists():
        subprocess.run([str(runtime.REPO_ROOT / "scripts" / "build_runtime.sh")], check=True)
