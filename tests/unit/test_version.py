# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""The VERSION file is the single source of the project version."""

import re

from training import project_version


def test_version_is_semver() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", project_version())
