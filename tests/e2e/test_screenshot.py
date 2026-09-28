# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""The screenshot tool produces light and dark PNGs."""

from pathlib import Path

import pytest

from tools import screenshot

pytestmark = pytest.mark.e2e


def test_screenshot_tool(web_models: dict[str, Path], tmp_path: Path) -> None:
    assert (
        screenshot.main(
            ["--model", f"greenhouse={web_models['greenhouse']}", "--out", str(tmp_path)]
        )
        == 0
    )
    for name in ("web-ui.png", "web-ui-dark.png"):
        assert (tmp_path / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
