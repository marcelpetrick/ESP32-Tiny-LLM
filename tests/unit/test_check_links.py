# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Tests for tools.check_links."""

from pathlib import Path

import pytest

from tools import check_links


def test_iter_links_skips_code_fences_images_and_inline_code() -> None:
    text = "\n".join(
        [
            "[a](one.md) and ![img](pic.png)",
            "```",
            "[b](inside-fence.md)",
            "```",
            '`[c](inline.md)` [d](two.md#part "title")',
        ]
    )
    assert check_links.iter_links(text) == [(1, "one.md"), (5, "two.md#part")]


def test_broken_links_reports_missing_relative_targets(tmp_path: Path) -> None:
    (tmp_path / "exists.md").write_text("x", encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text(
        "[ok](exists.md#h) [ext](https://example.com) [anchor](#top) [bad](missing.md)\n",
        encoding="utf-8",
    )
    assert check_links.broken_links(doc) == [f"{doc}:1: broken link -> missing.md"]


@pytest.mark.parametrize(("content", "code"), [("[ok](doc.md)", 0), ("[no](nope.md)", 1)])
def test_main_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str, code: int
) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(content, encoding="utf-8")
    assert check_links.main([str(doc)]) == code
    assert "checked 1 file(s)" in capsys.readouterr().out
