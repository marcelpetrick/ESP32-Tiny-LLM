# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Check that relative links in Markdown files point to existing files.

Usage: python -m tools.check_links FILE.md [FILE.md ...]

External links (http, https, mailto) and pure in-page anchors are ignored; for
``path#anchor`` links only the path part is checked. Exits 1 if any link is broken.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_FENCE = re.compile(r"^\s*(```|~~~)")
_EXTERNAL = ("http://", "https://", "mailto:")


def iter_links(text: str) -> list[tuple[int, str]]:
    """Return ``(line_number, target)`` for every Markdown link outside code fences."""
    links: list[tuple[int, str]] = []
    in_fence = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        stripped = re.sub(r"`[^`]*`", "", line)
        links.extend((number, match.group(1)) for match in _LINK.finditer(stripped))
    return links


def broken_links(path: Path) -> list[str]:
    """Return human-readable descriptions of broken relative links in ``path``."""
    problems: list[str] = []
    for number, target in iter_links(path.read_text(encoding="utf-8")):
        if target.startswith(_EXTERNAL) or target.startswith("#"):
            continue
        file_part = target.split("#", 1)[0]
        if not (path.parent / file_part).exists():
            problems.append(f"{path}:{number}: broken link -> {target}")
    return problems


def main(argv: list[str]) -> int:
    """Check all files given on the command line; print problems; return exit code."""
    problems = [p for name in argv for p in broken_links(Path(name))]
    for problem in problems:
        print(problem)
    print(f"checked {len(argv)} file(s), {len(problems)} broken link(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
