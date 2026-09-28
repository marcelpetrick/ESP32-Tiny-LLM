# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Spelling noise for the robustness suite (vision §19: spelling mistakes, noise)."""

from __future__ import annotations

import random

_KEYBOARD_NEIGHBOURS = {
    "a": "qs", "e": "wr", "i": "uo", "o": "ip", "u": "yi", "t": "ry", "n": "bm",
    "s": "ad", "r": "et", "h": "gj", "l": "k", "d": "sf", "m": "n", "p": "o",
}  # fmt: skip


def typo(word: str, rng: random.Random) -> str:
    """Apply one realistic typo (swap, drop, double, or neighbour key) to ``word``."""
    if len(word) < 4 or not word.isalpha():
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(("swap", "drop", "double", "neighbour"))
    if kind == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    if kind == "drop":
        return word[:i] + word[i + 1 :]
    if kind == "double":
        return word[:i] + word[i] + word[i:]
    options = _KEYBOARD_NEIGHBOURS.get(word[i], word[i])
    return word[:i] + rng.choice(options) + word[i + 1 :]


def add_typos(text: str, rng: random.Random, count: int = 1) -> str:
    """Introduce ``count`` typos into distinct eligible words of ``text``."""
    words = text.split(" ")
    eligible = [i for i, w in enumerate(words) if len(w) >= 4 and w.isalpha()]
    for i in rng.sample(eligible, min(count, len(eligible))):
        words[i] = typo(words[i], rng)
    return " ".join(words)
