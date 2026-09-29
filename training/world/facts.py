# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Device facts and deterministic keyword retrieval (vision §24 D: retrieval without a second LLM).

The firmware looks up at most one short fact by keyword in the user's text and injects it
into the prompt as ``<F> fact</F>``; the model learns to answer from injected text instead
of storing manual knowledge in its parameters. Mirrored exactly by ``runtime/src/facts.c``
(same table order, same matching rule); a parity test checks both.

Matching rule: lowercase the (ASCII) text, then return the first fact (table order) that has a
keyword occurring as a whole-word phrase (letters/digits delimit words).
"""

from __future__ import annotations

import string
from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    """One retrievable fact."""

    name: str
    keywords: tuple[str, ...]
    text: str
    heldout: bool = False  # never used in training data (tests answering from unseen facts)


FACTS: tuple[Fact, ...] = (
    Fact(
        "e3",
        ("e3", "error 3", "fault 3"),
        "e3 means the heater overheated or its sensor failed. check the heater, then clear the fault.",
    ),
    Fact(
        "e5",
        ("e5", "error 5", "fault 5"),
        "e5 means the window motor is blocked. remove the obstacle, then clear the fault.",
    ),
    Fact(
        "e7",
        ("e7", "error 7", "fault 7"),
        "e7 means the pump drew too much current. check the pump for a blockage, then clear the fault.",
    ),
    Fact(
        "humidity",
        ("ideal humidity", "good humidity", "best humidity"),
        "the ideal humidity for most plants is 50 to 70 percent.",
    ),
    Fact(
        "temperature",
        ("ideal temperature", "good temperature", "best temperature"),
        "the ideal temperature is 18 to 26 degrees.",
    ),
    Fact(
        "watering",
        ("how often", "when should i water", "watering"),
        "water when the soil moisture drops below 25 percent, about once a day in summer.",
    ),
    Fact(
        "fan",
        ("fan levels", "fan speeds", "how fast is the fan"),
        "the fan has levels 0 to 3. level 3 moves the most air but is loud.",
    ),
    Fact(
        "light",
        ("how long", "light hours", "hours of light"),
        "most plants need 12 to 16 hours of light a day.",
    ),
    Fact(
        "bearing",
        ("bearing", "maintenance", "service the fan"),
        "fan bearings wear after about two years. replace the fan when vibration stays high.",
    ),
    Fact(
        "tank",
        ("water tank", "refill", "tank"),
        "the water tank holds 20 litres and lasts about a week.",
    ),
    Fact(
        "heater",
        ("heater power", "how strong is the heater", "watts"),
        "the heater has 500 watts and is not allowed above 35 degrees.",
    ),
    Fact(
        "window",
        ("window open", "when to open", "ventilate"),
        "open the window when it is warmer than 28 degrees and not raining.",
    ),
    # held out: never in training, only in the facts_heldout evaluation suite
    Fact(
        "frost",
        ("frost", "freezing night"),
        "below 5 degrees at night, close the window and run the heater.",
        heldout=True,
    ),
    Fact(
        "soil",
        ("soil type", "which soil", "potting soil"),
        "use loose potting soil with good drainage.",
        heldout=True,
    ),
    Fact(
        "pests",
        ("pests", "insects", "bugs"),
        "check the leaves for insects once a week and remove them by hand.",
        heldout=True,
    ),
    Fact(
        "warranty",
        ("warranty", "guarantee"),
        "the controller has a two year warranty.",
        heldout=True,
    ),
)


_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


def _is_word_char(c: str) -> bool:
    return c.isascii() and c.isalnum()


def contains_phrase(text: str, phrase: str) -> bool:
    """``phrase`` occurs in ``text`` delimited by non-alphanumeric characters (or the ends)."""
    start = 0
    while True:
        i = text.find(phrase, start)
        if i < 0:
            return False
        before = i == 0 or not _is_word_char(text[i - 1])
        end = i + len(phrase)
        after = end == len(text) or not _is_word_char(text[end])
        if before and after:
            return True
        start = i + 1


def retrieve(user_text: str) -> Fact | None:
    """The first fact (table order) with a keyword phrase in the lowercased text.

    Only ASCII letters are lowercased, exactly like the C side (``str.lower`` would also map
    e.g. the Kelvin sign to ``k``).
    """
    text = user_text.translate(_ASCII_LOWER)
    for fact in FACTS:
        if any(contains_phrase(text, kw) for kw in fact.keywords):
            return fact
    return None
