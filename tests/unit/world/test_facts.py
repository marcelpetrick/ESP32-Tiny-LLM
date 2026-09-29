# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Fact table and keyword retrieval (vision §24 D)."""

import pytest

from training.world.facts import FACTS, contains_phrase, retrieve


@pytest.mark.parametrize(
    ("text", "phrase", "expected"),
    [
        ("what does e5 mean", "e5", True),
        ("e5", "e5", True),
        ("e55", "e5", False),
        ("xe5", "e5", False),
        ("see e55 and e5.", "e5", True),
        ("x-tank", "tank", True),
        ("tanks", "tank", False),
        ("", "tank", False),
        ("how often do i", "how often", True),
    ],
)
def test_contains_phrase(text: str, phrase: str, expected: bool) -> None:
    assert contains_phrase(text, phrase) is expected


def test_retrieve() -> None:
    def name(text: str) -> str | None:
        fact = retrieve(text)
        return fact.name if fact else None

    assert name("What does ERROR 7 mean?") == "e7"
    assert name("the e5, then e3") == "e3"  # table order wins
    assert name("hello there") is None
    assert name("is there a warranty") == "warranty"
    assert name("tan" + chr(0x212A)) is None  # Kelvin sign is not lowercased to "k" (C parity)


def test_table_is_consistent() -> None:
    names = [f.name for f in FACTS]
    assert len(names) == len(set(names))
    assert {f.name for f in FACTS if f.heldout} == {"frost", "soil", "pests", "warranty"}
    for fact in FACTS:
        assert 1 <= len(fact.keywords) <= 4  # C table: MAX_KEYWORDS
        assert len(fact.text.encode()) < 256  # C: TLLM_FACT_TEXT_MAX
        assert fact.text == fact.text.lower()
        for keyword in fact.keywords:
            assert keyword == keyword.lower()
            # every keyword retrieves its own fact unless an earlier fact shadows it
            found = retrieve(f"about {keyword}")
            assert found is not None
            assert FACTS.index(found) <= FACTS.index(fact)
