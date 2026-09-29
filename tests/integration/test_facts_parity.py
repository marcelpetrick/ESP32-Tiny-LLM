# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Python <-> C parity of the fact table and keyword retrieval (vision §24 D)."""

import ctypes
import random

from tools import runtime
from training.data.dialogue import FACT_FRAMES
from training.data.noise import add_typos
from training.world.facts import FACTS, retrieve


def _c_api() -> ctypes.CDLL:
    lib = runtime.load_library()
    lib.tllm_fact_retrieve.argtypes = [ctypes.c_char_p]
    lib.tllm_fact_text.restype = ctypes.c_char_p
    lib.tllm_fact_text.argtypes = [ctypes.c_int]
    lib.tllm_fact_name.restype = ctypes.c_char_p
    lib.tllm_fact_name.argtypes = [ctypes.c_int]
    return lib


def test_tables_match() -> None:
    lib = _c_api()
    assert lib.tllm_fact_count() == len(FACTS)
    for i, fact in enumerate(FACTS):
        assert lib.tllm_fact_name(i).decode() == fact.name
        assert lib.tllm_fact_text(i).decode() == fact.text


def test_retrieval_matches_on_many_inputs() -> None:
    lib = _c_api()
    rng = random.Random(7)
    texts = [
        "",
        "hello",
        "E5",
        "e55",
        "tan" + chr(0x212A),
        "café tank",
        "x-tank!",
        "the e5, then e3",
    ]
    for fact in FACTS:
        for keyword in fact.keywords:
            for frame in FACT_FRAMES:
                text = frame.format(k=keyword)
                texts += [text, text.upper(), add_typos(text, rng, 1), f"{text} and the tank"]
    alphabet = "abcdeton 3579?-"
    texts += ["".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40))) for _ in range(2000)]
    for text in texts:
        found = retrieve(text)
        expected = FACTS.index(found) if found is not None else -1
        assert lib.tllm_fact_retrieve(text.encode()) == expected, text
