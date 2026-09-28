# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Byte-level BPE: pre-tokenisation, training determinism, round trips, serialisation."""

from pathlib import Path

import pytest

from training.tokenizer import Tokenizer, pretokenize, train_bpe
from training.tokenizer.bpe import FORMAT_VERSION

SPECIALS = ("<pad>", "<bos>", "<eos>", "<U>", "</U>", "<unsupported>")
CORPUS = ["turn on the fan", "turn off the fan", "the fan is at level 2.", " fan=2 t=31.2"] * 20


@pytest.mark.parametrize(
    ("text", "pieces"),
    [
        ("turn on", [b"turn", b" on"]),
        ("fan=2 t=31.2", [b"fan=", b"2", b" t=", b"3", b"1", b".", b"2"]),
        ("a  b", [b"a", b" ", b" b"]),
        ("end  ", [b"end", b"  "]),
        ("  x", [b" ", b" x"]),
        ("hi, you!", [b"hi", b",", b" you", b"!"]),
        ("café", [b"caf", b"\xc3", b"\xa9"]),
        (" 7", [b" 7"]),
        ("", []),
    ],
)
def test_pretokenize(text: str, pieces: list[bytes]) -> None:
    assert pretokenize(text.encode()) == pieces


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return train_bpe(CORPUS, SPECIALS, 400)


def test_training_is_deterministic(tok: Tokenizer) -> None:
    assert train_bpe(CORPUS, SPECIALS, 400).merges == tok.merges
    assert tok.vocab_size <= 400


def test_common_words_become_single_tokens(tok: Tokenizer) -> None:
    assert len(tok.encode_text(" fan")) == 1
    assert len(tok.encode_text("turn")) == 1


@pytest.mark.parametrize(
    "text", ["turn on the fan", "Unseen WORDS ☃ and emoji \U0001f600", "  spaced  ", "x=1"]
)
def test_round_trip(tok: Tokenizer, text: str) -> None:
    assert tok.decode(tok.encode(text)) == text


def test_specials_are_atomic_only_in_encode(tok: Tokenizer) -> None:
    ids = tok.encode("<bos><U> turn on</U><unsupported>")
    assert ids[:2] == [1, 3]
    assert ids[-2:] == [4, 5]
    assert 3 not in tok.encode_text("<U>")
    assert tok.encode("a < b") == tok.encode_text("a < b")


def test_json_and_blob_round_trip(tok: Tokenizer, tmp_path: Path) -> None:
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = Tokenizer.load(path)
    assert loaded.merges == tok.merges
    assert loaded.specials == tok.specials
    blob = tok.to_blob()
    assert len(blob) % 4 == 0
    again = Tokenizer.from_blob(blob)
    assert again.merges == tok.merges
    assert again.specials == tok.specials


def test_rejects_bad_versions_and_sizes(tok: Tokenizer) -> None:
    with pytest.raises(ValueError, match="version"):
        Tokenizer.from_json('{"version": 99, "specials": [], "merges": []}')
    with pytest.raises(ValueError, match="magic"):
        Tokenizer.from_blob(b"NOPE" + bytes(12))
    blob = bytearray(tok.to_blob())
    blob[4] = FORMAT_VERSION + 1
    with pytest.raises(ValueError, match="version"):
        Tokenizer.from_blob(bytes(blob))
    with pytest.raises(ValueError, match="vocab_size"):
        train_bpe(CORPUS, SPECIALS, 10)


def test_training_stops_when_no_pairs_remain() -> None:
    small = train_bpe(["ab"], (), 10_000)
    assert small.merges == [(97, 98)]
