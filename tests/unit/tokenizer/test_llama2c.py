# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""llama2.c-compatible scored tokenizer (Python mirror of run.c's encode/decode)."""

from pathlib import Path

import pytest

from tools.runtime import REPO_ROOT
from training.export import tokenizer_from_blob
from training.tokenizer import Tokenizer
from training.tokenizer.llama2c import ScoredTokenizer

TOK512 = REPO_ROOT / "models" / "third_party" / "stories260K" / "tok512.bin"


@pytest.fixture(scope="module")
def tok() -> ScoredTokenizer:
    return ScoredTokenizer.from_llama2c(TOK512, 512)


def test_reads_llama2c_vocab(tok: ScoredTokenizer) -> None:
    assert tok.vocab_size == 512
    assert tok.pieces_raw[3] == b"<0x00>"
    assert tok.pieces[3 + 0x41] == b"A"
    assert tok.lookup(b" the") > 258
    assert tok.lookup(b"no-such-piece") == -1


def test_encode_merges_by_score_and_decodes_like_run_c(tok: ScoredTokenizer) -> None:
    ids = tok.encode("Once upon a time")
    assert len(ids) == 4
    assert tok.decode([1, *ids]) == "\n<s>\nOnce upon a time"
    assert tok.decode(ids, prev=1) == "Once upon a time"
    assert tok.encode("") == []
    fallback = tok.encode("\u2603")  # snowman: not in the 512 vocab -> byte fallback
    assert fallback[-3:] == [0xE2 + 3, 0x98 + 3, 0x83 + 3]


def test_blob_round_trip_and_dispatch(tok: ScoredTokenizer) -> None:
    blob = tok.to_blob()
    again = tokenizer_from_blob(blob)
    assert isinstance(again, ScoredTokenizer)
    assert again.pieces_raw == tok.pieces_raw
    assert again.scores == tok.scores
    bpe = Tokenizer(("<pad>",), [])
    assert isinstance(tokenizer_from_blob(bpe.to_blob()), Tokenizer)


def test_rejects_malformed(tok: ScoredTokenizer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="magic"):
        ScoredTokenizer.from_blob(b"XXXX" + bytes(16))
    blob = bytearray(tok.to_blob())
    blob[4] = 9
    with pytest.raises(ValueError, match="version"):
        ScoredTokenizer.from_blob(bytes(blob))
    with pytest.raises(ValueError, match="length"):
        ScoredTokenizer([b"a"], [])
