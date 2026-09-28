# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""llama2.c-compatible "scored pieces" tokenizer (tokenizer blob version 2).

The encode/decode algorithms follow ``encode()``/``decode()`` in Andrej Karpathy's
llama2.c ``run.c`` (MIT licence, see third_party/llama2c/LICENSE): a dummy-prefix space,
per-codepoint lookup with byte fallback (ids 3..258), then repeatedly merging the
adjacent pair whose concatenated piece exists with the highest score. It lets our
runtime load llama2.c checkpoints (vision M1) and compare against run.c as an oracle.

Blob v2 layout (little-endian): ``"TTOK" u32 version=2, u32 n_special, u32 vocab_size,
u32 max_piece_len, then per token: f32 score, u16 length, bytes`` (padded to 4 bytes).
"""

from __future__ import annotations

import re
import struct
from bisect import bisect_left
from pathlib import Path

BLOB_VERSION = 2
N_SPECIAL = 3  # <unk>, <s>, </s>
_BYTE_PIECE = re.compile(rb"^<0x([0-9A-Fa-f]{2})>$")


class ScoredTokenizer:
    """Vocabulary of byte strings with merge scores (llama2.c ``tokenizer.bin``)."""

    def __init__(
        self, pieces: list[bytes], scores: list[float], n_special: int = N_SPECIAL
    ) -> None:
        if len(pieces) != len(scores):
            raise ValueError("pieces and scores differ in length")
        self.pieces_raw = pieces
        self.scores = scores
        self.n_special = n_special
        self.specials = tuple(p.decode("utf-8", errors="replace") for p in pieces[:n_special])
        self._sorted = sorted(range(len(pieces)), key=lambda i: pieces[i])
        self._sorted_keys = [pieces[i] for i in self._sorted]
        self.pieces = [self._piece_bytes(p) for p in pieces]

    @property
    def vocab_size(self) -> int:
        return len(self.pieces_raw)

    @staticmethod
    def _piece_bytes(piece: bytes) -> bytes:
        match = _BYTE_PIECE.match(piece)
        return bytes([int(match.group(1), 16)]) if match else piece

    def lookup(self, piece: bytes) -> int:
        """Id of an exact vocabulary string, or -1."""
        i = bisect_left(self._sorted_keys, piece)
        if i < len(self._sorted_keys) and self._sorted_keys[i] == piece:
            return self._sorted[i]
        return -1

    def encode_text(self, text: str) -> list[int]:
        """Encode like llama2.c ``encode(t, text, bos=0, eos=0, ...)``."""
        data = text.encode("utf-8")
        ids: list[int] = []
        if data:
            ids.append(self.lookup(b" "))
        i = 0
        while i < len(data):
            j = i + 1
            while j < len(data) and (data[j] & 0xC0) == 0x80 and j - i < 4:
                j += 1
            cp = data[i:j]
            found = self.lookup(cp)
            if found != -1:
                ids.append(found)
            else:
                ids.extend(b + 3 for b in cp)
            i = j
        while True:
            best_score, best_id, best_i = -1e10, -1, -1
            for k in range(len(ids) - 1):
                cand = self.lookup(self.pieces_raw[ids[k]] + self.pieces_raw[ids[k + 1]])
                if cand != -1 and self.scores[cand] > best_score:
                    best_score, best_id, best_i = self.scores[cand], cand, k
            if best_i == -1:
                break
            ids[best_i : best_i + 2] = [best_id]
        return ids

    def encode(self, text: str) -> list[int]:
        return self.encode_text(text)

    def decode(self, ids: list[int], prev: int = -1) -> str:
        """Decode with llama2.c's rule: a leading space after BOS (id 1) is dropped."""
        out = bytearray()
        for token in ids:
            piece = self.pieces[token]
            if prev == 1 and piece.startswith(b" "):
                piece = piece[1:]
            out += piece
            prev = token
        return out.decode("utf-8", errors="replace")

    @classmethod
    def from_llama2c(cls, path: Path, vocab_size: int) -> ScoredTokenizer:
        """Read llama2.c's ``tokenizer.bin`` (vocab size comes from the checkpoint)."""
        data = path.read_bytes()
        pos = 4  # max_token_length
        pieces, scores = [], []
        for _ in range(vocab_size):
            score, length = struct.unpack_from("<fi", data, pos)
            pos += 8
            pieces.append(data[pos : pos + length])
            scores.append(score)
            pos += length
        return cls(pieces, scores)

    def to_blob(self) -> bytes:
        out = bytearray(b"TTOK")
        max_len = max(len(p) for p in self.pieces_raw)
        out += struct.pack("<IIII", BLOB_VERSION, self.n_special, self.vocab_size, max_len)
        for piece, score in zip(self.pieces_raw, self.scores, strict=True):
            out += struct.pack("<fH", score, len(piece)) + piece
        while len(out) % 4:
            out.append(0)
        return bytes(out)

    @classmethod
    def from_blob(cls, blob: bytes) -> ScoredTokenizer:
        if blob[:4] != b"TTOK":
            raise ValueError("bad tokenizer magic")
        version, n_special, vocab, _max_len = struct.unpack_from("<IIII", blob, 4)
        if version != BLOB_VERSION:
            raise ValueError(f"unsupported tokenizer version {version}")
        pos = 20
        pieces, scores = [], []
        for _ in range(vocab):
            score, length = struct.unpack_from("<fH", blob, pos)
            pos += 6
            pieces.append(blob[pos : pos + length])
            scores.append(score)
            pos += length
        return cls(pieces, scores, n_special)
