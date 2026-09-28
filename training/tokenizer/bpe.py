# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Deterministic byte-level BPE tokenizer (trainer, encoder, decoder, serialisation).

Token id layout::

    0 .. S-1          special tokens (see ``training.data.textformat.SPECIAL_TOKENS``)
    S .. S+255        raw bytes 0x00..0xFF
    S+256 ..          merges, in rank order

Pre-tokenisation works on UTF-8 *bytes* so that ``runtime/src/tokenizer.c`` can mirror it
exactly: a pre-token is an optional single leading space followed by either a run of ASCII
letters (plus one directly following ``=``), a single digit, or a single other byte. Runs
of spaces keep their last space for the next word. Merges never cross pre-token boundaries.

Encoding applies the lowest-rank adjacent merge repeatedly (leftmost on ties) — the same
algorithm as the C runtime.
"""

from __future__ import annotations

import json
import struct
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

FORMAT_VERSION = 1
BLOB_MAGIC = b"TTOK"


def _is_letter(b: int) -> bool:
    return 65 <= b <= 90 or 97 <= b <= 122


def pretokenize(data: bytes) -> list[bytes]:
    """Split UTF-8 bytes into pre-tokens (see module docstring)."""
    out: list[bytes] = []
    i, n = 0, len(data)
    while i < n:
        start = i
        if data[i] == 32:
            j = i
            while j < n and data[j] == 32:
                j += 1
            if j == n:  # trailing spaces
                out.append(data[i:j])
                break
            if j - i > 1:  # keep the last space for the next word
                out.append(data[i : j - 1])
                i = j - 1
                start = i
            i += 1  # the single leading space
        if _is_letter(data[i]):
            while i < n and _is_letter(data[i]):
                i += 1
            if i < n and data[i] == 61:  # "key=" stays one pre-token (state/action syntax)
                i += 1
        else:
            i += 1  # one digit or one other byte
        out.append(data[start:i])
    return out


@dataclass
class Tokenizer:
    """A trained BPE tokenizer."""

    specials: tuple[str, ...]
    merges: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.n_special = len(self.specials)
        self.byte_offset = self.n_special
        self.pieces: list[bytes] = [s.encode() for s in self.specials]
        self.pieces += [bytes([b]) for b in range(256)]
        self.ranks: dict[tuple[int, int], int] = {}
        for rank, (a, b) in enumerate(self.merges):
            self.ranks[(a, b)] = rank
            self.pieces.append(self.pieces[a] + self.pieces[b])
        self.special_ids = {s: i for i, s in enumerate(self.specials)}
        self._longest_first = sorted(self.specials, key=len, reverse=True)
        self._cache: dict[bytes, tuple[int, ...]] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.pieces)

    # ------------------------------------------------------------------ encoding
    def _encode_pretoken(self, piece: bytes) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached
        ids = [self.byte_offset + b for b in piece]
        while len(ids) > 1:
            best_rank, best_i = len(self.merges), -1
            for i in range(len(ids) - 1):
                rank = self.ranks.get((ids[i], ids[i + 1]))
                if rank is not None and rank < best_rank:
                    best_rank, best_i = rank, i
            if best_i < 0:
                break
            ids[best_i : best_i + 2] = [self.n_special + 256 + best_rank]
        result = tuple(ids)
        self._cache[piece] = result
        return result

    def encode_text(self, text: str) -> list[int]:
        """Encode plain text; special-token strings are treated as ordinary bytes."""
        out: list[int] = []
        for piece in pretokenize(text.encode("utf-8")):
            out.extend(self._encode_pretoken(piece))
        return out

    def encode(self, text: str) -> list[int]:
        """Encode text in which special-token strings (e.g. ``<U>``) are recognised."""
        out: list[int] = []
        i = 0
        plain_start = 0
        while i < len(text):
            if text[i] == "<":
                match = next((s for s in self._longest_first if text.startswith(s, i)), None)
                if match is not None:
                    out.extend(self.encode_text(text[plain_start:i]))
                    out.append(self.special_ids[match])
                    i += len(match)
                    plain_start = i
                    continue
            i += 1
        out.extend(self.encode_text(text[plain_start:]))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids to text (invalid UTF-8 is replaced)."""
        return b"".join(self.pieces[i] for i in ids).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ persistence
    def to_json(self) -> str:
        return json.dumps(
            {"version": FORMAT_VERSION, "specials": list(self.specials), "merges": self.merges},
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, text: str) -> Tokenizer:
        data = json.loads(text)
        if data.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer version {data.get('version')}")
        return cls(tuple(data["specials"]), [(int(a), int(b)) for a, b in data["merges"]])

    def save(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Tokenizer:
        return cls.from_json(path.read_text(encoding="utf-8"))

    def to_blob(self) -> bytes:
        """Binary form embedded in ``.tllm`` files (little-endian, see docs/05 §4)."""
        out = bytearray(BLOB_MAGIC)
        out += struct.pack("<III", FORMAT_VERSION, self.n_special, len(self.merges))
        for s in self.specials:
            raw = s.encode()
            out += struct.pack("<B", len(raw)) + raw
        for a, b in self.merges:
            out += struct.pack("<HH", a, b)
        while len(out) % 4:
            out.append(0)
        return bytes(out)

    @classmethod
    def from_blob(cls, blob: bytes) -> Tokenizer:
        if blob[:4] != BLOB_MAGIC:
            raise ValueError("bad tokenizer magic")
        version, n_special, n_merges = struct.unpack_from("<III", blob, 4)
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported tokenizer version {version}")
        pos = 16
        specials = []
        for _ in range(n_special):
            length = blob[pos]
            specials.append(blob[pos + 1 : pos + 1 + length].decode())
            pos += 1 + length
        merges = [struct.unpack_from("<HH", blob, pos + 4 * k) for k in range(n_merges)]
        return cls(tuple(specials), [(int(a), int(b)) for a, b in merges])


def train_bpe(texts: Iterable[str], specials: Sequence[str], vocab_size: int) -> Tokenizer:
    """Train merges on plain ``texts`` (special tokens must already be removed).

    Deterministic: the most frequent pair wins; ties go to the smallest ``(a, b)``.
    """
    n_base = len(specials) + 256
    if vocab_size < n_base:
        raise ValueError(f"vocab_size must be >= {n_base}")
    counts: Counter[bytes] = Counter()
    for text in texts:
        counts.update(pretokenize(text.encode("utf-8")))
    offset = len(specials)
    words: list[list[int]] = [[offset + b for b in piece] for piece in counts]
    freqs = list(counts.values())

    pair_counts: Counter[tuple[int, int]] = Counter()
    where: dict[tuple[int, int], set[int]] = {}
    for w_index, word in enumerate(words):
        for pair in pairwise(word):
            pair_counts[pair] += freqs[w_index]
            where.setdefault(pair, set()).add(w_index)

    merges: list[tuple[int, int]] = []
    while n_base + len(merges) < vocab_size:
        candidates = [(c, p) for p, c in pair_counts.items() if c > 0]
        if not candidates:
            break
        best_count = max(c for c, _ in candidates)
        best = min(p for c, p in candidates if c == best_count)
        new_id = n_base + len(merges)
        merges.append(best)
        for w_index in sorted(where.pop(best, set())):
            word = words[w_index]
            freq = freqs[w_index]
            for pair in pairwise(word):
                pair_counts[pair] -= freq
            merged: list[int] = []
            i = 0
            while i < len(word):
                if i + 1 < len(word) and (word[i], word[i + 1]) == best:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            words[w_index] = merged
            for pair in pairwise(merged):
                pair_counts[pair] += freq
                where.setdefault(pair, set()).add(w_index)
        pair_counts.pop(best, None)
    return Tokenizer(tuple(specials), merges)
