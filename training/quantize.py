# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Symmetric per-row INT8 quantisation (vision §11 phase 1).

Each output row ``r`` of a weight matrix gets its own scale ``s_r = max|W[r,:]| / 127``;
``q = round(W / s_r)`` clamped to ``[-127, 127]``. The C runtime dequantises as
``W ≈ q * s_r`` (W8A32) or accumulates ``q · q_x`` in int32 (W8A8).
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from training.model import TinyLM

QMAX = 127


def quantize_rows(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Quantise a 2-D float array per row; returns ``(int8 values, float32 scales)``."""
    w = np.asarray(weight, dtype=np.float32)
    absmax = np.abs(w).max(axis=1)
    scales = np.where(absmax > 0, absmax / QMAX, 1.0).astype(np.float32)
    q = np.clip(np.rint(w / scales[:, None]), -QMAX, QMAX).astype(np.int8)
    return q, scales


Q4_GROUP = 32
Q4_MAX = 7


def quantize_q4(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group-wise symmetric 4-bit quantisation (vision §11 phase 3, O9).

    Each row is split into groups of :data:`Q4_GROUP` columns with one float16 scale
    ``max|w| / 7``; values in ``[-7, 7]`` are stored offset by 8 as nibbles, two per byte
    (low nibble = even column). Returns ``(packed uint8 [rows, cols/2], float16 scales
    [rows, cols/Q4_GROUP])``.
    """
    w = np.asarray(weight, dtype=np.float32)
    rows, cols = w.shape
    if cols % Q4_GROUP:
        raise ValueError(f"columns ({cols}) must be a multiple of {Q4_GROUP}")
    groups = w.reshape(rows, cols // Q4_GROUP, Q4_GROUP)
    absmax = np.abs(groups).max(axis=2)
    scales = np.where(absmax > 0, absmax / Q4_MAX, 1.0).astype(np.float16)
    q = np.clip(np.rint(groups / scales.astype(np.float32)[:, :, None]), -Q4_MAX, Q4_MAX).astype(
        np.int16
    )
    nibbles = (q + 8).astype(np.uint8).reshape(rows, cols)
    packed = (nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)).astype(np.uint8)
    return packed, scales


def dequantize_q4(packed: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_q4` (up to rounding)."""
    rows = packed.shape[0]
    nibbles = np.empty((rows, packed.shape[1] * 2), dtype=np.int16)
    nibbles[:, 0::2] = packed & 0x0F
    nibbles[:, 1::2] = packed >> 4
    values = (nibbles - 8).astype(np.float32).reshape(rows, -1, Q4_GROUP)
    out: np.ndarray = (values * scales.astype(np.float32)[:, :, None]).reshape(rows, -1)
    return out


def dequantize_rows(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_rows` (up to rounding)."""
    out: np.ndarray = q.astype(np.float32) * scales[:, None]
    return out


def fake_quantize(model: TinyLM, bits: int = 8) -> TinyLM:
    """Copy of ``model`` whose 2-D weights (except positions) are INT8- or Q4-rounded.

    With ``bits=4``, matrices whose width is not a multiple of the group size stay INT8,
    exactly like the exporter.
    """
    import torch  # noqa: PLC0415 - optional heavy dependency

    clone = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in clone.named_parameters():
            if param.dim() == 2 and not name.startswith("pos_emb"):
                w = param.detach().cpu().numpy()
                if bits == 4 and w.shape[1] % Q4_GROUP == 0:
                    param.copy_(torch.from_numpy(dequantize_q4(*quantize_q4(w))))
                else:
                    q, s = quantize_rows(w)
                    param.copy_(torch.from_numpy(dequantize_rows(q, s)))
    return clone
