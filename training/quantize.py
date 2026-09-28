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


def dequantize_rows(q: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quantize_rows` (up to rounding)."""
    out: np.ndarray = q.astype(np.float32) * scales[:, None]
    return out


def fake_quantize(model: TinyLM) -> TinyLM:
    """Copy of ``model`` whose 2-D weights (except positions) are INT8-rounded (W8A32)."""
    import torch  # noqa: PLC0415 - optional heavy dependency

    clone = copy.deepcopy(model)
    with torch.no_grad():
        for name, param in clone.named_parameters():
            if param.dim() == 2 and not name.startswith("pos_emb"):
                q, s = quantize_rows(param.detach().cpu().numpy())
                param.copy_(torch.from_numpy(dequantize_rows(q, s)))
    return clone
