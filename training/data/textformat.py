# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Exact text layout of prompts and targets (mirrored by ``runtime/src/console.c``).

::

    <bos>[<U> user</U><A> reply</A>[<ACT> body</ACT>]]*<S> state</S><U> user</U><A>
    target:  reply</A>[<ACT> body</ACT>]<eos>

Plain text following a special token starts with a single space, unless it itself
starts with a special token (``<clarify>``/``<unsupported>``).
"""

from __future__ import annotations

from training.data.dialogue import Sample, Turn
from training.world import render_state

SPECIAL_TOKENS = (
    "<pad>", "<bos>", "<eos>", "<S>", "</S>", "<U>", "</U>", "<A>", "</A>",
    "<ACT>", "</ACT>", "<clarify>", "<unsupported>",
)  # fmt: skip


def _seg(text: str) -> str:
    return text if text.startswith("<") else " " + text


def assistant_text(reply: str, action: str | None) -> str:
    """Assistant part after ``<A>``: reply, closing tag, optional action block."""
    out = _seg(reply) + "</A>"
    if action is not None:
        out += f"<ACT>{_seg(action)}</ACT>"
    return out


def history_text(turn: Turn) -> str:
    """One completed exchange as it appears in the context window."""
    return f"<U>{_seg(turn.user)}</U><A>" + assistant_text(turn.reply, turn.action)


def prompt_text(sample: Sample, history: int | None = None) -> str:
    """Prompt for the final turn, keeping the last ``history`` exchanges (all if None)."""
    turns = sample.history if history is None else sample.history[len(sample.history) - history :]
    if history == 0:
        turns = ()
    body = "".join(history_text(t) for t in turns)
    return f"<bos>{body}{render_state(sample.state)}<U>{_seg(sample.turn.user)}</U><A>"


def target_text(turn: Turn) -> str:
    """Training target: what the model must generate after the prompt."""
    return assistant_text(turn.reply, turn.action) + "<eos>"
