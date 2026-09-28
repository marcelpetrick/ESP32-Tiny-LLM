# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""ctypes binding to the C runtime's host API (``runtime/include/tinyllm/host.h``).

Used by the web simulator, the evaluation harness, and the Python <-> C parity tests.
The shared library is found via ``$TINYLLM_LIB`` or ``build/runtime-release/libtinyllm.so``
(build it with ``scripts/build_runtime.sh``).
"""

from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LIB = REPO_ROOT / "build" / "runtime-release" / "libtinyllm.so"


class TinyLLMError(RuntimeError):
    """Raised when the C runtime reports an error."""


def library_path() -> Path:
    """Path of the shared library to load."""
    return Path(os.environ.get("TINYLLM_LIB", str(DEFAULT_LIB)))


_lib: ctypes.CDLL | None = None


def load_library() -> ctypes.CDLL:
    """Load (once) and declare the C host API."""
    global _lib  # noqa: PLW0603 - process-wide singleton for the shared library
    if _lib is not None:
        return _lib
    lib = ctypes.CDLL(str(library_path()))
    c_int32_p = ctypes.POINTER(ctypes.c_int32)
    lib.tllm_host_open.restype = ctypes.c_void_p
    lib.tllm_host_open.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.tllm_host_close.argtypes = [ctypes.c_void_p]
    lib.tllm_host_submit.restype = ctypes.c_char_p
    lib.tllm_host_submit.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.tllm_host_last_json.restype = ctypes.c_char_p
    lib.tllm_host_last_json.argtypes = [ctypes.c_void_p]
    lib.tllm_host_vocab_size.argtypes = [ctypes.c_void_p]
    lib.tllm_host_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, c_int32_p, ctypes.c_int]
    lib.tllm_host_token_bytes.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int32,
        ctypes.c_char_p,
        ctypes.c_int,
    ]
    lib.tllm_host_logits.argtypes = [
        ctypes.c_void_p,
        c_int32_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.tllm_host_generate.argtypes = [
        ctypes.c_void_p,
        c_int32_p,
        ctypes.c_int,
        ctypes.c_int,
        c_int32_p,
    ]
    lib.tllm_host_alloc_count.restype = ctypes.c_long
    lib.tllm_host_device_eval.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    _lib = lib
    return lib


class Runtime:
    """One loaded model with its console (conversation + simulated device)."""

    def __init__(self, model_path: Path, act_int8: bool = False, kv_int8: bool = False) -> None:
        self._lib = load_library()
        err = ctypes.create_string_buffer(128)
        handle = self._lib.tllm_host_open(
            str(model_path).encode(), int(act_int8), int(kv_int8), err, len(err)
        )
        if not handle:
            raise TinyLLMError(f"cannot open {model_path}: {err.value.decode()}")
        self._handle: int | None = handle

    def close(self) -> None:
        """Release the model (idempotent)."""
        if self._handle is not None:
            self._lib.tllm_host_close(self._handle)
            self._handle = None

    def __enter__(self) -> Runtime:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def handle(self) -> int:
        if self._handle is None:
            raise TinyLLMError("runtime is closed")
        return self._handle

    def submit(self, line: str) -> str:
        """Send one console line; returns the console output text."""
        out: bytes = self._lib.tllm_host_submit(self.handle, line.encode())
        return out.decode("utf-8", errors="replace")

    def last_json(self) -> dict[str, Any]:
        """The JSON payload of the last ``@@`` line."""
        raw: bytes = self._lib.tllm_host_last_json(self.handle)
        result: dict[str, Any] = json.loads(raw.decode())
        return result

    def command(self, line: str) -> dict[str, Any]:
        """Submit a line and return its JSON payload."""
        self.submit(line)
        return self.last_json()

    @property
    def vocab_size(self) -> int:
        return int(self._lib.tllm_host_vocab_size(self.handle))

    def tokenize(self, text: str) -> list[int]:
        """Tokenise plain text with the C tokenizer."""
        raw = text.encode()
        cap = len(raw) + 16
        buf = (ctypes.c_int32 * cap)()
        n = self._lib.tllm_host_tokenize(self.handle, raw, buf, cap)
        if n < 0:
            raise TinyLLMError("tokenize failed")
        return list(buf[:n])

    def token_bytes(self, token: int) -> bytes:
        buf = ctypes.create_string_buffer(256)
        n = self._lib.tllm_host_token_bytes(self.handle, token, buf, len(buf))
        if n < 0:
            raise TinyLLMError(f"invalid token {token}")
        return buf.raw[:n]

    def logits(self, tokens: list[int]) -> list[float]:
        """Logits after feeding ``tokens`` into a fresh KV cache."""
        arr = (ctypes.c_int32 * len(tokens))(*tokens)
        out = (ctypes.c_float * self.vocab_size)()
        if self._lib.tllm_host_logits(self.handle, arr, len(tokens), out) != 0:
            raise TinyLLMError("forward failed")
        return list(out)

    def generate(self, prompt: list[int], max_new: int) -> list[int]:
        """Greedy continuation of exact prompt token ids (fresh KV cache, stops at EOS)."""
        arr = (ctypes.c_int32 * len(prompt))(*prompt)
        out = (ctypes.c_int32 * max(1, max_new))()
        n = self._lib.tllm_host_generate(self.handle, arr, len(prompt), max_new, out)
        if n < 0:
            raise TinyLLMError("generate failed")
        return list(out[:n])


def alloc_count() -> int:
    """Heap allocations performed by the C host layer so far."""
    return int(load_library().tllm_host_alloc_count())


def device_eval(settings: str, action: str) -> tuple[str, str]:
    """Run the C firmware rules: returns ``(verdict code, rendered state after)``."""
    buf = ctypes.create_string_buffer(256)
    if load_library().tllm_host_device_eval(settings.encode(), action.encode(), buf, len(buf)) != 0:
        raise TinyLLMError(f"bad settings {settings!r}")
    verdict, state = buf.value.decode().split("|", 1)
    return verdict, state
