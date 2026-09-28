# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Write and read ``.tllm`` model files (format v1, see docs/05-architecture.md §3).

Usage::

    python -m training.export --checkpoint runs/x/best.pt --out models/x.tllm --dtype i8

Layout: 128-byte header | tokenizer blob | tensor table | tensor data (16-byte aligned).
Everything after the header is covered by the CRC-32 stored in the header.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from training.model import ModelConfig, TinyLM
from training.quantize import quantize_rows
from training.tokenizer import Tokenizer

MAGIC = b"TLLM"
FORMAT_VERSION = 1
HEADER_SIZE = 128
ARCH_DECODER = 1
ALIGN = 16
DTYPE_F32 = 0
DTYPE_I8 = 1
NO_SCALE = 0xFFFFFFFF
NAME_LEN = 32
_HEADER = struct.Struct("<4sIII7I4I2I2III16sff")
_ENTRY = struct.Struct(f"<{NAME_LEN}sII4IIII")
_DTYPES = {"f32": DTYPE_F32, "i8": DTYPE_I8}


def _align(buf: bytearray) -> None:
    while len(buf) % ALIGN:
        buf.append(0)


def tensor_names(cfg: ModelConfig) -> list[str]:
    """Tensor names in file order."""
    names = ["tok_emb"]
    if cfg.pos_type == "learned":
        names.append("pos_emb")
    for i in range(cfg.n_layers):
        layer = ["attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "w1", "w2"]
        if cfg.mlp_type == "swiglu":
            layer.append("w3")
        names += [f"l{i}.{n}" for n in layer]
    names.append("final_norm")
    return names


def state_arrays(model: TinyLM) -> dict[str, np.ndarray]:
    """Map file tensor names to float32 numpy arrays from a :class:`TinyLM`."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    out: dict[str, np.ndarray] = {}
    for name in tensor_names(model.cfg):
        if name in ("tok_emb", "pos_emb"):
            key = f"{name}.weight"
        elif name == "final_norm":
            key = "final_norm.weight"
        else:
            layer, rest = name.split(".", 1)
            key = f"blocks.{layer[1:]}.{rest}.weight"
        out[name] = sd[key]
    return out


def _quantize(name: str, array: np.ndarray, dtype: int) -> bool:
    return dtype == DTYPE_I8 and array.ndim == 2 and name != "pos_emb"


def write_tllm(
    path: Path,
    cfg: ModelConfig,
    arrays: dict[str, np.ndarray],
    tokenizer: Tokenizer,
    dtype: str = "f32",
    model_id: bytes = bytes(16),
) -> dict[str, object]:
    """Write a ``.tllm`` file and return its manifest (also written as ``<path>.json``)."""
    weight_dtype = _DTYPES[dtype]
    if tokenizer.vocab_size != cfg.vocab_size:
        raise ValueError(f"tokenizer has {tokenizer.vocab_size} tokens, model {cfg.vocab_size}")
    names = tensor_names(cfg)
    payload = bytearray()
    tok_offset = HEADER_SIZE
    tok_blob = tokenizer.to_blob()
    payload += tok_blob
    _align(payload)
    table_offset = HEADER_SIZE + len(payload)
    payload += bytes(_ENTRY.size * len(names))
    _align(payload)
    entries: list[bytes] = []
    manifest_tensors = []
    for name in names:
        array = np.ascontiguousarray(arrays[name], dtype=np.float32)
        dims = [*array.shape, 0, 0, 0, 0][:4]
        data_offset = HEADER_SIZE + len(payload)
        scale_offset = NO_SCALE
        if _quantize(name, array, weight_dtype):
            q, scales = quantize_rows(array)
            raw = q.tobytes()
            payload += raw
            _align(payload)
            scale_offset = HEADER_SIZE + len(payload)
            payload += scales.astype("<f4").tobytes()
            tensor_dtype = DTYPE_I8
        else:
            raw = array.astype("<f4").tobytes()
            payload += raw
            tensor_dtype = DTYPE_F32
        _align(payload)
        entries.append(
            _ENTRY.pack(
                name.encode().ljust(NAME_LEN, b"\0"),
                tensor_dtype,
                array.ndim,
                *dims,
                data_offset,
                len(raw),
                scale_offset,
            )
        )
        manifest_tensors.append(
            {
                "name": name,
                "dtype": "i8" if tensor_dtype == DTYPE_I8 else "f32",
                "shape": list(array.shape),
                "offset": data_offset,
                "bytes": len(raw),
                "scale_offset": None if scale_offset == NO_SCALE else scale_offset,
            }
        )
    table_pos = table_offset - HEADER_SIZE
    payload[table_pos : table_pos + len(entries) * _ENTRY.size] = b"".join(entries)
    header = _HEADER.pack(
        MAGIC,
        FORMAT_VERSION,
        HEADER_SIZE,
        ARCH_DECODER,
        cfg.vocab_size,
        cfg.ctx_len,
        cfg.n_layers,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_kv_heads,
        cfg.d_ff,
        0 if cfg.mlp_type == "gelu" else 1,
        0 if cfg.pos_type == "learned" else 1,
        weight_dtype,
        0,
        tok_offset,
        len(tok_blob),
        table_offset,
        len(names),
        len(payload),
        zlib.crc32(payload),
        model_id,
        cfg.norm_eps,
        cfg.rope_theta,
    ).ljust(HEADER_SIZE, b"\0")
    data = header + bytes(payload)
    path.write_bytes(data)
    manifest: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "config": json.loads(cfg.to_json()),
        "weight_dtype": dtype,
        "file_bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "crc32": zlib.crc32(payload),
        "model_id": model_id.hex(),
        "params": cfg.param_count(),
        "tensors": manifest_tensors,
    }
    path.with_suffix(path.suffix + ".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


@dataclass
class TllmFile:
    """A parsed ``.tllm`` file (weights dequantised to float32)."""

    cfg: ModelConfig
    weight_dtype: str
    tokenizer: Tokenizer
    tensors: dict[str, np.ndarray]
    model_id: bytes
    crc32: int


class TllmFormatError(ValueError):
    """Raised for malformed or incompatible model files."""


def read_tllm(path: Path) -> TllmFile:
    """Parse and verify a ``.tllm`` file."""
    data = path.read_bytes()
    if len(data) < HEADER_SIZE:
        raise TllmFormatError("file too short")
    fields = _HEADER.unpack_from(data)
    magic, version, header_size, arch = fields[:4]
    if magic != MAGIC:
        raise TllmFormatError("bad magic")
    if version != FORMAT_VERSION or header_size != HEADER_SIZE or arch != ARCH_DECODER:
        raise TllmFormatError(f"unsupported version/header/arch {version}/{header_size}/{arch}")
    vocab, ctx, n_layers, d_model, n_heads, n_kv, d_ff = fields[4:11]
    mlp, pos, wdtype, _flags = fields[11:15]
    tok_off, tok_size, table_off, count, payload_size, crc = fields[15:21]
    model_id, eps, theta = fields[21:24]
    payload = data[HEADER_SIZE:]
    if len(payload) != payload_size:
        raise TllmFormatError("payload size mismatch")
    if zlib.crc32(payload) != crc:
        raise TllmFormatError("CRC mismatch")
    cfg = ModelConfig(
        vocab_size=vocab,
        ctx_len=ctx,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv,
        d_ff=d_ff,
        mlp_type="gelu" if mlp == 0 else "swiglu",
        pos_type="learned" if pos == 0 else "rope",
        norm_eps=float(str(np.float32(eps))),  # shortest repr survives the f32 round trip
        rope_theta=float(theta),
    )
    tokenizer = Tokenizer.from_blob(data[tok_off : tok_off + tok_size])
    tensors: dict[str, np.ndarray] = {}
    for k in range(count):
        name_raw, tdtype, ndim, *rest = _ENTRY.unpack_from(data, table_off + k * _ENTRY.size)
        dims, (offset, size, scale_off) = rest[:4], rest[4:]
        name = name_raw.rstrip(b"\0").decode()
        shape = tuple(dims[:ndim])
        raw = data[offset : offset + size]
        if tdtype == DTYPE_I8:
            q = np.frombuffer(raw, dtype=np.int8).reshape(shape)
            scales = np.frombuffer(data, dtype="<f4", count=shape[0], offset=scale_off)
            tensors[name] = q.astype(np.float32) * scales[:, None]
        else:
            tensors[name] = np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    return TllmFile(cfg, "i8" if wdtype == DTYPE_I8 else "f32", tokenizer, tensors, model_id, crc)


def load_model_from_tllm(tllm: TllmFile) -> TinyLM:
    """Build a :class:`TinyLM` with the (dequantised) weights of a parsed file."""
    model = TinyLM(tllm.cfg)
    sd = model.state_dict()
    for name, array in tllm.tensors.items():
        if name in ("tok_emb", "pos_emb"):
            key = f"{name}.weight"
        elif name == "final_norm":
            key = "final_norm.weight"
        else:
            layer, rest = name.split(".", 1)
            key = f"blocks.{layer[1:]}.{rest}.weight"
        sd[key] = torch.from_numpy(array.copy())
    model.load_state_dict(sd)
    return model.eval()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dtype", choices=sorted(_DTYPES), default="i8")
    args = parser.parse_args(argv)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ModelConfig.from_json(ckpt["config"])
    model = TinyLM(cfg)
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer.from_json(ckpt["tokenizer"])
    model_id = hashlib.sha256(json.dumps(ckpt["manifest"], sort_keys=True).encode()).digest()[:16]
    manifest = write_tllm(args.out, cfg, state_arrays(model), tok, args.dtype, model_id)
    manifest["training"] = ckpt["manifest"]
    args.out.with_suffix(args.out.suffix + ".json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"wrote {args.out} ({manifest['file_bytes']} bytes, {manifest['params']} params)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
