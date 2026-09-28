# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Hardware-in-the-loop benchmark runner (vision §19, §25).

Usage::

    python -m tools.serial_runner --port /dev/ttyACM0 --board "ESP32-S3-DevKitC-1 N16R8" \\
        [--out docs/results/benchmark.md] [--json out.json]
    python -m tools.serial_runner --host-model models/greenhouse-m-int8.tllm   # dry run on the PC

Sends the fixed benchmark suite to the console (serial port or the host runtime), collects
the ``@@{json}`` lines, and renders the benchmark matrix of vision §25. Numbers from a
board are **measured**; the host dry run only exercises the pipeline.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol

CASES = ("read", "diagnose", "command", "chat")


class Transport(Protocol):
    """Sends one console line, returns the JSON payloads of the ``@@`` lines it produced."""

    def exchange(self, line: str) -> list[dict[str, Any]]: ...


def parse_events(text: str) -> list[dict[str, Any]]:
    """All ``@@{json}`` payloads in console output."""
    events = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("@@{"):
            events.append(json.loads(line[2:]))
    return events


class HostTransport:
    """The desktop runtime (dry run of the whole pipeline without a board)."""

    def __init__(self, model: Path) -> None:
        from tools.runtime import Runtime  # noqa: PLC0415 - optional

        self.runtime = Runtime(model)

    def exchange(self, line: str) -> list[dict[str, Any]]:
        return parse_events(self.runtime.submit(line))


class SerialTransport:  # pragma: no cover - needs a board
    """A serial port speaking the firmware console protocol."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 60.0) -> None:
        import serial  # noqa: PLC0415 - pyserial is only needed with hardware

        self.port = serial.Serial(port, baud, timeout=1)
        self.timeout = timeout
        self.boot = self._read_until(lambda e: e.get("event") == "boot", 10.0)

    def _read_until(self, done: Any, timeout: float) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            raw = self.port.readline().decode("utf-8", errors="replace")
            events.extend(parse_events(raw))
            if events and done(events[-1]):
                break
        return events

    def exchange(self, line: str) -> list[dict[str, Any]]:
        self.port.write(line.encode() + b"\r")
        return self._read_until(lambda e: e.get("event") != "boot", self.timeout)


def run_suite(transport: Transport, boot: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Collect model info, memory, board benchmarks and per-case generation timings."""
    results: dict[str, Any] = {"boot": (boot or [{}])[-1] if boot else {}}
    results["model"] = transport.exchange("/model-info")[-1]
    results["memory"] = transport.exchange("/memory")[-1]
    for command in ("/bandwidth", "/gemv"):
        events = transport.exchange(command)
        results[command[1:]] = events[-1] if events and events[-1].get("event") != "error" else None
    results["cases"] = {case: transport.exchange(f"/benchmark {case}")[-1] for case in CASES}
    return results


def git_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _per_token(case: dict[str, Any], stage: str) -> float:
    tokens = max(1, int(case.get("prompt_tokens", 0)) + int(case.get("gen_tokens", 0)))
    return float(case.get("profile_us", {}).get(stage, 0)) / 1000.0 / tokens


def _shape(model: dict[str, Any]) -> str:
    heads = f"{model['n_heads']} ({model['n_kv_heads']} kv)"
    return f"{model['n_layers']} / {model['d_model']} / {heads} / {model['d_ff']}"


def _bandwidth(bw: dict[str, Any] | None) -> str:
    if not bw:
        return "n/a"
    return f"SRAM {bw['sram_mbs']} / PSRAM {bw['psram_mbs']} / flash {bw['flash_mbs']} MB/s"


def matrix(results: dict[str, Any], board: str, measured: bool) -> str:
    """Vision §25 benchmark matrix as Markdown."""
    boot, model, memory = results["boot"], results["model"], results["memory"]
    case = results["cases"]["diagnose"]
    prefill_tok_s = (
        case["prompt_tokens"] / (case["prefill_ms"] / 1000.0) if case["prefill_ms"] else 0.0
    )
    attn = sum(_per_token(case, s) for s in ("attn_score", "softmax", "attn_value", "attn_out"))
    label = "measured" if measured else "host dry run (not an ESP32 measurement)"
    rows = [
        ("Board", board),
        ("CPU frequency", f"{boot.get('cpu_mhz', 'n/a')} MHz"),
        ("Flash", f"{boot.get('flash_bytes', 0) >> 20} MB" if boot else "n/a"),
        (
            "PSRAM",
            f"{boot.get('psram_bytes', 0) >> 20} MB {boot.get('psram_mode', '')} @ {boot.get('psram_mhz', '')} MHz"
            if boot
            else "n/a",
        ),
        ("ESP-IDF version", boot.get("idf", "n/a")),
        ("Firmware revision", git_revision()),
        ("Model version", model["model_id"]),
        ("Parameters (dense)", f"{model['params']:,}"),
        ("Model bytes", f"{model['bytes']:,}"),
        (
            "Quantization",
            f"weights {model['weights']}, activations {model['activations']}, KV {model['kv_cache']}",
        ),
        ("Vocabulary", model["vocab_size"]),
        ("Context", f"{model['ctx_len']} tokens"),
        ("Layers / width / heads / FFN", _shape(model)),
        ("KV cache bytes", f"{memory['cold_arena']:,}"),
        ("Hot arena (internal SRAM)", f"{memory['hot_arena']:,}"),
        ("Heap report", memory.get("platform") or "n/a"),
        ("Memory bandwidth", _bandwidth(results.get("bandwidth"))),
        (
            "GEMV throughput",
            "n/a"
            if not results.get("gemv")
            else f"{results['gemv']['mmacs_a32']} MMAC/s (A32), {results['gemv']['mmacs_a8']} MMAC/s (A8)",
        ),
        ("Prompt length", f"{case['prompt_tokens']} tokens"),
        ("Prefill", f"{prefill_tok_s:.1f} tok/s"),
        ("Decode", f"{case['decode_tok_s']:.1f} tok/s"),
        ("First-token latency", f"{case['first_token_ms']:.1f} ms"),
        (
            "Mean / p95 token latency",
            f"{case['mean_token_ms']:.2f} / {case['p95_token_ms']:.2f} ms",
        ),
        ("Output head per token", f"{_per_token(case, 'head'):.3f} ms"),
        ("Attention per token", f"{attn:.3f} ms"),
        ("FFN per token", f"{_per_token(case, 'ffn'):.3f} ms"),
        ("Power / energy", "not measured"),
    ]
    lines = [f"### Benchmark matrix — {label}", "", "| Dimension | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in rows]
    lines += [
        "",
        "| Case | prompt tokens | generated | prefill ms | decode tok/s | mean token ms | p95 token ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, c in results["cases"].items():
        lines.append(
            f"| {name} | {c['prompt_tokens']} | {c['gen_tokens']} | {c['prefill_ms']:.1f} | "
            f"{c['decode_tok_s']:.1f} | {c['mean_token_ms']:.2f} | {c['p95_token_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--port")
    source.add_argument("--host-model", type=Path)
    parser.add_argument("--board", default="desktop host")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.port:  # pragma: no cover - needs a board
        serial_transport = SerialTransport(args.port)
        results = run_suite(serial_transport, serial_transport.boot)
    else:
        results = run_suite(HostTransport(args.host_model))
    text = matrix(results, args.board, measured=bool(args.port))
    print(text)
    if args.out:
        args.out.write_text(text)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
