# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Web simulator: the ESP32 console protocol behind a small HTTP API and a browser UI.

Usage::

    python -m web.server --model greenhouse=models/greenhouse-m-int8.tllm \\
        [--model stories=models/stories260K.tllm] [--host 127.0.0.1] [--port 8080]

Every request goes through exactly the same C runtime (``libtinyllm.so``) and console
command layer the firmware runs, so what the page shows is what the device would print
over its serial port: the reply, the proposed action, the firmware verdict, the device
state, and timings. Only the Python standard library is used on the serving path.
"""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from tools.runtime import Runtime
from training import project_version
from training.world import parse_state

STATIC_DIR = Path(__file__).resolve().parent / "static"
_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css",
    ".js": "text/javascript",
    ".svg": "image/svg+xml",
}
# Console commands the UI may send; everything else is rejected.
ALLOWED_COMMANDS = (
    "/set ",
    "/reset",
    "/kv-reset",
    "/profile ",
    "/temp ",
    "/seed ",
    "/benchmark ",
    "/generate",
    "/max-tokens ",
)
MAX_TEXT = 240


class Simulator:
    """Loaded models; one console per model, serialised by a lock (the C host is single-threaded)."""

    def __init__(self, models: dict[str, Path], act_int8: bool = False) -> None:
        if not models:
            raise ValueError("at least one model is required")
        self.lock = threading.Lock()
        self.runtimes = {name: Runtime(path, act_int8=act_int8) for name, path in models.items()}
        self.info = {name: rt.command("/model-info") for name, rt in self.runtimes.items()}
        for name, rt in self.runtimes.items():
            self.info[name]["mode"] = "story" if "story prompt" in rt.submit("/help") else "chat"
            rt.submit("/profile on")
            if self.info[name]["mode"] == "story":
                rt.submit("/max-tokens 256")

    def close(self) -> None:
        for rt in self.runtimes.values():
            rt.close()

    def _runtime(self, model: str | None) -> tuple[str, Runtime]:
        name = model or next(iter(self.runtimes))
        if name not in self.runtimes:
            raise KeyError(f"unknown model {name!r}")
        return name, self.runtimes[name]

    def describe(self) -> dict[str, Any]:
        return {
            "version": project_version(),
            "models": self.info,
            "default": next(iter(self.runtimes)),
        }

    def state(self, model: str | None = None) -> dict[str, Any]:
        with self.lock:
            name, rt = self._runtime(model)
            payload = rt.command("/state")
        return {"model": name, "state": state_dict(payload["state"]), "raw": payload["state"]}

    def chat(self, text: str, model: str | None = None) -> dict[str, Any]:
        text = text.strip()[:MAX_TEXT]
        if not text or text.startswith("/"):
            raise ValueError("message must be non-empty text")
        with self.lock:
            name, rt = self._runtime(model)
            output = rt.submit(text)
            reply = rt.last_json()
        result: dict[str, Any] = {"model": name, "reply": reply, "console": output}
        if "state" in reply:
            result["state"] = state_dict(reply["state"])
        return result

    def command(self, line: str, model: str | None = None) -> dict[str, Any]:
        line = line.strip()[:MAX_TEXT]
        if not (line in ("/reset", "/kv-reset", "/generate") or line.startswith(ALLOWED_COMMANDS)):
            raise ValueError(f"command not allowed: {line!r}")
        with self.lock:
            name, rt = self._runtime(model)
            output = rt.submit(line)
            payload = rt.last_json()
            state = rt.command("/state")["state"]
        return {"model": name, "result": payload, "console": output, "state": state_dict(state)}


def state_dict(text: str) -> dict[str, Any]:
    """Canonical state block -> JSON-friendly dict (None for offline sensors)."""
    state = parse_state(text)
    return {name: getattr(state, name) for name in state.__dataclass_fields__}


def make_handler(sim: Simulator) -> type[BaseHTTPRequestHandler]:
    """Request handler class bound to one simulator."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "tinyllm-web"

        def log_message(self, format: str, *args: Any) -> None:
            return  # quiet; the console output is the interesting log

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'"
            )
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: Any) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path == "/healthz":
                self._json(HTTPStatus.OK, {"ok": True})
            elif path == "/api/info":
                self._json(HTTPStatus.OK, sim.describe())
            elif path.startswith("/api/state"):
                query = self.path.partition("?")[2]
                model = dict(p.split("=", 1) for p in query.split("&") if "=" in p).get("model")
                try:
                    self._json(HTTPStatus.OK, sim.state(model))
                except KeyError as exc:
                    self._json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
            else:
                self._static(path)

        def _static(self, path: str) -> None:
            name = "index.html" if path in ("", "/") else path.lstrip("/").removeprefix("static/")
            target = (STATIC_DIR / name).resolve()
            if STATIC_DIR not in target.parents or not target.is_file():
                self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
                return
            self._send(
                HTTPStatus.OK,
                target.read_bytes(),
                _CONTENT_TYPES.get(target.suffix, "application/octet-stream"),
            )

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 4096:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
                return
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                model = body.get("model")
                if self.path == "/api/chat":
                    self._json(HTTPStatus.OK, sim.chat(str(body.get("text", "")), model))
                elif self.path == "/api/command":
                    self._json(HTTPStatus.OK, sim.command(str(body.get("line", "")), model))
                else:
                    self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            except (ValueError, KeyError) as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    return Handler


def parse_models(specs: list[str]) -> dict[str, Path]:
    """``name=path`` pairs (a bare path is named after its file stem)."""
    models: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = Path(spec).stem, spec
        models[name] = Path(path)
    return models


def serve(sim: Simulator, host: str, port: int) -> ThreadingHTTPServer:
    """Create (not start) the HTTP server."""
    return ThreadingHTTPServer((host, port), make_handler(sim))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", action="append", required=True, help="name=path.tllm (repeatable)"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--act-int8", action="store_true", help="W8A8 kernels for int8 models")
    args = parser.parse_args(argv)
    sim = Simulator(parse_models(args.model), args.act_int8)
    server = serve(sim, args.host, args.port)
    print(
        f"tinyllm web simulator on http://{args.host}:{server.server_port} ({', '.join(sim.runtimes)})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        pass
    finally:
        server.server_close()
        sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
