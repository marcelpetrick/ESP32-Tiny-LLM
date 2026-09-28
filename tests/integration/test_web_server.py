# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Web simulator HTTP API (in-process server, real C runtime)."""

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from web import server


def get(url: str) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url) as response:
            body = response.read()
            ctype = response.headers["Content-Type"]
            return response.status, json.loads(body) if "json" in ctype else body.decode()
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def post(url: str, payload: object, raw: bytes | None = None) -> tuple[int, Any]:
    data = raw if raw is not None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as err:
        return err.code, json.loads(err.read())


def test_info_health_and_static(web_url: str) -> None:
    status, info = get(f"{web_url}/api/info")
    assert status == 200
    assert info["default"] == "greenhouse"
    assert info["models"]["greenhouse"]["mode"] == "chat"
    assert info["models"]["stories"]["mode"] == "story"
    assert get(f"{web_url}/healthz") == (200, {"ok": True})
    status, page = get(f"{web_url}/")
    assert status == 200
    assert "<title>ESP32 Tiny LLM</title>" in page
    assert get(f"{web_url}/static/app.js")[0] == 200
    assert get(f"{web_url}/static/../server.py")[0] == 404
    assert get(f"{web_url}/nope.txt")[0] == 404


def test_chat_command_and_state(web_url: str) -> None:
    status, data = post(f"{web_url}/api/command", {"line": "/set t=34 h=80"})
    assert status == 200
    assert data["state"]["t"] == 34.0
    status, data = post(f"{web_url}/api/chat", {"text": "why is it so humid?"})
    assert status == 200
    assert data["reply"]["event"] == "reply"
    assert "profile_us" in data["reply"]
    assert data["state"]["h"] == 80
    status, state = get(f"{web_url}/api/state?model=greenhouse")
    assert state["state"]["h"] == 80
    assert get(f"{web_url}/api/state?model=missing")[0] == 404
    status, story = post(f"{web_url}/api/chat", {"model": "stories", "text": "Once upon a time"})
    assert story["reply"]["event"] == "generate"
    assert story["reply"]["text"].startswith("Once upon a time")
    assert "state" not in story


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/api/chat", {"text": ""}),
        ("/api/chat", {"text": "/set t=1"}),
        ("/api/chat", {"text": "hi", "model": "nope"}),
        ("/api/command", {"line": "/quit"}),
        ("/api/unknown", {}),
    ],
)
def test_rejections(web_url: str, path: str, payload: dict[str, str]) -> None:
    status, body = post(f"{web_url}{path}", payload)
    assert status in (400, 404)
    assert "error" in body


def test_oversized_and_malformed_requests(web_url: str) -> None:
    assert post(f"{web_url}/api/chat", None, raw=b"x" * 5000)[0] == 413
    assert post(f"{web_url}/api/chat", None, raw=b"{not json")[0] == 400


def test_helpers_and_cli_parsing(
    web_models: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert server.parse_models(["a=x.tllm", "models/b.tllm"]) == {
        "a": Path("x.tllm"),
        "b": Path("models/b.tllm"),
    }
    assert (
        server.state_dict(
            "<S> t=na h=55 soil=50 fan=0 heat=0 pump=0 light=0 win=0 pa=0.0 vib=0 err=0</S>"
        )["t"]
        is None
    )
    with pytest.raises(ValueError, match="at least one"):
        server.Simulator({})

    started: dict[str, object] = {}

    class FakeServer:
        server_port = 1234

        def serve_forever(self) -> None:
            started["served"] = True

        def server_close(self) -> None:
            started["closed"] = True

    monkeypatch.setattr(server, "serve", lambda sim, host, port: FakeServer())
    assert server.main(["--model", f"greenhouse={web_models['greenhouse']}", "--port", "0"]) == 0
    assert started == {"served": True, "closed": True}
