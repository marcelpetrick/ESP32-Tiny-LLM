# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Shared fixtures: a running web simulator (in-process) with a chat and a story model."""

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from tools import convert_llama2c
from tools.runtime import REPO_ROOT
from training import export
from web.server import Simulator, serve

STORIES = REPO_ROOT / "models" / "third_party" / "stories260K"


@pytest.fixture(scope="session")
def web_models(tiny_run: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    out = tmp_path_factory.mktemp("web_models")
    chat = out / "greenhouse.tllm"
    export.main(["--checkpoint", str(tiny_run / "best.pt"), "--out", str(chat), "--dtype", "i8"])
    story = out / "stories.tllm"
    convert_llama2c.convert(STORIES / "stories260K.bin", STORIES / "tok512.bin", story)
    return {"greenhouse": chat, "stories": story}


@pytest.fixture(scope="session")
def web_url(web_models: dict[str, Path]) -> Iterator[str]:
    sim = Simulator(web_models)
    server = serve(sim, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    sim.close()
