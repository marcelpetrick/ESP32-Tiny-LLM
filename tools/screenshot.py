# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Take the README screenshots of the web simulator with a scripted conversation.

Usage: python -m tools.screenshot --model greenhouse=M.tllm [--model stories=S.tllm] --out docs/images

Starts the real server in-process, drives Chromium through a short scenario (hot and humid
greenhouse, a complaint, a diagnosis, a refused command) and writes ``web-ui.png`` (light)
and ``web-ui-dark.png``. Needs ``playwright install chromium``.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

from web.server import Simulator, parse_models, serve

SCRIPT = ("why is it so sticky in here?", "is something wrong?", "turn on the heater")


def capture(url: str, out: Path, dark: bool) -> None:
    """Run the scripted conversation and save one screenshot."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(
            viewport={"width": 1440, "height": 900}, color_scheme="dark" if dark else "light"
        )
        page.goto(url)
        page.wait_for_selector("body[data-ready='1']")
        page.click("[data-testid=scenario-hot-humid]")
        for i, text in enumerate(SCRIPT, start=1):
            page.fill("[data-testid=input]", text)
            page.click("[data-testid=send]")
            page.wait_for_function(
                f"document.querySelectorAll('[data-testid=msg-bot]').length=={i}"
            )
        page.screenshot(path=str(out), full_page=True)
        browser.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/images"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sim = Simulator(parse_models(args.model))
    server = serve(sim, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        for dark in (False, True):
            path = args.out / ("web-ui-dark.png" if dark else "web-ui.png")
            capture(url, path, dark)
            print(f"wrote {path}")
    finally:
        server.shutdown()
        server.server_close()
        sim.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
