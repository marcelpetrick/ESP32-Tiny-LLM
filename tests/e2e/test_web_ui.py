# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
"""Browser end-to-end tests: page -> HTTP server -> C runtime -> page (Playwright)."""

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def open_app(page: Page, url: str) -> None:
    page.goto(url)
    expect(page.locator("body")).to_have_attribute("data-ready", "1")


def test_dashboard_and_chat(page: Page, web_url: str) -> None:
    open_app(page, web_url)
    expect(page.get_by_test_id("model-chips")).to_contain_text("int8 weights")
    expect(page.get_by_test_id("sensors")).to_contain_text("Temperature")
    page.get_by_test_id("scenario-hot-humid").click()
    expect(page.get_by_test_id("sensor-t")).to_contain_text("33.5")
    expect(page.get_by_test_id("sensor-t")).to_have_class("card alert")
    page.get_by_test_id("input").fill("why is it so sticky in here?")
    page.get_by_test_id("send").click()
    expect(page.get_by_test_id("msg-user")).to_have_count(1)
    expect(page.get_by_test_id("msg-bot")).to_have_count(1)
    expect(page.get_by_test_id("msg-bot").first).to_contain_text("tok/s")
    expect(page.get_by_test_id("stats")).to_contain_text("prompt tokens")
    expect(page.get_by_test_id("profile")).to_contain_text("ffn")
    page.get_by_test_id("clear").click()
    expect(page.get_by_test_id("msg-bot")).to_have_count(0)


def test_suggestion_chip_sends_message(page: Page, web_url: str) -> None:
    open_app(page, web_url)
    page.get_by_role("button", name="what can you do?").click()
    expect(page.get_by_test_id("msg-bot")).to_have_count(1)


def test_story_model(page: Page, web_url: str) -> None:
    open_app(page, web_url)
    page.get_by_test_id("model-select").select_option("stories")
    expect(page.locator("#device-panel")).to_be_hidden()
    page.get_by_test_id("input").fill("Once upon a time")
    page.get_by_test_id("input").press("Enter")
    bot = page.get_by_test_id("msg-bot")
    expect(bot).to_have_count(1)
    expect(bot.first).to_contain_text("Once upon a time,")
