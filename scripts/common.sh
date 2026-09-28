#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# common.sh — shared helpers sourced by the other scripts (not executed directly).
# Provides: REPO_ROOT, log/warn/die, have <cmd>, find_chrome.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT

log() { printf '[INFO] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*" >&2; }
die() {
    printf '[FAIL] %s\n' "$*" >&2
    exit 1
}
have() { command -v "$1" >/dev/null 2>&1; }

# Print the path of a Chrome/Chromium binary usable by puppeteer/playwright.
find_chrome() {
    local candidate
    for candidate in "${CHROME_PATH:-}" /usr/bin/chromium /usr/bin/chromium-browser \
        /usr/bin/google-chrome /usr/bin/google-chrome-stable; do
        if [[ -n "${candidate}" && -x "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done
    return 1
}
