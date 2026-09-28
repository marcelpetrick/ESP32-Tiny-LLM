#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# update_plan.sh — refresh the auto-generated "Current state" block of plan.md
# (between <!-- ship:begin --> and <!-- ship:end -->): version, date, the commit being
# made, and the most recent history. scripts/ship.sh runs it for every commit.
#
# Usage: scripts/update_plan.sh ["subject of the commit being made"]
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

cd "${REPO_ROOT}"
plan="plan.md"
[[ -f "${plan}" ]] || die "plan.md missing"
grep -q '<!-- ship:begin -->' "${plan}" || die "plan.md has no ship block"
subject="${1:-}"
block="$(mktemp)"
{
    echo "<!-- ship:begin -->"
    echo
    echo "| Field | Value |"
    echo "|---|---|"
    echo "| Version | \`$(cat VERSION)\` |"
    echo "| Updated | $(date -u +%Y-%m-%d\ %H:%M) UTC |"
    [[ -n "${subject}" ]] && echo "| This commit | ${subject//|/\\|} |"
    echo
    echo "Recent commits:"
    echo
    # shellcheck disable=SC2016 # the backticks are Markdown code spans, not command substitution
    git log -8 --format='- `%h` %s' | sed 's/|/\\|/g'
    echo
    echo "<!-- ship:end -->"
} >"${block}"
python3 - "${plan}" "${block}" <<'PY'
import sys
plan, block = sys.argv[1], open(sys.argv[2], encoding="utf-8").read().rstrip("\n")
text = open(plan, encoding="utf-8").read()
start, end = text.index("<!-- ship:begin -->"), text.index("<!-- ship:end -->") + len("<!-- ship:end -->")
open(plan, "w", encoding="utf-8").write(text[:start] + block + text[end:])
PY
rm -f "${block}"
log "plan.md updated ($(cat VERSION))"
