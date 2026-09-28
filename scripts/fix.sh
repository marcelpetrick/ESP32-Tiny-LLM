#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# fix.sh — apply all automatic fixes: ruff format, safe ruff lint fixes, clang-format
# on C sources, and markdownlint --fix on Markdown.
#
# Usage: scripts/fix.sh
#
# Run before ./localPipeline.sh; remaining findings need manual attention.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

cd "${REPO_ROOT}"
uv run --frozen ruff check --fix --quiet . || true
uv run --frozen ruff format --quiet .
mapfile -t c_files < <(git ls-files --cached --others --exclude-standard '*.c' '*.h' |
    grep -v -E '(^|/)third_party/' || true)
if [[ ${#c_files[@]} -gt 0 ]]; then
    uv run --frozen clang-format -i "${c_files[@]}"
fi
mapfile -t md_files < <(git ls-files --cached --others --exclude-standard '*.md' |
    grep -v -E '^(vision\.md|node_modules/|third_party/)' || true)
npx --no-install markdownlint-cli2 --fix "${md_files[@]}" >/dev/null || true
log "automatic fixes applied"
