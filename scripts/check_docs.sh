#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# check_docs.sh — documentation gate: markdownlint, relative links, Mermaid rendering,
# and presence of the required top-level documents.
#
# Usage: scripts/check_docs.sh
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

cd "${REPO_ROOT}"
for required in README.md AGENTS.md LICENSE THIRD_PARTY_NOTICES.md vision.md docs/README.md \
    scripts/README.md; do
    [[ -f "${required}" ]] || die "required document missing: ${required}"
done

mapfile -t md_files < <(git ls-files --cached --others --exclude-standard '*.md' |
    grep -v -E '^(node_modules|third_party)/')
npx --no-install markdownlint-cli2 "${md_files[@]}"
uv run --frozen python -m tools.check_links "${md_files[@]}"
"${REPO_ROOT}/scripts/render_mermaid.sh"
