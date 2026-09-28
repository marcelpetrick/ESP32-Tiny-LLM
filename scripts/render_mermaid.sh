#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# render_mermaid.sh — render every Mermaid block in the given Markdown files.
#
# Usage: scripts/render_mermaid.sh [FILE.md ...]   (default: AGENTS.md README.md docs/*.md)
#
# Uses the pinned @mermaid-js/mermaid-cli from package.json with a system Chrome/Chromium
# (override with CHROME_PATH). SVGs land in .pipeline/mermaid/. Exits 1 if any diagram
# fails to render, so broken diagrams never reach GitHub.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

cd "${REPO_ROOT}"
chrome="$(find_chrome)" || die "no Chrome/Chromium found; set CHROME_PATH"
out_dir="${REPO_ROOT}/.pipeline/mermaid"
mkdir -p "${out_dir}"
config="${out_dir}/puppeteer.json"
printf '{"executablePath":"%s","args":["--no-sandbox"]}\n' "${chrome}" >"${config}"

if [[ $# -gt 0 ]]; then files=("$@"); else files=(AGENTS.md README.md docs/*.md); fi
failed=0
diagrams=0
for file in "${files[@]}"; do
    [[ -f "${file}" ]] || continue
    count="$(grep -c '^```mermaid' "${file}" || true)"
    [[ "${count}" -gt 0 ]] || continue
    name="$(echo "${file}" | tr '/' '_')"
    if npx --no-install mmdc -q -p "${config}" -i "${file}" -o "${out_dir}/${name}" \
        >"${out_dir}/${name}.log" 2>&1; then
        diagrams=$((diagrams + count))
    else
        echo "mermaid render failed: ${file}"
        sed -n '1,20p' "${out_dir}/${name}.log"
        failed=1
    fi
done
log "rendered ${diagrams} mermaid diagram(s)"
[[ "${failed}" -eq 0 ]]
