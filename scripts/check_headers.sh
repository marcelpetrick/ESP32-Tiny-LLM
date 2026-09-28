#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# check_headers.sh — verify every authored source file carries the GPL SPDX header.
#
# Usage: scripts/check_headers.sh
#
# Checks tracked and new (non-ignored) files with source extensions; files under
# third_party/ keep their upstream headers and are skipped. Exits 1 on violations.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

cd "${REPO_ROOT}"
missing=0
checked=0
while IFS= read -r file; do
    [[ -f "${file}" ]] || continue
    case "${file}" in
        third_party/* | */third_party/* | */__init__.py) continue ;;
    esac
    case "${file}" in
        *.py | *.c | *.h | *.sh | *.js | *.css | *.cmake | *CMakeLists.txt | *Dockerfile | \
            *.yml | *.yaml | *.toml | *.html) ;;
        *) continue ;;
    esac
    checked=$((checked + 1))
    if ! head -n 6 "${file}" | grep -q 'SPDX-License-Identifier: GPL-3.0-or-later'; then
        echo "missing SPDX header: ${file}"
        missing=$((missing + 1))
    fi
done < <(git ls-files --cached --others --exclude-standard)

log "checked ${checked} file(s), ${missing} without header"
[[ "${missing}" -eq 0 ]]
