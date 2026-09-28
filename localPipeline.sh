#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# localPipeline.sh — the single quality gate for this repository.
# GitHub Actions runs exactly this script; never commit unless it is green.
#
# Usage: ./localPipeline.sh [--list] [--only a,b] [--skip a,b] [-h|--help]
#
#   --list        print the stages in order and exit
#   --only a,b    run only the named stages (setup always runs)
#   --skip a,b    run everything except the named stages (for local iteration only;
#                 a commit requires the full run)
#
# Logs of each stage are kept in .pipeline/logs/<stage>.log. The script prints a
# summary table and exits non-zero if any stage failed.
# shellcheck disable=SC2329 # stage_* functions are invoked indirectly via "stage_${name}"
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}" || exit 1
LOG_DIR="${ROOT_DIR}/.pipeline/logs"
mkdir -p "${LOG_DIR}"

# Stage name | description. Order matters.
STAGES=(
    "setup|install pinned Python (uv) and Node (npm) tooling"
    "headers|SPDX GPL-3.0-or-later headers on authored files"
    "shell|shellcheck on all shell scripts"
    "docs|markdownlint, relative links, Mermaid rendering"
    "c-format|clang-format --dry-run on the C runtime"
    "c-lint|cppcheck + clang-tidy on the C runtime"
    "c-tests|C unit tests (ASan/UBSan), coverage >= 95 %, release build"
    "py-format|ruff format --check"
    "py-lint|ruff check"
    "py-types|mypy --strict"
    "py-tests|pytest unit + integration with coverage >= 95 %"
    "e2e|browser end-to-end tests (Playwright + Chromium) against the web simulator"
)

usage() { sed -n '4,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

ONLY=""
SKIP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h | --help) usage; exit 0 ;;
        --list)
            for entry in "${STAGES[@]}"; do printf '  %-12s %s\n' "${entry%%|*}" "${entry#*|}"; done
            exit 0
            ;;
        --only) ONLY=",${2:-},"; shift 2 ;;
        --skip) SKIP=",${2:-},"; shift 2 ;;
        *) usage; exit 2 ;;
    esac
done

# ---------------------------------------------------------------- stage bodies
stage_setup() {
    uv sync --frozen && npm ci --no-audit --no-fund --ignore-scripts &&
        uv run --frozen playwright install chromium
}

stage_headers() { scripts/check_headers.sh; }

stage_shell() {
    mapfile -t scripts < <(git ls-files --cached --others --exclude-standard '*.sh')
    uv run --frozen shellcheck -x "${scripts[@]}"
}

stage_docs() { scripts/check_docs.sh; }

c_sources() {
    git ls-files --cached --others --exclude-standard 'runtime/*.c' 'runtime/*.h' 'firmware/*.c' 'firmware/*.h' |
        grep -v -E '(^|/)third_party/'
}

stage_c-format() {
    mapfile -t files < <(c_sources)
    uv run --frozen clang-format --dry-run --Werror "${files[@]}"
}

stage_c-lint() {
    cppcheck --enable=warning,style,performance,portability --error-exitcode=1 --inline-suppr --std=c99 \
        --quiet -I runtime/include runtime/src runtime/cli &&
        cmake -S runtime -B build/runtime -DCMAKE_BUILD_TYPE=Debug >/dev/null &&
        uv run --frozen clang-tidy -p build/runtime --quiet runtime/src/*.c runtime/cli/main.c
}

stage_c-tests() { scripts/c_tests.sh; }

stage_py-format() { uv run --frozen ruff format --check .; }

stage_py-lint() { uv run --frozen ruff check .; }

stage_py-types() { uv run --frozen mypy training tools web tests; }

stage_py-tests() {
    uv run --frozen pytest -m "not e2e" --cov --cov-report=term --cov-report=xml:.pipeline/coverage.xml \
        --cov-report=html:.pipeline/htmlcov --junitxml=.pipeline/pytest.xml
}

stage_e2e() { uv run --frozen pytest -m e2e -p no:cacheprovider --browser chromium; }

# ---------------------------------------------------------------- runner
declare -a RESULTS=()
FAILED=0

selected() {
    local name="$1"
    [[ "${name}" == "setup" ]] && return 0
    [[ -n "${ONLY}" && "${ONLY}" != *",${name},"* ]] && return 1
    [[ -n "${SKIP}" && "${SKIP}" == *",${name},"* ]] && return 1
    return 0
}

for entry in "${STAGES[@]}"; do
    name="${entry%%|*}"
    if ! selected "${name}"; then
        RESULTS+=("$(printf '%-12s %-6s %6s' "${name}" "SKIP" "-")")
        continue
    fi
    printf '\n==> [%s] %s\n' "${name}" "${entry#*|}"
    start="${SECONDS}"
    "stage_${name}" 2>&1 | tee "${LOG_DIR}/${name}.log"
    status="${PIPESTATUS[0]}"
    elapsed="$((SECONDS - start))s"
    if [[ "${status}" -eq 0 ]]; then
        RESULTS+=("$(printf '%-12s %-6s %6s' "${name}" "OK" "${elapsed}")")
    else
        RESULTS+=("$(printf '%-12s %-6s %6s' "${name}" "FAIL" "${elapsed}")")
        FAILED=1
        [[ "${name}" == "setup" ]] && break
    fi
done

printf '\n==================== pipeline summary ====================\n'
printf '%-12s %-6s %6s\n' "stage" "result" "time"
printf '%s\n' "${RESULTS[@]}"
if [[ "${FAILED}" -eq 0 ]]; then
    printf 'PIPELINE GREEN (version %s)\n' "$(cat VERSION)"
else
    printf 'PIPELINE RED — see .pipeline/logs/\n'
fi
exit "${FAILED}"
