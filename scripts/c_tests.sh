#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# c_tests.sh — C runtime test gate: unit tests under AddressSanitizer + UBSan, then a
# coverage build whose line coverage of runtime/src must be >= 95 % (gcovr), then the
# optimised release build used by the Python integration tests.
#
# Usage: scripts/c_tests.sh
#
# Reports: .pipeline/c-coverage/index.html and .pipeline/c-coverage.xml
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

cd "${REPO_ROOT}"
generator=()
have ninja && generator=(-G Ninja)

log "sanitizer build (ASan + UBSan)"
cmake -S runtime -B build/runtime-asan "${generator[@]}" -DCMAKE_BUILD_TYPE=Debug -DTLLM_SANITIZE=ON >/dev/null
cmake --build build/runtime-asan --parallel >/dev/null
(cd build/runtime-asan && ASAN_OPTIONS=detect_leaks=1 ./tests/tinyllm_tests)

log "coverage build"
cmake -S runtime -B build/runtime-cov "${generator[@]}" -DCMAKE_BUILD_TYPE=Debug -DTLLM_COVERAGE=ON >/dev/null
cmake --build build/runtime-cov --parallel >/dev/null
find build/runtime-cov -name '*.gcda' -delete
(cd build/runtime-cov && ./tests/tinyllm_tests >/dev/null)
mkdir -p .pipeline/c-coverage
uv run --frozen gcovr -r runtime build/runtime-cov --filter runtime/src --print-summary \
    --fail-under-line 95 --html-details .pipeline/c-coverage/index.html --xml .pipeline/c-coverage.xml

"${REPO_ROOT}/scripts/build_runtime.sh"
