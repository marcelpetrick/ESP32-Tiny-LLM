#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# build_runtime.sh — optimised host build of the C runtime (libtinyllm.so, tinyllm-cli).
#
# Usage: scripts/build_runtime.sh [BUILD_DIR]   (default: build/runtime-release)
#
# Used by the Python integration tests, the web simulator, and the Docker image.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

build_dir="${1:-${REPO_ROOT}/build/runtime-release}"
generator=()
have ninja && generator=(-G Ninja)
cmake -S "${REPO_ROOT}/runtime" -B "${build_dir}" "${generator[@]}" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "${build_dir}" --parallel >/dev/null
log "built ${build_dir}/libtinyllm.so and ${build_dir}/tinyllm-cli"
