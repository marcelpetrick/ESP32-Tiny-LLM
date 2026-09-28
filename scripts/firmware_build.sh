#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# firmware_build.sh — build the ESP32-S3 firmware in the official ESP-IDF container
# (pinned version), so local builds and CI are identical and no local IDF is needed.
#
# Usage: scripts/firmware_build.sh [idf.py arguments]   (default: build)
#   scripts/firmware_build.sh build            -> build/firmware/esp32_tiny_llm.bin
#   IDF_IMAGE=espressif/idf:v5.5.1 scripts/firmware_build.sh size-components
#
# Flashing needs the board attached: see docs/08-firmware.md (idf.py flash monitor).
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

image="${IDF_IMAGE:-espressif/idf:v5.5.1}"
have docker || die "docker is not installed"
[[ $# -gt 0 ]] || set -- build
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e IDF_COMPONENT_MANAGER=0 \
    -v "${REPO_ROOT}:/project" -w /project/firmware "${image}" \
    idf.py -B /project/build/firmware -DSDKCONFIG=/project/build/firmware/sdkconfig "$@"
if [[ "$1" == "build" ]]; then
    size="$(stat -c %s "${REPO_ROOT}/build/firmware/esp32_tiny_llm.bin")"
    log "firmware image build/firmware/esp32_tiny_llm.bin (${size} bytes)"
fi
