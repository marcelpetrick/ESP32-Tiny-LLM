#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# firmware_qemu.sh — boot the built firmware in Espressif's QEMU (esp32s3 machine with
# PSRAM) inside the ESP-IDF container and drive the serial console with a fixed script.
# Proves the firmware boots, loads and verifies the model and answers — no board needed.
# (Timing inside QEMU is meaningless; performance numbers need real hardware.)
#
# Usage: scripts/firmware_qemu.sh [LOG_FILE]
#
# Builds a QEMU flavour (quad PSRAM overlay, build/firmware-qemu) and boots it.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

image="${IDF_IMAGE:-espressif/idf:v5.5.1}"
fixture="turn on the fan, t=31.2 please!"
log_file="${1:-${REPO_ROOT}/.pipeline/firmware-qemu.log}"
mkdir -p "$(dirname "${log_file}")"
docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp -e IDF_COMPONENT_MANAGER=0 -v "${REPO_ROOT}:/project" \
    -w /project/firmware "${image}" idf.py -B /project/build/firmware-qemu -DSDKCONFIG=/project/build/firmware-qemu/sdkconfig \
    -DSDKCONFIG_DEFAULTS="sdkconfig.defaults;sdkconfig.qemu" build >/dev/null
docker run --rm -i -u "$(id -u):$(id -g)" -e HOME=/tmp -v "${REPO_ROOT}:/project" -w /project/build/firmware-qemu "${image}" \
    bash -c '. "$IDF_PATH/export.sh" >/dev/null 2>&1
        esptool.py --chip esp32s3 merge_bin --fill-flash-size 16MB -o /tmp/flash.bin @flash_args >/dev/null
        ( sleep 12; printf "/model-info\r"; sleep 6; printf "what is the temperature?\r"; sleep 20;
          printf "/tokenize turn on the fan, t=31.2 please!\r"; sleep 5 ) |
            timeout 60 qemu-system-xtensa -nographic -machine esp32s3 -m 4M \
                -drive file=/tmp/flash.bin,if=mtd,format=raw 2>&1 || true' | tee "${log_file}" >/dev/null
grep -q '@@{"event":"boot"' "${log_file}" || die "firmware did not boot (see ${log_file})"
grep -q '@@{"event":"model-info"' "${log_file}" || die "no /model-info answer (see ${log_file})"
grep -q '@@{"event":"reply"' "${log_file}" || die "no chat reply (see ${log_file})"
# tokenizer round trip: the chip must produce exactly the host runtime's ids (vision §7)
[[ -x "${REPO_ROOT}/build/runtime-release/tinyllm-cli" ]] || "${REPO_ROOT}/scripts/build_runtime.sh"
host_ids="$("${REPO_ROOT}/build/runtime-release/tinyllm-cli" "${REPO_ROOT}/models/greenhouse-m-int8.tllm" \
    -c "/tokenize ${fixture}" | grep '@@{"event":"tokens"' | tr -d '\r')"
chip_ids="$(grep -a '@@{"event":"tokens"' "${log_file}" | head -n1 | tr -d '\r')"
[[ -n "${host_ids}" && "${host_ids}" == "${chip_ids}" ]] || die "tokenizer differs: host ${host_ids} vs chip ${chip_ids}"
log "firmware boots in QEMU, answers over the serial console, tokenizes like the host (${log_file})"
