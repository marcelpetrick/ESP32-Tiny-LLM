#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# docker_smoke.sh — build the web simulator image and prove it works: health check,
# a chat turn through the C runtime, a story from the llama2.c model, and the CLI.
#
# Usage: scripts/docker_smoke.sh [IMAGE_TAG]   (default: esp32-tiny-llm:local)
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,9p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

tag="${1:-esp32-tiny-llm:local}"
cd "${REPO_ROOT}"
have docker || die "docker is not installed"
docker build -q -t "${tag}" . >/dev/null
name="tinyllm-smoke-$$"
docker run -d --rm --name "${name}" -p 127.0.0.1::8080 "${tag}" >/dev/null
trap 'docker stop "${name}" >/dev/null 2>&1 || true' EXIT
port="$(docker port "${name}" 8080/tcp | head -n1 | sed 's/.*://')"
url="http://127.0.0.1:${port}"
for _ in $(seq 1 60); do
    curl -fsS "${url}/healthz" >/dev/null 2>&1 && break
    sleep 0.5
done
curl -fsS "${url}/healthz" | grep -q '"ok": true' || die "health check failed"
reply="$(curl -fsS -X POST "${url}/api/chat" -H 'Content-Type: application/json' \
    -d '{"text":"what is the temperature?"}')"
grep -q '"event": "reply"' <<<"${reply}" || die "chat failed: ${reply}"
story="$(curl -fsS -X POST "${url}/api/chat" -H 'Content-Type: application/json' \
    -d '{"model":"stories","text":"Once upon a time"}')"
grep -q '"text": "Once upon a time' <<<"${story}" || die "story failed: ${story}"
docker exec "${name}" tinyllm-cli /app/models/greenhouse-m-int8.tllm -c /model-info | grep -q 'int8 weights' ||
    die "cli failed"
docker exec "${name}" tinyllm-cli /app/models/greenhouse-m-facts-int8.tllm -c "is there a warranty?" |
    grep -q '"fact":"warranty"' || die "facts model did not retrieve the fact"
log "docker image ${tag} OK (${url})"
