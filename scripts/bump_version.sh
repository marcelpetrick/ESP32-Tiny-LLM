#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# bump_version.sh — bump the SemVer in VERSION (mirrored into pyproject.toml and uv.lock).
#
# Usage: scripts/bump_version.sh [patch|minor|major]   (default: patch)
#        scripts/bump_version.sh --print               (print current version)
#
# Every commit bumps patch; completing a major feature/milestone bumps minor.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() { sed -n '4,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

part="${1:-patch}"
version_file="${REPO_ROOT}/VERSION"
current="$(tr -d '[:space:]' <"${version_file}")"
[[ "${current}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || die "VERSION is not SemVer: ${current}"
major="${BASH_REMATCH[1]}" minor="${BASH_REMATCH[2]}" patch="${BASH_REMATCH[3]}"

case "${part}" in
    -h | --help) usage; exit 0 ;;
    --print) echo "${current}"; exit 0 ;;
    patch) patch=$((patch + 1)) ;;
    minor) minor=$((minor + 1)) patch=0 ;;
    major) major=$((major + 1)) minor=0 patch=0 ;;
    *) usage; die "unknown part: ${part}" ;;
esac

next="${major}.${minor}.${patch}"
echo "${next}" >"${version_file}"
sed -i "s/^version = \"[0-9.]*\"/version = \"${next}\"/" "${REPO_ROOT}/pyproject.toml"
if [[ -f "${REPO_ROOT}/uv.lock" ]]; then
    sed -i "/^name = \"esp32-tiny-llm\"$/{n;s/^version = \".*\"/version = \"${next}\"/}" "${REPO_ROOT}/uv.lock"
fi
log "version ${current} -> ${next}"
