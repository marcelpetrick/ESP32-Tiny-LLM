#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# ship.sh — run the full pipeline on the staged change, bump the version, commit, push.
#
# Usage: scripts/ship.sh [--minor|--major] [--no-push] "type(scope): subject" [BODY]
#
# Stage exactly the files of one atomic change first (git add ...). The working tree
# must not contain other unstaged or untracked changes, so the pipeline verifies
# precisely what gets committed. The subject must follow Conventional Commits.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() { sed -n '4,11p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

part="patch"
push=1
while [[ $# -gt 0 && "$1" == -* ]]; do
    case "$1" in
        -h | --help) usage; exit 0 ;;
        --minor) part="minor" ;;
        --major) part="major" ;;
        --no-push) push=0 ;;
        *) usage; die "unknown option: $1" ;;
    esac
    shift
done
subject="${1:-}"
body="${2:-}"
[[ -n "${subject}" ]] || { usage; die "missing commit subject"; }
conventional='^(feat|fix|docs|test|refactor|perf|build|ci|chore|style|revert)(\([a-z0-9-]+\))?!?: .+'
[[ "${subject}" =~ ${conventional} ]] || die "not a Conventional Commit subject: ${subject}"

cd "${REPO_ROOT}"
git diff --cached --quiet && die "nothing staged"
git diff --quiet || die "unstaged changes present; stage or stash them first"
[[ -z "$(git ls-files --others --exclude-standard)" ]] || die "untracked files present"

./localPipeline.sh || die "pipeline red — not committing"

scripts/bump_version.sh "${part}"
git add VERSION pyproject.toml uv.lock
if [[ -n "${body}" ]]; then
    git commit -q -m "${subject}" -m "${body}"
else
    git commit -q -m "${subject}"
fi
log "committed $(git log -1 --format='%h %s') (v$(cat VERSION))"
if [[ "${push}" -eq 1 ]]; then
    git push -q origin HEAD
    log "pushed to origin"
fi
