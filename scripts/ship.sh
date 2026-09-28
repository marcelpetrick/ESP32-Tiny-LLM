#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# ship.sh — run the full pipeline on the staged change, bump the version, update
# plan.md, commit, push.
#
# Usage: scripts/ship.sh [--minor|--major] [--no-push] "type(scope): subject" [BODY]
#
# Stage exactly the files of one atomic change first (git add ...). Unstaged and
# untracked changes are stashed while the pipeline runs and restored afterwards, so the
# pipeline verifies precisely what gets committed. The subject must follow
# Conventional Commits.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

usage() { sed -n '4,13p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

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
stashed=0
# plan.md is never stashed: every commit carries it (AGENTS.md §2.4) and it is rewritten below
if ! git diff --quiet -- . ':!plan.md' || [[ -n "$(git ls-files --others --exclude-standard -- . ':!plan.md')" ]]; then
    git stash push -q --keep-index --include-untracked -m "ship.sh: unrelated work" -- . ':!plan.md'
    stashed=1
    log "stashed unrelated work while the pipeline runs"
fi
restore() {
    if [[ "${stashed}" -eq 1 ]]; then
        git stash pop -q || warn "could not restore stash automatically; see 'git stash list'"
        stashed=0
    fi
}
trap restore EXIT

./localPipeline.sh || die "pipeline red — not committing"

scripts/bump_version.sh "${part}"
scripts/update_plan.sh "${subject}" # AGENTS.md §2.4: every commit updates plan.md
git add VERSION pyproject.toml uv.lock plan.md
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
