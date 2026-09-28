#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# run_experiments.sh — the distillation experiments of docs/04 §4 (vision M6) on one GPU.
# Every variant uses the tier-M architecture, the shipped tokenizer, and the identical
# evaluation files of data/generated, so only the training signal differs:
#
#   E1  scratch, oracle templates only (no teacher)      data/e1-templates
#   E2  TinyStories pretraining, then E1 data fine-tune  data/pretrain -> data/e1-templates
#   E3  teacher phrasing (= the shipped model, runs/m-v2)
#   E4  E3 data + logit distillation from tier L (teacher assistant, runs/l-v2)
#
# Usage: scripts/run_experiments.sh [STEPS]   (default 6000; resumable: finished runs are skipped)
# Needs: data/generated (+ tokenizer), data/cache/TinyStories-valid.txt, runs/l-v2, .venv-gpu.
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

steps="${1:-6000}"
cd "${REPO_ROOT}"
py=".venv-gpu/bin/python"
[[ -x "${py}" ]] || die "run scripts/gpu_env.sh first"
out="runs/experiments"
mkdir -p "${out}"

train() { # name data [extra args...]
    local name="$1" data="$2"
    shift 2
    if [[ -f "${out}/${name}/best.pt" ]]; then
        log "${name}: already trained"
    else
        log "${name}: training"
        "${py}" -u -m training.train --data "${data}" --out "${out}/${name}" --preset M "$@" >"${out}/${name}.log" 2>&1
    fi
}

evaluate() { # name checkpoint
    local name="$1" ckpt="$2"
    uv run --frozen python -m training.export --checkpoint "${ckpt}" --out "${out}/${name}.tllm" --dtype i8 >/dev/null
    uv run --frozen python -m training.eval --model "${out}/${name}.tllm" --data data/generated \
        --splits test_id teacher_test heldout robust multiturn safety --limit 1500 \
        --json "${out}/${name}.json" --title "${name}" >"${out}/${name}.md"
    log "${name}: evaluated"
}

[[ -d data/e1-templates ]] || uv run --frozen python -m training.data.dataset --out data/e1-templates \
    --tokenizer data/generated/tokenizer.json >/dev/null
[[ -d data/pretrain ]] || uv run --frozen python -m training.data.pretrain --text data/cache/TinyStories-valid.txt \
    --tokenizer data/generated/tokenizer.json --out data/pretrain >/dev/null

train e1-scratch data/e1-templates --steps "${steps}"
train e2-pretrain data/pretrain --steps 3000
train e2-finetune data/e1-templates --steps "${steps}" --init-from "${out}/e2-pretrain/best.pt"
train e4-ta-kd data/generated --steps "${steps}" --teacher runs/l-v2/best.pt --kd-alpha 0.5 --kd-temp 2.0

evaluate e1-scratch "${out}/e1-scratch/best.pt"
evaluate e2-finetune "${out}/e2-finetune/best.pt"
evaluate e3-teacher runs/m-v2/best.pt
evaluate e4-ta-kd "${out}/e4-ta-kd/best.pt"
log "results in ${out}/*.md and ${out}/*.json"
