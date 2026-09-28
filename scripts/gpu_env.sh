#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# gpu_env.sh — create/refresh .venv-gpu with the CUDA build of the pinned PyTorch, for
# training on an NVIDIA GPU. The default .venv (used by the pipeline and CI) stays CPU-only.
#
# Usage: scripts/gpu_env.sh [CUDA_TAG]   (default: cu130)
#        .venv-gpu/bin/python -m training.train ...
set -euo pipefail
# shellcheck source=scripts/common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
[[ "${1:-}" =~ ^(-h|--help)$ ]] && { sed -n '4,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

cuda="${1:-cu130}"
cd "${REPO_ROOT}"
torch_version="$(sed -n 's/^ *"torch==\([0-9.]*\)",$/\1/p' pyproject.toml)"
numpy_version="$(sed -n 's/^ *"numpy==\([0-9.]*\)",$/\1/p' pyproject.toml)"
[[ -n "${torch_version}" && -n "${numpy_version}" ]] || die "could not read pins from pyproject.toml"
[[ -d .venv-gpu ]] || uv venv -q --python "$(cat .python-version)" .venv-gpu
uv pip install -q --python .venv-gpu/bin/python "numpy==${numpy_version}"
uv pip install -q --python .venv-gpu/bin/python "torch==${torch_version}" \
    --index-url "https://download.pytorch.org/whl/${cuda}"
.venv-gpu/bin/python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
