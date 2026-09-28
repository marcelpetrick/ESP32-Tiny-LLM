# scripts/

Small, documented helpers for humans, agents, and CI. Every script prints its usage with
`-h`/`--help`, starts with an SPDX header, and passes `shellcheck`.

| Script | Purpose |
|---|---|
| [`../localPipeline.sh`](../localPipeline.sh) | the full quality gate (`--list` shows stages); CI runs exactly this |
| [`ship.sh`](ship.sh) | pipeline on the staged change → bump version → conventional commit → push |
| [`bump_version.sh`](bump_version.sh) | bump `VERSION` (patch/minor/major), mirrored into `pyproject.toml` and `uv.lock` |
| [`check_headers.sh`](check_headers.sh) | verify SPDX GPL-3.0-or-later headers on authored files |
| [`check_docs.sh`](check_docs.sh) | markdownlint, relative-link check, Mermaid rendering, required docs present |
| [`gpu_env.sh`](gpu_env.sh) | create `.venv-gpu` with the CUDA build of the pinned PyTorch for GPU training |
| [`fix.sh`](fix.sh) | apply automatic fixes (ruff format/fix, clang-format, markdownlint --fix) |
| [`render_mermaid.sh`](render_mermaid.sh) | render all Mermaid diagrams to `.pipeline/mermaid/` (fails on syntax errors) |
| [`common.sh`](common.sh) | shared helpers (`log`, `die`, `find_chrome`) sourced by the others |

## Typical flow

```bash
git add <files of one logical change>
scripts/ship.sh "feat(runtime): add KV cache"            # patch bump
scripts/ship.sh --minor "feat(web): add web simulator"   # milestone → minor bump
```
