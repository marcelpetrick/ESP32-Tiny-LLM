# Next steps (handover)

Paused by the maintainer on 2026-09-30. This file says what is ongoing and what comes
next; the full status (done / waiting for the board / research) is [plan.md](plan.md).

## Ongoing: chat-lite small talk (docs/03-chat-model.md §2)

| Step | State |
|---|---|
| Small-talk generator, `--chat` dataset option, `smalltalk_test` suite, eval accepting any valid reply of a topic | **done**, committed (v0.10.4) |
| Curated teacher bank `data/teacher/smalltalk.json` (20 topics, ~730 user lines, 20 % held out, ~390 replies) | **done**, committed (v0.10.7) |
| Chat dataset `data/generated-chat` (local, git-ignored) | **done** |
| Tier-M chat model `runs/m-chat` (8000 steps, 1187 s, best val loss 0.024 at step 7500) | **done** (local, not committed) |
| INT8 export `runs/eval/m-chat.tllm` | **done** (local) |
| Evaluation on all suites incl. `smalltalk_test` | **stopped mid-run** on request — rerun |

## Next, in order

1. Evaluate (CPU, ~15 min):

   ```bash
   uv run python -m training.eval --model runs/eval/m-chat.tllm --data data/generated-chat \
       --splits test_id teacher_test heldout robust multiturn safety smalltalk_test \
       --json runs/eval/chat.json
   ```

   Compare with v3 (`docs/results/greenhouse-m.md`); a default model must keep safety at
   100 %. Try a few lines in `build/runtime-release/tinyllm-cli runs/eval/m-chat.tllm`.
2. If small talk is weak, train tier L once (`--preset L`, ~40 min on the GPU) and compare.
3. Document the results in `docs/results/greenhouse-m.md` and `docs/03-chat-model.md`,
   update `docs/09-vision-status.md`, move chat-lite to *Done* in `plan.md`.
4. If good: ship `models/greenhouse-m-chat-int8.tllm` as a research model, add it to the
   Docker web UI (`Dockerfile` CMD + `scripts/docker_smoke.sh`) and an integration test.
5. Release checks: final vision check (`docs/09-vision-status.md`), `./localPipeline.sh`
   green, GitHub Actions green, `docker pull ghcr.io/marcelpetrick/esp32-tiny-llm` + run
   (needs `gh auth refresh -h github.com -s read:packages`), clean tree, all pushed.

## Waiting for others

- The ESP32-S3 board (phase P10): measured bandwidth, tok/s, kernels, dual-core default,
  PSRAM 120 MHz, HIL CI — see plan.md.
