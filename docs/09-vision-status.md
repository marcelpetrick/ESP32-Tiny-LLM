# 09 — Vision status: every item, honestly

This page checks [`vision.md`](../vision.md) item by item. Legend: **done** (implemented and
tested), **host** (done and verified on the desktop / in QEMU; the on-device measurement
needs the board), **partial**, **planned** (documented next step, not built). Nothing is
marked done that is not in the repository with a test or a reproducible script.

## Milestones (vision §21)

| Milestone | Status | Evidence |
|---|---|---|
| M0 hardware + benchmark harness | host | firmware with `/bandwidth`, `/gemv`, per-stage profiler, model partition, serial shell ([08-firmware.md](08-firmware.md)); numbers need the board |
| M1 reproduce a known tiny transformer | host | stories260K converted and run by our runtime; identical greedy output to upstream `run.c` (`tests/integration/test_llama2c_oracle.py`) |
| M2 our scalar runtime | done | `.tllm` format, tokenizer, C runtime; Python ↔ C logits within 1e-4 (`tests/integration/test_parity.py`); runs under ESP-IDF (QEMU) |
| M3 our first trained model | done | 4×128 transformer, byte-level BPE, trained and evaluated ([results](results/greenhouse-m.md)); a TinyStories-pretrained variant exists as experiment E2 |
| M4 INT8 production path | host | per-row INT8, W8A32 and W8A8 kernels, INT8 KV option, ≤ 0.1 pp quality loss; ≥ 5–10 tok/s is **estimated** (33–59 tok/s) until measured |
| M5 constrained chat dataset | done | ontology (`training/world`), generator, state tokens, multi-turn, held-out sets, teacher generation |
| M6 three-way training comparison | done | E1 scratch / E2 pretrained+fine-tuned / E3 teacher-distilled / E4 teacher-assistant KD ([results](results/experiments.md)) |
| M7 useful on-device assistant | host | language in, state injection, short answers, action/diag tokens, deterministic validator, regression suites; in QEMU the firmware answers over serial |
| M8 quality/performance sweep | done (host quality, estimated speed) | `tools/sweep.py`, [Pareto chart](results/sweep.md) |

## Success definitions (vision §23)

| Technical success | Status |
|---|---|
| one ESP32-S3, no network | firmware is fully local; runs in QEMU; board pending |
| custom, inspectable decoder transformer | done — every stage is a named C function ([05-architecture.md](05-architecture.md) §2) |
| local tokenization, causal attention, KV cache, autoregressive generation | done |
| ~1 MB INT8 class model | done — 804 KiB |
| interactive speed | estimated 26–48 tok/s; to be measured |
| reproducible benchmark numbers | runner + matrix ready (`tools/serial_runner.py`); host dry run works |

| Useful-model success | Status |
|---|---|
| natural phrasing | 85–88 % action accuracy on unseen phrasing, 98.5 % with typos |
| multi-turn references | 99.5 % |
| current device state | state block in every prompt; numbers grounded in 87–99 % of replies |
| short diagnostic explanations | yes (diagnosis rules as ground truth) |
| correct structured actions | 99.3 % in distribution; every action validated by firmware |
| safe fallback when unsure | `<clarify>` / `<unsupported>`; safety suite 99.6 % |
| 95 %+ of a carefully designed suite | met in distribution, multi-turn, typos and safety; unseen phrasing still 85–88 % |

## Architecture and runtime items (§2, §6–§7, §10–§16)

| Item | Status |
|---|---|
| decoder components of §2 | done |
| 4×128, RMSNorm, tied embeddings, learned positions, 64–128 context | done |
| RoPE, MQA/GQA, SwiGLU variants | done in runtime + training; compared in the sweep |
| 1K tokenizer with control tokens, digits, domain words, deterministic C/Python | done (1024) — 2K comparison: domain vocabulary saturates near 1024, see §Open items |
| language + structured intent, strict parser, firmware validation | done |
| FP32 reference → INT8 → INT4 | done; mixed precision (norms, positions, logits in f32); Q4 group-wise kernels (W4A32, W4A8) with no measurable quality loss at 446 KiB ([results](results/greenhouse-m.md)) |
| QAT | not needed: INT8 PTQ loses ≤ 0.1 pp |
| ESP-IDF, FreeRTOS pinning, `esp_timer`, heap caps, custom model partition | done |
| ESP-DSP / ESP-NN kernels | planned for board bring-up (P10); reference kernels are the differential baseline |
| model file with magic, version, CRC, offsets, alignment, manifest | done |
| inference loop with KV cache, prefill vs decode reported separately | done |
| memory placement (hot SRAM / PSRAM / flash) | done in firmware; measured placement report at boot |
| no heap allocation after init | done — asserted by tests |
| watchdog-safe scheduling | done — yield hook, watchdog stays enabled |
| sampling: greedy, temperature, top-k, repetition penalty, seed | done |
| stop conditions: EOS, turn end, max tokens, malformed grammar, context | done |
| observability commands (`/model-info`, `/memory`, `/profile`, `/kv-reset`, `/seed`, `/temp`, `/topk`, `/max-tokens`, `/benchmark`, per-layer checksums) | done |
| reproducible training manifests | done |
| licensing and dataset provenance | done ([THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md), [07](07-prior-art.md), manifests) |
| host CI | done — `./localPipeline.sh` mirrored by GitHub Actions |
| hardware CI / HIL with regression thresholds | planned — needs a self-hosted runner with the board |

## "What we should not do" (vision §22) — all respected

30 K vocabulary (we use 1024) · 512+ context (128) · INT4 first (INT8 first) · early
assembly (reference C first) · generic internet text (closed domain + TinyStories prior
only in E2) · direct hardware authority (validator) · assuming two cores double speed
(single inference core; dual-core only after measurement) · ignoring tokenizer/output
head (both profiled) · comparing parameter counts alone (bytes/token roofline) · hiding
behind a framework (custom runtime).

## Research branches (vision §24) and the 20 researched items (vision §25 appendix)

| # | Item | Status |
|---|---|---|
| 1 | llama2.c as permanent oracle | **done** — vendored `run.c` built in tests; identical greedy stories |
| 2 | bytes-per-token roofline | **done** — `tools/estimate.py`, feasibility doc; measured bandwidth via `/bandwidth` on the board |
| 3 | dual-core by arithmetic intensity | partial — row-split GEMV executor on both cores with a size threshold (`/parallel N`), verified identical in QEMU; the crossover needs the board |
| 4 | PSRAM/flash topology as a design variable | partial — placement logic + boot report + `/bandwidth` for all tiers |
| 5 | SRAM-only profile | partial — tier S in estimator and sweep (`s-2x96`); firmware already runs without PSRAM (QEMU) |
| 6 | output head as its own problem | partial — head profiled per token; vocabulary kept at 1024; factorised head planned |
| 7 | dense Q4 upper boundary | partial — Q4 format, kernels and a Q4 model shipped; the 1–8 M Q4 capacity sweep is planned |
| 8 | on-device learning on tiny surfaces | planned |
| 9 | purpose-built curriculum | **done** — oracle + staged generator + held-out templates + teacher style |
| 10 | deeper-and-thinner at fixed bytes | **done** — `6x96-deep-thin`, `8x80-deep-thin` in the sweep |
| 11 | factorised embeddings / layer sharing | planned |
| 12 | MQA/GQA as an early A/B | **done** — `mqa`, `gqa` in the sweep |
| 13 | SmoothQuant-style rescaling | not needed yet (W8A8 already within 0.1 pp) |
| 14 | activation-aware Q4 | not needed yet: plain group-wise Q4 already loses nothing measurable |
| 15 | rotation-based low-bit quantisation | planned |
| 16 | ternary/BitNet student | planned |
| 17 | LUT-based low-bit GEMV | planned |
| 18 | hybrid recurrent/local attention | planned |
| 19 | Espressif stack as operator oracle | planned (P10) |
| 20 | PSRAM speed/cache constraints, startup self-test | partial — boot self-test reports PSRAM mode/speed, cache configured; 120 MHz trial needs the board |
| §24 A | per-layer embeddings in flash | planned |
| §24 B | multi-query attention | **done** (runtime + sweep) |
| §24 C | sliding-window chat memory | **done** — console trims oldest exchanges; state is always current |
| §24 D | retrieval without a second LLM | planned |
| §24 E | hybrid classifier + generator | partial — keyword baseline exists for comparison |
| §24 F | sensor-token multimodality | planned (format does not assume text-only tokens) |
| §24 G | on-device adaptation | planned |
| §24 H | energy-aware inference | planned — needs a power meter |

## Open items

- **Everything measured on silicon** (M0 bandwidth, M4 tok/s, benchmark matrix, HIL CI):
  blocked only on attaching an ESP32-S3 N16R8; the firmware, runner and checklist are
  ready ([08-firmware.md](08-firmware.md) §5).
- **2K vocabulary comparison**: the domain corpus needs only ~1000 merges; a 2K vocabulary
  becomes meaningful with the chat-lite corpus (TinyStories-style text + domain).
- **Research items marked planned** above are deliberate follow-ups, not blockers of the
  product (vision §24: "explicitly out of the critical path").
