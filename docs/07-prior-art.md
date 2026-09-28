# 07 — Prior art, attribution, and how we use it

We build on other people's work where it helps and credit it clearly. This page lists
every project, paper, and dataset the design relies on, its licence (as checked on
2026-09-28 via the GitHub/Hugging Face APIs), and **exactly how we use it**. Code or data
that we actually ship is additionally listed in
[`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

Rule of thumb from [AGENTS.md](../AGENTS.md) §2.1: *no licence file means all rights
reserved* — we may read, cite, and benchmark against such projects, but we do not copy
their code.

## 1. Transformers on ESP32 and other MCUs

| Project | What it showed | Licence | How we use it |
|---|---|---|---|
| [karpathy/llama2.c](https://github.com/karpathy/llama2.c) | minimal Llama-2 training + single-file C inference | MIT | **numerical oracle**: vendored `run.c` in tests, format converter for its checkpoints (P3) |
| [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) | `stories260K` / `stories15M` checkpoints | MIT | hardware sanity model (vision M1) |
| [DaveBben/esp32-llm](https://github.com/DaveBben/esp32-llm) | llama2.c on ESP32-S3, 260 K params, 19.13 tok/s with ESP-DSP + dual core | none stated | cited for results and ideas only; no code reused |
| [doryiii/esp32-llm](https://github.com/doryiii/esp32-llm) | INT8 PIE path, 3.3 M params ≈ 12 tok/s, "bandwidth limit 13.1 tok/s", single core | none stated | cited; source of our 43 MB/s bandwidth estimate |
| [Circuit-Digest/ESP32-Tiny-LLM](https://github.com/Circuit-Digest/ESP32-Tiny-LLM) | flash streaming vs PSRAM comparison for a 260 K model | none stated | cited for the hardware-matrix idea |
| [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai) | 28.9 M stored / 559 K dense params via per-layer embeddings in flash, ≈ 9.5 tok/s; output head 57.6 ms/token | MIT | cited; per-component timings motivate our small vocabulary; PLE is research branch A |
| [manjunathshiva/esp32-tinyllm](https://github.com/manjunathshiva/esp32-tinyllm) | interactive storytelling with the PLE model | MIT | cited |
| [JARACH-209/esp32-30.7M](https://github.com/JARACH-209/esp32-30.7M) | 30.7 M dense Q4 (W4A8) at ≈ 0.95 tok/s | MIT | cited; upper-capacity reference for the sweep |
| [Ngducok/esp32s3-Super-mini-AI](https://github.com/Ngducok/esp32s3-Super-mini-AI) | micro-transformer inside 384 KB SRAM, no PSRAM, Wi-Fi web UI | MIT | cited; inspiration for tier S and the web UI |
| [Carloscodix/qapla](https://github.com/Carloscodix/qapla) | full training loop with hand-written backprop on the S3 | Apache-2.0 | cited; on-device learning is research branch G |
| [lspr98/conformer-stt-s3](https://github.com/lspr98/conformer-stt-s3) | 13.1 M-param Conformer speech recognition on the S3 | Apache-2.0 | cited; sensor/speech-sequence direction |

## 2. Espressif libraries

| Library | Licence | Planned use |
|---|---|---|
| [ESP-IDF](https://github.com/espressif/esp-idf) | Apache-2.0 | firmware framework |
| [ESP-DSP](https://github.com/espressif/esp-dsp) | Apache-2.0 | optimised fp32 dot products (P10) |
| [ESP-NN](https://github.com/espressif/esp-nn) | Apache-2.0 | optimised INT8 kernels (P10) |
| [ESP-DL](https://github.com/espressif/esp-dl) | MIT | operator oracle / benchmark reference (research item 19) |
| [ESP-SR](https://github.com/espressif/esp-sr) | Espressif licence (restricted to Espressif chips) | optional voice front-end; check terms before use |

## 3. Papers behind design decisions

| Topic | Paper | Where it shows up |
|---|---|---|
| Attention / transformer | [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762) | the whole runtime |
| Small models on simple data | [TinyStories, Eldan & Li 2023](https://arxiv.org/abs/2305.07759) | tier M demo, data design |
| Simple dialogue data | [TinyDialogues, Feng et al. 2024](https://arxiv.org/abs/2408.03617) | chat-lite |
| Simplified curricula | [TinyHelen, 2025](https://arxiv.org/abs/2501.00522) | teacher style guide |
| RMSNorm | [Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467) | norm layers |
| RoPE | [Su et al. 2021](https://arxiv.org/abs/2104.09864) | position variant |
| SwiGLU | [Shazeer 2020](https://arxiv.org/abs/2002.05202) | MLP variant |
| MQA / GQA | [Shazeer 2019](https://arxiv.org/abs/1911.02150), [Ainslie et al. 2023](https://aclanthology.org/2023.emnlp-main.298/) | `n_kv_heads` |
| Deep-and-thin small LMs | [MobileLLM, Liu et al. 2024](https://proceedings.mlr.press/v235/liu24ce.html) | sweep |
| Factorised embeddings, sharing | [ALBERT, Lan et al. 2019](https://arxiv.org/abs/1909.11942) | sweep |
| Knowledge distillation | [Hinton et al. 2015](https://arxiv.org/abs/1503.02531), [Kim & Rush 2016](https://arxiv.org/abs/1606.07947), [TAKD 2019](https://arxiv.org/abs/1902.03393), [GKD 2023](https://arxiv.org/abs/2306.13649), [MiniLLM 2023](https://arxiv.org/abs/2306.08543), [ULD 2024](https://arxiv.org/abs/2402.12030) | [04-distillation.md](04-distillation.md) |
| Quantisation | [SmoothQuant](https://proceedings.mlr.press/v202/xiao23c.html), [AWQ](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html), [QuaRot](https://www.microsoft.com/en-us/research/publication/quarot-outlier-free-4-bit-inference-in-rotated-llms/), [BitNet b1.58](https://arxiv.org/abs/2402.17764), [T-MAC](https://github.com/microsoft/T-MAC) | P4, research |
| Retrieval | [RAG, Lewis et al. 2020](https://arxiv.org/abs/2005.11401) | chat-lite facts |
| Constrained decoding | [llama.cpp grammars](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md) (MIT) | action-grammar masking idea |

## 4. Datasets and teachers

See the licence matrix in [04-distillation.md](04-distillation.md) §2.

## 5. Similar ideas elsewhere (for context)

- Coverage of the 2026 PLE result:
  [CNX Software](https://www.cnx-software.com/2026/08/03/28-9m-parameter-llm-runs-locally-on-esp32-s3-at-9-tokens-s/),
  [Hackster.io](https://www.hackster.io/news/running-a-28-9m-parameter-llm-on-an-8-microcontroller-173f1f370708).
- Overview article:
  [Running an LLM on an $8 microcontroller: what's real in 2026](https://dev.to/aiexplore369zoho/running-an-llm-on-an-8-microcontroller-whats-real-in-2026-35ak).

**What is different here:** those projects demonstrate story generation. Our focus is a
*useful* closed-domain assistant with state grounding, multi-turn references, a strict
action grammar with firmware validation, reproducible training/distillation, and a
host-verifiable pipeline — while reusing their measurements and, where licences allow,
their code.
