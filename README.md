# ESP32-Tiny-LLM

[![Local Pipeline](https://github.com/marcelpetrick/ESP32-Tiny-LLM/actions/workflows/local-pipeline.yml/badge.svg?branch=main)](https://github.com/marcelpetrick/ESP32-Tiny-LLM/actions/workflows/local-pipeline.yml)
[![License: GPL v3 or later](https://img.shields.io/badge/license-GPLv3%20or%20later-blue.svg)](LICENSE)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-3776ab.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-ee4c2c.svg)](https://pytorch.org/)
[![C99](https://img.shields.io/badge/C-99-00599c.svg)](https://en.cppreference.com/w/c/99)
[![ESP32-S3](https://img.shields.io/badge/target-ESP32--S3-e7352c.svg)](https://www.espressif.com/en/products/socs/esp32-s3)

A **real decoder-only transformer language model** — our own tokenizer, attention with a
KV cache, INT8 weights, a portable C runtime — designed to run fully locally on a single
**ESP32-S3** and to be *useful*: a narrow-domain conversational assistant that explains
its device's state, diagnoses faults, and proposes actions that firmware validates.

**Author: Marcel Petrick <mail@marcelpetrick.it>**

**License: GPLv3 or later. See [`LICENSE`](LICENSE) and
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).**

**Note: project is generated with AI.** Rules for AI agents: [`AGENTS.md`](AGENTS.md).

## Status

Current version: see [`VERSION`](VERSION). Phase **P0 (docs + scaffolding)** of the
[implementation plan](docs/06-implementation-plan.md).

## Documentation

| Page | What it covers |
|---|---|
| [Vision](vision.md) | the original project vision |
| [01 Feasibility](docs/01-feasibility.md) | hardware budget, bandwidth roofline, model tiers, speed estimates |
| [02 Use cases](docs/02-use-cases.md) | what transformers can usefully do on this MCU |
| [03 Toward a real chat model](docs/03-chat-model.md) | how far toward "real chat" the S3 goes |
| [04 Distillation](docs/04-distillation.md) | learning from larger LLMs, teacher licences |
| [05 Architecture](docs/05-architecture.md) | runtime, file format, safety boundary, firmware, web |
| [06 Implementation plan](docs/06-implementation-plan.md) | phases, tests, CI, vision traceability |
| [07 Prior art](docs/07-prior-art.md) | projects and papers we build on, with licences |

## Quality pipeline

```bash
./localPipeline.sh          # the full gate; GitHub Actions runs the same script
./localPipeline.sh --list   # show stages
```

Requirements: [uv](https://docs.astral.sh/uv/), Node.js ≥ 22, a Chrome/Chromium
(for Mermaid rendering). All Python and Node tool versions are pinned (`uv.lock`,
`package-lock.json`). Helper scripts are documented in [`scripts/README.md`](scripts/README.md).

## Versioning

[SemVer](https://semver.org/) in [`VERSION`](VERSION): every commit bumps the patch
number, completed milestones bump the minor number. Commits follow
[Conventional Commits](https://www.conventionalcommits.org/).
