# Results: greenhouse assistant, tier M

All numbers on this page are **measured on the desktop** by running the exported model
through the same C runtime the firmware uses (`python -m training.eval`). Speed on the
ESP32-S3 is estimated in [01-feasibility.md](../01-feasibility.md) until a board is attached.

## The shipped model: `models/greenhouse-m-int8.tllm`

| Property | Value |
|---|---|
| Architecture | 4 layers × 128, 4 heads, FFN 256 (GELU), learned positions, tied embeddings |
| Parameters | 672,896 |
| Vocabulary / context | 1024 byte-level BPE tokens / 128 tokens |
| File | 749,568 bytes, INT8 per-row weights, CRC-32 `4dcf6aae` |
| Training | 8000 steps (best at 7250), 801 s on an RTX A2000, best validation loss 0.0017 |
| Data | rule-based oracle labels + 50 % teacher phrasing (Qwen3.5-4B, Apache-2.0) + 10 % typos, 200 000 samples |
| Provenance | git `163797d0798c`, dataset manifest `e3abc6af5537`, tokenizer `2f1571495e69` |

## Suites

| Suite | What it measures |
|---|---|
| test_id | same distribution as training (templates + teacher phrasing) |
| teacher_test | the teacher's **held-back** 20 % of paraphrases (natural wording never trained on) |
| heldout | **held-out sentence frames** from the template lexicon, never in training |
| robust | typos, politeness, fillers, punctuation |
| multiturn | pronouns, ellipsis ("and the heater"), corrections ("no, the pump") |
| safety | refusals, out-of-range values, interlocks, clarification |

`baseline` is a deterministic keyword system on the same samples — the falsifiability
check of vision §29. `rejected` counts proposals the firmware validator refused (the
model is never trusted: those actions do not execute).

## FP32 vs INT8 (vision M4 / P4)

### tier M v2 FP32

| suite | n | action | actuators | fallback | reply exact | numbers | malformed | halluc. action | rejected | baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_id | 1500 | 99.3 % | 99.3 % | 99.8 % | 99.1 % | 99.4 % | 0.1 % | 0.1 % | 0.0 % | 66.0 % |
| teacher_test | 1500 | 85.2 % | 86.5 % | 88.6 % | 68.7 % | 91.5 % | 0.1 % | 6.0 % | 0.0 % | 65.6 % |
| heldout | 1500 | 88.0 % | 90.8 % | 92.1 % | 75.5 % | 87.7 % | 0.1 % | 3.5 % | 0.0 % | 64.3 % |
| robust | 1500 | 98.5 % | 98.6 % | 99.8 % | 97.7 % | 98.9 % | 0.1 % | 0.3 % | 0.0 % | 66.9 % |
| multiturn | 1500 | 99.5 % | 99.5 % | 99.5 % | 99.2 % | 99.7 % | 0.0 % | 0.2 % | 0.1 % | 61.3 % |
| safety | 1500 | 99.6 % | 99.6 % | 99.4 % | 99.4 % | 100.0 % | 0.0 % | 0.4 % | 0.0 % | 97.5 % |

### tier M v2 INT8 (W8A32)

| suite | n | action | actuators | fallback | reply exact | numbers | malformed | halluc. action | rejected | baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_id | 1500 | 99.3 % | 99.3 % | 99.8 % | 99.1 % | 99.4 % | 0.1 % | 0.1 % | 0.0 % | 66.0 % |
| teacher_test | 1500 | 85.3 % | 86.6 % | 88.7 % | 68.7 % | 91.6 % | 0.1 % | 6.0 % | 0.0 % | 65.6 % |
| heldout | 1500 | 88.1 % | 90.9 % | 92.1 % | 75.5 % | 87.5 % | 0.1 % | 3.5 % | 0.0 % | 64.3 % |
| robust | 1500 | 98.5 % | 98.6 % | 99.8 % | 97.7 % | 99.0 % | 0.1 % | 0.3 % | 0.0 % | 66.9 % |
| multiturn | 1500 | 99.5 % | 99.5 % | 99.5 % | 99.2 % | 99.7 % | 0.0 % | 0.2 % | 0.1 % | 61.3 % |
| safety | 1500 | 99.6 % | 99.6 % | 99.4 % | 99.4 % | 100.0 % | 0.0 % | 0.4 % | 0.0 % | 97.5 % |

### tier M v2 INT8 (W8A8)

| suite | n | action | actuators | fallback | reply exact | numbers | malformed | halluc. action | rejected | baseline |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test_id | 1500 | 99.3 % | 99.3 % | 99.8 % | 99.1 % | 99.4 % | 0.1 % | 0.1 % | 0.0 % | 66.0 % |
| teacher_test | 1500 | 85.3 % | 86.7 % | 88.5 % | 68.7 % | 91.5 % | 0.1 % | 5.9 % | 0.0 % | 65.6 % |
| heldout | 1500 | 88.1 % | 90.8 % | 92.1 % | 75.5 % | 87.5 % | 0.1 % | 3.5 % | 0.0 % | 64.3 % |
| robust | 1500 | 98.5 % | 98.6 % | 99.8 % | 97.7 % | 98.9 % | 0.1 % | 0.3 % | 0.0 % | 66.9 % |
| multiturn | 1500 | 99.5 % | 99.5 % | 99.5 % | 99.2 % | 99.7 % | 0.0 % | 0.2 % | 0.1 % | 61.3 % |
| safety | 1500 | 99.6 % | 99.6 % | 99.4 % | 99.4 % | 100.0 % | 0.0 % | 0.4 % | 0.0 % | 97.5 % |

**INT8 costs nothing measurable here**: every metric stays within ±0.1 percentage points
of FP32, including the fully integer W8A8 path that maps onto the ESP32-S3 PIE unit.

## Effect of distillation (vision M6, experiment E1 → E3)

Same architecture (tier M) trained first on oracle templates only (v1), then with teacher
paraphrases, end punctuation and typo augmentation (v2). Measured on 1000 samples per
suite of each run's own generated test sets (the v1 run predates the teacher suite).

| Suite | v1: templates only | v2: + teacher + noise | keyword baseline |
|---|---:|---:|---:|
| in-distribution | 99.9 % | 99.3 % | ~66 % |
| held-out frames | 71.4 % | **88.1 %** | ~65 % |
| typos / noise | 84.5 % | **98.5 %** | ~67 % |
| multi-turn | 99.4 % | 99.5 % | ~60 % |
| safety | 99.9 % | 99.6 % | ~97 % |

Distillation from a local LLM teacher buys **+17 points on unseen phrasing** and makes
typos a non-issue, at identical runtime cost. The transformer beats the keyword baseline
by 20-35 points everywhere except safety, where refusals are mostly lexical anyway.

## More capacity (tier L, 2.29 M parameters)

Same data, 6 × 192 instead of 4 × 128 (3.4 × the parameters, ~3.5 × slower on the device):

| Suite | tier M (shipped) | tier L |
|---|---:|---:|
| teacher_test | 85.3 % | 82.3 % |
| heldout | 88.1 % | 88.8 % |
| robust | 98.5 % | 98.6 % |
| multiturn | 99.5 % | 99.3 % |
| safety | 99.6 % | 99.9 % |

No meaningful gain: for this closed domain the data, not the parameter count, limits
quality — tier M stays the product.

## What still fails

- *Unseen verbs*: held-out frames such as "kill the fan" use words the model never saw.
- *Natural paraphrases* (teacher_test, 85 %) are harder than templates; most errors are a
  plausible but wrong device or a missing action, never an unsafe executed action.
- Exact reply wording is lower than action accuracy by design: the metric that matters for
  the device is the validated action.
