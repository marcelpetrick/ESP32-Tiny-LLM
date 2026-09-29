# Results: distillation experiments (vision M6)

The three-way comparison the vision asks for — *scratch domain* vs *simple-English
pretraining + domain fine-tune* vs *teacher-distilled* — plus a fourth arm with logit
distillation from a bigger teacher assistant ([04-distillation.md](../04-distillation.md)
§4). **Same architecture** (tier M, 4 × 128, 673 k parameters), **same tokenizer**, **same
INT8 export**, same step budget (6000), evaluated through the C runtime on **identical**
test files (1500 samples per suite). Reproduce with `scripts/run_experiments.sh`.

| Run | Training signal |
|---|---|
| E1 | oracle templates only (politeness noise, punctuation, 10 % typos) |
| E2 | 3000 steps on 4 M tokens of lowercased TinyStories, then E1's data |
| E3 | E1 + 50 % teacher phrasing from Qwen3.5-4B — the shipped model |
| E4 | E3's data + soft targets from the tier-L model (α = 0.5, T = 2) |

## Action accuracy per suite

| Run | test_id | teacher_test | heldout | robust | multiturn | safety |
|---|---:|---:|---:|---:|---:|---:|
| E1 scratch, templates only | 87.8 % | 70.5 % | 76.9 % | 98.5 % | 99.5 % | 99.8 % |
| E2 TinyStories prior + templates | 86.5 % | 70.3 % | 78.4 % | **99.1 %** | 99.0 % | **100.0 %** |
| E3 teacher phrasing (shipped) | 99.3 % | **85.3 %** | 88.1 % | 98.5 % | 99.5 % | 99.6 % |
| E4 E3 + logit KD from tier L | **99.3 %** | 84.2 % | **88.7 %** | 98.5 % | **99.7 %** | 99.6 % |

Best value per suite in bold. `test_id` contains 50 % teacher phrasing, which E1/E2 never
saw — that is why their in-distribution number is lower.

Hallucinated actions (an action where none was expected) on unseen phrasing:

| Run | held-out frames | teacher paraphrases |
|---|---:|---:|
| E1 | 6.6 % | 9.5 % |
| E2 | 7.6 % | 9.1 % |
| E3 | 3.5 % | 6.0 % |
| E4 | 3.7 % | 4.9 % |

## Findings

1. **Distillation from a local LLM teacher is the decisive lever** (E1 → E3): +14.8 points
   on unseen natural phrasing, +11.2 on held-out frames, half the hallucinated actions — at
   zero runtime cost, since architecture, size and speed are identical.
2. **A TinyStories language prior barely helps this domain** (E1 → E2): +1.5 points on
   held-out frames, nothing on paraphrases. Generic simple English is not the missing
   knowledge; domain wording is. (The prior is tokenized with the domain tokenizer, so
   much of it falls back to bytes — a fairer E2 needs a joint tokenizer; see open items.)
3. **Soft targets from a 3.4× larger teacher assistant are roughly neutral** (E3 → E4):
   +0.6 on held-out frames, −1.1 on paraphrases, slightly fewer hallucinated actions on
   paraphrases. The tier-L teacher is not better than the student on this data (see the
   [tier-L result](greenhouse-m.md)), so there is little dark knowledge to transfer.
4. Multi-turn reference resolution, typo robustness and safety are saturated (≥ 98.5 %)
   in every arm; the differences are all in **generalisation to unseen wording**.

**Decision:** keep E3 as the product (simplest pipeline, best on natural paraphrases).
Next levers, in order: more teacher paraphrases and more semantic keys, on-policy
correction of the student's own mistakes (experiment E5 in docs/04), a joint tokenizer
for a fair language-prior test.

## E5: on-policy correction (measured, host, INT8)

`training/onpolicy.py` runs the shipped v3 model through the C runtime on 20 000 fresh
dialogues, the oracle grades every answer, and each mistake becomes a corrected training
example (×4, plus 60 000 replayed original examples). v3 is then fine-tuned for 1 500
steps (LR 5e-4, best checkpoint at step 1 000).

- **Mining on already-seen phrasing is useless:** only 47 of 20 000 answers were wrong.
- **Mining on unseen phrasing works as a probe:** a new teacher round (Qwen3.5-4B, seeds
  23 and 31) gave 3 173 lines that are in neither the train nor the test paraphrases; the
  student got 2 383 of 20 000 dialogues wrong (1 525 actions, 1 158 fallbacks).

Evaluation on the unchanged v3 test files (`data/generated-vocab2048`):

| Model | suite | action | fallback | reply exact | halluc. action |
|---|---|---:|---:|---:|---:|
| v3 (shipped) | teacher_test | **87.0 %** | 86.8 % | 70.8 % | **4.5 %** |
| E5 | teacher_test | 86.9 % | **94.8 %** | **77.3 %** | 6.0 % |
| v3 (shipped) | heldout | 88.9 % | 86.2 % | 71.1 % | **3.2 %** |
| E5 | heldout | **89.2 %** | **90.5 %** | **73.9 %** | 4.4 % |
| v3 (shipped) | safety | **100.0 %** | **100.0 %** | **100.0 %** | **0.0 %** |
| E5 | safety | 99.8 % | 99.7 % | 99.7 % | 0.2 % |

In-distribution, robust and multi-turn suites stay within ±0.4 points.

**Finding:** on-policy correction mainly teaches *when to fall back* (+8 points on unseen
phrasing) and improves replies, but action accuracy does not move, and the model
proposes more actions it should not (hallucinated actions +1.2 to +1.5 points; safety
drops from 100 % to 99.8 %). The firmware validator still blocks every unsafe command,
but a safety regression is not acceptable for the shipped model.
**Decision:** v3 stays the product. A next attempt would weight fallback and safety
examples in the replay, or mine only actions, and would need several rounds (DAgger).
