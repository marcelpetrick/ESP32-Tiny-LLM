# Vision: A Real Tiny Transformer / Language Model on a Single ESP32-S3

**Status:** working project vision and implementation plan  
**Target platform:** ESP32-S3, ESP-IDF, single node, fully local inference  
**Primary engineering target:** a genuine decoder-only transformer language model that fits and runs at useful interactive speed on one microcontroller  
**Secondary target:** turn the same runtime into a useful narrow-domain conversational/diagnostic model rather than a toy text generator  
**Last reviewed:** 2026-09-28

---

## 1. Executive decision

The project should **not** start by trying to make a general-purpose miniature ChatGPT. That is the wrong optimization target for an ESP32.

The project should instead proceed in two deliberate stages:

1. **Prove the platform with a real autoregressive transformer.** Reproduce or independently implement a small generative model that tokenizes text, uses causal self-attention, maintains a KV cache, predicts next-token logits, samples tokens, and generates text entirely on the ESP32-S3. This gives us a hard, auditable statement: *a real transformer language model is running locally on the microcontroller*.
2. **Turn the same runtime into something useful.** Train our own tiny model for a constrained conversational domain: device control, state explanation, diagnostics, or another small closed world. This is the model whose quality actually matters.

The best end-state is therefore not “chat model versus diagnostics.” It is a **narrow conversational diagnostic/control model**. It still chats and generates text, so it exercises the language-model machinery, but its world is intentionally small enough that a sub-megabyte to few-megabyte model can become competent.

A useful product-shaped example is:

```text
<state>
temperature=31.2
humidity=73
fan=off
window=open
motor_current=0.0
</state>
<user>
Why is it still uncomfortable in here?
</user>
<assistant>
Humidity is high and the fan is off. I can start the fan at level 2.
<action fan=2>
</assistant>
```

The neural network interprets language and context. **Firmware remains the authority** for whether an action is allowed. The model should never directly own safety-critical GPIO decisions.

---

## 2. What counts as a “real transformer” here

We should be able to point to the following components in our firmware and model and say exactly where the transformer is:

- token embeddings;
- positional information;
- repeated decoder blocks;
- normalization;
- causal self-attention;
- query/key/value projections;
- causal masking;
- a KV cache for autoregressive inference;
- residual connections;
- a feed-forward network / MLP;
- final normalization;
- vocabulary projection to logits;
- next-token selection and autoregressive generation.

For an input hidden state matrix `X`, self-attention computes projections:

```text
Q = X Wq
K = X Wk
V = X Wv
```

and conceptually:

```text
Attention(Q,K,V) = softmax((Q K^T) / sqrt(d_head)) V
```

with a causal mask that prevents a token from reading future tokens.

The distinctive transformer property is not “lots of parameters.” It is the **content-addressable attention mechanism**: the current token can directly decide which previous representations matter, rather than relying only on a single recurrent state passed forward one timestep at a time.

A minimal decoder block is:

```text
                 ┌─────────────────────── residual ───────────────────────┐
                 │                                                        │
input ──> norm ──> Q/K/V projections ──> causal attention ──> output ───> +
  │                                                                       │
  └───────────────────────────────────────────────────────────────────────┘
                                      │
                                      v
                 ┌─────────────────────── residual ───────────────────────┐
                 │                                                        │
              norm ──> feed-forward / gated MLP ───────────────────────> +
                 │                                                        │
                 └────────────────────────────────────────────────────────┘
```

At the end of all layers:

```text
hidden state -> final norm -> vocabulary projection -> logits -> sampler -> next token
```

Append the token and repeat.

That is enough to call the device a transformer language model without qualification.

---

## 3. Hardware target

### Recommended development board

Use an **ESP32-S3 with 16 MB flash and 8 MB Octal PSRAM (N16R8 class)** for development even if the initial design budget is only 4 MB of PSRAM.

Reasons:

- dual-core Xtensa LX7 at up to 240 MHz;
- vector/SIMD instructions that Espressif's optimized libraries can exploit;
- 512 KB on-chip SRAM;
- external PSRAM support;
- enough flash to hold multiple model variants, tokenizer data, test corpora, and logging builds;
- enough PSRAM headroom to profile before aggressively optimizing memory.

The ESP32-S3 can map substantially more external RAM than older ESP32 variants, and ESP-IDF supports placing allocations and selected sections in PSRAM. Octal PSRAM at 80 MHz is materially better for this workload than repeatedly streaming large model blocks through a slow filesystem path.

### Design constraint

Even if the dev board has 8 MB PSRAM, the **v1 model should target a 4 MB runtime memory ceiling**. That forces good engineering and keeps the design viable on smaller modules.

Suggested memory policy:

- **Internal SRAM:** hot scratch buffers, stacks, latency-sensitive vectors, synchronization objects, frequently used quantization metadata.
- **PSRAM:** model weights if not flash-mapped, KV cache, larger activation buffers, tokenizer tables when needed.
- **Flash:** immutable model files, large cold tables, optional embedding/per-layer embedding tables, tokenizer vocabulary, test prompts.

### CPU usage

Both cores matter, but we should not start with a complicated symmetric parallel runtime.

Initial division:

```text
Core 0: ESP-IDF / I/O / serial / networking / command handling / token streaming
Core 1: inference task
```

Later, after profiling, selected GEMV/GEMM kernels can use both cores when the matrix dimensions are large enough to amortize synchronization. The critical principle is: **do not parallelize by instinct; parallelize only measured hotspots.**

---

## 4. What the first model should do

There should be three model milestones, not one.

### Milestone A — transformer proof

Purpose: validate the runtime, numerical correctness, tokenizer, memory map, KV cache, sampling, and throughput.

Task:

- TinyStories-style continuation, or another tiny synthetic English corpus;
- no claim of useful conversation;
- short prompts and short continuations;
- quality can be modest as long as the model is recognizably autoregressive and coherent enough to validate the pipeline.

This lets us benchmark against existing ESP32 transformer work and isolates firmware/runtime problems from dataset problems.

### Milestone B — constrained chat

Purpose: prove conversational context under extreme constraints.

Task domain examples:

- one device or appliance;
- one embedded controller;
- a small home-automation subsystem;
- a motor controller;
- a sensor hub;
- a lab instrument.

The model must understand short contextual turns:

```text
User: Turn the fan on.
Assistant: Fan is on.
User: Lower it a bit.
Assistant: Setting fan to level 2.
User: What is the temperature now?
Assistant: 29.4 degrees.
```

This proves the model can resolve references such as “it,” “that,” and “now,” which is a meaningful test of attention/context rather than simple one-shot classification.

### Milestone C — useful diagnostic assistant

This should be the real target.

Inputs can combine:

- text;
- current device state;
- a small recent sensor history summarized into tokens;
- fault flags;
- prior conversational turns.

Outputs can combine:

- a terse natural-language explanation;
- a structured intent/action token;
- a confidence or fallback token;
- a request for clarification.

Example:

```text
<state motor=running current=2.8A vibration=high temp=normal>
<user>Why does it sound different today?</user>
<assistant>Vibration is above its normal range while current is also elevated. Check mechanical load or alignment.<diag vibration_load></assistant>
```

This is a much better use of scarce parameters than teaching the chip geography, history, celebrity names, programming languages, and general world knowledge.

---

## 5. Other useful transformer use cases we should preserve for later

The runtime should be designed so the language-model project does not block other sequence tasks.

Potential later models:

- vibration anomaly detection;
- predictive maintenance from current/vibration/temperature sequences;
- accelerometer gesture classification;
- protocol/message normalization;
- log-line interpretation;
- time-series event classification;
- intent detection;
- command completion;
- contextual automation from a short event history.

The same conceptual advantage applies: transformers are sequence models. Text is only one possible sequence.

A useful long-term architecture could prepend sensor-derived tokens to the language context:

```text
sensor feature encoder -> state tokens ─┐
                                        ├-> tiny decoder transformer -> reply/action
user text -> tokenizer -> text tokens ──┘
```

We do **not** need to build this in v1, but the model format should not assume that every token came from UTF-8 text.

---

## 6. Recommended v1 model architecture

Start with a deliberately small decoder-only transformer that is close enough to Llama-like designs to reuse known ideas, but simplified for MCU constraints.

### Baseline configuration

| Parameter | Initial target |
|---|---:|
| Architecture | decoder-only causal transformer |
| Layers | 4 |
| Model width (`d_model`) | 128 |
| Attention heads | 4 |
| Head width | 32 |
| FFN hidden width | 256 |
| Context | 64 tokens initially; 128 target |
| Vocabulary | 1,024–2,048 subword tokens |
| Weight tying | yes, embedding and output projection if practical |
| Normalization | RMSNorm preferred |
| Positional method | learned positions first, RoPE variant later |
| Activation | start with ReLU/GELU-like simple path; compare SwiGLU later |
| Weight precision | FP32 reference -> INT8 production -> optional INT4 experiments |
| KV cache | INT8 or INT16 experiment; MHA first, MQA/GQA later |

### Why this size

At `d_model=128`, four layers, and a 2K vocabulary, the model can remain around the order of **one million parameters** depending on the FFN choice.

Approximate parameter accounting for a compact gated design:

- token embedding: `2048 × 128 ≈ 262k`;
- attention projections per layer: roughly `4 × 128 × 128 ≈ 65k`;
- compact three-matrix gated FFN at width 256: roughly `98k` per layer;
- four layers: roughly `650k` for transformer blocks;
- norms and small metadata: negligible relative to matrices;
- tied output embedding avoids another 262k matrix.

Total: around **0.9–1.0M parameters**.

At INT8 that is roughly **0.9–1.0 MB of raw weights**. At INT4 it is roughly **0.45–0.5 MB**, plus scale tables and packing overhead.

This is small enough to be an honest ESP32 experiment and large enough to exhibit transformer behavior.

### Context and KV cache

For ordinary multi-head attention with four layers, `d_model=128`, context 128, and an 8-bit KV cache:

```text
4 layers × 2(K,V) × 128 positions × 128 channels × 1 byte ≈ 128 KB
```

At 16-bit it is about 256 KB.

This is manageable. If context becomes expensive, multi-query attention can cut K/V storage substantially because all query heads can share a smaller K/V representation.

### Architecture variants worth A/B testing

Do not decide these theoretically. Train and measure them.

1. **Learned position embeddings vs RoPE**  
   Learned positions are trivial at runtime for short fixed contexts. RoPE generalizes positions more naturally and resembles modern LLMs, but adds rotation work or lookup tables.

2. **Standard MHA vs MQA/GQA**  
   MQA/GQA reduces KV cache and some bandwidth. At a 64–128 token context, the benefit may be modest, so simplicity may win initially.

3. **ReLU/GELU-like FFN vs SwiGLU**  
   SwiGLU can improve language-model quality per parameter but needs more operations and a nonlinear gate. We should compare quality per millisecond, not quality alone.

4. **LayerNorm vs RMSNorm**  
   RMSNorm is simpler and a good default for our custom runtime. ESP-DL currently supports both LayerNormalization and RMSNormalization in quantized form, which is useful if we prototype through that path.

5. **512 / 1K / 2K / 4K vocabulary**  
   Vocabulary size is one of the most important MCU design variables because it affects both embedding size and the output head cost every generated token.

---

## 7. Vocabulary and tokenizer strategy

A normal server LLM vocabulary of 30K–100K tokens is inappropriate here. On an MCU, the output head can become one of the dominant costs.

### Recommended first tokenizer

Train a small BPE or unigram tokenizer with **1,024 tokens**, then compare with **2,048 tokens**.

Requirements:

- English only for v1;
- preserve digits and common unit strings efficiently;
- preserve domain words efficiently;
- reserve compact control tokens such as `<user>`, `<assistant>`, `<state>`, `<action>`, `<eos>`;
- tokenize sensor/state syntax predictably;
- tokenizer implementation must be deterministic and small enough for C/C++.

### Why not character-level?

Character-level gives a tiny vocabulary but makes sequences much longer. Attention cost grows with context length, and 64 characters carry far less conversational content than 64 subword tokens.

### Why not byte-level only?

A 256-byte vocabulary is attractive for firmware simplicity, but sequence inflation again becomes costly. It is a useful baseline, not the likely final design.

### Tokenizer implementation

The training-side tokenizer can use SentencePiece or a small custom BPE trainer, but the deployed format should be simple:

- token string table;
- token IDs;
- merge table or trie;
- optional hash accelerator;
- no heap allocation in the hot path after initialization.

We should write tokenizer round-trip tests that run identically in Python and on ESP32.

---

## 8. “Train from scratch” versus “distill an existing LLM”

The best answer is: **design the student architecture from scratch, then use a larger model as a teacher.**

Do not mechanically cut layers out of a 7B model and expect it to become a good 1M model. The geometry, vocabulary, and capacity allocation of a billion-parameter network are wrong for this hardware budget.

### Student architecture

Our student is ours:

```text
4 layers
128 width
small vocabulary
short context
MCU-aware quantization
custom runtime
```

### Teacher contribution

The teacher supplies:

- high-quality domain conversations;
- paraphrases;
- examples of ambiguous commands;
- examples requiring conversational reference resolution;
- diagnostic explanations;
- clarification behavior;
- negative examples and refusal/fallback behavior;
- structured action labels.

The simplest and most practical form of distillation is **sequence-level distillation**:

```text
prompt / state -> teacher generates canonical response -> student trains on that response
```

This works even when teacher and student use different tokenizers.

### Logit distillation

True soft-target/logit distillation can transfer more information per example, but becomes operationally harder when teacher and student vocabularies differ. It is worth trying only after the simpler pipeline works.

Possible approaches later:

- use an open teacher locally and train a matching/reduced vocabulary;
- map teacher distributions to student token strings;
- distill hidden representations into auxiliary student losses;
- use teacher ranking/preferences over candidate student responses.

### Required experiment

Train at least these three variants with the **same deployed architecture**:

1. scratch model on the domain corpus;
2. scratch pretraining on simple English / TinyStories, then domain fine-tuning;
3. teacher-generated/distilled domain model.

That gives us an actual answer to: *how much does distillation buy at the same RAM, flash, and runtime cost?*

---

## 9. Training-data plan

The model will be far more sensitive to dataset design than a normal large model. Every useless behavior consumes scarce capacity.

### Stage 1 corpus: basic English structure

Use a compact corpus that teaches simple grammar and local coherence. TinyStories is a proven type of dataset for studying very small language models. We should not assume it creates an assistant; its job is to give the model a basic prior over English sequences.

### Stage 2 corpus: domain language

Create a formal domain ontology first.

Example device world:

```text
entities:
  fan
  heater
  pump
  motor
  light
  sensor

properties:
  on/off
  level
  temperature
  humidity
  current
  vibration
  pressure
  fault code

actions:
  start
  stop
  set level
  acknowledge
  read
  explain
  diagnose
```

Then generate many linguistic variations over the same semantics.

### Stage 3: contextual dialogue

Explicitly include phenomena the attention mechanism should learn:

- pronouns: “it,” “that,” “the other one”;
- ellipsis: “and upstairs?”;
- relative commands: “a little lower,” “two more,” “same as before”;
- correction: “No, the other fan.”;
- state-dependent interpretation;
- multi-turn diagnostic follow-up;
- clarification when required.

### Stage 4: structured state tokens

The model should see state in a compact canonical format rather than free-form English generated by firmware.

Example:

```text
<S> t=31.2 h=73 fan=0 fanlvl=0 win=1 err=0 </S>
```

The exact syntax should be chosen by tokenizer efficiency, not human beauty.

### Stage 5: hard negatives

We need examples where the correct answer is *not* an action:

- unsafe request;
- impossible request;
- ambiguous device;
- out-of-range value;
- sensor unavailable;
- conflicting state;
- unknown terminology.

The model can emit a compact fallback token such as `<clarify>` or `<unsupported>`, followed by a short explanation.

---

## 10. Output design: language plus structured intent

A pure text generator is fun but brittle for firmware integration. A pure classifier loses the interesting language-model behavior.

Use both.

Example output grammar:

```text
<reply>The fan is currently off.</reply><action fan=2></action>
```

or a more token-efficient variant:

```text
R:The fan is currently off.|A:fan=2
```

The runtime should parse actions with a deterministic state machine. Firmware validates:

- action ID is known;
- parameters are within range;
- target exists;
- operation is allowed in current state;
- safety interlocks pass.

Only then does normal firmware execute the command.

This preserves an important architecture boundary:

```text
model = semantic proposal
firmware = authority
```

---

## 11. Quantization plan

Quantization should be treated as part of model design, not a last-minute compression step.

### Phase 0 — FP32 reference

Train and export a full-precision model. Maintain a tiny host reference implementation that can produce exact intermediate tensors for unit tests.

### Phase 1 — INT8 weights, floating or wider accumulators

The first deployable model should use INT8 weights. This is the most natural point for ESP32-S3 vector acceleration and is well supported by Espressif's optimized NN ecosystem.

Potential representation:

- weights: signed INT8;
- per-channel or per-row scales where useful;
- activations: INT8 or INT16 depending on accuracy;
- accumulators: INT32;
- normalization/softmax: fixed-point, LUT, or selective float depending on measured cost.

### Phase 2 — mixed precision

Do not force every tensor to 8-bit if a small number of layers lose too much quality.

Candidates to keep wider:

- normalization statistics;
- attention logits before softmax;
- residual stream;
- final logits;
- selected first/last layers.

ESP-DL's current toolchain supports `w8a8`, `w16a16`, and `w8a16` for supported operators, which makes it useful as a comparison/prototyping path even if the final runtime is custom.

### Phase 3 — INT4 weights

Only after the INT8 model is stable.

INT4 may halve weight traffic, which could matter more than arithmetic, but costs include:

- unpacking overhead;
- more complex kernels;
- scale metadata;
- potential quality loss;
- alignment complexity.

The actual win must be measured end-to-end in tokens/sec.

### Quantization-aware training

If post-training quantization loses too much quality, use QAT for the final student. With a model this small, recovering a few percent quality can matter a lot.

---

## 12. ESP-IDF software stack

Use **ESP-IDF**, not Arduino, for the main implementation.

### Core stack

- ESP-IDF project and component model;
- FreeRTOS tasks and task pinning;
- `esp_timer_get_time()` / cycle counters for profiling;
- heap capabilities API for internal SRAM vs PSRAM placement;
- custom flash partition for model assets;
- optional memory mapping for read-only model data;
- serial console first; network/UI later.

### ESP-DSP

Use Espressif's **ESP-DSP** library as an early source of optimized primitives and benchmarking references.

Relevant operations include:

- dot products;
- matrix multiplication;
- vector math.

ESP-DSP includes optimized assembly implementations for supported chips as well as reference C implementations. It is especially useful for quickly replacing a naive inner loop and establishing what the S3 can do.

### ESP-NN

Use **ESP-NN** as another source of optimized INT8 kernels. On ESP32-S3 it contains assembly implementations designed to use S3 vector instructions. It is tightly associated with TFLite Micro, but kernels or implementation patterns may be reusable even if our runtime is not TFLM.

Important engineering rule: do not wrap the whole transformer in a heavy framework merely to call one fast dot-product function. If a kernel is useful, integrate at the correct abstraction level.

### ESP-DL

ESP-DL is worth evaluating, not blindly adopting.

Current ESP-DL has useful properties:

- quantized Gemm/MatMul;
- LayerNormalization and RMSNormalization;
- Softmax;
- a static memory planner;
- model loading and profiling;
- ESP-PPQ quantization flow;
- ESP32-S3-specific optimized operator implementations;
- current mixed `w8a16` support for suitable operators.

However, autoregressive transformer inference has unusual runtime requirements:

- incremental one-token execution;
- persistent KV cache;
- tiny repeated GEMV-like operations rather than one static image-style graph;
- custom sampling;
- aggressively controlled memory placement;
- sometimes custom quantization formats.

Therefore the recommended approach is:

1. implement a **small custom inference runtime** for the language model;
2. use ESP-DSP / ESP-NN primitives where they fit;
3. keep an ESP-DL prototype branch as a reference and possible future backend;
4. reuse ESP-DL's quantization/operator knowledge rather than forcing our entire architecture into it.

### TFLite Micro

TFLM is useful for conventional TinyML models and as a benchmark path for sensor classifiers. It is not the preferred first backend for this autoregressive LLM because we want explicit control of KV caching, token-by-token scheduling, and the memory hierarchy.

---

## 13. Firmware architecture

Suggested project structure:

```text
firmware/
  CMakeLists.txt
  sdkconfig.defaults
  partitions.csv
  main/
    app_main.cpp
    console.cpp
  components/
    tinyllm/
      include/
        tinyllm.h
        model_format.h
        tokenizer.h
        sampler.h
      model_loader.cpp
      transformer.cpp
      attention.cpp
      ffn.cpp
      norm.cpp
      tokenizer.cpp
      sampler.cpp
      kv_cache.cpp
      quant.cpp
      profile.cpp
    tinyllm_kernels/
      reference/
        gemv_ref.c
      esp32s3/
        gemv_i8.c
        dot_i8.c
        softmax.c
        rmsnorm.c
```

Training side:

```text
training/
  model.py
  train.py
  distill.py
  tokenizer/
  datasets/
  eval/
  export_model.py
  quantize.py
  compare_intermediates.py
```

Shared format tools:

```text
tools/
  pack_model.py
  inspect_model.py
  benchmark_prompts.py
  serial_runner.py
```

---

## 14. Model file format

Do not start with a complicated generic model format.

Use a simple versioned binary container optimized for the runtime:

```text
header
  magic
  version
  architecture ID
  vocab size
  context length
  layer count
  d_model
  head count
  kv head count
  d_ff
  quantization mode
  offsets

tokenizer block
quant metadata
embedding weights
layer 0 weights
layer 1 weights
...
final norm
output head if untied
```

Requirements:

- fixed endianness;
- alignment suitable for vector loads;
- CRC/hash for model integrity;
- explicit per-tensor scales;
- no parsing allocations during generation;
- support memory-mapped tensors where practical;
- versioning from day one.

An export script should emit both the binary and a manifest with exact tensor names, shapes, dtypes, and offsets.

---

## 15. Inference loop

At a high level:

```text
load model
load tokenizer
allocate KV cache
allocate scratch buffers

prompt_tokens = tokenize(prompt)

for token in prompt_tokens:
    transformer_step(token, kv_cache)

while not stop_condition:
    logits = transformer_step(last_token, kv_cache)
    next = sample(logits)
    stream detokenized next token
    last_token = next
```

Inside a token step:

```text
x = embedding[token] + position_info

for layer in layers:
    r = x
    x = rmsnorm(x)

    q = Wq * x
    k = Wk * x
    v = Wv * x

    write k,v to layer KV cache

    scores = q dot cached_K
    probs = causal_softmax(scores)
    attn = probs weighted cached_V
    x = r + Wo * attn

    r = x
    x = rmsnorm(x)
    x = r + FFN(x)

x = final_norm(x)
logits = output_projection(x)
return logits
```

For generation, most big “matrix multiplications” are actually **matrix-vector multiplications (GEMV)** because only one new token is processed at a time. That fact should strongly influence kernel design.

---

## 16. Memory-placement strategy

Memory bandwidth is likely to dominate before raw multiply throughput does.

### Hot data in internal SRAM

Prefer internal RAM for:

- current hidden vector;
- Q/K/V temporary vectors;
- attention score/probability vectors;
- quantized unpack scratch;
- small scale tables;
- active layer tile;
- stacks for inference task.

### PSRAM

Use PSRAM for:

- most dense weights during early implementations;
- KV cache if it does not fit comfortably internally;
- tokenizer tables;
- optional output head;
- diagnostic logging buffers.

### Flash

Use flash for:

- cold immutable weights;
- model binary;
- large embedding tables;
- optional per-layer embedding tables.

A 2026 ESP32-S3 experiment demonstrated an interesting memory-hierarchy technique using **per-layer embeddings**: a very large stored parameter table can stay in flash while only a small number of rows are read per token. That does not magically create general reasoning capability—the dense computational core remains small—but it is a legitimate future direction for increasing lexical/coherence capacity without loading all stored parameters into RAM.

This should be considered a **later architecture branch**, not the v1 baseline.

---

## 17. Speed expectations and performance targets

Do not define success from theoretical peak MAC/s. Define it from measured token latency.

Recent public ESP32-S3 experiments give useful sanity checks:

- a roughly 260K-parameter / ~1.05 MB FP32 TinyStories transformer has been reported at about **22 tokens/s** on an ESP32-S3 N16R8 with Octal PSRAM;
- another 2026 experiment with a much larger *stored* parameter count but only roughly a 559K dense computational core reports around **9–10 tokens/s**, with the output head taking a large share of token time.

These are external project measurements, not guarantees for our architecture.

### Project performance gates

For the approximately 0.9–1.0M-parameter INT8 model:

- **minimum viable:** >= 5 tok/s;
- **good:** >= 10 tok/s;
- **strong:** >= 15 tok/s;
- **stretch:** >= 20 tok/s.

Interactive feel matters more than benchmark vanity. A 10-token answer at 10 tok/s takes about one second after prompt processing, which is already usable for a tiny embedded assistant.

### Latency buckets to measure separately

For every build record:

```text
tokenizer
embedding/position
QKV projections
attention score
softmax
attention value reduction
attention output projection
FFN
norms
output head
sampling
PSRAM/flash transfers
other
```

Also separate:

- prompt prefill speed;
- single-token decode speed;
- first-token latency;
- steady-state tokens/sec.

---

## 18. Optimization roadmap

Optimization should happen in layers so correctness is never lost.

### O0 — correct host model

- PyTorch model;
- deterministic inference;
- fixed test prompts;
- save intermediate tensors.

### O1 — reference C/C++ on desktop

- same binary model format as ESP32;
- scalar FP32 implementation;
- bit/close comparison against Python;
- no ESP32-specific code yet.

This is critical. It lets us debug runtime math on a desktop before debugging embedded memory at the same time.

### O2 — reference ESP32 FP32

- same code compiled under ESP-IDF;
- model in PSRAM;
- serial generation;
- profile everything;
- establish memory high-water mark.

### O3 — INT8 weight-only path

- quantized matrices;
- dequantize or mixed accumulation initially;
- compare quality and speed.

### O4 — native INT8 kernels

Replace hotspots with optimized kernels:

- vectorized dot products;
- vectorized GEMV;
- tiled reads;
- aligned loads;
- INT32 accumulation;
- reduce unnecessary requantization.

Use ESP-DSP/ESP-NN where they beat custom code.

### O5 — memory bandwidth optimization

- reorder weights in storage format expected by the kernel;
- pack rows to vector alignment;
- prefetch/tile from PSRAM;
- avoid copying weights through temporary buffers;
- keep repeatedly used scale/metadata in internal SRAM;
- compare PSRAM resident vs memory-mapped flash tensors.

### O6 — output-head optimization

The vocabulary projection is easy to underestimate.

Experiments:

- shrink vocabulary;
- tie embeddings;
- quantize head aggressively;
- hierarchical/adaptive output head if quality justifies complexity;
- restrict candidate vocabulary in a domain model only if the restriction is semantically valid;
- cache/static bias handling.

### O7 — attention/KV optimization

- MQA/GQA;
- INT8 KV cache;
- sliding context window;
- ring-buffer KV layout;
- incremental RoPE if used;
- specialized attention kernel for `d_head=32`.

### O8 — dual-core optimization

Only now consider splitting large GEMV rows across cores.

Measure:

- task wakeup overhead;
- barrier cost;
- cache contention;
- PSRAM contention;
- matrix size threshold where two cores actually win.

Potential model:

```text
core 1 computes first half of output rows
core 0 computes second half / handles next lightweight stage
barrier
continue
```

Some stages may never benefit from dual-core execution.

### O9 — INT4

Only after INT8 is well optimized. The goal is lower memory traffic, not merely smaller files.

---

## 19. Quality evaluation

Perplexity alone is not sufficient for the useful model.

### Generative baseline metrics

- validation loss / perplexity;
- repetition rate;
- malformed UTF-8/token output rate;
- EOS behavior;
- short human review of coherence.

### Domain assistant metrics

Build a fixed test suite with thousands of cases and track:

- exact action accuracy;
- parameter accuracy;
- clarification accuracy;
- unsupported-request accuracy;
- context reference resolution;
- state grounding;
- hallucinated device/action rate;
- natural-language explanation quality;
- maximum response length violations.

### Robustness suites

Include:

- spelling mistakes;
- synonyms;
- unusual word order;
- extra politeness/noise;
- incomplete commands;
- contradictory requests;
- adversarial values;
- rapid conversation state changes.

### Hardware-in-the-loop evaluation

The final benchmark runner should send test prompts over serial, capture:

- generated text;
- action parse result;
- token timing;
- heap/PSRAM high-water marks;
- model version;
- firmware git revision.

This prevents performance and quality claims from becoming anecdotal.

---

## 20. Safety and deterministic boundaries

Even though this is primarily an engineering experiment, the architecture should be sane from the beginning.

The model must **not** directly perform unrestricted hardware writes.

Use:

```text
language model
    ↓
structured proposal
    ↓
strict parser
    ↓
firmware validation
    ↓
state machine / safety interlocks
    ↓
GPIO / actuator
```

The deterministic layer owns:

- allowable pins;
- voltage/current constraints;
- allowed ranges;
- mutually exclusive states;
- emergency stop;
- rate limits;
- actuator timing;
- authentication if networking is later enabled.

The model can explain and propose; firmware decides.

---

## 21. Development milestones

### M0 — hardware and benchmark harness

Deliverables:

- ESP32-S3 N16R8 board configured under ESP-IDF;
- PSRAM verified and measured;
- flash partition for models;
- serial command shell;
- microbenchmark for internal SRAM, PSRAM, and flash read bandwidth;
- dot/GEMV baseline benchmarks;
- per-stage profiler infrastructure.

Exit criterion: we know the actual memory bandwidth and compute behavior of our board.

### M1 — reproduce a known tiny transformer

Deliverables:

- run an existing tiny TinyStories-style checkpoint or architecture;
- generation entirely on device;
- measured tok/s;
- memory report.

Purpose: validate the board and toolchain before blaming our own model.

### M2 — our scalar runtime

Deliverables:

- our model binary format;
- our tokenizer;
- our C/C++ transformer implementation;
- host and ESP32 intermediate-output tests;
- FP32 generation.

Exit criterion: Python, desktop C, and ESP32 agree numerically within defined tolerance.

### M3 — our first trained model

Deliverables:

- 4×128 transformer;
- 1K tokenizer baseline;
- basic TinyStories/simple-English model;
- quality report;
- ESP32 speed report.

Exit criterion: recognizable coherent short continuation on our runtime.

### M4 — INT8 production path

Deliverables:

- calibrated quantization;
- INT8 model;
- optimized GEMV/dot path;
- memory layout optimized for S3;
- >= 5 tok/s target, preferably >= 10 tok/s.

### M5 — constrained chat dataset

Deliverables:

- domain ontology;
- synthetic teacher generation scripts;
- compact state-token format;
- multi-turn dialogue dataset;
- held-out evaluation set.

### M6 — three-way training comparison

Train and compare:

- scratch domain;
- simple-English-pretrained + domain fine-tune;
- teacher-distilled.

Same architecture. Same quantization. Same hardware.

Exit criterion: empirical choice rather than ideology about distillation.

### M7 — useful on-device assistant

Deliverables:

- natural-language input;
- state injection;
- short answer generation;
- structured action/diagnostic tokens;
- deterministic action validator;
- regression suite.

### M8 — quality/performance sweep

Sweep:

- 2 / 3 / 4 / 6 layers;
- width 96 / 128 / 160;
- vocabulary 512 / 1K / 2K / 4K;
- context 32 / 64 / 128;
- MHA vs MQA;
- FFN variants;
- INT8 vs mixed vs INT4;
- learned positions vs RoPE.

Plot quality against:

- model bytes;
- peak RAM;
- tok/s;
- energy/token if we measure power.

The deliverable should be a Pareto frontier, not one arbitrary configuration.

---

## 22. What we should *not* do initially

Avoid these traps:

1. **Do not start with a 30K vocabulary.** The embedding/output head can consume the project.
2. **Do not start with 512+ token context.** It complicates memory and hides the core experiment.
3. **Do not start with INT4.** Debugging quantization and packing simultaneously will slow us down.
4. **Do not start with hand-written assembly.** First identify which kernels deserve it.
5. **Do not train on generic internet text and hope for a chatbot.** A 1M model does not have enough capacity to store the world.
6. **Do not give the model direct hardware authority.** Use structured proposals plus deterministic validation.
7. **Do not assume both cores automatically double speed.** PSRAM bandwidth and synchronization can make this false.
8. **Do not optimize only token generation and ignore the tokenizer/output head.** Both can dominate small models.
9. **Do not compare parameter count alone.** Stored lookup parameters and dense computed parameters are not equivalent.
10. **Do not hide behind a framework.** We should understand and be able to profile each transformer stage.

---

## 23. Reasonable success definitions

### Technical success

We can demonstrate:

- one ESP32-S3;
- no network required;
- a custom or clearly inspectable decoder transformer;
- local tokenization;
- causal attention;
- KV cache;
- autoregressive text generation;
- model roughly around 1 MB INT8 class for the core baseline;
- interactive generation speed;
- reproducible benchmark numbers.

### Useful-model success

The model can handle a small closed domain with:

- natural phrasing;
- multi-turn references;
- current device state;
- short diagnostic explanations;
- correct structured actions;
- safe fallback when unsure.

A narrow model that gets 95%+ of a carefully designed domain suite correct is much more impressive on this hardware than a “general chatbot” that produces fluent nonsense.

---

## 24. Research branches after v1

These are explicitly out of the critical path but should not be forgotten.

### A. Per-layer embeddings / flash-resident lexical capacity

A 2026 ESP32-S3 project reported a 28.9M stored-parameter TinyStories model at around 9.5 tok/s by keeping a large embedding-style table in flash while retaining only a much smaller dense core for per-token computation.

Research question: can we use a similar memory hierarchy to improve coherence/domain vocabulary without materially increasing dense compute?

### B. Multi-query attention

Reduce KV cache and bandwidth while keeping multiple query heads.

### C. Sliding-window chat memory

Keep the latest 64/128 tokens and optionally summarize old state into compact firmware-provided state tokens.

### D. Retrieval without a second LLM

Store a tiny table of device facts/manual snippets and retrieve one or two compact records before generation. This is not classic vector-RAG at server scale; it can be a deterministic keyword/hash/embedding lookup that injects only a few tokens.

### E. Hybrid classifier + generator

Use a tiny classifier first for obvious actions and invoke the generative transformer only for ambiguous/diagnostic turns. This can save power and improve determinism.

### F. Sensor-token multimodality

Train a small encoder that converts recent sensor windows into a few learned tokens consumed by the language model.

### G. On-device adaptation

Do not initially train the transformer on-device. Later, explore tiny adapters, counters, prototypes, or a very small learned preference layer. Full backpropagation on ESP32 has been demonstrated experimentally in 2026, but it is a research branch, not the product path.

### H. Energy-aware inference

Measure mJ/token and investigate frequency scaling, sleep between interactions, and model selection by task complexity.

---

## 25. Initial benchmark matrix

Every serious build should report the following table:

| Dimension | Value |
|---|---|
| Board | exact ESP32-S3 module/board |
| CPU frequency | MHz |
| Flash type/mode/frequency | exact |
| PSRAM type/mode/frequency | exact |
| ESP-IDF version | exact git/tag |
| Model version | hash |
| Parameters | dense / stored |
| Weight bytes | exact |
| Quantization | exact |
| Vocabulary | count |
| Context | tokens |
| Layers / width / heads / FFN | exact |
| KV cache bytes | exact |
| Peak internal RAM | bytes |
| Peak PSRAM | bytes |
| Prompt length | tokens |
| Prefill tok/s | measured |
| Decode tok/s | measured |
| First-token latency | ms |
| Mean token latency | ms |
| p95 token latency | ms |
| Output-head time/token | ms |
| Attention time/token | ms |
| FFN time/token | ms |
| Power/energy | optional |

Without this, “runs at N tokens/sec” is not comparable across experiments.

---

## 26. Suggested first concrete configuration

If we had to freeze the first custom model today:

```text
chip:           ESP32-S3 N16R8
framework:      ESP-IDF
runtime:        custom C/C++ token-by-token transformer
layers:         4
d_model:        128
heads:          4
kv_heads:       4 initially
head_dim:       32
d_ff:           256
norm:           RMSNorm
position:       learned positional embeddings initially
activation:     simple FFN first; SwiGLU comparison branch
vocab:          1024 first, 2048 comparison
context:        64 first, 128 target
weight tying:   yes
weights:        INT8 target
KV cache:       INT8 experiment, INT16 fallback
sampling:       greedy + temperature/top-k for debug
target weights: <~1 MB INT8 class
target RAM:     <4 MB total external+internal runtime footprint
target speed:   >=10 tok/s desirable, >=5 tok/s minimum
```

The v1 domain should be one controlled device/sensor environment with 20–50 meaningful actions/diagnoses and enough multi-turn language variation to prove the model is doing semantic contextual work rather than string matching.

---

## 27. Immediate next actions

1. Select one exact ESP32-S3 N16R8 board and freeze its memory configuration.
2. Create an ESP-IDF repository with reproducible `sdkconfig.defaults` and custom model partition.
3. Measure SRAM/PSRAM/flash bandwidth and basic dot/GEMV performance.
4. Port/reproduce a known 260K TinyStories transformer benchmark as a hardware sanity check.
5. Implement our model format and desktop C reference runtime.
6. Train the 4×128 baseline in PyTorch with a 1K tokenizer.
7. Get exact numerical agreement between PyTorch and C.
8. Run the same runtime on ESP32 in FP32.
9. Introduce INT8 and optimize the measured GEMV hotspots.
10. In parallel, define the first useful device/diagnostic ontology and start generating teacher data.
11. Train scratch / pretrained / distilled variants with identical student architecture.
12. Pick the best quality-speed point from measured results, not assumptions.

---

## 28. Current Espressif-specific references

The exact versions should be pinned in the repository when implementation begins. These references were checked while preparing this plan.

- [ESP-IDF: ESP32-S3 external RAM support](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-guides/external-ram.html)
- [ESP-IDF: ESP32-S3 flash and PSRAM configuration](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-guides/flash_psram_config.html)
- [ESP-DL latest documentation](https://docs.espressif.com/projects/esp-dl/en/latest/)
- [ESP-DL operator support](https://github.com/espressif/esp-dl/blob/master/operator_support_state.md)
- [ESP-DSP](https://github.com/espressif/esp-dsp)
- [ESP-NN](https://github.com/espressif/esp-nn)
- [ESP32-S3 datasheet](https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf)

Useful external proof points / prior art:

- [Circuit-Digest ESP32 Tiny-LLM / llama2.c benchmark repository](https://github.com/Circuit-Digest/ESP32-Tiny-LLM)
- [2026 ESP32-S3 28.9M stored-parameter / per-layer-embedding experiment](https://github.com/manjunathshiva/esp32-tinyllm)
- [Conformer speech recognition on ESP32-S3](https://github.com/lspr98/conformer-stt-s3)

These projects are references, not architectural requirements. The goal is to understand the techniques, reproduce useful baselines, then own the student architecture, training pipeline, binary format, and runtime ourselves.

---

## 29. Implementation details that are easy to forget

This section is the result of a final review pass. These items are not the glamorous part of the model, but they can decide whether the project is reproducible or frustrating.

### Reproducible training

Every training run should save:

- git revision;
- tokenizer hash;
- dataset manifest/hash;
- random seed;
- architecture JSON;
- optimizer and learning-rate schedule;
- validation metrics;
- quantization configuration;
- exported binary hash.

A reasonable first training recipe is AdamW on the workstation/GPU, sequence length 64 initially, gradient accumulation as needed, cosine or warmup+cosine learning-rate schedule, and early stopping on a fixed validation set. Exact hyperparameters should be selected by small sweeps rather than copied from large-model recipes. Once context 64 is stable, continue or retrain at 128 if the quality gain is worth the runtime cost.

### Static memory after initialization

The inference loop should perform **no general heap allocation** after model initialization if possible. Preallocate:

- hidden vectors;
- Q/K/V buffers;
- attention scores;
- FFN scratch;
- logits buffer or tiled logits workspace;
- KV cache;
- tokenizer workspace.

This avoids fragmentation and makes peak memory measurable.

### Watchdogs and task scheduling

Long compute loops can trigger FreeRTOS/task watchdogs or starve I/O. The inference task must either yield at safe boundaries or be configured deliberately. Do not “fix” watchdog resets by globally disabling protection without understanding the scheduling impact.

### Benchmark conditions

For comparable performance measurements:

- lock CPU frequency for the run;
- record compiler optimization flags;
- record PSRAM/flash mode and frequency;
- benchmark with Wi-Fi/Bluetooth disabled first;
- then repeat with the intended connectivity enabled;
- use the same prompts and generated-token count;
- discard or separately report model-load time;
- report warm and cold-cache behavior where it materially differs.

### Build optimization

After correctness:

- compare `-O2`, `-O3`, and size-oriented builds rather than assuming one wins;
- evaluate LTO if compatible with debugging needs;
- align hot arrays and packed weights for the vector kernel;
- inspect generated assembly for the handful of kernels that dominate runtime;
- keep a scalar reference kernel compiled into debug builds for differential testing.

### Prompt prefill is different from decode

Generation is one token at a time, but an initial prompt contains many known tokens. A later optimization can use a more efficient **prefill path** that processes prompt tokens in blocks, while decode remains GEMV-heavy and incremental. We should report prefill and decode separately from the beginning so this optimization remains visible.

### Sampling

Implement these in increasing order of complexity:

1. greedy decoding;
2. temperature;
3. top-k;
4. optional repetition penalty / recent-token suppression.

Greedy mode is essential for deterministic regression testing. Sampling code should use a deterministic seed when requested.

### Stop conditions

The model format/training corpus must have explicit turn and sequence markers. Runtime stopping rules should include:

- EOS token;
- assistant-turn end token;
- maximum generated tokens;
- malformed action grammar fallback;
- optional timeout.

Without disciplined stop behavior, a tiny model can waste time generating repetition.

### Flash partition and OTA planning

A 16 MB flash part sounds large until firmware, filesystem, model, NVS, crash data, and OTA slots compete for it. Decide early whether the prototype needs OTA.

Possible layouts:

- development: one large app partition + one large model partition;
- product: dual app OTA partitions + separate versioned model partition;
- model update path: firmware and model updated independently, with compatibility/version checks.

Never load a model whose architecture/version header is incompatible with the runtime.

### Model integrity and security

For prototypes, CRC/hash validation is enough to catch corruption. If the design becomes a product, consider ESP-IDF secure boot, flash encryption, signed updates, and model authenticity as part of the normal firmware threat model.

### CI and hardware-in-the-loop

CI should have two layers:

```text
host CI:
  tokenizer tests
  binary format tests
  PyTorch vs C tensor comparisons
  quantization regression
  domain evaluation

hardware CI / bench:
  flash board
  run fixed prompt suite
  collect speed and memory
  compare against thresholds
```

A performance regression of 20% should fail or at least flag the build just as a quality regression would.

### Licensing and dataset provenance

Before publishing or productizing:

- record licenses for base datasets;
- record the teacher/model terms used for generated training data;
- record licenses for borrowed runtime code or kernels;
- keep our model weights and training-data provenance auditable.

This matters especially if we start from public TinyStories/llama2.c-style examples and later combine them with teacher-generated domain data.

### Product/runtime observability

Useful debug commands should expose:

```text
/model-info
/memory
/profile on|off
/kv-reset
/seed N
/temp X
/topk N
/max-tokens N
/benchmark <case>
```

Debug builds should be able to print per-layer checksums rather than entire tensors. This is enough to localize the first layer where ESP32 output diverges from the desktop reference.

### Keep the experiment falsifiable

Define failure conditions up front. Examples:

- if the useful-domain model cannot exceed a deterministic non-neural baseline by a meaningful margin, the transformer may not justify its complexity for that task;
- if INT8 quality collapses, use mixed precision before spending weeks on assembly;
- if the output head consumes most token time, reduce vocabulary before optimizing attention;
- if dual-core execution is slower, keep inference mostly single-core;
- if a 1M model is too weak, first improve data/distillation, then consider 1.5–2M parameters before exotic architectures.

The goal is not to prove the initial architecture correct. The goal is to find the best quality/latency/memory point honestly.

---

## 30. Final project thesis

The project is not about pretending an ESP32 can replace a server LLM.

The interesting engineering question is:

> **How much contextual language behavior can we preserve when a transformer is designed from first principles for a few megabytes of memory, a 240 MHz dual-core MCU, and a tiny closed world?**

The transformer gives us learned contextual attention. Distillation gives us a way to transfer behavior from a much larger teacher. Quantization and MCU-specific kernels make the computation fit the hardware. A deliberately narrow domain makes the remaining capacity useful.

If the final device can hold a short conversation about its own state, understand varied phrasing and references, explain a diagnostic condition, and propose a validated action at interactive speed, then it is both a legitimate tiny LLM and a genuinely useful embedded system—not merely a stunt.

---

## 25. Additional researched technical possibilities and prior-art notes

**Research pass:** 2026-09-28. The following 20 items are additions to the plan, based on public implementations, papers, and current Espressif documentation. They are hypotheses or engineering directions to test, not assumptions that results from other hardware/model sizes will transfer unchanged. Short quotations are included only as attribution anchors; where code is reused rather than independently reimplemented, we must check and preserve the upstream license and notices.

- **1. Keep `llama2.c` as a permanent numerical/reference oracle even after our runtime diverges.** Andrej Karpathy describes `llama2.c` as having a “**focus on minimalism and simplicity**” and explicitly positions it as a small, hackable training-plus-inference reference. ([Karpathy, `llama2.c`](https://github.com/karpathy/llama2.c)) We should exploit that property rather than merely use it once for inspiration: freeze one or two known tiny checkpoints, add a host-side adapter that can dump every intermediate tensor from `llama2.c`, and make our desktop C runtime and ESP32 runtime compare against those tensors layer by layer. This gives us a regression oracle for RMSNorm, RoPE, Q/K/V layout, MQA/GQA indexing, SwiGLU, KV-cache position handling, logits, and tokenization. When an optimization changes output, we can identify the first diverging tensor rather than debug generated prose. The test corpus should include first-token, cache-wrap, maximum-context, repeated-token, and EOS cases. This is also the cleanest defense against the common embedded failure mode where an optimization is “fast” because it is subtly wrong.

- **2. Build a bytes-per-token roofline model before optimizing arithmetic.** The `doryiii/esp32-llm` ESP32-S3 port reports an INT8 3.3M model at roughly 12 tok/s and says this is “**very close to the memory bandwidth limit of 13.1 tok/s**”; it also keeps hot activations in on-chip RAM and dequantizes token embeddings on demand. ([doryiii, `esp32-llm`](https://github.com/doryiii/esp32-llm)) We should therefore add a simple analytical model to every benchmark: for each token, estimate the bytes of weights, scales, KV data, and output-head data that must be read, then divide by measured latency to obtain effective bandwidth. Compare that with a sequential PSRAM bandwidth microbenchmark. If the ratio is already near the board's achievable bandwidth, more MAC parallelism cannot materially help; the next optimization must reduce bytes moved, improve locality, or change the model architecture. Conversely, if effective bandwidth is low, kernel instruction count or access pattern is probably the problem. This turns “compute-bound versus memory-bound” from an intuition into a measured classification per layer.

- **3. Make dual-core execution conditional on arithmetic intensity, not a global compile-time choice.** DaveBben's ESP32-S3 port attributes part of its reported 19.13 tok/s result to “**Utilizing both cores of the ESP32 during math heavy operations**,” while the newer `doryiii/esp32-llm` implementation says it deliberately does not use the second core once the INT8 path reaches the memory-bandwidth ceiling. ([DaveBben, `esp32-llm`](https://github.com/DaveBben/esp32-llm)) ([doryiii, `esp32-llm`](https://github.com/doryiii/esp32-llm)) Both can be true. We should benchmark a runtime policy where large compute-heavy rows are split across two cores, but PSRAM-bound stages stay single-core so the second LX7 can handle I/O or remain idle. Record the crossover matrix size at which dual-core wins, including barrier cost and PSRAM contention. The output head may have a different crossover than FFN/QKV kernels. A small static decision table generated from board benchmarks is likely better than either “always use two cores” or “never use two cores.”

- **4. Treat PSRAM/flash topology as a model-design variable and maintain a hardware matrix, not just a board name.** Circuit Digest's ESP32 Tiny-LLM benchmark explicitly studies the “**raw memory bandwidth limits of direct SPI Flash streaming versus PSRAM cached execution**” and reports materially different token rates across Octal-PSRAM and older QSPI configurations for the same ~1.05 MB FP32 model. ([Circuit-Digest, `ESP32-Tiny-LLM`](https://github.com/Circuit-Digest/ESP32-Tiny-LLM)) Our benchmark manifest should therefore include flash mode/frequency, PSRAM bus width/frequency, cache configuration, model placement, compiler flags, and ESP-IDF version. A result such as “15 tok/s on ESP32-S3” is otherwise underspecified. We should explicitly test at least: model in PSRAM, model memory-mapped from flash, hot-layer staging into internal SRAM, and hybrid placement. That data can drive an automatic packer that places tensors by reuse frequency and size rather than using one memory tier for all weights.

- **5. Preserve an SRAM-only ultra-small profile as a second target.** The `esp32s3-Super-mini-AI` project claims a local autoregressive micro-transformer operating “**strictly within the 384KB internal SRAM boundary**” and without external PSRAM. ([Ngducok, `esp32s3-Super-mini-AI`](https://github.com/Ngducok/esp32s3-Super-mini-AI)) Even if our main N16R8 build uses PSRAM, an internal-SRAM-only profile would be extremely valuable: it isolates the raw cost of external-memory traffic, proves the runtime can degrade gracefully onto cheaper boards, and gives us a latency-oriented configuration for very narrow tasks. Concretely, target perhaps a 50K–150K parameter model, 256–512 token vocabulary, 32-token context, one or two blocks, and aggressively reused scratch buffers. The point is not to maximize linguistic quality; it is to establish the lower corner of the Pareto frontier and quantify how many tokens/sec are lost when capacity moves from SRAM to PSRAM.

- **6. Treat the vocabulary/output head as an independent architecture problem, because real ESP32 measurements show it can dominate token time.** The `slvDev/esp32-ai` PLE TinyLM report measures “**output head | 59.4 ms/token**” out of about 94.9 ms/token total compute on its ESP32-S3 configuration. ([slvDev, `esp32-ai`](https://github.com/slvDev/esp32-ai)) This makes the head too important to regard as a final linear layer we simply accept. In addition to our existing vocabulary sweep, research: a lower-dimensional tied embedding/head with a projection into `d_model`; clustered or two-stage vocabularies; domain-vocabulary pruning; and a deterministic candidate shortlist only when firmware context makes that mathematically safe. We should report both head top-1 agreement and full-language perplexity so a fast shortlist does not silently remove valid words. For narrow chat, a 512–2K vocabulary may outperform a larger model with a 10K+ vocabulary simply because more of the compute budget remains available for contextual layers.

- **7. Add a dense-Q4 “how far can we push it?” branch to establish the upper capacity boundary.** `JARACH-209/esp32-30.7M` describes a 30.72M-parameter ESP32-S3 model where “**Every stored parameter is multiplied on every token**,” using group-128 Q4 weights and W4A8 integer dot products; the project reports roughly 0.95 tok/s. ([JARACH-209, `esp32-30.7M`](https://github.com/JARACH-209/esp32-30.7M)) We should not make 30M dense parameters the main product target, but it is useful evidence that the feasible region extends much farther than ~1 MB if low latency is relaxed. A controlled 1M/2M/4M/8M Q4 sweep using the *same corpus and tokenizer* would tell us whether quality gains per extra megabyte are worth the inverse throughput. It would also validate our W4A8 pack format and reveal when flash capacity, PSRAM staging, or bandwidth becomes the dominant constraint. This gives the project a measured capacity/latency curve rather than a single arbitrary model size.

- **8. Keep on-device learning as a later experiment, but narrow it to tiny trainable surfaces rather than the full model.** The `qapla` project is useful prior art because “**the chip runs the full training loop**,” including forward pass, cross-entropy, hand-written backpropagation, updates, checkpointing, and generation on an ESP32-S3. ([Carloscodix, `qapla`](https://github.com/Carloscodix/qapla)) That establishes technical possibility, not product desirability. For our system, the first adaptation experiment should instead freeze the transformer and train only something tiny: a response bias vector, a low-rank adapter on one projection, a prototype/classification head, or a few learned state-token embeddings. This keeps RAM for gradients/optimizer state bounded and limits flash writes. We should measure flash endurance implications, training energy, catastrophic forgetting, and whether a deterministic non-neural personalization table works better. Full on-device backprop remains an excellent science demo, but it should not complicate the inference-first architecture.

- **9. Create a purpose-built “TinyDiagnostics” synthetic curriculum rather than assuming generic English pretraining is the best use of parameters.** The TinyStories paper showed coherent generation in models “**below 10 million total parameters**” and even explored models with only one transformer block. ([Eldan & Li, *TinyStories*](https://arxiv.org/abs/2305.07759)) The transferable lesson is not that our assistant should tell stories; it is that reducing lexical and conceptual entropy can make very small language models qualitatively better. We should generate a staged corpus whose language complexity grows deliberately: basic command grammar, state descriptions, pronouns/reference, causality, diagnostic chains, ambiguity, and recovery. Keep a clean held-out set generated from different templates/teacher prompts so we do not measure template memorization. Also compare a “simple English” teacher style with unrestricted prose: for a 1M model, canonical concise language may free capacity for actual state reasoning.

- **10. Test deeper-and-thinner students at fixed bytes and fixed decode MACs.** MobileLLM reports gains from “**deep and thin architectures**” together with embedding sharing and grouped-query attention in the sub-billion regime. ([Liu et al., *MobileLLM*, ICML 2024](https://proceedings.mlr.press/v235/liu24ce.html)) That scale is vastly larger than ours, so this is a hypothesis, not a direct prescription. Still, our current 4×128 baseline should be challenged by iso-parameter alternatives such as 6×96, 8×80, or recurrently shared 8 logical layers. Small models may benefit from additional nonlinear depth even when width shrinks, especially for multi-turn transformations. The correct comparison is a Pareto plot with model bytes, measured token latency, validation loss, domain action accuracy, and reference-resolution accuracy. If deeper/thinner improves quality but destroys tokens/sec because more sequential layer passes amplify PSRAM latency, we will have direct evidence for where the MCU regime diverges from mobile-class findings.

- **11. Factorize token embeddings and experiment with cross-layer sharing.** ALBERT's two headline compression ideas are “**factorized embedding parametrization**” and “**cross-layer parameter sharing**.” ([Google Research, *ALBERT*](https://research.google/blog/albert-a-lite-bert-for-self-supervised-learning-of-language-representations/)) Both map unusually well to our constraints. Instead of a `V × d_model` embedding, try `V × d_embed` followed by a small `d_embed × d_model` projection, with `d_embed` perhaps 32–64 while `d_model` remains 96–160. Separately, share one transformer block across multiple logical depths, or share every pair of blocks. Parameter sharing reduces stored bytes but does **not** proportionally reduce compute because the shared block is still executed repeatedly, which is exactly why it is worth separating “model capacity in flash” from “decode cost per token.” A 2-unique-block / 6-logical-layer model could be especially interesting if flash/PSRAM bandwidth, rather than MACs, is the limiter.

- **12. Promote MQA/GQA from a later optimization to an early architectural A/B test.** Shazeer's original MQA paper is literally titled “**One Write-Head is All You Need**,” and the later GQA work reports that grouped-query attention can approach multi-head quality with MQA-like decode efficiency. ([Shazeer, *Fast Transformer Decoding*](https://arxiv.org/abs/1911.02150)) ([Ainslie et al., *GQA*, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.298/)) For our four-query-head baseline, the useful experiments are concrete: `n_kv_heads=4` (MHA), `2` (GQA), and `1` (MQA). At only 64–128 context tokens the KV-cache capacity saving is not huge, but MQA also shrinks K/V projection weights and reduces K/V memory traffic every decode step. Because tiny models can react strongly to seemingly small architecture changes, train all three from scratch under the same token budget rather than converting only after training.

- **13. Try SmoothQuant-style offline rescaling before giving up on W8A8 activations.** SmoothQuant's key idea is “**migrating the quantization difficulty from activations to weights**” through an equivalent offline transformation. ([Xiao et al., *SmoothQuant*, ICML 2023](https://proceedings.mlr.press/v202/xiao23c.html)) Our current plan already includes INT8, but this gives us a specific method when activation outliers make W8A8 unstable. Instrument calibration runs to record per-channel activation maxima for Q/K/V, FFN inputs, and residual branches; then test whether offline channel scaling reduces INT8 error enough to avoid INT16 activations. The attractive property for ESP32 is that much of the transformation can be folded into stored weights/scales, so runtime complexity need not increase. Tiny models may have different outlier behavior than 7B+ LLMs, so the outcome must be measured, not assumed.

- **14. Use activation-aware scaling as the first serious W4 weight experiment instead of naïve round-to-nearest INT4.** AWQ reports that “**Protecting only 1% salient weights can greatly reduce quantization error**” and identifies important channels from activation statistics rather than weight magnitude alone. ([Lin et al., *AWQ*, MLSys 2024](https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html)) We do not need to copy the full GPU-oriented stack. The relevant MCU experiment is export-time per-channel or per-group rescaling followed by a simple uniform W4 pack that our kernel can read efficiently. Compare plain groupwise Q4, AWQ-inspired scaled Q4, and QAT-Q4 at identical group sizes. If activation-aware scaling recovers most of the quality while preserving a single packed kernel format, that is much more valuable than a theoretically superior mixed-precision scheme that branches per channel during every token.

- **15. Keep rotation-based quantization as a research path for 4-bit activations and KV cache.** QuaRot says it “**removes outliers from the hidden state without changing the output**,” enabling end-to-end low-bit quantization in its target models. ([Ashkboos et al., *QuaRot*, NeurIPS 2024](https://www.microsoft.com/en-us/research/publication/quarot-outlier-free-4-bit-inference-in-rotated-llms/)) For us, the interesting target is not merely smaller static weights; it is reducing residual/KV bandwidth enough that longer context becomes cheap. Investigate whether fixed Hadamard-like rotations can be absorbed into adjacent matrices at export time, leaving little or no runtime rotation cost for some paths. If a runtime transform is still necessary, benchmark it against the bytes saved. A 4-bit KV cache is only useful if quantization/dequantization does not cost more than the PSRAM traffic it eliminates. This should come after the W8A8 baseline and after we have per-layer error instrumentation.

- **16. Train one ternary/BitNet-style student from scratch as a hardware co-design experiment.** BitNet b1.58 constrains weights to the ternary set “**{-1, 0, 1}**” rather than taking a conventional model and post-quantizing it. ([Ma et al., *The Era of 1-bit LLMs*, 2024](https://arxiv.org/abs/2402.17764)) This is unusually relevant to an MCU because a ternary matrix-vector product can in principle replace general weight multiplies with add/subtract/skip operations and very dense bit packing. The open question is whether unpacking, scale application, activation precision, and irregular zeros erase the win on Xtensa LX7. Build a tiny fixed-shape ternary GEMV microkernel and benchmark it *before* training a large sweep; if the kernel is promising, train an iso-parameter ternary student and compare perplexity/domain accuracy, bytes/token, energy/token, and speed with INT8 and Q4. Treat this as co-design, not as an automatic consequence of a 1.58-bit paper result on other hardware.

- **17. Investigate LUT-based low-bit GEMV inspired by T-MAC for 2–4 bit weights.** T-MAC's central idea is to use “**table lookup**” for mixed-precision low-bit inference and avoid repeatedly dequantizing weights before multiplication. ([Microsoft/Opera, `T-MAC`](https://github.com/microsoft/T-MAC)) The published kernels target ARM/x86 features that the ESP32-S3 does not share, so direct code reuse is unlikely to be appropriate. The algorithmic idea is still worth testing: for a small group of packed 2- or 4-bit weights and an INT8 activation fragment, precompute a compact table of partial sums in internal SRAM, then replace some multiplies/dequant operations with indexed additions. The experiment should be a standalone dot-product benchmark across realistic `d_model`/`d_ff` shapes. If it loses to PIE/SIMD INT8, abandon it quickly; if it wins, it could make low-bit models faster rather than merely smaller.

- **18. Keep a hybrid recurrent/local-attention model as a comparator if context length becomes the dominant limitation.** RecurrentGemma/Griffin uses a “**fixed-sized state**” together with local sliding-window attention, reducing memory growth with sequence length. ([Google DeepMind, *RecurrentGemma*](https://deepmind.google/models/gemma/recurrentgemma/)) This would no longer be a pure decoder-transformer baseline, so it must not replace the model that proves “transformer on ESP32.” But as a v2 comparison it addresses a real engineering question: if a useful diagnostic conversation needs hundreds or thousands of historical tokens, is maintaining a growing KV cache still the right architecture? A tiny hybrid could keep the last 16–32 tokens in local attention while compressing older history into recurrent state. Compare it with our sliding-window transformer at equal model bytes and similar compute. The result would tell us whether the transformer's context mechanism or the language modeling capacity is the real bottleneck in this MCU regime.

- **19. Use Espressif's own quantization/runtime stack as an operator oracle, even if the final LLM engine stays custom.** Current ESP-DL documentation labels `w8a8` as “**Highest inference speed**” for its supported models and also supports `w8a16` and `w16a16`; it can export test values with the model for board-side verification. ESP-NN separately provides ESP32-S3 assembly paths “**optimised to benefit from vector instructions**.” ([ESP-DL quantization guide](https://docs.espressif.com/projects/esp-dl/en/latest/tutorials/how_to_quantize_model.html)) ([Espressif, `esp-nn`](https://github.com/espressif/esp-nn)) We should build a small conformance/benchmark harness that feeds identical vectors into our RMSNorm/MatMul/Softmax/quantization code and the closest Espressif implementation, measuring cycles and numerical error. This gives us a vendor-optimized performance reference without forcing the entire autoregressive model into ESP-DL/TFLM. When our custom kernel is slower, we have a concrete baseline; when it is faster, we can explain why the LLM-specific shape/layout benefits.

- **20. Turn ESP-IDF's external-memory behavior into explicit runtime constraints and benchmark 80 MHz versus 120 MHz Octal PSRAM where the board supports it.** Espressif warns that for “**large chunks of data (> 32 KB), the cache can be insufficient**”; current ESP32-S3 documentation also notes optional 120 MHz Octal-PSRAM operation with restrictions, and that PSRAM becomes inaccessible when flash cache is disabled during flash writes/erases. ([ESP-IDF: ESP32-S3 external RAM](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-guides/external-ram.html)) This has direct LLM consequences. During inference, avoid NVS/OTA/model writes on the critical path; keep the inference task stack and truly hot buffers internal; benchmark `CONFIG_SPIRAM_SPEED`, cache sizes, `.rodata`/instruction placement, and XIP-from-PSRAM options; and measure whether moving code/rodata into PSRAM helps or instead competes with weight traffic. Add a startup self-test that reports actual PSRAM mode/speed and refuses to publish benchmark numbers without that configuration, because the memory subsystem is part of the model's effective hardware.

