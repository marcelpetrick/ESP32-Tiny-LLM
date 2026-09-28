/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * tinyllm — portable C99 inference runtime for tiny decoder-only transformers.
 *
 * Design rules (docs/05-architecture.md):
 *   - the model blob is used in place (no copies, no parsing allocations);
 *   - all working memory comes from two caller-provided arenas ("hot" for small,
 *     latency-critical vectors — internal SRAM on the ESP32 — and "cold" for the KV
 *     cache — PSRAM); nothing allocates after tllm_ctx_init();
 *   - every transformer stage is a named function that can be profiled.
 */
#ifndef TINYLLM_TINYLLM_H
#define TINYLLM_TINYLLM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TLLM_FORMAT_VERSION 1u
#define TLLM_HEADER_SIZE 128u
#define TLLM_MAX_LAYERS 16u
#define TLLM_MAX_SPECIAL 32u
#define TLLM_MAX_CTX 1024u
#define TLLM_MAX_DIM 1024u
#define TLLM_MAX_VOCAB 32768u

typedef enum {
    TLLM_OK = 0,
    TLLM_ERR_ARG,         /* invalid argument */
    TLLM_ERR_MAGIC,       /* not a .tllm file */
    TLLM_ERR_VERSION,     /* unsupported format version / header size / architecture */
    TLLM_ERR_TRUNCATED,   /* sizes or offsets out of range */
    TLLM_ERR_CRC,         /* payload checksum mismatch */
    TLLM_ERR_UNSUPPORTED, /* hyper-parameters beyond compile-time limits */
    TLLM_ERR_TENSOR,      /* missing tensor or wrong shape/dtype */
    TLLM_ERR_TOKENIZER,   /* malformed tokenizer block */
    TLLM_ERR_ARENA,       /* arena too small */
    TLLM_ERR_CONTEXT_FULL /* no room left in the context window */
} tllm_status;

const char *tllm_status_str(tllm_status status);

typedef enum { TLLM_DTYPE_F32 = 0, TLLM_DTYPE_I8 = 1 } tllm_dtype;
typedef enum { TLLM_MLP_GELU = 0, TLLM_MLP_SWIGLU = 1 } tllm_mlp_type;
typedef enum { TLLM_POS_LEARNED = 0, TLLM_POS_ROPE = 1 } tllm_pos_type;
typedef enum { TLLM_ACT_F32 = 0, TLLM_ACT_I8 = 1 } tllm_act_mode;

typedef struct {
    uint32_t vocab_size, ctx_len, n_layers, d_model, n_heads, n_kv_heads, d_ff;
    uint32_t mlp_type, pos_type, weight_dtype, flags;
    float norm_eps, rope_theta;
} tllm_config;

/* A weight tensor living inside the model blob. */
typedef struct {
    const void *data;    /* float32 or int8, row-major [rows, cols] */
    const float *scales; /* one per row for int8, else NULL */
    uint32_t dtype, rows, cols;
} tllm_tensor;

typedef struct {
    tllm_tensor attn_norm, wq, wk, wv, wo, mlp_norm, w1, w2, w3;
} tllm_layer;

#define TLLM_MAX_PIECE 64u

typedef enum {
    TLLM_TOK_BPE = 1,   /* byte-level BPE with ranked merges (training/tokenizer/bpe.py) */
    TLLM_TOK_SCORED = 2 /* llama2.c-style scored pieces (training/tokenizer/llama2c.py) */
} tllm_tokenizer_kind;

typedef struct {
    uint32_t kind, n_special, n_merges, vocab_size;
    int32_t bos_id, eos_id;
    const char *special[TLLM_MAX_SPECIAL];
    uint8_t special_len[TLLM_MAX_SPECIAL];
    /* BPE */
    const uint8_t *merges; /* n_merges pairs of little-endian u16 (a, b) */
    uint32_t *hash_keys;   /* open-addressing pair -> rank table (workspace) */
    uint16_t *hash_ranks;
    uint32_t hash_mask;
    /* scored pieces */
    const uint8_t *records; /* per token: f32 score, u16 length, bytes */
    uint32_t max_piece_len;
    uint32_t *piece_off; /* workspace: record offset per token */
    uint16_t *sorted;    /* workspace: token ids sorted by piece bytes */
    uint8_t attached;    /* workspace built */
} tllm_tokenizer;

typedef struct {
    tllm_config cfg;
    tllm_tensor tok_emb, pos_emb, final_norm;
    tllm_layer layers[TLLM_MAX_LAYERS];
    tllm_tokenizer tok;
    const uint8_t *blob;
    size_t size;
    uint8_t model_id[16];
    uint32_t crc32;
} tllm_model;

/* Parse and verify a model blob (kept by reference; must outlive the model). The blob
 * must be 4-byte aligned. The tokenizer hash table is attached later by tllm_ctx_init(). */
tllm_status tllm_model_load(tllm_model *model, const void *blob, size_t size);

/* Per-stage profile buckets (vision §17). Times are microseconds, accumulated. */
typedef enum {
    TLLM_PROF_TOKENIZER = 0,
    TLLM_PROF_EMBEDDING,
    TLLM_PROF_NORM,
    TLLM_PROF_QKV,
    TLLM_PROF_ATTN_SCORE,
    TLLM_PROF_SOFTMAX,
    TLLM_PROF_ATTN_VALUE,
    TLLM_PROF_ATTN_OUT,
    TLLM_PROF_FFN,
    TLLM_PROF_HEAD,
    TLLM_PROF_SAMPLING,
    TLLM_PROF_COUNT
} tllm_prof_stage;

const char *tllm_prof_stage_name(tllm_prof_stage stage);

typedef uint64_t (*tllm_clock_fn)(void);

typedef struct {
    tllm_act_mode act_mode; /* int8 weights: TLLM_ACT_F32 (W8A32) or TLLM_ACT_I8 (W8A8) */
    int kv_int8;            /* 1: int8 KV cache with per-row scales, 0: float32 */
    tllm_clock_fn clock;    /* microsecond clock for profiling, may be NULL */
} tllm_ctx_options;

typedef struct {
    const tllm_model *model;
    tllm_ctx_options opt;
    /* hot arena */
    float *x, *xb, *xb2, *q, *k, *v, *att, *hb, *hb2, *logits;
    int8_t *qx;
    int32_t *tokens;   /* tokens currently in the KV cache (for prefix reuse) */
    double *checksums; /* per-layer output sums of the last forward, n_layers + 1 */
    /* cold arena */
    float *key_f32, *val_f32;
    int8_t *key_i8, *val_i8;
    float *key_scale, *val_scale;
    uint32_t n_cached; /* number of positions in the KV cache */
    int profiling;
    uint64_t prof_us[TLLM_PROF_COUNT];
    uint32_t prof_tokens;
} tllm_ctx;

/* Exact arena sizes (bytes) needed for a model with the given options. */
size_t tllm_hot_arena_size(const tllm_model *model);
size_t tllm_cold_arena_size(const tllm_model *model, int kv_int8);

/* Bind a context to a model and arenas (16-byte aligned recommended). Builds the
 * tokenizer hash table in the hot arena. */
tllm_status tllm_ctx_init(tllm_ctx *ctx, tllm_model *model, const tllm_ctx_options *opt, void *hot, size_t hot_size,
                          void *cold, size_t cold_size);

/* Forget the KV cache (new conversation). */
void tllm_ctx_reset(tllm_ctx *ctx);

/* Run one token at position ctx->n_cached; on success *logits points to vocab_size floats
 * owned by the context. */
tllm_status tllm_forward(tllm_ctx *ctx, int32_t token, const float **logits);

/* Feed prompt tokens, reusing the longest prefix already in the KV cache. Returns the
 * logits after the last token via *logits (NULL if nothing was run and no cache). */
tllm_status tllm_prefill(tllm_ctx *ctx, const int32_t *tokens, uint32_t n, const float **logits);

void tllm_profile_reset(tllm_ctx *ctx);

/* ---------------------------------------------------------------- tokenizer */
/* Lookup tables live in caller memory (4-byte aligned): the merge hash table for BPE,
 * piece offsets + sorted index for scored tokenizers. tllm_ctx_init() does this. */
size_t tllm_tokenizer_workspace_size(const tllm_tokenizer *tok);
void tllm_tokenizer_attach(tllm_tokenizer *tok, void *workspace);

/* Encode plain text (special-token strings are NOT recognised). Scored tokenizers add
 * llama2.c's dummy-prefix space. Returns the number of tokens written, or -1 if max_out is
 * too small or the tokenizer has no workspace. */
int tllm_tokenize(const tllm_tokenizer *tok, const char *text, size_t len, int32_t *out, int max_out);

/* Id of a special token such as "<U>", or -1. */
int32_t tllm_special_id(const tllm_tokenizer *tok, const char *name);

/* Write the bytes of one token (not NUL-terminated; "<0xHH>" pieces become the byte).
 * Returns the byte count, or -1 if the id is invalid or max_out is too small. */
int tllm_token_bytes(const tllm_tokenizer *tok, int32_t id, char *out, int max_out);

/* ---------------------------------------------------------------- sampler */
typedef struct {
    float temperature;      /* <= 0: greedy */
    uint32_t top_k;         /* 0: disabled */
    float repeat_penalty;   /* 1.0: disabled */
    uint32_t repeat_window; /* recent tokens considered by the penalty */
} tllm_sampler_cfg;

typedef struct {
    uint64_t state;
} tllm_rng;

void tllm_rng_seed(tllm_rng *rng, uint64_t seed);
float tllm_rng_float(tllm_rng *rng); /* [0, 1) */

/* Pick the next token. `logits` is modified in place (penalty, temperature). `recent`
 * holds the last `n_recent` generated tokens. */
int32_t tllm_sample(float *logits, uint32_t n, const tllm_sampler_cfg *cfg, tllm_rng *rng, const int32_t *recent,
                    uint32_t n_recent);

/* ---------------------------------------------------------------- kernels */
void tllm_rmsnorm(float *out, const float *x, const float *weight, uint32_t n, float eps);
void tllm_softmax(float *x, uint32_t n);
/* out[rows] = W[rows, cols] * x[cols] for f32 or int8 (W8A32) weights */
void tllm_matvec(float *out, const tllm_tensor *w, const float *x);
/* int8 weights and int8-quantised activations, int32 accumulation (W8A8) */
void tllm_matvec_q8(float *out, const tllm_tensor *w, const int8_t *qx, float x_scale);
float tllm_quantize_vec(int8_t *out, const float *x, uint32_t n); /* returns scale */
float tllm_gelu(float x);
float tllm_silu(float x);
uint32_t tllm_crc32(const uint8_t *data, size_t len);

#ifdef __cplusplus
}
#endif
#endif /* TINYLLM_TINYLLM_H */
