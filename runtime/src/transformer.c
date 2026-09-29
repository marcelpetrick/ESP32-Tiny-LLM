/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Token-by-token decoder forward pass with a KV cache (vision §15). Every stage named in
 * vision §2 is a visible step below; timings go to the profile buckets of vision §17.
 */
#include <math.h>
#include <string.h>

#include "tinyllm/tinyllm.h"

/* ------------------------------------------------------------------ arena layout */
typedef struct {
    uint8_t *base; /* NULL while only measuring */
    size_t used;
} bump;

static void *take(bump *b, size_t bytes) {
    size_t start = (b->used + 15u) & ~(size_t)15u;
    b->used = start + bytes;
    return b->base == NULL ? NULL : b->base + start;
}

static uint32_t kv_dim(const tllm_config *c) { return c->n_kv_heads * (c->d_model / c->n_heads); }

static uint32_t max_u32(uint32_t a, uint32_t b) { return a > b ? a : b; }

/* Lay out the hot arena; returns the tokenizer workspace. */
static void *layout_hot(tllm_ctx *ctx, const tllm_model *m, bump *b) {
    const tllm_config *c = &m->cfg;
    const uint32_t kvd = kv_dim(c);
    tllm_ctx scratch;
    tllm_ctx *t = ctx != NULL ? ctx : &scratch;
    t->x = take(b, sizeof(float) * c->d_model);
    t->xb = take(b, sizeof(float) * c->d_model);
    t->xb2 = take(b, sizeof(float) * c->d_model);
    t->q = take(b, sizeof(float) * c->d_model);
    t->k = take(b, sizeof(float) * kvd);
    t->v = take(b, sizeof(float) * kvd);
    t->att = take(b, sizeof(float) * c->n_heads * c->ctx_len);
    t->hb = take(b, sizeof(float) * c->d_ff);
    t->hb2 = take(b, sizeof(float) * c->d_ff);
    t->logits = take(b, sizeof(float) * c->vocab_size);
    t->qx = take(b, max_u32(c->d_model, c->d_ff));
    t->tokens = take(b, sizeof(int32_t) * c->ctx_len);
    t->checksums = take(b, sizeof(double) * (c->n_layers + 1u));
    return take(b, tllm_tokenizer_workspace_size(&m->tok));
}

static void layout_cold(tllm_ctx *ctx, const tllm_model *m, int kv_int8, bump *b) {
    const tllm_config *c = &m->cfg;
    const size_t cells = (size_t)c->n_layers * c->ctx_len * kv_dim(c);
    const size_t rows = (size_t)c->n_layers * c->ctx_len * c->n_kv_heads;
    tllm_ctx scratch;
    tllm_ctx *t = ctx != NULL ? ctx : &scratch;
    t->key_f32 = t->val_f32 = NULL;
    t->key_i8 = t->val_i8 = NULL;
    t->key_scale = t->val_scale = NULL;
    if (kv_int8) {
        t->key_i8 = take(b, cells);
        t->val_i8 = take(b, cells);
        t->key_scale = take(b, sizeof(float) * rows);
        t->val_scale = take(b, sizeof(float) * rows);
    } else {
        t->key_f32 = take(b, sizeof(float) * cells);
        t->val_f32 = take(b, sizeof(float) * cells);
    }
}

size_t tllm_hot_arena_size(const tllm_model *m) {
    bump b = {NULL, 0};
    (void)layout_hot(NULL, m, &b);
    return b.used + 16u;
}

size_t tllm_cold_arena_size(const tllm_model *m, int kv_int8) {
    bump b = {NULL, 0};
    layout_cold(NULL, m, kv_int8, &b);
    return b.used + 16u;
}

static uint8_t *align16(void *p) { return (uint8_t *)p + ((16u - ((uintptr_t)p & 15u)) & 15u); }

tllm_status tllm_ctx_init(tllm_ctx *ctx, tllm_model *m, const tllm_ctx_options *opt, void *hot, size_t hot_size,
                          void *cold, size_t cold_size) {
    if (ctx == NULL || m == NULL || m->blob == NULL || hot == NULL || cold == NULL) return TLLM_ERR_ARG;
    const int kv_int8 = opt != NULL && opt->kv_int8;
    if (hot_size < tllm_hot_arena_size(m) || cold_size < tllm_cold_arena_size(m, kv_int8)) return TLLM_ERR_ARENA;
    memset(ctx, 0, sizeof *ctx);
    ctx->model = m;
    if (opt != NULL) ctx->opt = *opt;
    bump h = {align16(hot), 0};
    tllm_tokenizer_attach(&m->tok, layout_hot(ctx, m, &h));
    bump c = {align16(cold), 0};
    layout_cold(ctx, m, kv_int8, &c);
    tllm_ctx_reset(ctx);
    return TLLM_OK;
}

void tllm_ctx_reset(tllm_ctx *ctx) { ctx->n_cached = 0; }

void tllm_profile_reset(tllm_ctx *ctx) {
    memset(ctx->prof_us, 0, sizeof ctx->prof_us);
    ctx->prof_tokens = 0;
}

const char *tllm_prof_stage_name(tllm_prof_stage stage) {
    static const char *const names[TLLM_PROF_COUNT] = {"tokenizer",  "embedding", "norm",       "qkv",
                                                       "attn_score", "softmax",   "attn_value", "attn_out",
                                                       "ffn",        "head",      "sampling"};
    return (unsigned)stage < TLLM_PROF_COUNT ? names[stage] : "?";
}

/* ------------------------------------------------------------------ helpers */
static uint64_t now(const tllm_ctx *ctx) { return (ctx->profiling && ctx->opt.clock != NULL) ? ctx->opt.clock() : 0u; }

static void prof_add(tllm_ctx *ctx, tllm_prof_stage stage, uint64_t start) {
    if (ctx->profiling && ctx->opt.clock != NULL) ctx->prof_us[stage] += ctx->opt.clock() - start;
}

/* y = W x, using the W8A8 path when the context asks for it and W is int8. */
static void project(const tllm_ctx *ctx, float *out, const tllm_tensor *w, const float *x, int8_t *qx, float *qscale,
                    int *have_q) {
    if (w->dtype != TLLM_DTYPE_F32 && ctx->opt.act_mode == TLLM_ACT_I8) {
        if (!*have_q) {
            *qscale = tllm_quantize_vec(qx, x, w->cols);
            *have_q = 1;
        }
        tllm_matvec_q8(out, w, qx, *qscale);
    } else {
        tllm_matvec(out, w, x);
    }
}

static void embed_row(float *out, const tllm_tensor *t, uint32_t row) {
    const uint32_t d = t->cols;
    if (t->dtype == TLLM_DTYPE_F32) {
        memcpy(out, (const float *)t->data + (size_t)row * d, sizeof(float) * d);
    } else if (t->dtype == TLLM_DTYPE_Q4) {
        const uint8_t *p = (const uint8_t *)t->data + (size_t)row * (d / 2u);
        const uint16_t *s = t->scales16 + (size_t)row * (d / TLLM_Q4_GROUP);
        for (uint32_t i = 0; i < d; ++i) {
            int q = (i & 1u) ? (int)(p[i / 2u] >> 4) : (int)(p[i / 2u] & 0x0Fu);
            out[i] = (float)(q - 8) * tllm_f16_to_f32(s[i / TLLM_Q4_GROUP]);
        }
    } else {
        const int8_t *q = (const int8_t *)t->data + (size_t)row * d;
        for (uint32_t i = 0; i < d; ++i) out[i] = (float)q[i] * t->scales[row];
    }
}

static void rope(float *vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta) {
    for (uint32_t i = 0; i < head_dim; i += 2u) {
        float freq = 1.0f / powf(theta, (float)i / (float)head_dim);
        float angle = (float)pos * freq;
        float c = cosf(angle), s = sinf(angle);
        for (uint32_t h = 0; h < n_heads; ++h) {
            float *p = vec + h * head_dim + i;
            float a = p[0], b = p[1];
            p[0] = a * c - b * s;
            p[1] = a * s + b * c;
        }
    }
}

static double sum_of(const float *x, uint32_t n) {
    double s = 0.0;
    for (uint32_t i = 0; i < n; ++i) s += (double)x[i];
    return s;
}

/* Store this position's K/V rows and return pointers usable for attention. */
static void kv_store(tllm_ctx *ctx, uint32_t layer, uint32_t pos) {
    const tllm_config *c = &ctx->model->cfg;
    const uint32_t kvd = kv_dim(c), hd = c->d_model / c->n_heads;
    const size_t base = ((size_t)layer * c->ctx_len + pos) * kvd;
    if (ctx->key_f32 != NULL) {
        memcpy(ctx->key_f32 + base, ctx->k, sizeof(float) * kvd);
        memcpy(ctx->val_f32 + base, ctx->v, sizeof(float) * kvd);
        return;
    }
    const size_t row = ((size_t)layer * c->ctx_len + pos) * c->n_kv_heads;
    for (uint32_t g = 0; g < c->n_kv_heads; ++g) {
        ctx->key_scale[row + g] = tllm_quantize_vec(ctx->key_i8 + base + g * hd, ctx->k + g * hd, hd);
        ctx->val_scale[row + g] = tllm_quantize_vec(ctx->val_i8 + base + g * hd, ctx->v + g * hd, hd);
    }
}

static void attention(tllm_ctx *ctx, uint32_t layer, uint32_t pos) {
    const tllm_config *c = &ctx->model->cfg;
    const uint32_t hd = c->d_model / c->n_heads, kvd = kv_dim(c);
    const uint32_t group = c->n_heads / c->n_kv_heads;
    const float inv_sqrt = 1.0f / sqrtf((float)hd);
    const size_t layer_base = (size_t)layer * c->ctx_len * kvd;
    const size_t layer_rows = (size_t)layer * c->ctx_len * c->n_kv_heads;
    for (uint32_t h = 0; h < c->n_heads; ++h) {
        const uint32_t g = h / group;
        const float *q = ctx->q + h * hd;
        float *att = ctx->att + h * c->ctx_len;
        uint64_t t0 = now(ctx);
        for (uint32_t t = 0; t <= pos; ++t) { /* causal: only positions 0..pos exist */
            const size_t off = layer_base + (size_t)t * kvd + g * hd;
            float score = 0.0f;
            if (ctx->key_f32 != NULL) {
                const float *k = ctx->key_f32 + off;
                for (uint32_t i = 0; i < hd; ++i) score += q[i] * k[i];
            } else {
                const int8_t *k = ctx->key_i8 + off;
                for (uint32_t i = 0; i < hd; ++i) score += q[i] * (float)k[i];
                score *= ctx->key_scale[layer_rows + (size_t)t * c->n_kv_heads + g];
            }
            att[t] = score * inv_sqrt;
        }
        prof_add(ctx, TLLM_PROF_ATTN_SCORE, t0);
        t0 = now(ctx);
        tllm_softmax(att, pos + 1u);
        prof_add(ctx, TLLM_PROF_SOFTMAX, t0);
        t0 = now(ctx);
        float *out = ctx->xb2 + h * hd;
        memset(out, 0, sizeof(float) * hd);
        for (uint32_t t = 0; t <= pos; ++t) {
            const size_t off = layer_base + (size_t)t * kvd + g * hd;
            if (ctx->val_f32 != NULL) {
                const float *v = ctx->val_f32 + off;
                for (uint32_t i = 0; i < hd; ++i) out[i] += att[t] * v[i];
            } else {
                const int8_t *v = ctx->val_i8 + off;
                const float w = att[t] * ctx->val_scale[layer_rows + (size_t)t * c->n_kv_heads + g];
                for (uint32_t i = 0; i < hd; ++i) out[i] += w * (float)v[i];
            }
        }
        prof_add(ctx, TLLM_PROF_ATTN_VALUE, t0);
    }
}

/* ------------------------------------------------------------------ forward */
tllm_status tllm_forward(tllm_ctx *ctx, int32_t token, const float **logits) {
    if (ctx == NULL || ctx->model == NULL) return TLLM_ERR_ARG;
    const tllm_model *m = ctx->model;
    const tllm_config *c = &m->cfg;
    if (token < 0 || (uint32_t)token >= c->vocab_size) return TLLM_ERR_ARG;
    const uint32_t pos = ctx->n_cached;
    if (pos >= c->ctx_len) return TLLM_ERR_CONTEXT_FULL;
    const uint32_t d = c->d_model, hd = d / c->n_heads;
    float qscale = 1.0f;
    int have_q;

    uint64_t t0 = now(ctx);
    embed_row(ctx->x, &m->tok_emb, (uint32_t)token);
    if (c->pos_type == TLLM_POS_LEARNED) {
        const float *p = (const float *)m->pos_emb.data + (size_t)pos * d;
        for (uint32_t i = 0; i < d; ++i) ctx->x[i] += p[i];
    }
    prof_add(ctx, TLLM_PROF_EMBEDDING, t0);

    for (uint32_t l = 0; l < c->n_layers; ++l) {
        const tllm_layer *L = &m->layers[l];
        t0 = now(ctx);
        tllm_rmsnorm(ctx->xb, ctx->x, (const float *)L->attn_norm.data, d, c->norm_eps);
        prof_add(ctx, TLLM_PROF_NORM, t0);

        t0 = now(ctx);
        have_q = 0;
        project(ctx, ctx->q, &L->wq, ctx->xb, ctx->qx, &qscale, &have_q);
        project(ctx, ctx->k, &L->wk, ctx->xb, ctx->qx, &qscale, &have_q);
        project(ctx, ctx->v, &L->wv, ctx->xb, ctx->qx, &qscale, &have_q);
        if (c->pos_type == TLLM_POS_ROPE) {
            rope(ctx->q, c->n_heads, hd, pos, c->rope_theta);
            rope(ctx->k, c->n_kv_heads, hd, pos, c->rope_theta);
        }
        kv_store(ctx, l, pos);
        prof_add(ctx, TLLM_PROF_QKV, t0);

        attention(ctx, l, pos);

        t0 = now(ctx);
        have_q = 0;
        project(ctx, ctx->xb, &L->wo, ctx->xb2, ctx->qx, &qscale, &have_q);
        for (uint32_t i = 0; i < d; ++i) ctx->x[i] += ctx->xb[i]; /* residual */
        prof_add(ctx, TLLM_PROF_ATTN_OUT, t0);

        t0 = now(ctx);
        tllm_rmsnorm(ctx->xb, ctx->x, (const float *)L->mlp_norm.data, d, c->norm_eps);
        prof_add(ctx, TLLM_PROF_NORM, t0);

        t0 = now(ctx);
        have_q = 0;
        project(ctx, ctx->hb, &L->w1, ctx->xb, ctx->qx, &qscale, &have_q);
        if (c->mlp_type == TLLM_MLP_SWIGLU) {
            project(ctx, ctx->hb2, &L->w3, ctx->xb, ctx->qx, &qscale, &have_q);
            for (uint32_t i = 0; i < c->d_ff; ++i) ctx->hb[i] = tllm_silu(ctx->hb[i]) * ctx->hb2[i];
        } else {
            for (uint32_t i = 0; i < c->d_ff; ++i) ctx->hb[i] = tllm_gelu(ctx->hb[i]);
        }
        have_q = 0;
        project(ctx, ctx->xb, &L->w2, ctx->hb, ctx->qx, &qscale, &have_q);
        for (uint32_t i = 0; i < d; ++i) ctx->x[i] += ctx->xb[i]; /* residual */
        prof_add(ctx, TLLM_PROF_FFN, t0);
        ctx->checksums[l] = sum_of(ctx->x, d);
    }

    t0 = now(ctx);
    tllm_rmsnorm(ctx->x, ctx->x, (const float *)m->final_norm.data, d, c->norm_eps);
    prof_add(ctx, TLLM_PROF_NORM, t0);
    t0 = now(ctx);
    have_q = 0;
    project(ctx, ctx->logits, &m->tok_emb, ctx->x, ctx->qx, &qscale, &have_q); /* tied head */
    prof_add(ctx, TLLM_PROF_HEAD, t0);
    ctx->checksums[c->n_layers] = sum_of(ctx->x, d);

    ctx->tokens[pos] = token;
    ctx->n_cached = pos + 1u;
    ctx->prof_tokens++;
    if (ctx->opt.yield != NULL) ctx->opt.yield();
    if (logits != NULL) *logits = ctx->logits;
    return TLLM_OK;
}

tllm_status tllm_prefill(tllm_ctx *ctx, const int32_t *tokens, uint32_t n, const float **logits) {
    if (ctx == NULL || (tokens == NULL && n > 0u)) return TLLM_ERR_ARG;
    if (logits != NULL) *logits = NULL;
    if (n == 0u) return TLLM_OK;
    if (n > ctx->model->cfg.ctx_len) return TLLM_ERR_CONTEXT_FULL;
    uint32_t common = 0;
    while (common < ctx->n_cached && common < n && ctx->tokens[common] == tokens[common]) ++common;
    if (common == n) --common; /* recompute the last token to obtain its logits */
    ctx->n_cached = common;
    for (uint32_t i = common; i < n; ++i) {
        tllm_status st = tllm_forward(ctx, tokens[i], logits);
        if (st != TLLM_OK) return st;
    }
    return TLLM_OK;
}
