/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include <stdlib.h>

#include "builder.h"
#include "test.h"
#include "tinyllm/tinyllm.h"

void test_transformer(void);

typedef struct {
    test_blob blob;
    tllm_model model;
    tllm_ctx ctx;
    void *hot, *cold;
} fixture;

static uint64_t g_fake_us;
static uint64_t fake_clock(void) { return g_fake_us += 3u; }

static int setup(fixture *f, const test_model_spec *spec, const tllm_ctx_options *opt) {
    f->blob = test_build_model(spec);
    if (tllm_model_load(&f->model, f->blob.data, f->blob.size) != TLLM_OK) return -1;
    int kv8 = opt != NULL && opt->kv_int8;
    size_t hs = tllm_hot_arena_size(&f->model), cs = tllm_cold_arena_size(&f->model, kv8);
    f->hot = malloc(hs);
    f->cold = malloc(cs);
    return tllm_ctx_init(&f->ctx, &f->model, opt, f->hot, hs, f->cold, cs) == TLLM_OK ? 0 : -1;
}

static void teardown(fixture *f) {
    free(f->hot);
    free(f->cold);
    test_free_blob(&f->blob);
}

static float max_abs_diff(const float *a, const float *b, uint32_t n) {
    float m = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float d = fabsf(a[i] - b[i]);
        if (d > m) m = d;
    }
    return m;
}

static double correlation(const float *a, const float *b, uint32_t n) {
    double ma = 0.0, mb = 0.0, sab = 0.0, saa = 0.0, sbb = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        ma += a[i];
        mb += b[i];
    }
    ma /= n;
    mb /= n;
    for (uint32_t i = 0; i < n; ++i) {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma) * (a[i] - ma);
        sbb += (b[i] - mb) * (b[i] - mb);
    }
    return sab / sqrt(saa * sbb);
}

static const int32_t SEQ[6] = {1, 20, 40, 60, 80, 100};

/* Logits after running SEQ through a fixture with the given spec/options. */
static void run_seq(const test_model_spec *spec, const tllm_ctx_options *opt, float *out) {
    fixture f;
    CHECK_EQ_INT(setup(&f, spec, opt), 0);
    const float *logits = NULL;
    CHECK_EQ_INT(tllm_prefill(&f.ctx, SEQ, 6, &logits), TLLM_OK);
    memcpy(out, logits, sizeof(float) * f.model.cfg.vocab_size);
    teardown(&f);
}

void test_transformer(void) {
    test_model_spec spec = test_default_spec();
    fixture f;
    CHECK_EQ_INT(setup(&f, &spec, NULL), 0);
    const uint32_t V = f.model.cfg.vocab_size;
    float *seq_logits = malloc(sizeof(float) * V), *ref = malloc(sizeof(float) * V);
    const float *logits = NULL;

    /* token by token == prefill */
    for (int i = 0; i < 6; ++i) CHECK_EQ_INT(tllm_forward(&f.ctx, SEQ[i], &logits), TLLM_OK);
    memcpy(seq_logits, logits, sizeof(float) * V);
    CHECK_EQ_INT(f.ctx.n_cached, 6);
    tllm_ctx_reset(&f.ctx);
    CHECK_EQ_INT(tllm_prefill(&f.ctx, SEQ, 6, &logits), TLLM_OK);
    CHECK_NEAR(max_abs_diff(seq_logits, logits, V), 0.0, 1e-6);
    /* prefix reuse: same prompt again recomputes only the last token */
    CHECK_EQ_INT(tllm_prefill(&f.ctx, SEQ, 6, &logits), TLLM_OK);
    CHECK_NEAR(max_abs_diff(seq_logits, logits, V), 0.0, 1e-6);
    /* diverging suffix after a shared prefix */
    int32_t other[6] = {1, 20, 40, 61, 80, 100};
    CHECK_EQ_INT(tllm_prefill(&f.ctx, other, 6, &logits), TLLM_OK);
    CHECK(max_abs_diff(seq_logits, logits, V) > 1e-4f);
    CHECK_EQ_INT(tllm_prefill(&f.ctx, SEQ, 0, &logits), TLLM_OK);
    CHECK(logits == NULL);
    CHECK(isfinite(f.ctx.checksums[0]) && isfinite(f.ctx.checksums[2]));

    /* argument and capacity errors */
    CHECK_EQ_INT(tllm_forward(&f.ctx, -1, &logits), TLLM_ERR_ARG);
    CHECK_EQ_INT(tllm_forward(&f.ctx, (int32_t)V, &logits), TLLM_ERR_ARG);
    CHECK_EQ_INT(tllm_forward(NULL, 1, &logits), TLLM_ERR_ARG);
    CHECK_EQ_INT(tllm_prefill(NULL, SEQ, 1, &logits), TLLM_ERR_ARG);
    CHECK_EQ_INT(tllm_prefill(&f.ctx, NULL, 1, &logits), TLLM_ERR_ARG);
    int32_t *longseq = calloc(spec.ctx_len + 1u, sizeof(int32_t));
    CHECK_EQ_INT(tllm_prefill(&f.ctx, longseq, spec.ctx_len + 1u, &logits), TLLM_ERR_CONTEXT_FULL);
    tllm_ctx_reset(&f.ctx);
    CHECK_EQ_INT(tllm_prefill(&f.ctx, longseq, spec.ctx_len, &logits), TLLM_OK);
    CHECK_EQ_INT(tllm_forward(&f.ctx, 1, &logits), TLLM_ERR_CONTEXT_FULL);
    int32_t bad[2] = {1, -5};
    tllm_ctx_reset(&f.ctx);
    CHECK_EQ_INT(tllm_prefill(&f.ctx, bad, 2, &logits), TLLM_ERR_ARG);
    free(longseq);

    tllm_ctx other_ctx;
    CHECK_EQ_INT(tllm_ctx_init(&other_ctx, &f.model, NULL, f.hot, 8, f.cold, 1u << 20), TLLM_ERR_ARENA);
    CHECK_EQ_INT(tllm_ctx_init(NULL, &f.model, NULL, f.hot, 8, f.cold, 8), TLLM_ERR_ARG);
    teardown(&f);

    /* int8 weights, W8A8, int8 KV cache stay close to the float reference */
    run_seq(&spec, NULL, ref);
    test_model_spec q = spec;
    q.dtype = 1;
    run_seq(&q, NULL, seq_logits);
    CHECK(max_abs_diff(ref, seq_logits, V) < 0.05f);
    tllm_ctx_options a8 = {TLLM_ACT_I8, 0, NULL, NULL, NULL, NULL, 0};
    run_seq(&q, &a8, seq_logits);
    CHECK(max_abs_diff(ref, seq_logits, V) < 0.1f);
    /* Q4 weights (32-wide groups need a wider model) stay close to the float reference */
    test_model_spec wide = spec;
    wide.d_model = 32;
    wide.d_ff = 64;
    float *wide_ref = malloc(sizeof(float) * V);
    run_seq(&wide, NULL, wide_ref);
    wide.dtype = 2;
    run_seq(&wide, NULL, seq_logits);
    CHECK(correlation(wide_ref, seq_logits, V) > 0.95); /* exact Q4 numerics: Python parity test */
    CHECK(max_abs_diff(wide_ref, seq_logits, V) > 0.0f);
    run_seq(&wide, &a8, seq_logits);
    CHECK(correlation(wide_ref, seq_logits, V) > 0.95);
    free(wide_ref);
    tllm_ctx_options kv8 = {TLLM_ACT_F32, 1, NULL, NULL, NULL, NULL, 0};
    run_seq(&spec, &kv8, seq_logits);
    CHECK(max_abs_diff(ref, seq_logits, V) < 0.05f);
    CHECK(max_abs_diff(ref, seq_logits, V) > 0.0f);

    /* RoPE + SwiGLU + MQA variant runs and differs by position */
    test_model_spec r = spec;
    r.pos_type = 1;
    r.mlp_type = 1;
    r.n_kv_heads = 1;
    r.n_layers = 11; /* two-digit layer names */
    r.ctx_len = 16;
    tllm_ctx_options prof = {TLLM_ACT_F32, 0, fake_clock, NULL, NULL, NULL, 0};
    CHECK_EQ_INT(setup(&f, &r, &prof), 0);
    f.ctx.profiling = 1;
    CHECK_EQ_INT(tllm_forward(&f.ctx, 31, &logits), TLLM_OK);
    memcpy(ref, logits, sizeof(float) * V);
    tllm_ctx_reset(&f.ctx);
    CHECK_EQ_INT(tllm_forward(&f.ctx, 30, &logits), TLLM_OK);
    CHECK_EQ_INT(tllm_forward(&f.ctx, 31, &logits), TLLM_OK);
    CHECK(max_abs_diff(ref, logits, V) > 1e-5f); /* attention mixes in the earlier token */
    CHECK(f.ctx.prof_us[TLLM_PROF_QKV] > 0u && f.ctx.prof_us[TLLM_PROF_FFN] > 0u);
    CHECK_EQ_INT(f.ctx.prof_tokens, 3);
    tllm_profile_reset(&f.ctx);
    CHECK_EQ_INT(f.ctx.prof_us[TLLM_PROF_HEAD], 0);
    CHECK_STR(tllm_prof_stage_name(TLLM_PROF_HEAD), "head");
    CHECK_STR(tllm_prof_stage_name(TLLM_PROF_COUNT), "?");
    teardown(&f);

    free(seq_logits);
    free(ref);
}
