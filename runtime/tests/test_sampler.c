/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include "test.h"
#include "tinyllm/tinyllm.h"

void test_sampler(void);

void test_sampler(void) {
    float logits[5] = {0.1f, 3.0f, 2.9f, -1.0f, 0.0f};
    tllm_sampler_cfg greedy = {0.0f, 0, 1.0f, 0};
    tllm_rng rng;
    tllm_rng_seed(&rng, 0);
    CHECK(rng.state != 0u);
    CHECK_EQ_INT(tllm_sample(logits, 5, NULL, NULL, NULL, 0), 1);
    CHECK_EQ_INT(tllm_sample(logits, 5, &greedy, &rng, NULL, 0), 1);

    /* repetition penalty pushes the recent token down (positive and negative logits) */
    float pen[3] = {2.0f, 1.9f, -1.0f};
    int32_t recent[3] = {0, 2, 7};
    tllm_sampler_cfg p = {0.0f, 0, 2.0f, 8};
    CHECK_EQ_INT(tllm_sample(pen, 3, &p, &rng, recent, 3), 1);
    CHECK_NEAR(pen[0], 1.0, 1e-6);
    CHECK_NEAR(pen[2], -2.0, 1e-6);

    /* temperature sampling is deterministic per seed and respects top-k */
    tllm_sampler_cfg topk = {1.0f, 2, 1.0f, 0};
    int counts[5] = {0};
    tllm_rng_seed(&rng, 42);
    for (int i = 0; i < 400; ++i) {
        float l[5] = {0.1f, 3.0f, 2.9f, -1.0f, 2.8f};
        int32_t t = tllm_sample(l, 5, &topk, &rng, NULL, 0);
        counts[t]++;
    }
    CHECK_EQ_INT(counts[0] + counts[3] + counts[4], 0);
    CHECK(counts[1] > 100 && counts[2] > 100);
    tllm_rng a, c;
    tllm_rng_seed(&a, 7);
    tllm_rng_seed(&c, 7);
    CHECK_NEAR(tllm_rng_float(&a), tllm_rng_float(&c), 0.0);
    float f = tllm_rng_float(&a);
    CHECK(f >= 0.0f && f < 1.0f);

    /* ties at the top-k boundary keep all tied tokens; plain temperature sampling */
    tllm_sampler_cfg tie = {0.5f, 1, 1.0f, 0};
    float t2[3] = {1.0f, 1.0f, 0.0f};
    int32_t pick = tllm_sample(t2, 3, &tie, &rng, NULL, 0);
    CHECK(pick == 0 || pick == 1);
    tllm_sampler_cfg all = {1.0f, 0, 1.0f, 0};
    float t3[2] = {0.0f, 50.0f};
    CHECK_EQ_INT(tllm_sample(t3, 2, &all, &rng, NULL, 0), 1);
    tllm_sampler_cfg big_k = {1.0f, 10, 1.0f, 0};
    float t4[2] = {50.0f, 0.0f};
    CHECK_EQ_INT(tllm_sample(t4, 2, &big_k, &rng, NULL, 0), 0);
    /* degenerate: all -inf except none -> falls back to argmax */
    float t5[2] = {-INFINITY, -INFINITY};
    tllm_sampler_cfg k1 = {1.0f, 1, 1.0f, 0};
    CHECK_EQ_INT(tllm_sample(t5, 2, &k1, &rng, NULL, 0), 0);
}
