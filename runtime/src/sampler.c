/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Next-token selection in the order of vision §29: greedy, temperature, top-k, and a
 * recent-token repetition penalty. Deterministic for a given seed. No allocation: top-k
 * uses a selection over the logits array itself.
 */
#include <math.h>

#include "tinyllm/tinyllm.h"

void tllm_rng_seed(tllm_rng *rng, uint64_t seed) { rng->state = seed ? seed : 0x9E3779B97F4A7C15ull; }

static uint64_t next_u64(tllm_rng *rng) { /* xorshift64* */
    uint64_t x = rng->state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng->state = x;
    return x * 0x2545F4914F6CDD1Dull;
}

float tllm_rng_float(tllm_rng *rng) { return (float)(next_u64(rng) >> 40) / 16777216.0f; }

static int32_t argmax(const float *x, uint32_t n) {
    int32_t best = 0;
    for (uint32_t i = 1; i < n; ++i)
        if (x[i] > x[best]) best = (int32_t)i;
    return best;
}

/* k-th largest value (1-based) without modifying x: repeated max below a ceiling. */
static float kth_largest(const float *x, uint32_t n, uint32_t k) {
    float ceiling = INFINITY;
    float value = -INFINITY;
    uint32_t taken = 0;
    while (taken < k) {
        value = -INFINITY;
        uint32_t count = 0;
        for (uint32_t i = 0; i < n; ++i)
            if (x[i] < ceiling && x[i] > value) value = x[i];
        for (uint32_t i = 0; i < n; ++i)
            if (x[i] == value) ++count;
        taken += count;
        ceiling = value;
        if (value == -INFINITY) break;
    }
    return value;
}

int32_t tllm_sample(float *logits, uint32_t n, const tllm_sampler_cfg *cfg, tllm_rng *rng, const int32_t *recent,
                    uint32_t n_recent) {
    if (cfg != NULL && cfg->repeat_penalty > 1.0f && recent != NULL) {
        uint32_t window = cfg->repeat_window < n_recent ? cfg->repeat_window : n_recent;
        for (uint32_t i = n_recent - window; i < n_recent; ++i) {
            int32_t t = recent[i];
            if (t < 0 || (uint32_t)t >= n) continue;
            logits[t] = logits[t] > 0.0f ? logits[t] / cfg->repeat_penalty : logits[t] * cfg->repeat_penalty;
        }
    }
    if (cfg == NULL || cfg->temperature <= 0.0f || rng == NULL) return argmax(logits, n);

    float cutoff = -INFINITY;
    if (cfg->top_k > 0u && cfg->top_k < n) cutoff = kth_largest(logits, n, cfg->top_k);
    float max = logits[argmax(logits, n)];
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        logits[i] = logits[i] >= cutoff ? expf((logits[i] - max) / cfg->temperature) : 0.0f;
        sum += logits[i];
    }
    float r = tllm_rng_float(rng) * sum;
    float acc = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        acc += logits[i];
        if (r < acc) return (int32_t)i;
    }
    return argmax(logits, n); /* numerical edge case: r == sum */
}
