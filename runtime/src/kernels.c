/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Reference scalar kernels. These are the differential-testing baseline; optimised
 * ESP32-S3 versions (ESP-DSP / ESP-NN / PIE) must match them (vision §18 O4, §29).
 */
#include <math.h>

#include "tinyllm/tinyllm.h"

void tllm_rmsnorm(float *out, const float *x, const float *weight, uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; ++i) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; ++i) out[i] = x[i] * inv * weight[i];
}

void tllm_softmax(float *x, uint32_t n) {
    float max = x[0];
    for (uint32_t i = 1; i < n; ++i)
        if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    float inv = 1.0f / sum;
    for (uint32_t i = 0; i < n; ++i) x[i] *= inv;
}

void tllm_matvec(float *out, const tllm_tensor *w, const float *x) {
    const uint32_t rows = w->rows, cols = w->cols;
    if (w->dtype == TLLM_DTYPE_F32) {
        const float *W = (const float *)w->data;
        for (uint32_t r = 0; r < rows; ++r) {
            const float *row = W + (size_t)r * cols;
            float acc = 0.0f;
            for (uint32_t c = 0; c < cols; ++c) acc += row[c] * x[c];
            out[r] = acc;
        }
        return;
    }
    const int8_t *Q = (const int8_t *)w->data;
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *row = Q + (size_t)r * cols;
        float acc = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) acc += (float)row[c] * x[c];
        out[r] = acc * w->scales[r];
    }
}

void tllm_matvec_q8(float *out, const tllm_tensor *w, const int8_t *qx, float x_scale) {
    const uint32_t rows = w->rows, cols = w->cols;
    const int8_t *Q = (const int8_t *)w->data;
    for (uint32_t r = 0; r < rows; ++r) {
        const int8_t *row = Q + (size_t)r * cols;
        int32_t acc = 0;
        for (uint32_t c = 0; c < cols; ++c) acc += (int32_t)row[c] * (int32_t)qx[c];
        out[r] = (float)acc * w->scales[r] * x_scale;
    }
}

float tllm_quantize_vec(int8_t *out, const float *x, uint32_t n) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
    float inv = 1.0f / scale;
    for (uint32_t i = 0; i < n; ++i) {
        long q = lrintf(x[i] * inv);
        if (q > 127) q = 127;
        if (q < -127) q = -127;
        out[i] = (int8_t)q;
    }
    return scale;
}

float tllm_gelu(float x) {
    const float k = 0.7978845608028654f; /* sqrt(2 / pi) */
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

float tllm_silu(float x) { return x / (1.0f + expf(-x)); }

uint32_t tllm_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int b = 0; b < 8; ++b) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
    }
    return ~crc;
}
