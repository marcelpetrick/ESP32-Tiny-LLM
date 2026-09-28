/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include "test.h"
#include "tinyllm/tinyllm.h"

void test_kernels(void);

void test_kernels(void) {
    /* rmsnorm: x / rms(x) * w */
    float x[4] = {1.0f, -2.0f, 3.0f, -4.0f}, w[4] = {1.0f, 1.0f, 2.0f, 0.5f}, out[4];
    tllm_rmsnorm(out, x, w, 4, 0.0f);
    float rms = sqrtf((1.0f + 4.0f + 9.0f + 16.0f) / 4.0f);
    CHECK_NEAR(out[0], 1.0f / rms, 1e-6);
    CHECK_NEAR(out[2], 6.0f / rms, 1e-6);
    CHECK_NEAR(out[3], -2.0f / rms, 1e-6);

    /* softmax sums to one and is shift-invariant */
    float s[3] = {1000.0f, 1001.0f, 1002.0f};
    tllm_softmax(s, 3);
    CHECK_NEAR(s[0] + s[1] + s[2], 1.0, 1e-6);
    CHECK_NEAR(s[2], 0.66524096, 1e-5);

    /* f32 and int8 GEMV */
    float W[6] = {1, 2, 3, -1, 0, 1}, v[3] = {1, 1, 2}, y[2];
    tllm_tensor tf = {W, NULL, TLLM_DTYPE_F32, 2, 3};
    tllm_matvec(y, &tf, v);
    CHECK_NEAR(y[0], 9.0, 1e-6);
    CHECK_NEAR(y[1], 1.0, 1e-6);
    int8_t Q[6] = {127, 0, -127, 64, 64, 64};
    float scales[2] = {0.5f, 0.25f};
    tllm_tensor ti = {Q, scales, TLLM_DTYPE_I8, 2, 3};
    tllm_matvec(y, &ti, v);
    CHECK_NEAR(y[0], (127.0 - 254.0) * 0.5, 1e-4);
    CHECK_NEAR(y[1], 256.0 * 0.25, 1e-4);

    /* activation quantisation and W8A8 */
    int8_t qx[3];
    float sx = tllm_quantize_vec(qx, v, 3);
    CHECK_NEAR(sx, 2.0 / 127.0, 1e-7);
    CHECK_EQ_INT(qx[2], 127);
    CHECK_EQ_INT(qx[0], 64);
    tllm_matvec_q8(y, &ti, qx, sx);
    CHECK_NEAR(y[1], 64.0, 0.6);
    float zeros[2] = {0.0f, 0.0f};
    CHECK_NEAR(tllm_quantize_vec(qx, zeros, 2), 1.0, 0.0);
    CHECK_EQ_INT(qx[0], 0);
    float big[2] = {1000.0f, -1000.0f};
    tllm_quantize_vec(qx, big, 2);
    CHECK_EQ_INT(qx[1], -127);

    /* activations */
    CHECK_NEAR(tllm_gelu(0.0f), 0.0, 1e-7);
    CHECK_NEAR(tllm_gelu(1.0f), 0.841192, 1e-5);
    CHECK_NEAR(tllm_silu(0.0f), 0.0, 1e-7);
    CHECK_NEAR(tllm_silu(2.0f), 1.761594, 1e-5);

    /* CRC-32 check value */
    CHECK_EQ_INT(tllm_crc32((const uint8_t *)"123456789", 9), 0xCBF43926u);
    CHECK_EQ_INT(tllm_crc32((const uint8_t *)"", 0), 0u);
}
