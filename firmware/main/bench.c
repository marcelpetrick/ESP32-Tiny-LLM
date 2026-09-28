/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Memory-bandwidth and GEMV microbenchmarks. Results feed the roofline model of
 * docs/01-feasibility.md with *measured* numbers instead of estimates.
 */
#include "bench.h"

#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_timer.h"

static const uint8_t *s_blob, *s_flash;
static size_t s_size;
static const tllm_model *s_model;

void bench_set_model(const uint8_t *blob, size_t size, const uint8_t *flash_blob, const tllm_model *model) {
    s_blob = blob;
    s_size = size;
    s_flash = flash_blob;
    s_model = model;
}

/* Sequential 32-bit reads; returns MB/s (volatile sink defeats the optimiser). */
static double read_mbs(const uint8_t *p, size_t bytes, int rounds) {
    volatile uint32_t sink = 0;
    const uint32_t *w = (const uint32_t *)(const void *)p;
    size_t n = bytes / 4u;
    int64_t t0 = esp_timer_get_time();
    for (int r = 0; r < rounds; ++r) {
        uint32_t acc = 0;
        for (size_t i = 0; i < n; ++i) acc += w[i];
        sink += acc;
    }
    int64_t us = esp_timer_get_time() - t0;
    (void)sink;
    return us > 0 ? (double)bytes * rounds / (double)us : 0.0;
}

static void bandwidth(tllm_console *con) {
    char buf[256];
    size_t internal_bytes = 64u * 1024u, psram_bytes = 1024u * 1024u;
    uint8_t *internal = heap_caps_malloc(internal_bytes, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    uint8_t *psram = heap_caps_malloc(psram_bytes, MALLOC_CAP_SPIRAM);
    double sram = 0.0, ext = 0.0, flash = 0.0;
    if (internal != NULL) {
        memset(internal, 1, internal_bytes);
        sram = read_mbs(internal, internal_bytes, 16);
    }
    if (psram != NULL) {
        memset(psram, 1, psram_bytes);
        ext = read_mbs(psram, psram_bytes, 4); /* 1 MiB >> 64 KiB cache: streaming */
    }
    size_t flash_bytes = s_size < psram_bytes ? s_size & ~(size_t)3u : psram_bytes;
    flash = read_mbs(s_flash, flash_bytes, 2);
    heap_caps_free(internal);
    heap_caps_free(psram);
    snprintf(buf, sizeof buf, "read bandwidth: internal SRAM %.1f MB/s, PSRAM %.1f MB/s, flash XIP %.1f MB/s\n", sram,
             ext, flash);
    tllm_console_write(con, buf);
    snprintf(buf, sizeof buf, "@@{\"event\":\"bandwidth\",\"sram_mbs\":%.1f,\"psram_mbs\":%.1f,\"flash_mbs\":%.1f}\n",
             sram, ext, flash);
    tllm_console_write(con, buf);
}

static void gemv(tllm_console *con) {
    char buf[256];
    const tllm_tensor *w = &s_model->layers[0].w1;
    static float x[TLLM_MAX_DIM], y[4 * TLLM_MAX_DIM];
    static int8_t qx[TLLM_MAX_DIM];
    for (uint32_t i = 0; i < w->cols; ++i) x[i] = (float)(i % 7) * 0.1f;
    const int reps = 50;
    int64_t t0 = esp_timer_get_time();
    for (int r = 0; r < reps; ++r) tllm_matvec(y, w, x);
    int64_t us_a32 = esp_timer_get_time() - t0;
    int64_t us_a8 = 0;
    if (w->dtype == TLLM_DTYPE_I8) {
        float s = tllm_quantize_vec(qx, x, w->cols);
        t0 = esp_timer_get_time();
        for (int r = 0; r < reps; ++r) tllm_matvec_q8(y, w, qx, s);
        us_a8 = esp_timer_get_time() - t0;
    }
    double macs = (double)w->rows * w->cols * reps;
    snprintf(buf, sizeof buf, "gemv %ux%u (%s): %.1f MMAC/s float activations, %.1f MMAC/s int8 activations\n",
             (unsigned)w->rows, (unsigned)w->cols, w->dtype == TLLM_DTYPE_I8 ? "int8" : "f32",
             us_a32 ? macs / (double)us_a32 : 0.0, us_a8 ? macs / (double)us_a8 : 0.0);
    tllm_console_write(con, buf);
    snprintf(buf, sizeof buf, "@@{\"event\":\"gemv\",\"rows\":%u,\"cols\":%u,\"mmacs_a32\":%.1f,\"mmacs_a8\":%.1f}\n",
             (unsigned)w->rows, (unsigned)w->cols, us_a32 ? macs / (double)us_a32 : 0.0,
             us_a8 ? macs / (double)us_a8 : 0.0);
    tllm_console_write(con, buf);
}

int bench_command(tllm_console *con, const char *line) {
    if (strcmp(line, "/bandwidth") == 0) {
        bandwidth(con);
        return 1;
    }
    if (strcmp(line, "/gemv") == 0) {
        gemv(con);
        return 1;
    }
    return 0;
}
