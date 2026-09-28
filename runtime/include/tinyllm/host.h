/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Desktop/host convenience layer: loads a .tllm file, allocates the arenas once, and
 * exposes an opaque handle API that is easy to call from Python (ctypes) — used by the
 * CLI, the web simulator, and the Python <-> C integration tests. Not used on the ESP32.
 */
#ifndef TINYLLM_HOST_H
#define TINYLLM_HOST_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tllm_host tllm_host;

/* act_int8: W8A8 for int8 models; kv_int8: int8 KV cache. NULL on failure (reason in err). */
tllm_host *tllm_host_open(const char *path, int act_int8, int kv_int8, char *err, size_t err_cap);
void tllm_host_close(tllm_host *h);

/* Run one console line; returns everything the console wrote (valid until the next call). */
const char *tllm_host_submit(tllm_host *h, const char *line);
/* JSON payload of the last "@@" line. */
const char *tllm_host_last_json(const tllm_host *h);

/* Parity helpers for tests. */
int tllm_host_vocab_size(const tllm_host *h);
int tllm_host_tokenize(tllm_host *h, const char *text, int32_t *out, int max_out);
int tllm_host_token_bytes(tllm_host *h, int32_t id, char *out, int max_out);
/* Fresh KV cache, run all tokens, copy the last logits into out (vocab_size floats). */
int tllm_host_logits(tllm_host *h, const int32_t *tokens, int n, float *out);
/* Number of heap allocations performed by the runtime since open (must stay constant). */
long tllm_host_alloc_count(void);

/* Device parity: start from the default state, apply space-separated "key=value" settings,
 * then parse+validate+apply `action`. Writes "<verdict>|<rendered state>" into out. */
int tllm_host_device_eval(const char *settings, const char *action, char *out, size_t cap);

uint64_t tllm_host_clock_us(void);

#ifdef __cplusplus
}
#endif
#endif /* TINYLLM_HOST_H */
