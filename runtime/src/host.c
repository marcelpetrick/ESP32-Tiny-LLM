/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Host convenience layer (see tinyllm/host.h). All allocation happens in
 * tllm_host_open(); the counter lets tests prove the token loop never allocates.
 */
#define _POSIX_C_SOURCE 200809L
#include "tinyllm/host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "tinyllm/console.h"

struct tllm_host {
    void *blob, *hot, *cold;
    tllm_model model;
    tllm_ctx ctx;
    tllm_console con;
    char *output;
    size_t out_len, out_cap;
};

static long g_allocs;

static void *host_alloc(size_t size) {
    ++g_allocs;
    return calloc(1, size);
}

long tllm_host_alloc_count(void) { return g_allocs; }

uint64_t tllm_host_clock_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000u + (uint64_t)ts.tv_nsec / 1000u;
}

/* Output goes into a buffer sized at open; overflow is truncated, never reallocated. */
static void capture(void *user, const char *text, size_t len) {
    tllm_host *h = (tllm_host *)user;
    if (h->out_len + len + 1u > h->out_cap) len = h->out_cap - h->out_len - 1u;
    memcpy(h->output + h->out_len, text, len);
    h->out_len += len;
    h->output[h->out_len] = '\0';
}

static int fail(char *err, size_t cap, const char *msg) {
    if (err != NULL && cap > 0u) snprintf(err, cap, "%s", msg);
    return 0;
}

tllm_host *tllm_host_open(const char *path, int act_int8, int kv_int8, char *err, size_t err_cap) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fail(err, err_cap, "cannot open model file");
        return NULL;
    }
    long size = -1;
    if (fseek(f, 0, SEEK_END) == 0) size = ftell(f);
    if (size <= 0 || fseek(f, 0, SEEK_SET) != 0) {
        (void)fclose(f);
        fail(err, err_cap, "cannot read model file");
        return NULL;
    }
    tllm_host *h = host_alloc(sizeof *h);
    h->blob = host_alloc((size_t)size);
    if (h->blob == NULL || fread(h->blob, 1, (size_t)size, f) != (size_t)size) {
        (void)fclose(f);
        fail(err, err_cap, "cannot read model file");
        tllm_host_close(h);
        return NULL;
    }
    (void)fclose(f);
    tllm_status st = tllm_model_load(&h->model, h->blob, (size_t)size);
    if (st != TLLM_OK) {
        fail(err, err_cap, tllm_status_str(st));
        tllm_host_close(h);
        return NULL;
    }
    size_t hot = tllm_hot_arena_size(&h->model), cold = tllm_cold_arena_size(&h->model, kv_int8);
    h->hot = host_alloc(hot);
    h->cold = host_alloc(cold);
    tllm_ctx_options opt = {act_int8 ? TLLM_ACT_I8 : TLLM_ACT_F32, kv_int8, tllm_host_clock_us, NULL};
    st = tllm_ctx_init(&h->ctx, &h->model, &opt, h->hot, hot, h->cold, cold);
    if (st == TLLM_OK) st = tllm_console_init(&h->con, &h->ctx, capture, h);
    if (st != TLLM_OK) {
        fail(err, err_cap, tllm_status_str(st));
        tllm_host_close(h);
        return NULL;
    }
    h->out_cap = 64u * 1024u;
    h->output = host_alloc(h->out_cap);
    return h;
}

void tllm_host_close(tllm_host *h) {
    if (h == NULL) return;
    free(h->blob);
    free(h->hot);
    free(h->cold);
    free(h->output);
    free(h);
}

const char *tllm_host_submit(tllm_host *h, const char *line) {
    h->out_len = 0;
    h->output[0] = '\0';
    tllm_console_line(&h->con, line);
    return h->output;
}

const char *tllm_host_last_json(const tllm_host *h) { return tllm_console_last_json(&h->con); }

int tllm_host_vocab_size(const tllm_host *h) { return (int)h->model.cfg.vocab_size; }

int tllm_host_tokenize(tllm_host *h, const char *text, int32_t *out, int max_out) {
    return tllm_tokenize(&h->model.tok, text, strlen(text), out, max_out);
}

int tllm_host_token_bytes(tllm_host *h, int32_t id, char *out, int max_out) {
    return tllm_token_bytes(&h->model.tok, id, out, max_out);
}

int tllm_host_logits(tllm_host *h, const int32_t *tokens, int n, float *out) {
    const float *logits = NULL;
    tllm_ctx_reset(&h->ctx);
    if (n <= 0 || tllm_prefill(&h->ctx, tokens, (uint32_t)n, &logits) != TLLM_OK || logits == NULL) return -1;
    memcpy(out, logits, sizeof(float) * h->model.cfg.vocab_size);
    tllm_ctx_reset(&h->ctx);
    return 0;
}

int tllm_host_generate(tllm_host *h, const int32_t *prompt, int n, int max_new, int32_t *out) {
    const float *logits = NULL;
    tllm_ctx_reset(&h->ctx);
    if (n <= 0 || max_new < 0 || tllm_prefill(&h->ctx, prompt, (uint32_t)n, &logits) != TLLM_OK) return -1;
    const uint32_t vocab = h->model.cfg.vocab_size;
    int produced = 0;
    while (produced < max_new) {
        int32_t tok = tllm_sample(h->ctx.logits, vocab, NULL, NULL, NULL, 0);
        out[produced++] = tok;
        if (tok == h->model.tok.eos_id || tllm_forward(&h->ctx, tok, &logits) != TLLM_OK) break;
    }
    tllm_ctx_reset(&h->ctx);
    return produced;
}

int tllm_host_device_eval(const char *settings, const char *action, char *out, size_t cap) {
    tllm_device_state s;
    tllm_device_default(&s);
    char buf[256];
    snprintf(buf, sizeof buf, "%s", settings);
    for (const char *tok = strtok(buf, " "); tok != NULL; tok = strtok(NULL, " "))
        if (tllm_device_set(&s, tok) != 0) return -1;
    tllm_action a;
    tllm_verdict v = TLLM_VERDICT_MALFORMED;
    if (tllm_action_parse(action, strlen(action), &a) == 0) {
        v = tllm_device_validate(&s, &a);
        if (v == TLLM_VERDICT_OK) tllm_device_apply(&s, &a);
    }
    char state[TLLM_STATE_TEXT_MAX];
    tllm_device_render(&s, state, sizeof state);
    int n = snprintf(out, cap, "%s|%s", tllm_verdict_code(v), state);
    return (n < 0 || (size_t)n >= cap) ? -1 : 0;
}
