/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include <stdlib.h>

#include "builder.h"
#include "test.h"
#include "tinyllm/host.h"

void test_host(void);

void test_host(void) {
    char err[64];
    CHECK(tllm_host_open("/nonexistent/model.tllm", 0, 0, err, sizeof err) == NULL);
    CHECK_STR(err, "cannot open model file");

    test_model_spec spec = test_default_spec();
    spec.dtype = 1;
    spec.ctx_len = 256; /* no merges for the state block: it needs room */
    test_blob b = test_build_model(&spec);
    const char *path = test_write_temp(&b, "tinyllm_host_test.tllm");
    CHECK(path != NULL);
    tllm_host *h = tllm_host_open(path, 1, 1, err, sizeof err);
    CHECK(h != NULL);
    long allocs = tllm_host_alloc_count();
    CHECK_EQ_INT(tllm_host_vocab_size(h), TEST_N_SPECIAL + 256 + 8);
    int32_t ids[32];
    int n = tllm_host_tokenize(h, "the fan", ids, 32);
    CHECK_EQ_INT(n, 3); /* "the", " ", "fan" */
    char buf[16];
    CHECK_EQ_INT(tllm_host_token_bytes(h, ids[2], buf, 16), 3);
    float *logits = malloc(sizeof(float) * (size_t)tllm_host_vocab_size(h));
    CHECK_EQ_INT(tllm_host_logits(h, ids, n, logits), 0);
    CHECK_EQ_INT(tllm_host_logits(h, ids, 0, logits), -1);
    const char *out = tllm_host_submit(h, "/model-info");
    CHECK_CONTAINS(out, "int8 weights");
    CHECK_CONTAINS(tllm_host_last_json(h), "\"activations\":\"int8\",\"kv_cache\":\"int8\"");
    out = tllm_host_submit(h, "hello");
    CHECK_CONTAINS(out, "@@{\"event\":\"reply\"");
    CHECK_EQ_INT(tllm_host_alloc_count(), allocs); /* no allocation while chatting */
    CHECK(tllm_host_clock_us() > 0u);
    tllm_host_close(h);
    tllm_host_close(NULL);
    free(logits);

    /* corrupted file: load error is reported */
    b.data[b.size - 1] ^= 0xFFu;
    path = test_write_temp(&b, "tinyllm_host_bad.tllm");
    CHECK(tllm_host_open(path, 0, 0, err, sizeof err) == NULL);
    CHECK_STR(err, "crc mismatch");
    /* empty file */
    test_blob empty = b;
    empty.size = 0;
    path = test_write_temp(&empty, "tinyllm_host_empty.tllm");
    CHECK(tllm_host_open(path, 0, 0, NULL, 0) == NULL);
    /* no chat specials */
    test_free_blob(&b);
    spec.omit_chat_specials = 1;
    b = test_build_model(&spec);
    path = test_write_temp(&b, "tinyllm_host_nochat.tllm");
    CHECK(tllm_host_open(path, 0, 0, err, sizeof err) == NULL);
    CHECK_STR(err, "malformed tokenizer");
    test_free_blob(&b);

    /* device parity helper */
    char res[160];
    CHECK_EQ_INT(tllm_host_device_eval("t=36", "heat=1", res, sizeof res), 0);
    CHECK_CONTAINS(res, "overheat|<S> t=36.0");
    CHECK_EQ_INT(tllm_host_device_eval("", "fan=2", res, sizeof res), 0);
    CHECK_CONTAINS(res, "ok|<S> t=22.0 h=55 soil=50 fan=2");
    CHECK_EQ_INT(tllm_host_device_eval("", "fan=7", res, sizeof res), 0);
    CHECK_CONTAINS(res, "malformed|");
    CHECK_EQ_INT(tllm_host_device_eval("bogus=1", "fan=1", res, sizeof res), -1);
    CHECK_EQ_INT(tllm_host_device_eval("", "fan=1", res, 8), -1);
}
