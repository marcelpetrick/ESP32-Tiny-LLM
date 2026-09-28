/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include <stdlib.h>

#include "builder.h"
#include "test.h"
#include "tinyllm/tinyllm.h"

void test_tokenizer(void);

#define S TEST_N_SPECIAL
#define M(r) (S + 256 + (r))

static int encode(const tllm_tokenizer *t, const char *text, int32_t *out, int max) {
    return tllm_tokenize(t, text, strlen(text), out, max);
}

static void roundtrip(const tllm_tokenizer *t, const char *text) {
    int32_t ids[128];
    char buf[512];
    int n = encode(t, text, ids, 128);
    CHECK(n >= 0);
    size_t len = 0;
    for (int i = 0; i < n; ++i) {
        int k = tllm_token_bytes(t, ids[i], buf + len, (int)(sizeof buf - len));
        CHECK(k > 0);
        len += (size_t)k;
    }
    buf[len] = '\0';
    CHECK_STR(buf, text);
}

void test_tokenizer(void) {
    test_model_spec spec = test_default_spec();
    test_blob b = test_build_model(&spec);
    tllm_model m;
    CHECK_EQ_INT(tllm_model_load(&m, b.data, b.size), TLLM_OK);
    size_t ws_size = tllm_tokenizer_workspace_size(&m.tok);
    CHECK_EQ_INT(ws_size, 16 * 6); /* 16 hash entries: u32 key + u16 rank */
    void *ws = malloc(ws_size);
    int32_t ids[64];
    CHECK_EQ_INT(encode(&m.tok, "x", ids, 64), -1); /* no workspace yet */
    tllm_tokenizer_attach(&m.tok, ws);
    CHECK_EQ_INT(m.tok.kind, TLLM_TOK_BPE);
    CHECK_EQ_INT(m.tok.bos_id, 1);
    CHECK_EQ_INT(m.tok.eos_id, 2);

    /* merges: th, the, " t", fa, fan, an, " the", "fan=" */
    CHECK_EQ_INT(encode(&m.tok, "the", ids, 64), 1);
    CHECK_EQ_INT(ids[0], M(1));
    CHECK_EQ_INT(encode(&m.tok, " the", ids, 64), 1);
    CHECK_EQ_INT(ids[0], M(6));
    CHECK_EQ_INT(encode(&m.tok, "fan", ids, 64), 1); /* rank 3 (fa) beats rank 5 (an) */
    CHECK_EQ_INT(ids[0], M(4));
    CHECK_EQ_INT(encode(&m.tok, "fan=2", ids, 64), 2); /* "fan=" is one pre-token */
    CHECK_EQ_INT(ids[0], M(7));
    CHECK_EQ_INT(ids[1], S + '2');
    CHECK_EQ_INT(encode(&m.tok, "an", ids, 64), 1);
    CHECK_EQ_INT(ids[0], M(5));
    /* digits are single pre-tokens: "12" -> '1','2' */
    CHECK_EQ_INT(encode(&m.tok, "12", ids, 64), 2);
    /* space runs keep their last space for the next word: "a  the" -> a, ' ', ' the' */
    CHECK_EQ_INT(encode(&m.tok, "a  the", ids, 64), 3);
    CHECK_EQ_INT(ids[1], S + ' ');
    CHECK_EQ_INT(ids[2], M(6));
    /* trailing spaces form one pre-token (no merges on spaces) */
    CHECK_EQ_INT(encode(&m.tok, "a   ", ids, 64), 4);
    /* special strings are plain bytes here */
    CHECK_EQ_INT(encode(&m.tok, "<U>", ids, 64), 3);
    CHECK_EQ_INT(encode(&m.tok, "", ids, 64), 0);

    roundtrip(&m.tok, "the fan=3 is on, t=21.5!");
    roundtrip(&m.tok, "  leading and trailing  ");
    roundtrip(&m.tok, "caf\xc3\xa9 \xe2\x98\x83");

    /* capacity errors */
    CHECK_EQ_INT(encode(&m.tok, "abc", ids, 2), -1);
    CHECK_EQ_INT(encode(&m.tok, "a   b", ids, 1), -1);
    CHECK_EQ_INT(tllm_tokenize(NULL, "a", 1, ids, 4), -1);
    CHECK_EQ_INT(tllm_tokenize(&m.tok, NULL, 1, ids, 4), -1);

    CHECK_EQ_INT(tllm_special_id(&m.tok, "<U>"), 5);
    CHECK_EQ_INT(tllm_special_id(&m.tok, "<eos>"), 2);
    CHECK_EQ_INT(tllm_special_id(&m.tok, "<nope>"), -1);
    char buf[16];
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 5, buf, 16), 3);
    CHECK(memcmp(buf, "<U>", 3) == 0);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 5, buf, 2), -1);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, S + 'q', buf, 0), -1);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, M(6), buf, 3), -1);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, M(6), buf, 1), -1);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, -1, buf, 16), -1);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 100000, buf, 16), -1);
    free(ws);
    test_free_blob(&b);

    /* a larger merge table gets a larger hash table */
    tllm_tokenizer big = m.tok;
    big.n_merges = 100;
    CHECK_EQ_INT(tllm_tokenizer_workspace_size(&big), 256 * 6);

    /* llama2.c-style scored pieces */
    spec.scored = 1;
    b = test_build_model(&spec);
    CHECK_EQ_INT(tllm_model_load(&m, b.data, b.size), TLLM_OK);
    CHECK_EQ_INT(m.tok.kind, TLLM_TOK_SCORED);
    CHECK_EQ_INT(m.tok.bos_id, 1);
    CHECK_EQ_INT(m.tok.eos_id, 2);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 263, buf, 16), -1); /* no workspace yet */
    ws = malloc(tllm_tokenizer_workspace_size(&m.tok));
    tllm_tokenizer_attach(&m.tok, ws);
    CHECK_EQ_INT(encode(&m.tok, "ab c", ids, 64), 3); /* " " a b " " c -> " ab" " " c */
    CHECK_EQ_INT(ids[0], 264);
    CHECK_EQ_INT(ids[1], 259);
    CHECK_EQ_INT(ids[2], 262);
    CHECK_EQ_INT(encode(&m.tok, "abc", ids, 64), 2); /* "abc" (4) beats " ab" (3) */
    CHECK_EQ_INT(ids[1], 265);
    CHECK_EQ_INT(encode(&m.tok, "\xc3\xa9", ids, 64), 2); /* multi-byte codepoint piece */
    CHECK_EQ_INT(ids[1], 266);
    CHECK_EQ_INT(encode(&m.tok, "x\xff", ids, 64), 3); /* byte fallback: byte + 3 */
    CHECK_EQ_INT(ids[1], 'x' + 3);
    CHECK_EQ_INT(ids[2], 0xFF + 3);
    CHECK_EQ_INT(encode(&m.tok, "", ids, 64), 0);
    CHECK_EQ_INT(encode(&m.tok, "abc", ids, 1), -1);
    CHECK_EQ_INT(encode(&m.tok, "abc", ids, 0), -1);
    CHECK_EQ_INT(encode(&m.tok, "x\xff", ids, 2), -1);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 3 + 0x41, buf, 16), 1); /* "<0x41>" -> 'A' */
    CHECK_EQ_INT(buf[0], 'A');
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 3 + 0x0A, buf, 16), 1); /* lowercase hex */
    CHECK_EQ_INT(buf[0], '\n');
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 3, buf, 0), -1);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 265, buf, 16), 3);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 265, buf, 2), -1);
    CHECK_EQ_INT(tllm_token_bytes(&m.tok, 1, buf, 16), 5); /* "\n<s>\n" */
    free(ws);
    test_free_blob(&b);
}
