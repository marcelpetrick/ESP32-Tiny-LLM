/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include <stdlib.h>

#include "builder.h"
#include "test.h"
#include "tinyllm/tinyllm.h"

void test_loader(void);

static void put32(uint8_t *p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t)(v >> (8 * i));
}

static int g_scored;

/* Apply `edit` to a fresh blob and return the load status. */
static tllm_status load_edited(void (*edit)(test_blob *), int fix_crc) {
    test_model_spec spec = test_default_spec();
    spec.scored = g_scored;
    test_blob b = test_build_model(&spec);
    edit(&b);
    if (fix_crc) test_fix_crc(&b);
    tllm_model m;
    tllm_status st = tllm_model_load(&m, b.data, b.size);
    test_free_blob(&b);
    return st;
}

static void e_magic(test_blob *b) { b->data[0] = 'X'; }
static void e_version(test_blob *b) { put32(b->data + 4, 2); }
static void e_arch(test_blob *b) { put32(b->data + 12, 7); }
static void e_payload(test_blob *b) { put32(b->data + 76, 5); }
static void e_crc(test_blob *b) { b->data[b->size - 1] ^= 0x55u; }
static void e_layers(test_blob *b) { put32(b->data + 24, TLLM_MAX_LAYERS + 1u); }
static void e_heads(test_blob *b) { put32(b->data + 32, 3); }
static void e_kvheads(test_blob *b) { put32(b->data + 36, 3); }
static void e_mlp(test_blob *b) { put32(b->data + 44, 9); }
static void e_eps(test_blob *b) { put32(b->data + 100, 0); }
static void e_tok_off(test_blob *b) { put32(b->data + 60, 0xFFFFFF00u); }
static void e_table(test_blob *b) { put32(b->data + 72, 0x0FFFFFFFu); }
static void e_vocab(test_blob *b) { put32(b->data + 16, 999); }
static void e_tok_magic(test_blob *b) { b->data[b->tokenizer_offset] = 'X'; }
static void e_tok_version(test_blob *b) { put32(b->data + b->tokenizer_offset + 4, 3); }
static void e_tok_specials(test_blob *b) { put32(b->data + b->tokenizer_offset + 8, 0); }
static void e_tok_merges(test_blob *b) { put32(b->data + b->tokenizer_offset + 12, 100000); }
static void e_tok_special_len(test_blob *b) { b->data[b->tokenizer_offset + 16] = 0; }
static void e_tok_merge_order(test_blob *b) {
    /* first merge refers to a not-yet-existing token */
    size_t pos = b->tokenizer_offset + 16;
    for (int i = 0; i < TEST_N_SPECIAL; ++i) pos += 1u + b->data[pos];
    b->data[pos] = 0xFF;
    b->data[pos + 1] = 0x7F;
}
static void e_tensor_name(test_blob *b) { b->data[b->tensor_table_offset] = 'X'; }
static void e_tensor_dtype(test_blob *b) { put32(b->data + b->tensor_table_offset + 32, 5); }
static void e_tensor_rows(test_blob *b) { put32(b->data + b->tensor_table_offset + 40, 3); }
static void e_tensor_bytes(test_blob *b) { put32(b->data + b->tensor_table_offset + 60, 3); }
static void e_tensor_align(test_blob *b) { put32(b->data + b->tensor_table_offset + 56, 130); }
static void e_tensor_ndim(test_blob *b) { put32(b->data + b->tensor_table_offset + 36, 3); }
static void e_norm_ndim(test_blob *b) {
    /* entry 2 is l0.attn_norm (tok_emb, pos_emb, l0.attn_norm): claim it is 2-D */
    put32(b->data + b->tensor_table_offset + 2 * 68 + 36, 2);
}

static void e_v2_vocab(test_blob *b) { put32(b->data + b->tokenizer_offset + 12, 0); }
static void e_v2_maxlen(test_blob *b) { put32(b->data + b->tokenizer_offset + 16, TLLM_MAX_PIECE + 1u); }
static void e_v2_specials(test_blob *b) { put32(b->data + b->tokenizer_offset + 8, TLLM_MAX_SPECIAL + 1u); }
static void e_v2_piece_len(test_blob *b) { b->data[b->tokenizer_offset + 20 + 4] = 60; } /* > max_piece_len 6 */
static void e_v2_truncated(test_blob *b) { put32(b->data + b->tokenizer_offset + 12, 60000); }

static void e_q4_ndim(test_blob *b) { put32(b->data + b->tensor_table_offset + 32, 2); } /* tok_emb cols 16 */
static void e_q4_scale(test_blob *b) {
    put32(b->data + b->tensor_table_offset + 32, 2);
    put32(b->data + b->tensor_table_offset + 60, TEST_N_SPECIAL * 0u + 8u); /* wrong byte count */
}

void test_loader(void) {
    test_model_spec spec = test_default_spec();
    test_blob b = test_build_model(&spec);
    tllm_model m;
    CHECK_EQ_INT(tllm_model_load(&m, b.data, b.size), TLLM_OK);
    CHECK_EQ_INT(m.cfg.n_layers, 2);
    CHECK_EQ_INT(m.cfg.vocab_size, TEST_N_SPECIAL + 256 + 8);
    CHECK_EQ_INT(m.tok.n_merges, 8);
    CHECK_EQ_INT(m.layers[1].wk.rows, 8);
    CHECK_NEAR(m.cfg.norm_eps, 1e-5, 1e-12);
    CHECK_EQ_INT(m.model_id[0], 0xA0);
    CHECK_EQ_INT(tllm_model_load(NULL, b.data, b.size), TLLM_ERR_ARG);
    CHECK_EQ_INT(tllm_model_load(&m, NULL, b.size), TLLM_ERR_ARG);
    CHECK_EQ_INT(tllm_model_load(&m, b.data, 10), TLLM_ERR_TRUNCATED);
    test_free_blob(&b);

    spec.dtype = 1;
    spec.mlp_type = 1;
    spec.pos_type = 1;
    b = test_build_model(&spec);
    CHECK_EQ_INT(tllm_model_load(&m, b.data, b.size), TLLM_OK);
    CHECK(m.layers[0].w3.data != NULL);
    CHECK(m.layers[0].wq.scales != NULL);
    test_free_blob(&b);

    CHECK_EQ_INT(load_edited(e_magic, 1), TLLM_ERR_MAGIC);
    CHECK_EQ_INT(load_edited(e_version, 1), TLLM_ERR_VERSION);
    CHECK_EQ_INT(load_edited(e_arch, 1), TLLM_ERR_VERSION);
    CHECK_EQ_INT(load_edited(e_payload, 1), TLLM_ERR_TRUNCATED);
    CHECK_EQ_INT(load_edited(e_crc, 0), TLLM_ERR_CRC);
    CHECK_EQ_INT(load_edited(e_layers, 1), TLLM_ERR_UNSUPPORTED);
    CHECK_EQ_INT(load_edited(e_heads, 1), TLLM_ERR_UNSUPPORTED);
    CHECK_EQ_INT(load_edited(e_kvheads, 1), TLLM_ERR_UNSUPPORTED);
    CHECK_EQ_INT(load_edited(e_mlp, 1), TLLM_ERR_UNSUPPORTED);
    CHECK_EQ_INT(load_edited(e_eps, 1), TLLM_ERR_UNSUPPORTED);
    CHECK_EQ_INT(load_edited(e_tok_off, 1), TLLM_ERR_TRUNCATED);
    CHECK_EQ_INT(load_edited(e_table, 1), TLLM_ERR_TRUNCATED);
    CHECK_EQ_INT(load_edited(e_vocab, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_tok_magic, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_tok_version, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_tok_specials, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_tok_merges, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_tok_special_len, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_tok_merge_order, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_tensor_name, 1), TLLM_ERR_TENSOR);
    CHECK_EQ_INT(load_edited(e_tensor_dtype, 1), TLLM_ERR_TENSOR);
    CHECK_EQ_INT(load_edited(e_tensor_rows, 1), TLLM_ERR_TENSOR);
    CHECK_EQ_INT(load_edited(e_tensor_bytes, 1), TLLM_ERR_TENSOR);
    CHECK_EQ_INT(load_edited(e_tensor_align, 1), TLLM_ERR_TENSOR);
    CHECK_EQ_INT(load_edited(e_tensor_ndim, 1), TLLM_ERR_TENSOR);
    CHECK_EQ_INT(load_edited(e_norm_ndim, 1), TLLM_ERR_TENSOR);

    CHECK_EQ_INT(load_edited(e_q4_ndim, 1), TLLM_ERR_TENSOR);
    CHECK_EQ_INT(load_edited(e_q4_scale, 1), TLLM_ERR_TENSOR);
    g_scored = 1;
    CHECK_EQ_INT(load_edited(e_v2_vocab, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_v2_maxlen, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_v2_specials, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_v2_piece_len, 1), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(load_edited(e_v2_truncated, 1), TLLM_ERR_TOKENIZER);
    g_scored = 0;

    for (int s = 0; s <= TLLM_ERR_CONTEXT_FULL; ++s) CHECK(tllm_status_str((tllm_status)s)[0] != '\0');
    CHECK_STR(tllm_status_str((tllm_status)99), "unknown");
}
