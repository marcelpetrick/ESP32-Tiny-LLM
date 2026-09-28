/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Synthetic .tllm model builder for unit tests (writes the same format as
 * training/export.py). Weights are pseudo-random with a fixed seed.
 */
#ifndef TINYLLM_TEST_BUILDER_H
#define TINYLLM_TEST_BUILDER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t n_layers, d_model, n_heads, n_kv_heads, d_ff, ctx_len;
    uint32_t mlp_type, pos_type, dtype;
    int n_merges;           /* merges from TEST_MERGES to include (max 8) */
    int zero_blocks;        /* 1: attention/MLP outputs are zero (embedding-only model) */
    int omit_chat_specials; /* 1: only <pad> <bos> <eos> specials */
    int scored;             /* 1: llama2.c-style scored tokenizer (blob v2, 3 specials, 8 pieces) */
    uint32_t seed;
} test_model_spec;

typedef struct {
    uint8_t *data; /* malloc'd, 16-byte aligned */
    size_t size;
    size_t pos_emb_offset; /* byte offset of pos_emb data (0 if rope) */
    size_t tok_emb_offset;
    size_t header_crc_offset;
    size_t tensor_table_offset;
    size_t tokenizer_offset;
} test_blob;

test_model_spec test_default_spec(void);
test_blob test_build_model(const test_model_spec *spec);
void test_free_blob(test_blob *b);
/* Recompute the payload CRC after editing the blob. */
void test_fix_crc(test_blob *b);
/* Write the blob to a temporary file; returns the path (static buffer). */
const char *test_write_temp(const test_blob *b, const char *name);

/* Number of specials the builder emits and their layout: <pad> <bos> <eos> <S> </S> <U> </U> <A> </A>
 * <ACT> </ACT> <clarify> <unsupported> (13) unless omit_chat_specials. */
#define TEST_N_SPECIAL 13

#endif /* TINYLLM_TEST_BUILDER_H */
