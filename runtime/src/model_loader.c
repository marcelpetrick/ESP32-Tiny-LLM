/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * .tllm v1 loader — parses and verifies the header, tokenizer block and tensor table
 * (docs/05-architecture.md §3). Tensors are referenced in place; nothing is copied.
 */
#include <string.h>

#include "tinyllm/tinyllm.h"

#define ENTRY_SIZE 68u
#define NAME_LEN 32u
#define NO_SCALE 0xFFFFFFFFu

static uint32_t rd32(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float rdf32(const uint8_t *p) {
    uint32_t bits = rd32(p);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

const char *tllm_status_str(tllm_status status) {
    switch (status) {
    case TLLM_OK: return "ok";
    case TLLM_ERR_ARG: return "invalid argument";
    case TLLM_ERR_MAGIC: return "bad magic";
    case TLLM_ERR_VERSION: return "unsupported version";
    case TLLM_ERR_TRUNCATED: return "truncated or inconsistent file";
    case TLLM_ERR_CRC: return "crc mismatch";
    case TLLM_ERR_UNSUPPORTED: return "unsupported hyper-parameters";
    case TLLM_ERR_TENSOR: return "missing or malformed tensor";
    case TLLM_ERR_TOKENIZER: return "malformed tokenizer";
    case TLLM_ERR_ARENA: return "arena too small";
    case TLLM_ERR_CONTEXT_FULL: return "context full";
    }
    return "unknown";
}

static int32_t find_special(const tllm_tokenizer *tok, const char *name) {
    size_t len = strlen(name);
    for (uint32_t i = 0; i < tok->n_special; ++i)
        if (tok->special_len[i] == len && memcmp(tok->special[i], name, len) == 0) return (int32_t)i;
    return -1;
}

/* Version 1: ranked BPE merges (training/tokenizer/bpe.py). */
static tllm_status parse_bpe(tllm_tokenizer *tok, const uint8_t *p, size_t size) {
    tok->n_special = rd32(p + 8);
    tok->n_merges = rd32(p + 12);
    if (tok->n_special == 0u || tok->n_special > TLLM_MAX_SPECIAL) return TLLM_ERR_TOKENIZER;
    size_t pos = 16u;
    for (uint32_t i = 0; i < tok->n_special; ++i) {
        if (pos >= size) return TLLM_ERR_TOKENIZER;
        uint8_t len = p[pos];
        if (len == 0u || pos + 1u + len > size) return TLLM_ERR_TOKENIZER;
        tok->special[i] = (const char *)(p + pos + 1u);
        tok->special_len[i] = len;
        pos += 1u + len;
    }
    if (pos + (size_t)tok->n_merges * 4u > size) return TLLM_ERR_TOKENIZER;
    tok->merges = p + pos;
    tok->vocab_size = tok->n_special + 256u + tok->n_merges;
    for (uint32_t r = 0; r < tok->n_merges; ++r) {
        uint32_t a = (uint32_t)p[pos + 4u * r] | ((uint32_t)p[pos + 4u * r + 1u] << 8);
        uint32_t b = (uint32_t)p[pos + 4u * r + 2u] | ((uint32_t)p[pos + 4u * r + 3u] << 8);
        uint32_t new_id = tok->n_special + 256u + r;
        if (a >= new_id || b >= new_id || a < tok->n_special || b < tok->n_special) return TLLM_ERR_TOKENIZER;
    }
    tok->kind = TLLM_TOK_BPE;
    tok->bos_id = find_special(tok, "<bos>");
    tok->eos_id = find_special(tok, "<eos>");
    return TLLM_OK;
}

/* Version 2: llama2.c scored pieces (training/tokenizer/llama2c.py). */
static tllm_status parse_scored(tllm_tokenizer *tok, const uint8_t *p, size_t size) {
    if (size < 20u) return TLLM_ERR_TOKENIZER;
    tok->n_special = rd32(p + 8);
    tok->vocab_size = rd32(p + 12);
    tok->max_piece_len = rd32(p + 16);
    if (tok->n_special > TLLM_MAX_SPECIAL || tok->n_special > tok->vocab_size || tok->vocab_size == 0u ||
        tok->vocab_size > 65535u || tok->max_piece_len == 0u || tok->max_piece_len > TLLM_MAX_PIECE)
        return TLLM_ERR_TOKENIZER;
    size_t pos = 20u;
    for (uint32_t i = 0; i < tok->vocab_size; ++i) {
        if (pos + 6u > size) return TLLM_ERR_TOKENIZER;
        uint32_t len = (uint32_t)p[pos + 4u] | ((uint32_t)p[pos + 5u] << 8);
        if (len > tok->max_piece_len || pos + 6u + len > size) return TLLM_ERR_TOKENIZER;
        if (i < tok->n_special) {
            tok->special[i] = (const char *)(p + pos + 6u);
            tok->special_len[i] = (uint8_t)len;
        }
        pos += 6u + len;
    }
    tok->records = p + 20u;
    tok->kind = TLLM_TOK_SCORED;
    tok->bos_id = tok->n_special > 1u ? 1 : -1; /* llama2.c: 0 <unk>, 1 <s>, 2 </s> */
    tok->eos_id = tok->n_special > 2u ? 2 : -1;
    return TLLM_OK;
}

static tllm_status parse_tokenizer(tllm_tokenizer *tok, const uint8_t *p, size_t size) {
    memset(tok, 0, sizeof *tok);
    if (size < 16u || memcmp(p, "TTOK", 4) != 0) return TLLM_ERR_TOKENIZER;
    uint32_t version = rd32(p + 4);
    if (version == 1u) return parse_bpe(tok, p, size);
    if (version == 2u) return parse_scored(tok, p, size);
    return TLLM_ERR_TOKENIZER;
}

typedef struct {
    const uint8_t *blob;
    size_t size;
    uint32_t table, count;
} table_view;

/* Find a tensor by name and check its shape; rows/cols of 0 skip the check (1-D: cols=0). */
static tllm_status find_tensor(const table_view *tv, const char *name, uint32_t rows, uint32_t cols, tllm_tensor *out) {
    size_t name_len = strlen(name);
    for (uint32_t i = 0; i < tv->count; ++i) {
        const uint8_t *e = tv->blob + tv->table + (size_t)i * ENTRY_SIZE;
        if (memcmp(e, name, name_len) != 0 || (name_len < NAME_LEN && e[name_len] != 0)) continue;
        uint32_t dtype = rd32(e + 32), ndim = rd32(e + 36);
        uint32_t d0 = rd32(e + 40), d1 = rd32(e + 44);
        uint32_t offset = rd32(e + 56), bytes = rd32(e + 60), scale_off = rd32(e + 64);
        uint32_t r = d0, c = ndim == 2u ? d1 : 1u;
        if (ndim < 1u || ndim > 2u || dtype > TLLM_DTYPE_Q4) return TLLM_ERR_TENSOR;
        if (r != rows || (cols != 0u && c != cols) || (cols == 0u && ndim != 1u)) return TLLM_ERR_TENSOR;
        if (dtype == TLLM_DTYPE_Q4 && (ndim != 2u || c % TLLM_Q4_GROUP != 0u)) return TLLM_ERR_TENSOR;
        size_t expect = dtype == TLLM_DTYPE_F32  ? (size_t)r * c * 4u
                        : dtype == TLLM_DTYPE_I8 ? (size_t)r * c
                                                 : (size_t)r * c / 2u;
        if (expect != bytes || (size_t)offset + bytes > tv->size || (offset & 3u) != 0u) return TLLM_ERR_TENSOR;
        out->data = tv->blob + offset;
        out->dtype = dtype;
        out->rows = r;
        out->cols = c;
        out->scales = NULL;
        out->scales16 = NULL;
        if (dtype == TLLM_DTYPE_Q4) {
            size_t groups = (size_t)r * c / TLLM_Q4_GROUP;
            if (scale_off == NO_SCALE || (scale_off & 1u) != 0u || (size_t)scale_off + 2u * groups > tv->size)
                return TLLM_ERR_TENSOR;
            out->scales16 = (const uint16_t *)(const void *)(tv->blob + scale_off);
        } else if (dtype == TLLM_DTYPE_I8) {
            if (scale_off == NO_SCALE || (scale_off & 3u) != 0u || (size_t)scale_off + 4u * r > tv->size)
                return TLLM_ERR_TENSOR;
            out->scales = (const float *)(const void *)(tv->blob + scale_off);
        }
        return TLLM_OK;
    }
    return TLLM_ERR_TENSOR;
}

static tllm_status load_tensors(tllm_model *m, const table_view *tv) {
    const tllm_config *c = &m->cfg;
    uint32_t kv_dim = c->n_kv_heads * (c->d_model / c->n_heads);
    tllm_status st = find_tensor(tv, "tok_emb", c->vocab_size, c->d_model, &m->tok_emb);
    if (st == TLLM_OK && c->pos_type == TLLM_POS_LEARNED)
        st = find_tensor(tv, "pos_emb", c->ctx_len, c->d_model, &m->pos_emb);
    if (st == TLLM_OK) st = find_tensor(tv, "final_norm", c->d_model, 0, &m->final_norm);
    for (uint32_t l = 0; st == TLLM_OK && l < c->n_layers; ++l) {
        tllm_layer *L = &m->layers[l];
        char name[NAME_LEN];
        struct {
            const char *suffix;
            tllm_tensor *t;
            uint32_t rows, cols;
        } spec[] = {
            {"attn_norm", &L->attn_norm, c->d_model, 0}, {"wq", &L->wq, c->d_model, c->d_model},
            {"wk", &L->wk, kv_dim, c->d_model},          {"wv", &L->wv, kv_dim, c->d_model},
            {"wo", &L->wo, c->d_model, c->d_model},      {"mlp_norm", &L->mlp_norm, c->d_model, 0},
            {"w1", &L->w1, c->d_ff, c->d_model},         {"w2", &L->w2, c->d_model, c->d_ff},
            {"w3", &L->w3, c->d_ff, c->d_model},
        };
        size_t n_spec = c->mlp_type == TLLM_MLP_SWIGLU ? 9u : 8u;
        for (size_t s = 0; st == TLLM_OK && s < n_spec; ++s) {
            size_t len = 0;
            name[len++] = 'l';
            if (l >= 10u) name[len++] = (char)('0' + l / 10u);
            name[len++] = (char)('0' + l % 10u);
            name[len++] = '.';
            size_t sl = strlen(spec[s].suffix);
            memcpy(name + len, spec[s].suffix, sl + 1u);
            st = find_tensor(tv, name, spec[s].rows, spec[s].cols, spec[s].t);
        }
        /* norms must be float32 */
        if (st == TLLM_OK && (L->attn_norm.dtype != TLLM_DTYPE_F32 || L->mlp_norm.dtype != TLLM_DTYPE_F32))
            st = TLLM_ERR_TENSOR;
    }
    if (st == TLLM_OK && m->final_norm.dtype != TLLM_DTYPE_F32) st = TLLM_ERR_TENSOR;
    if (st == TLLM_OK && c->pos_type == TLLM_POS_LEARNED && m->pos_emb.dtype != TLLM_DTYPE_F32) st = TLLM_ERR_TENSOR;
    return st;
}

tllm_status tllm_model_load(tllm_model *m, const void *blob_v, size_t size) {
    if (m == NULL || blob_v == NULL) return TLLM_ERR_ARG;
    memset(m, 0, sizeof *m);
    const uint8_t *blob = (const uint8_t *)blob_v;
    if (size < TLLM_HEADER_SIZE) return TLLM_ERR_TRUNCATED;
    if (memcmp(blob, "TLLM", 4) != 0) return TLLM_ERR_MAGIC;
    if (rd32(blob + 4) != TLLM_FORMAT_VERSION || rd32(blob + 8) != TLLM_HEADER_SIZE || rd32(blob + 12) != 1u)
        return TLLM_ERR_VERSION;
    tllm_config *c = &m->cfg;
    c->vocab_size = rd32(blob + 16);
    c->ctx_len = rd32(blob + 20);
    c->n_layers = rd32(blob + 24);
    c->d_model = rd32(blob + 28);
    c->n_heads = rd32(blob + 32);
    c->n_kv_heads = rd32(blob + 36);
    c->d_ff = rd32(blob + 40);
    c->mlp_type = rd32(blob + 44);
    c->pos_type = rd32(blob + 48);
    c->weight_dtype = rd32(blob + 52);
    c->flags = rd32(blob + 56);
    uint32_t tok_off = rd32(blob + 60), tok_size = rd32(blob + 64);
    uint32_t table = rd32(blob + 68), count = rd32(blob + 72);
    uint32_t payload = rd32(blob + 76), crc = rd32(blob + 80);
    memcpy(m->model_id, blob + 84, 16);
    c->norm_eps = rdf32(blob + 100);
    c->rope_theta = rdf32(blob + 104);

    if ((size_t)payload + TLLM_HEADER_SIZE != size) return TLLM_ERR_TRUNCATED;
    if (tllm_crc32(blob + TLLM_HEADER_SIZE, payload) != crc) return TLLM_ERR_CRC;
    if (c->n_layers == 0u || c->n_layers > TLLM_MAX_LAYERS || c->d_model == 0u || c->d_model > TLLM_MAX_DIM ||
        c->d_ff == 0u || c->d_ff > 4u * TLLM_MAX_DIM || c->ctx_len == 0u || c->ctx_len > TLLM_MAX_CTX ||
        c->vocab_size == 0u || c->vocab_size > TLLM_MAX_VOCAB || c->n_heads == 0u || c->n_kv_heads == 0u ||
        c->d_model % c->n_heads != 0u || c->n_heads % c->n_kv_heads != 0u || c->mlp_type > 1u || c->pos_type > 1u ||
        c->weight_dtype > 2u || !(c->norm_eps > 0.0f) ||
        (c->pos_type == TLLM_POS_ROPE && ((c->d_model / c->n_heads) % 2u != 0u || !(c->rope_theta > 0.0f))))
        return TLLM_ERR_UNSUPPORTED;
    if ((size_t)tok_off + tok_size > size || (size_t)table + (size_t)count * ENTRY_SIZE > size)
        return TLLM_ERR_TRUNCATED;

    tllm_status st = parse_tokenizer(&m->tok, blob + tok_off, tok_size);
    if (st != TLLM_OK) return st;
    if (m->tok.vocab_size != c->vocab_size) return TLLM_ERR_TOKENIZER;

    table_view tv = {blob, size, table, count};
    st = load_tensors(m, &tv);
    if (st != TLLM_OK) return st;
    m->blob = blob;
    m->size = size;
    m->crc32 = crc;
    return TLLM_OK;
}
