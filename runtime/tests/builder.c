/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#define _POSIX_C_SOURCE 200809L
#include "builder.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm/tinyllm.h"

static const char *const SPECIALS[TEST_N_SPECIAL + 2] = {"<pad>",  "<bos>",     "<eos>",         "<S>",  "</S>",
                                                         "<U>",    "</U>",      "<A>",           "</A>", "<ACT>",
                                                         "</ACT>", "<clarify>", "<unsupported>", "<F>",  "</F>"};

typedef struct {
    uint8_t *buf;
    size_t len, cap;
} growbuf;

static void put(growbuf *g, const void *src, size_t n) {
    if (g->len + n > g->cap) {
        size_t cap = g->cap ? g->cap : 4096u;
        while (cap < g->len + n) cap *= 2u;
        g->buf = realloc(g->buf, cap);
        g->cap = cap;
    }
    if (src != NULL)
        memcpy(g->buf + g->len, src, n);
    else
        memset(g->buf + g->len, 0, n);
    g->len += n;
}

static void put32(growbuf *g, uint32_t v) {
    uint8_t b[4] = {(uint8_t)v, (uint8_t)(v >> 8), (uint8_t)(v >> 16), (uint8_t)(v >> 24)};
    put(g, b, 4);
}

static void putf(growbuf *g, float f) {
    uint32_t v;
    memcpy(&v, &f, 4);
    put32(g, v);
}

static void align16(growbuf *g) {
    while (g->len % 16u) put(g, NULL, 1);
}

/* float32 -> float16 (normal range, round to nearest) for Q4 test models */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xFFu) - 127 + 15;
    uint32_t mant = (x >> 13) & 0x3FFu;
    if ((x >> 12) & 1u) ++mant; /* round */
    if (mant == 0x400u) {
        mant = 0;
        ++exp;
    }
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7C00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | mant);
}

static uint32_t g_rng;
static float frand(void) {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 17;
    g_rng ^= g_rng << 5;
    return (float)(g_rng % 2000001u) / 1000000.0f - 1.0f;
}

test_model_spec test_default_spec(void) {
    test_model_spec s;
    memset(&s, 0, sizeof s);
    s.n_layers = 2;
    s.d_model = 16;
    s.n_heads = 4;
    s.n_kv_heads = 2;
    s.d_ff = 24;
    s.ctx_len = 64;
    s.n_merges = 8;
    s.seed = 12345u;
    return s;
}

typedef struct {
    char name[32];
    uint32_t rows, cols; /* cols 0 for 1-D */
    float *values;
    int quantize;
} tensor_def;

static void set_name(tensor_def *t, const char *name) {
    memset(t->name, 0, sizeof t->name);
    snprintf(t->name, sizeof t->name, "%s", name);
}

test_blob test_build_model(const test_model_spec *s) {
    test_blob out;
    memset(&out, 0, sizeof out);
    g_rng = s->seed ? s->seed : 1u;
    const uint32_t n_special =
        (s->omit_chat_specials || s->scored) ? 3u : TEST_N_SPECIAL + (s->fact_specials ? 2u : 0u);
    const uint32_t S = n_special;
    const uint32_t merges[8][2] = {{S + 't', S + 'h'}, {S + 256, S + 'e'}, {S + ' ', S + 't'}, {S + 'f', S + 'a'},
                                   {S + 259, S + 'n'}, {S + 'a', S + 'n'}, {S + ' ', S + 257}, {S + 260, S + '='}};
    const uint32_t n_merges = (uint32_t)s->n_merges;
    const uint32_t vocab = n_special + 256u + n_merges;
    const uint32_t d = s->d_model, hd = d / s->n_heads, kvd = s->n_kv_heads * hd;

    /* tensors */
    tensor_def defs[3 + 9 * TLLM_MAX_LAYERS];
    int nt = 0;
    set_name(&defs[nt], "tok_emb");
    defs[nt].rows = vocab;
    defs[nt].cols = d;
    defs[nt++].quantize = 1;
    if (s->pos_type == 0) {
        set_name(&defs[nt], "pos_emb");
        defs[nt].rows = s->ctx_len;
        defs[nt].cols = d;
        defs[nt++].quantize = 0;
    }
    for (uint32_t l = 0; l < s->n_layers; ++l) {
        const char *names[9] = {"attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "w1", "w2", "w3"};
        uint32_t rows[9] = {d, d, kvd, kvd, d, d, s->d_ff, d, s->d_ff};
        uint32_t cols[9] = {0, d, d, d, d, 0, d, s->d_ff, d};
        int n = s->mlp_type ? 9 : 8;
        for (int i = 0; i < n; ++i) {
            char name[32];
            snprintf(name, sizeof name, "l%u.%s", l, names[i]);
            set_name(&defs[nt], name);
            defs[nt].rows = rows[i];
            defs[nt].cols = cols[i];
            defs[nt++].quantize = cols[i] != 0u;
        }
    }
    set_name(&defs[nt], "final_norm");
    defs[nt].rows = d;
    defs[nt].cols = 0;
    defs[nt++].quantize = 0;

    for (int i = 0; i < nt; ++i) {
        size_t n = (size_t)defs[i].rows * (defs[i].cols ? defs[i].cols : 1u);
        defs[i].values = malloc(sizeof(float) * n);
        int is_norm = defs[i].cols == 0u;
        int zero = s->zero_blocks && (strstr(defs[i].name, ".wo") != NULL || strstr(defs[i].name, ".w2") != NULL);
        for (size_t k = 0; k < n; ++k) {
            float r = frand();
            defs[i].values[k] = is_norm ? 1.0f : zero ? 0.0f : r * 0.5f;
        }
        if (strcmp(defs[i].name, "pos_emb") == 0 && s->zero_blocks)
            for (size_t k = 0; k < n; ++k) defs[i].values[k] = 0.0f;
    }

    growbuf g = {NULL, 0, 0};
    put(&g, NULL, TLLM_HEADER_SIZE);
    /* tokenizer */
    size_t tok_off = g.len;
    put(&g, "TTOK", 4);
    if (s->scored) {
        /* llama2.c layout: <unk> <s> </s>, 256 byte pieces, then 8 scored pieces */
        static const char *const pieces[8] = {" ", "a", "b", "c", "ab", " ab", "abc", "\xc3\xa9"};
        static const float scores[8] = {0.0f, 0.0f, 0.0f, 0.0f, 5.0f, 3.0f, 4.0f, 0.0f};
        const char *names[3] = {"<unk>", "\n<s>\n", "\n</s>\n"};
        put32(&g, 2);
        put32(&g, 3);
        put32(&g, vocab);
        put32(&g, 6);
        for (uint32_t i = 0; i < vocab; ++i) {
            char text[16];
            const char *p = text;
            if (i < 3u)
                p = names[i];
            else if (i < 259u)
                (void)snprintf(text, sizeof text, i - 3u == 0x0Au ? "<0x%02x>" : "<0x%02X>", i - 3u);
            else
                p = pieces[i - 259u];
            uint16_t len = (uint16_t)strlen(p);
            putf(&g, i >= 259u ? scores[i - 259u] : 0.0f);
            uint8_t lb[2] = {(uint8_t)len, (uint8_t)(len >> 8)};
            put(&g, lb, 2);
            put(&g, p, len);
        }
    } else {
        put32(&g, 1);
        put32(&g, n_special);
        put32(&g, n_merges);
        for (uint32_t i = 0; i < n_special; ++i) {
            uint8_t len = (uint8_t)strlen(SPECIALS[i]);
            put(&g, &len, 1);
            put(&g, SPECIALS[i], len);
        }
        for (uint32_t r = 0; r < n_merges; ++r) {
            uint8_t b[4] = {(uint8_t)merges[r][0], (uint8_t)(merges[r][0] >> 8), (uint8_t)merges[r][1],
                            (uint8_t)(merges[r][1] >> 8)};
            put(&g, b, 4);
        }
    }
    while (g.len % 4u) put(&g, NULL, 1);
    size_t tok_size = g.len - tok_off;
    align16(&g);
    size_t table_off = g.len;
    put(&g, NULL, 68u * (size_t)nt);
    align16(&g);
    for (int i = 0; i < nt; ++i) {
        tensor_def *t = &defs[i];
        size_t n = (size_t)t->rows * (t->cols ? t->cols : 1u);
        int q4 = t->quantize && s->dtype == 2u && t->cols % 32u == 0u;
        int q = t->quantize && s->dtype >= 1u && !q4;
        size_t data_off = g.len, scale_off = 0xFFFFFFFFu, bytes;
        if (strcmp(t->name, "pos_emb") == 0) out.pos_emb_offset = data_off;
        if (strcmp(t->name, "tok_emb") == 0) out.tok_emb_offset = data_off;
        if (q4) { /* group-wise 4-bit, float16 scales (same layout as training/quantize.py) */
            uint32_t groups = t->rows * (t->cols / 32u);
            uint16_t *s16 = malloc(sizeof(uint16_t) * groups);
            for (uint32_t r = 0; r < t->rows; ++r)
                for (uint32_t gi = 0; gi < t->cols / 32u; ++gi) {
                    const float *v = t->values + (size_t)r * t->cols + gi * 32u;
                    float amax = 0.0f;
                    for (int k = 0; k < 32; ++k)
                        if (fabsf(v[k]) > amax) amax = fabsf(v[k]);
                    uint16_t h = f32_to_f16(amax > 0.0f ? amax / 7.0f : 1.0f);
                    float sc = tllm_f16_to_f32(h);
                    s16[r * (t->cols / 32u) + gi] = h;
                    for (int k = 0; k < 32; k += 2) {
                        long lo = lrintf(v[k] / sc), hi = lrintf(v[k + 1] / sc);
                        lo = lo < -7 ? -7 : lo > 7 ? 7 : lo;
                        hi = hi < -7 ? -7 : hi > 7 ? 7 : hi;
                        uint8_t byte = (uint8_t)((lo + 8) | ((hi + 8) << 4));
                        put(&g, &byte, 1);
                    }
                }
            bytes = n / 2u;
            align16(&g);
            scale_off = g.len;
            for (uint32_t k = 0; k < groups; ++k) {
                uint8_t le[2] = {(uint8_t)s16[k], (uint8_t)(s16[k] >> 8)};
                put(&g, le, 2);
            }
            free(s16);
        } else if (q) {
            float *scales = malloc(sizeof(float) * t->rows);
            for (uint32_t r = 0; r < t->rows; ++r) {
                float amax = 0.0f;
                for (uint32_t c = 0; c < t->cols; ++c) {
                    float a = fabsf(t->values[(size_t)r * t->cols + c]);
                    if (a > amax) amax = a;
                }
                scales[r] = amax > 0.0f ? amax / 127.0f : 1.0f;
                for (uint32_t c = 0; c < t->cols; ++c) {
                    int8_t v = (int8_t)lrintf(t->values[(size_t)r * t->cols + c] / scales[r]);
                    put(&g, &v, 1);
                }
            }
            bytes = n;
            align16(&g);
            scale_off = g.len;
            for (uint32_t r = 0; r < t->rows; ++r) putf(&g, scales[r]);
            free(scales);
        } else {
            for (size_t k = 0; k < n; ++k) putf(&g, t->values[k]);
            bytes = n * 4u;
        }
        align16(&g);
        /* table entry */
        uint8_t *e = g.buf + table_off + 68u * (size_t)i;
        memcpy(e, t->name, 32);
        uint32_t fields[9] = {q4 ? 2u : q ? 1u : 0u, t->cols ? 2u : 1u,  t->rows, t->cols, 0, 0, (uint32_t)data_off,
                              (uint32_t)bytes,       (uint32_t)scale_off};
        for (int k = 0; k < 9; ++k)
            for (int b = 0; b < 4; ++b) e[32 + 4 * k + b] = (uint8_t)(fields[k] >> (8 * b));
        free(t->values);
    }
    /* header */
    growbuf h = {NULL, 0, 0};
    put(&h, "TLLM", 4);
    put32(&h, TLLM_FORMAT_VERSION);
    put32(&h, TLLM_HEADER_SIZE);
    put32(&h, 1);
    uint32_t hv[] = {vocab,
                     s->ctx_len,
                     s->n_layers,
                     d,
                     s->n_heads,
                     s->n_kv_heads,
                     s->d_ff,
                     s->mlp_type,
                     s->pos_type,
                     s->dtype,
                     0,
                     (uint32_t)tok_off,
                     (uint32_t)tok_size,
                     (uint32_t)table_off,
                     (uint32_t)nt,
                     (uint32_t)(g.len - TLLM_HEADER_SIZE),
                     0};
    for (size_t i = 0; i < sizeof hv / sizeof hv[0]; ++i) put32(&h, hv[i]);
    uint8_t id[16];
    for (int i = 0; i < 16; ++i) id[i] = (uint8_t)(0xA0 + i);
    put(&h, id, 16);
    putf(&h, 1e-5f);
    putf(&h, 10000.0f);
    memcpy(g.buf, h.buf, h.len);
    free(h.buf);

    /* copy into an aligned buffer */
    void *aligned = NULL;
    if (posix_memalign(&aligned, 16, g.len) != 0) abort();
    memcpy(aligned, g.buf, g.len);
    free(g.buf);
    out.data = aligned;
    out.size = g.len;
    out.header_crc_offset = 80;
    out.tensor_table_offset = table_off;
    out.tokenizer_offset = tok_off;
    test_fix_crc(&out);
    return out;
}

void test_fix_crc(test_blob *b) {
    uint32_t crc = tllm_crc32(b->data + TLLM_HEADER_SIZE, b->size - TLLM_HEADER_SIZE);
    for (int i = 0; i < 4; ++i) b->data[b->header_crc_offset + (size_t)i] = (uint8_t)(crc >> (8 * i));
}

void test_free_blob(test_blob *b) {
    free(b->data);
    b->data = NULL;
}

const char *test_write_temp(const test_blob *b, const char *name) {
    static char path[256];
    const char *dir = getenv("TMPDIR");
    snprintf(path, sizeof path, "%s/%s", dir ? dir : "/tmp", name);
    FILE *f = fopen(path, "wb");
    if (f == NULL) return NULL;
    fwrite(b->data, 1, b->size, f);
    fclose(f);
    return path;
}
