/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Tokenizers.
 *
 * TLLM_TOK_BPE — byte-level BPE, a line-by-line mirror of training/tokenizer/bpe.py.
 *   Pre-tokens: optional single leading space + (ASCII letter run [+ '='] | one byte);
 *   space runs keep their last space for the next word. Merges are applied lowest rank
 *   first, leftmost on ties.
 *
 * TLLM_TOK_SCORED — llama2.c-compatible scored pieces. The encode/decode algorithm follows
 *   encode()/decode() in Andrej Karpathy's llama2.c run.c (MIT licence, see
 *   third_party/llama2c/LICENSE): dummy-prefix space, per-codepoint lookup with byte
 *   fallback (ids 3..258), then repeatedly merge the adjacent pair whose concatenation is
 *   a vocabulary piece with the highest score. Implemented without heap allocation.
 */
#include <string.h>

#include "tinyllm/tinyllm.h"

static uint32_t rd16(const uint8_t *p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8); }

/* ================================================================== BPE */
static int is_letter(uint8_t b) { return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z'); }

static uint32_t merge_a(const tllm_tokenizer *t, uint32_t r) { return rd16(t->merges + 4u * r); }
static uint32_t merge_b(const tllm_tokenizer *t, uint32_t r) { return rd16(t->merges + 4u * r + 2u); }

static uint32_t slot_of(uint32_t key, uint32_t mask) { return (key * 2654435761u) & mask; }

static uint32_t bpe_entries(const tllm_tokenizer *t) {
    uint32_t entries = 16u;
    while (entries < 2u * t->n_merges) entries <<= 1;
    return entries;
}

static void bpe_attach(tllm_tokenizer *t, uint32_t *keys, uint16_t *ranks, uint32_t entries) {
    t->hash_keys = keys;
    t->hash_ranks = ranks;
    t->hash_mask = entries - 1u;
    memset(keys, 0, sizeof(uint32_t) * entries);
    for (uint32_t r = 0; r < t->n_merges; ++r) {
        uint32_t key = ((merge_a(t, r) << 16) | merge_b(t, r)) + 1u;
        uint32_t s = slot_of(key, t->hash_mask);
        while (keys[s] != 0u && keys[s] != key) s = (s + 1u) & t->hash_mask;
        if (keys[s] == 0u) { /* keep the first (lowest) rank for duplicate pairs */
            keys[s] = key;
            ranks[s] = (uint16_t)r;
        }
    }
}

static int rank_of(const tllm_tokenizer *t, int32_t a, int32_t b) {
    uint32_t key = (((uint32_t)a << 16) | (uint32_t)b) + 1u;
    uint32_t s = slot_of(key, t->hash_mask);
    while (t->hash_keys[s] != 0u) {
        if (t->hash_keys[s] == key) return (int)t->hash_ranks[s];
        s = (s + 1u) & t->hash_mask;
    }
    return -1;
}

/* Encode one pre-token into out (capacity checked by the caller); returns token count. */
static int encode_piece(const tllm_tokenizer *t, const uint8_t *p, size_t len, int32_t *out) {
    int n = (int)len;
    for (int i = 0; i < n; ++i) out[i] = (int32_t)(t->n_special + p[i]);
    while (n > 1) {
        int best_rank = -1, best_i = -1;
        for (int i = 0; i + 1 < n; ++i) {
            int r = rank_of(t, out[i], out[i + 1]);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) {
                best_rank = r;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        out[best_i] = (int32_t)(t->n_special + 256u + (uint32_t)best_rank);
        for (int i = best_i + 1; i + 1 < n; ++i) out[i] = out[i + 1];
        --n;
    }
    return n;
}

static int bpe_tokenize(const tllm_tokenizer *t, const uint8_t *d, size_t len, int32_t *out, int max_out) {
    size_t i = 0;
    int n_out = 0;
    while (i < len) {
        size_t start = i;
        if (d[i] == ' ') {
            size_t j = i;
            while (j < len && d[j] == ' ') ++j;
            if (j == len) { /* trailing spaces form one pre-token */
                i = j;
                goto emit;
            }
            if (j - i > 1u) { /* all but the last space */
                i = j - 1u;
                if ((size_t)(max_out - n_out) < i - start) return -1;
                n_out += encode_piece(t, d + start, i - start, out + n_out);
                start = i;
            }
            ++i;
        }
        if (is_letter(d[i])) {
            while (i < len && is_letter(d[i])) ++i;
            if (i < len && d[i] == '=') ++i;
        } else {
            ++i;
        }
    emit:
        if ((size_t)(max_out - n_out) < i - start) return -1;
        n_out += encode_piece(t, d + start, i - start, out + n_out);
    }
    return n_out;
}

static int bpe_bytes(const tllm_tokenizer *t, uint32_t u, char *out, int max_out) {
    if (u < t->n_special + 256u) {
        if (max_out < 1) return -1;
        out[0] = (char)(uint8_t)(u - t->n_special);
        return 1;
    }
    uint32_t r = u - t->n_special - 256u;
    int left = tllm_token_bytes(t, (int32_t)merge_a(t, r), out, max_out);
    if (left < 0) return -1;
    int right = tllm_token_bytes(t, (int32_t)merge_b(t, r), out + left, max_out - left);
    return right < 0 ? -1 : left + right;
}

/* ================================================================== scored pieces */
static const uint8_t *piece(const tllm_tokenizer *t, uint32_t id, uint32_t *len) {
    const uint8_t *rec = t->records + t->piece_off[id];
    *len = rd16(rec + 4);
    return rec + 6;
}

static float score_of(const tllm_tokenizer *t, uint32_t id) {
    const uint8_t *rec = t->records + t->piece_off[id];
    uint32_t bits = (uint32_t)rec[0] | ((uint32_t)rec[1] << 8) | ((uint32_t)rec[2] << 16) | ((uint32_t)rec[3] << 24);
    float f;
    memcpy(&f, &bits, sizeof f);
    return f;
}

/* Lexicographic byte order, shorter first on a common prefix (same as Python bytes). */
static int cmp_bytes(const uint8_t *a, uint32_t la, const uint8_t *b, uint32_t lb) {
    int c = memcmp(a, b, la < lb ? la : lb);
    if (c != 0) return c;
    return la < lb ? -1 : la > lb ? 1 : 0;
}

static int cmp_ids(const tllm_tokenizer *t, uint16_t a, uint16_t b) {
    uint32_t la, lb;
    const uint8_t *pa = piece(t, a, &la), *pb = piece(t, b, &lb);
    return cmp_bytes(pa, la, pb, lb);
}

static void sift_down(const tllm_tokenizer *t, uint16_t *v, uint32_t root, uint32_t n) {
    for (;;) {
        uint32_t child = 2u * root + 1u;
        if (child >= n) return;
        if (child + 1u < n && cmp_ids(t, v[child], v[child + 1u]) < 0) ++child;
        if (cmp_ids(t, v[root], v[child]) >= 0) return;
        uint16_t tmp = v[root];
        v[root] = v[child];
        v[child] = tmp;
        root = child;
    }
}

static void scored_attach(tllm_tokenizer *t, uint32_t *offsets, uint16_t *sorted) {
    t->piece_off = offsets;
    t->sorted = sorted;
    uint32_t pos = 0;
    for (uint32_t i = 0; i < t->vocab_size; ++i) {
        offsets[i] = pos;
        pos += 6u + rd16(t->records + pos + 4u);
        sorted[i] = (uint16_t)i;
    }
    /* heapsort: in place, no allocation */
    uint32_t n = t->vocab_size;
    for (uint32_t i = n / 2u; i-- > 0u;) sift_down(t, sorted, i, n);
    for (uint32_t end = n; end-- > 1u;) {
        uint16_t tmp = sorted[0];
        sorted[0] = sorted[end];
        sorted[end] = tmp;
        sift_down(t, sorted, 0, end);
    }
}

static int32_t scored_lookup(const tllm_tokenizer *t, const uint8_t *s, uint32_t len) {
    uint32_t lo = 0, hi = t->vocab_size;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2u, lm;
        const uint8_t *pm = piece(t, t->sorted[mid], &lm);
        int c = cmp_bytes(pm, lm, s, len);
        if (c == 0) return (int32_t)t->sorted[mid];
        if (c < 0)
            lo = mid + 1u;
        else
            hi = mid;
    }
    return -1;
}

static int scored_tokenize(const tllm_tokenizer *t, const uint8_t *d, size_t len, int32_t *out, int max_out) {
    int n = 0;
    if (len > 0u) {
        int32_t space = scored_lookup(t, (const uint8_t *)" ", 1u);
        if (space >= 0) {
            if (n >= max_out) return -1;
            out[n++] = space;
        }
    }
    for (size_t i = 0; i < len;) {
        size_t j = i + 1u;
        while (j < len && (d[j] & 0xC0u) == 0x80u && j - i < 4u) ++j;
        int32_t id = scored_lookup(t, d + i, (uint32_t)(j - i));
        if (id >= 0) {
            if (n >= max_out) return -1;
            out[n++] = id;
        } else {
            if ((size_t)(max_out - n) < j - i) return -1;
            for (size_t k = i; k < j; ++k) out[n++] = (int32_t)d[k] + 3; /* byte fallback */
        }
        i = j;
    }
    uint8_t buf[2u * TLLM_MAX_PIECE];
    for (;;) {
        float best_score = -1e10f;
        int best_i = -1;
        int32_t best_id = -1;
        for (int i = 0; i + 1 < n; ++i) {
            uint32_t la, lb;
            const uint8_t *pa = piece(t, (uint32_t)out[i], &la), *pb = piece(t, (uint32_t)out[i + 1], &lb);
            memcpy(buf, pa, la);
            memcpy(buf + la, pb, lb);
            int32_t id = scored_lookup(t, buf, la + lb);
            if (id >= 0 && score_of(t, (uint32_t)id) > best_score) {
                best_score = score_of(t, (uint32_t)id);
                best_id = id;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        out[best_i] = best_id;
        for (int i = best_i + 1; i + 1 < n; ++i) out[i] = out[i + 1];
        --n;
    }
    return n;
}

static int hex_digit(uint8_t c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    return -1;
}

static int scored_bytes(const tllm_tokenizer *t, uint32_t id, char *out, int max_out) {
    uint32_t len;
    const uint8_t *p = piece(t, id, &len);
    if (len == 6u && memcmp(p, "<0x", 3) == 0 && p[5] == '>' && hex_digit(p[3]) >= 0 && hex_digit(p[4]) >= 0) {
        if (max_out < 1) return -1;
        out[0] = (char)(uint8_t)(hex_digit(p[3]) * 16 + hex_digit(p[4]));
        return 1;
    }
    if ((int)len > max_out) return -1;
    memcpy(out, p, len);
    return (int)len;
}

/* ================================================================== public API */
size_t tllm_tokenizer_workspace_size(const tllm_tokenizer *t) {
    if (t->kind == TLLM_TOK_SCORED) return sizeof(uint32_t) * t->vocab_size + sizeof(uint16_t) * t->vocab_size;
    uint32_t entries = bpe_entries(t);
    return sizeof(uint32_t) * entries + sizeof(uint16_t) * entries;
}

void tllm_tokenizer_attach(tllm_tokenizer *t, void *workspace) {
    if (t->kind == TLLM_TOK_SCORED) {
        uint32_t *offsets = (uint32_t *)workspace;
        scored_attach(t, offsets, (uint16_t *)(void *)(offsets + t->vocab_size));
    } else {
        uint32_t entries = bpe_entries(t);
        uint32_t *keys = (uint32_t *)workspace;
        bpe_attach(t, keys, (uint16_t *)(void *)(keys + entries), entries);
    }
    t->attached = 1;
}

int tllm_tokenize(const tllm_tokenizer *t, const char *text, size_t len, int32_t *out, int max_out) {
    if (t == NULL || !t->attached || (text == NULL && len > 0u) || out == NULL || max_out < 0) return -1;
    if (t->kind == TLLM_TOK_SCORED) return scored_tokenize(t, (const uint8_t *)text, len, out, max_out);
    return bpe_tokenize(t, (const uint8_t *)text, len, out, max_out);
}

int32_t tllm_special_id(const tllm_tokenizer *t, const char *name) {
    size_t len = strlen(name);
    for (uint32_t i = 0; i < t->n_special; ++i)
        if (t->special_len[i] == len && memcmp(t->special[i], name, len) == 0) return (int32_t)i;
    return -1;
}

int tllm_token_bytes(const tllm_tokenizer *t, int32_t id, char *out, int max_out) {
    if (id < 0 || (uint32_t)id >= t->vocab_size || max_out < 0) return -1;
    uint32_t u = (uint32_t)id;
    if (t->kind == TLLM_TOK_SCORED) return t->attached ? scored_bytes(t, u, out, max_out) : -1;
    if (u < t->n_special) {
        int len = t->special_len[u];
        if (len > max_out) return -1;
        memcpy(out, t->special[u], (size_t)len);
        return len;
    }
    return bpe_bytes(t, u, out, max_out);
}
