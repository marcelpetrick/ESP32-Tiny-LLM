/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Byte-level BPE encoder/decoder — a line-by-line mirror of training/tokenizer/bpe.py.
 * Pre-tokens: optional single leading space + (ASCII letter run [+ '='] | one byte);
 * space runs keep their last space for the next word. Merges are applied lowest rank
 * first, leftmost on ties. Round-trip fixtures are shared with the Python tests.
 */
#include <string.h>

#include "tinyllm/tinyllm.h"

static int is_letter(uint8_t b) { return (b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z'); }

static uint32_t merge_a(const tllm_tokenizer *t, uint32_t r) {
    return (uint32_t)t->merges[4u * r] | ((uint32_t)t->merges[4u * r + 1u] << 8);
}

static uint32_t merge_b(const tllm_tokenizer *t, uint32_t r) {
    return (uint32_t)t->merges[4u * r + 2u] | ((uint32_t)t->merges[4u * r + 3u] << 8);
}

static uint32_t slot_of(uint32_t key, uint32_t mask) { return (key * 2654435761u) & mask; }

uint32_t tllm_tokenizer_table_entries(const tllm_tokenizer *t) {
    uint32_t entries = 16u;
    while (entries < 2u * t->n_merges) entries <<= 1;
    return entries;
}

void tllm_tokenizer_build(tllm_tokenizer *t, uint32_t *keys, uint16_t *ranks, uint32_t entries) {
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

int tllm_tokenize(const tllm_tokenizer *t, const char *text, size_t len, int32_t *out, int max_out) {
    if (t == NULL || t->hash_keys == NULL || (text == NULL && len > 0u) || out == NULL || max_out < 0) return -1;
    const uint8_t *d = (const uint8_t *)text;
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

int32_t tllm_special_id(const tllm_tokenizer *t, const char *name) {
    size_t len = strlen(name);
    for (uint32_t i = 0; i < t->n_special; ++i)
        if (t->special_len[i] == len && memcmp(t->special[i], name, len) == 0) return (int32_t)i;
    return -1;
}

int tllm_token_bytes(const tllm_tokenizer *t, int32_t id, char *out, int max_out) {
    if (id < 0 || (uint32_t)id >= t->vocab_size || max_out < 0) return -1;
    uint32_t u = (uint32_t)id;
    if (u < t->n_special) {
        int len = t->special_len[u];
        if (len > max_out) return -1;
        memcpy(out, t->special[u], (size_t)len);
        return len;
    }
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
