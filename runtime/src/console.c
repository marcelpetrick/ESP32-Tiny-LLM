/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Console: prompt construction, generation with stop rules, action parsing and
 * validation, observability commands (vision §29 "Product/runtime observability").
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm/console.h"

#define REPLY_RESERVE 48u /* tokens kept free for the answer when trimming history */
#define ACTION_MAX_TOKENS 24u

/* ------------------------------------------------------------------ output helpers */
static void out(tllm_console *con, const char *text) {
    if (con->write != NULL) con->write(con->user, text, strlen(text));
}

static void outf(tllm_console *con, const char *fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n > 0) out(con, buf);
}

typedef struct {
    char *buf;
    size_t cap, len;
} jbuf;

static void jraw(jbuf *j, const char *s) {
    size_t n = strlen(s);
    if (j->len + n + 1u > j->cap) n = j->cap - j->len - 1u;
    memcpy(j->buf + j->len, s, n);
    j->len += n;
    j->buf[j->len] = '\0';
}

static void jfmt(jbuf *j, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(j->buf + j->len, j->cap - j->len, fmt, ap);
    va_end(ap);
    if (n > 0) j->len = j->len + (size_t)n < j->cap ? j->len + (size_t)n : j->cap - 1u;
}

/* Length of the valid UTF-8 sequence starting at s[0] (0 if invalid). */
static size_t utf8_len(const unsigned char *s, size_t n) {
    size_t len;
    if (s[0] < 0x80u) return 1u;
    if (s[0] >= 0xC2u && s[0] <= 0xDFu)
        len = 2u;
    else if (s[0] >= 0xE0u && s[0] <= 0xEFu)
        len = 3u;
    else if (s[0] >= 0xF0u && s[0] <= 0xF4u)
        len = 4u;
    else
        return 0u;
    if (len > n) return 0u;
    for (size_t i = 1; i < len; ++i)
        if ((s[i] & 0xC0u) != 0x80u) return 0u;
    return len;
}

/* JSON string with escapes; invalid UTF-8 (possible with sampled byte tokens) -> U+FFFD. */
static void jstr(jbuf *j, const char *s, size_t n) {
    char tmp[8];
    jraw(j, "\"");
    for (size_t i = 0; i < n;) {
        const unsigned char *u = (const unsigned char *)s + i;
        size_t len = utf8_len(u, n - i);
        if (len == 0u) {
            jraw(j, "\\ufffd");
            ++i;
            continue;
        }
        if (len > 1u) {
            memcpy(tmp, u, len);
            tmp[len] = '\0';
        } else if (u[0] == '"' || u[0] == '\\') {
            tmp[0] = '\\';
            tmp[1] = (char)u[0];
            tmp[2] = '\0';
        } else if (u[0] < 0x20u) {
            (void)snprintf(tmp, sizeof tmp, "\\u%04x", u[0]);
        } else {
            tmp[0] = (char)u[0];
            tmp[1] = '\0';
        }
        jraw(j, tmp);
        i += len;
    }
    jraw(j, "\"");
}

static void jkey_str(jbuf *j, const char *key, const char *s) {
    jfmt(j, ",\"%s\":", key);
    if (s == NULL)
        jraw(j, "null");
    else
        jstr(j, s, strlen(s));
}

static jbuf json_begin(tllm_console *con, const char *event) {
    jbuf j = {con->json, sizeof con->json, 0};
    con->json[0] = '\0';
    jfmt(&j, "{\"event\":\"%s\"", event);
    return j;
}

static void json_close(jbuf *j) { jraw(j, "}"); }

static void json_end(tllm_console *con, jbuf *j) {
    json_close(j);
    out(con, "@@");
    out(con, con->json);
    out(con, "\n");
}

static void emit_error(tllm_console *con, const char *message) {
    outf(con, "error: %s\n", message);
    jbuf j = json_begin(con, "error");
    jkey_str(&j, "message", message);
    json_end(con, &j);
}

const char *tllm_console_last_json(const tllm_console *con) { return con->json; }

void tllm_console_write(tllm_console *con, const char *text) { out(con, text); }

static uint64_t clock_us(const tllm_console *con) { return con->ctx->opt.clock != NULL ? con->ctx->opt.clock() : 0u; }

/* ------------------------------------------------------------------ init */
tllm_status tllm_console_init(tllm_console *con, tllm_ctx *ctx, tllm_write_fn write, void *user) {
    if (con == NULL || ctx == NULL) return TLLM_ERR_ARG;
    memset(con, 0, sizeof *con);
    con->ctx = ctx;
    con->write = write;
    con->user = user;
    con->max_tokens = 64u;
    con->execute = 1;
    con->seed = 1u;
    con->sampler.repeat_penalty = 1.0f;
    tllm_rng_seed(&con->rng, con->seed);
    tllm_device_default(&con->state);
    const tllm_tokenizer *t = &ctx->model->tok;
    con->bos = t->bos_id;
    con->eos = t->eos_id;
    if (con->bos < 0 || con->eos < 0) return TLLM_ERR_TOKENIZER;
    struct {
        int32_t *id;
        const char *name;
    } specials[] = {{&con->s_open, "<S>"},        {&con->s_close, "</S>"},
                    {&con->u_open, "<U>"},        {&con->u_close, "</U>"},
                    {&con->a_open, "<A>"},        {&con->a_close, "</A>"},
                    {&con->act_open, "<ACT>"},    {&con->act_close, "</ACT>"},
                    {&con->clarify, "<clarify>"}, {&con->unsupported, "<unsupported>"}};
    con->chat = 1;
    for (size_t i = 0; i < sizeof specials / sizeof specials[0]; ++i) {
        *specials[i].id = tllm_special_id(t, specials[i].name);
        if (*specials[i].id < 0) con->chat = 0;
    }
    return TLLM_OK;
}

/* ------------------------------------------------------------------ prompt building */
static int append_text(const tllm_tokenizer *t, const char *text, int32_t *buf, uint32_t *n, uint32_t cap) {
    int got = tllm_tokenize(t, text, strlen(text), buf + *n, (int)(cap - *n));
    if (got < 0) return -1;
    *n += (uint32_t)got;
    return 0;
}

static int append_id(int32_t id, int32_t *buf, uint32_t *n, uint32_t cap) {
    if (*n >= cap) return -1;
    buf[(*n)++] = id;
    return 0;
}

/* Python's _seg(): text following a special token gets one leading space. */
static void seg(const char *text, char *dst, size_t cap) {
    snprintf(dst, cap, "%s%s", text[0] == '<' ? "" : " ", text);
}

static int build_prompt(tllm_console *con, const char *user, uint32_t *n_prompt, uint32_t *user_start) {
    const tllm_tokenizer *t = &con->ctx->model->tok;
    const uint32_t cap = con->ctx->model->cfg.ctx_len;
    char state_text[TLLM_STATE_TEXT_MAX];
    char body[TLLM_STATE_TEXT_MAX];
    char user_seg[TLLM_CONSOLE_LINE_MAX + 2];
    int32_t tail[TLLM_MAX_CTX];
    uint32_t n_tail = 0;
    int len = tllm_device_render(&con->state, state_text, sizeof state_text);
    /* inner text between "<S>" and "</S>" */
    memcpy(body, state_text + 3, (size_t)len - 7u);
    body[len - 7] = '\0';
    seg(user, user_seg, sizeof user_seg);
    uint32_t tail_cap = cap < TLLM_MAX_CTX ? cap : TLLM_MAX_CTX;
    if (append_id(con->s_open, tail, &n_tail, tail_cap) || append_text(t, body, tail, &n_tail, tail_cap) ||
        append_id(con->s_close, tail, &n_tail, tail_cap))
        return -1;
    uint32_t u_at = n_tail;
    if (append_id(con->u_open, tail, &n_tail, tail_cap) || append_text(t, user_seg, tail, &n_tail, tail_cap) ||
        append_id(con->u_close, tail, &n_tail, tail_cap) || append_id(con->a_open, tail, &n_tail, tail_cap))
        return -1;
    /* drop the oldest exchanges until prompt + reply reserve fits */
    uint32_t reserve = con->max_tokens < REPLY_RESERVE ? con->max_tokens : REPLY_RESERVE;
    uint32_t first = 0;
    while (1u + (con->hist_len - (first < con->n_turns ? con->turn_start[first] : con->hist_len)) + n_tail + reserve >
               cap &&
           first < con->n_turns)
        ++first;
    uint32_t hist_from = first < con->n_turns ? con->turn_start[first] : con->hist_len;
    uint32_t n = 0;
    if (1u + (con->hist_len - hist_from) + n_tail > cap) return -1;
    con->prompt[n++] = con->bos;
    memcpy(con->prompt + n, con->hist + hist_from, sizeof(int32_t) * (con->hist_len - hist_from));
    n += con->hist_len - hist_from;
    *user_start = n + u_at;
    memcpy(con->prompt + n, tail, sizeof(int32_t) * n_tail);
    n += n_tail;
    /* forget trimmed exchanges for good */
    if (first > 0u) {
        memmove(con->hist, con->hist + hist_from, sizeof(int32_t) * (con->hist_len - hist_from));
        con->hist_len -= hist_from;
        for (uint32_t i = first; i < con->n_turns; ++i) con->turn_start[i - first] = con->turn_start[i] - hist_from;
        con->n_turns -= first;
        /* the prefix changed: the KV cache can't be reused beyond <bos> */
    }
    *n_prompt = n;
    return 0;
}

static void remember_exchange(tllm_console *con, uint32_t user_start, uint32_t n_prompt) {
    /* exchange = prompt[user_start .. n_prompt) (i.e. "<U> ..</U><A>") + generated tokens w/o <eos> */
    uint32_t n_gen = con->n_gen;
    while (n_gen > 0u && con->gen[n_gen - 1u] == con->eos) --n_gen;
    int has_close = 0;
    for (uint32_t i = 0; i < n_gen; ++i)
        if (con->gen[i] == con->a_close) has_close = 1;
    uint32_t need = (n_prompt - user_start) + n_gen + (has_close ? 0u : 1u);
    if (need > TLLM_MAX_CTX) return;
    while ((con->hist_len + need > TLLM_MAX_CTX || con->n_turns == TLLM_CONSOLE_MAX_TURNS) && con->n_turns > 0u) {
        uint32_t cut = con->n_turns > 1u ? con->turn_start[1] : con->hist_len;
        memmove(con->hist, con->hist + cut, sizeof(int32_t) * (con->hist_len - cut));
        con->hist_len -= cut;
        for (uint32_t i = 1; i < con->n_turns; ++i) con->turn_start[i - 1u] = con->turn_start[i] - cut;
        con->n_turns--;
    }
    con->turn_start[con->n_turns++] = con->hist_len;
    memcpy(con->hist + con->hist_len, con->prompt + user_start, sizeof(int32_t) * (n_prompt - user_start));
    con->hist_len += n_prompt - user_start;
    memcpy(con->hist + con->hist_len, con->gen, sizeof(int32_t) * n_gen);
    con->hist_len += n_gen;
    if (!has_close) con->hist[con->hist_len++] = con->a_close;
}

/* Detokenise gen[from..to) skipping special tokens into out; returns length. */
static size_t detok(const tllm_console *con, uint32_t from, uint32_t to, char *outbuf, size_t cap) {
    const tllm_tokenizer *t = &con->ctx->model->tok;
    size_t len = 0;
    for (uint32_t i = from; i < to && len + 1u < cap; ++i) {
        if ((uint32_t)con->gen[i] < t->n_special) continue;
        int n = tllm_token_bytes(t, con->gen[i], outbuf + len, (int)(cap - len - 1u));
        if (n < 0) break;
        len += (size_t)n;
    }
    outbuf[len] = '\0';
    return len;
}

static char *trim(char *s) {
    while (*s == ' ') ++s;
    size_t n = strlen(s);
    while (n > 0u && s[n - 1u] == ' ') s[--n] = '\0';
    return s;
}

/* ------------------------------------------------------------------ chat */
typedef struct {
    uint32_t prompt_tokens, reused, gen_tokens;
    uint64_t prefill_us, decode_us, first_token_us;
} run_stats;

static int generate(tllm_console *con, const char *user, run_stats *rs, int stream) {
    tllm_ctx *ctx = con->ctx;
    uint32_t n_prompt = 0, user_start = 0;
    uint64_t tok0 = clock_us(con);
    if (build_prompt(con, user, &n_prompt, &user_start) != 0) return -1;
    if (ctx->profiling) ctx->prof_us[TLLM_PROF_TOKENIZER] += clock_us(con) - tok0;
    uint32_t common = 0;
    while (common < ctx->n_cached && common < n_prompt && ctx->tokens[common] == con->prompt[common]) ++common;
    rs->prompt_tokens = n_prompt;
    rs->reused = common < n_prompt ? common : n_prompt - 1u;
    const float *logits = NULL;
    uint64_t t0 = clock_us(con);
    if (tllm_prefill(ctx, con->prompt, n_prompt, &logits) != TLLM_OK) return -1;
    rs->prefill_us = clock_us(con) - t0;
    const uint32_t vocab = ctx->model->cfg.vocab_size;
    con->n_gen = 0;
    int in_action = 0, after_reply = 0;
    uint32_t action_tokens = 0;
    uint64_t decode_start = clock_us(con);
    uint64_t last = decode_start;
    while (con->n_gen < con->max_tokens) {
        uint64_t s0 = clock_us(con);
        int32_t tok = tllm_sample(ctx->logits, vocab, &con->sampler, &con->rng, con->gen, con->n_gen);
        if (ctx->profiling && ctx->opt.clock != NULL) ctx->prof_us[TLLM_PROF_SAMPLING] += clock_us(con) - s0;
        con->gen[con->n_gen++] = tok;
        uint64_t now_us = clock_us(con);
        con->token_us[con->n_gen - 1u] = (uint32_t)(now_us - last);
        if (con->n_gen == 1u) rs->first_token_us = rs->prefill_us + (now_us - decode_start);
        last = now_us;
        if (tok == con->eos) break;
        if (after_reply && !in_action && tok != con->act_open) break; /* malformed: stop */
        if (tok == con->a_close) after_reply = 1;
        if (tok == con->act_open) in_action = 1;
        if (in_action && ++action_tokens > ACTION_MAX_TOKENS) break;
        if (tok == con->act_close) in_action = 0;
        if (stream && !after_reply && (uint32_t)tok >= ctx->model->tok.n_special) {
            char piece[64];
            int n = tllm_token_bytes(&ctx->model->tok, tok, piece, (int)sizeof piece - 1);
            if (n > 0 && con->write != NULL) con->write(con->user, piece, (size_t)n);
        }
        if (tllm_forward(ctx, tok, &logits) != TLLM_OK) break; /* context full */
    }
    rs->decode_us = clock_us(con) - decode_start;
    rs->gen_tokens = con->n_gen;
    remember_exchange(con, user_start, n_prompt);
    return 0;
}

static void chat(tllm_console *con, const char *line, int stream, const char *event, int emit) {
    char user[TLLM_CONSOLE_LINE_MAX + 1];
    size_t n = 0;
    for (const char *p = line; *p != '\0' && n < TLLM_CONSOLE_LINE_MAX; ++p) {
        char c = *p;
        if (c == '\r' || c == '\n') continue;
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a'); /* the model was trained on lowercase */
        user[n++] = c;
    }
    user[n] = '\0';
    const char *text = trim(user);
    if (*text == '\0') {
        emit_error(con, "empty input");
        return;
    }
    run_stats rs;
    memset(&rs, 0, sizeof rs);
    if (con->ctx->profiling) tllm_profile_reset(con->ctx);
    if (generate(con, text, &rs, stream) != 0) {
        emit_error(con, "input too long for the context window");
        return;
    }
    /* split reply / action */
    uint32_t reply_end = con->n_gen, act_from = 0, act_to = 0;
    const char *fallback = NULL;
    for (uint32_t i = 0; i < con->n_gen; ++i) {
        int32_t tok = con->gen[i];
        if (tok == con->a_close && reply_end == con->n_gen) reply_end = i;
        if (tok == con->clarify && i < reply_end && fallback == NULL) fallback = "clarify";
        if (tok == con->unsupported && i < reply_end && fallback == NULL) fallback = "unsupported";
        if (tok == con->act_open && act_from == 0u) act_from = i + 1u;
        if (tok == con->act_close && act_from != 0u && act_to == 0u) act_to = i;
    }
    char reply_buf[1024], action_buf[256], state_text[TLLM_STATE_TEXT_MAX];
    detok(con, 0, reply_end, reply_buf, sizeof reply_buf);
    const char *reply = trim(reply_buf);
    const char *action = NULL;
    tllm_verdict verdict = TLLM_VERDICT_OK;
    int have_action = act_from != 0u;
    if (have_action) {
        if (act_to == 0u) act_to = con->n_gen;
        detok(con, act_from, act_to, action_buf, sizeof action_buf);
        action = trim(action_buf);
        tllm_action parsed;
        if (tllm_action_parse(action, strlen(action), &parsed) != 0) {
            verdict = TLLM_VERDICT_MALFORMED;
        } else {
            verdict = tllm_device_validate(&con->state, &parsed);
            if (verdict == TLLM_VERDICT_OK && con->execute) tllm_device_apply(&con->state, &parsed);
        }
    }
    if (stream) {
        out(con, "\n");
        if (have_action)
            outf(con, "[action] %s -> %s%s%s\n", action, tllm_verdict_code(verdict),
                 verdict == TLLM_VERDICT_OK ? "" : ": ",
                 verdict == TLLM_VERDICT_OK ? "" : tllm_verdict_message(verdict));
    }
    tllm_device_render(&con->state, state_text, sizeof state_text);
    jbuf j = json_begin(con, event);
    jkey_str(&j, "text", reply);
    jkey_str(&j, "fallback", fallback);
    jkey_str(&j, "action", action);
    jkey_str(&j, "verdict", have_action ? tllm_verdict_code(verdict) : NULL);
    jkey_str(&j, "message", have_action ? tllm_verdict_message(verdict) : NULL);
    jkey_str(&j, "state", state_text);
    double decode_s = (double)rs.decode_us / 1e6;
    uint32_t decoded = rs.gen_tokens > 0u ? rs.gen_tokens - 1u : 0u; /* the first token comes from prefill */
    jfmt(&j, ",\"prompt_tokens\":%u,\"reused_tokens\":%u,\"gen_tokens\":%u", rs.prompt_tokens, rs.reused,
         rs.gen_tokens);
    jfmt(&j, ",\"prefill_ms\":%.3f,\"first_token_ms\":%.3f,\"decode_ms\":%.3f,\"decode_tok_s\":%.2f",
         (double)rs.prefill_us / 1e3, (double)rs.first_token_us / 1e3, (double)rs.decode_us / 1e3,
         decode_s > 0.0 ? (double)decoded / decode_s : 0.0);
    if (con->ctx->profiling) {
        jraw(&j, ",\"profile_us\":{");
        for (int s = 0; s < TLLM_PROF_COUNT; ++s)
            jfmt(&j, "%s\"%s\":%llu", s ? "," : "", tllm_prof_stage_name((tllm_prof_stage)s),
                 (unsigned long long)con->ctx->prof_us[s]);
        jraw(&j, "}");
    }
    if (emit)
        json_end(con, &j);
    else
        json_close(&j);
}

/* ------------------------------------------------------------------ story mode */
typedef struct {
    char *buf;
    size_t cap, len;
    int stream;
} text_sink;

/* Emit one token's bytes with llama2.c's rule: a leading space right after BOS is dropped. */
static void emit_token(tllm_console *con, text_sink *sink, int32_t prev, int32_t tok) {
    char piece[TLLM_MAX_PIECE + 1];
    int n = tllm_token_bytes(&con->ctx->model->tok, tok, piece, (int)sizeof piece);
    if (n <= 0) return;
    const char *p = piece;
    if (prev == con->bos && p[0] == ' ') {
        ++p;
        --n;
    }
    if (sink->stream && con->write != NULL && n > 0) con->write(con->user, p, (size_t)n);
    size_t room = sink->cap - sink->len - 1u;
    size_t take_n = (size_t)n < room ? (size_t)n : room;
    memcpy(sink->buf + sink->len, p, take_n);
    sink->len += take_n;
    sink->buf[sink->len] = '\0';
}

/* Continue `prompt` (llama2.c run.c semantics): fresh context, BOS + prompt, then sample. */
static void story(tllm_console *con, const char *prompt, int stream, const char *event, int emit) {
    tllm_ctx *ctx = con->ctx;
    const tllm_tokenizer *t = &ctx->model->tok;
    const uint32_t cap = ctx->model->cfg.ctx_len < TLLM_MAX_CTX ? ctx->model->cfg.ctx_len : TLLM_MAX_CTX;
    char text[TLLM_CONSOLE_LINE_MAX + 2], story_text[1024] = {0};
    if (t->kind == TLLM_TOK_BPE && prompt[0] != '\0')
        seg(prompt, text, sizeof text);
    else
        (void)snprintf(text, sizeof text, "%s", prompt);
    uint32_t n = 0;
    con->prompt[n++] = con->bos;
    uint64_t tok0 = clock_us(con);
    int got = tllm_tokenize(t, text, strlen(text), con->prompt + 1, (int)cap - 1);
    if (got < 0) {
        emit_error(con, "input too long for the context window");
        return;
    }
    n += (uint32_t)got;
    if (ctx->profiling) {
        tllm_profile_reset(ctx);
        ctx->prof_us[TLLM_PROF_TOKENIZER] = clock_us(con) - tok0;
    }
    tllm_ctx_reset(ctx);
    text_sink sink = {story_text, sizeof story_text, 0, stream};
    story_text[0] = '\0';
    for (uint32_t i = 1; i < n; ++i) emit_token(con, &sink, con->prompt[i - 1], con->prompt[i]);
    const float *logits = NULL;
    uint64_t t0 = clock_us(con);
    (void)tllm_prefill(ctx, con->prompt, n, &logits);
    uint64_t prefill_us = clock_us(con) - t0, first_us = 0;
    uint64_t decode_start = clock_us(con);
    int32_t prev = con->prompt[n - 1u];
    con->n_gen = 0;
    while (con->n_gen < con->max_tokens) {
        int32_t tok =
            tllm_sample(ctx->logits, ctx->model->cfg.vocab_size, &con->sampler, &con->rng, con->gen, con->n_gen);
        if (con->n_gen == 0u) first_us = prefill_us + (clock_us(con) - decode_start);
        if (tok == con->eos || tok == con->bos) break;
        con->gen[con->n_gen++] = tok;
        emit_token(con, &sink, prev, tok);
        prev = tok;
        if (tllm_forward(ctx, tok, &logits) != TLLM_OK) break; /* context full */
    }
    uint64_t decode_us = clock_us(con) - decode_start;
    if (stream) out(con, "\n");
    jbuf j = json_begin(con, event);
    jkey_str(&j, "text", story_text);
    double decode_s = (double)decode_us / 1e6;
    jfmt(&j, ",\"prompt_tokens\":%u,\"reused_tokens\":0,\"gen_tokens\":%u", n, con->n_gen);
    jfmt(&j, ",\"prefill_ms\":%.3f,\"first_token_ms\":%.3f,\"decode_ms\":%.3f,\"decode_tok_s\":%.2f",
         (double)prefill_us / 1e3, (double)first_us / 1e3, (double)decode_us / 1e3,
         decode_s > 0.0 ? (double)con->n_gen / decode_s : 0.0);
    if (emit)
        json_end(con, &j);
    else
        json_close(&j);
    tllm_ctx_reset(ctx);
}

/* ------------------------------------------------------------------ commands */
static void cmd_model_info(tllm_console *con) {
    const tllm_model *m = con->ctx->model;
    const tllm_config *c = &m->cfg;
    char id[33];
    for (int i = 0; i < 16; ++i) snprintf(id + 2 * i, 3, "%02x", m->model_id[i]);
    uint64_t params = (uint64_t)c->vocab_size * c->d_model + c->d_model;
    uint32_t kvd = c->n_kv_heads * (c->d_model / c->n_heads);
    params += (uint64_t)c->n_layers * (2ull * c->d_model * c->d_model + 2ull * c->d_model * kvd +
                                       (c->mlp_type ? 3ull : 2ull) * c->d_model * c->d_ff + 2ull * c->d_model);
    if (c->pos_type == TLLM_POS_LEARNED) params += (uint64_t)c->ctx_len * c->d_model;
    outf(con, "model %s: %u layers x %u, heads %u/%u, ff %u, vocab %u, ctx %u, %s, %s, %s weights, %llu params\n", id,
         c->n_layers, c->d_model, c->n_heads, c->n_kv_heads, c->d_ff, c->vocab_size, c->ctx_len,
         c->mlp_type ? "swiglu" : "gelu", c->pos_type ? "rope" : "learned", c->weight_dtype ? "int8" : "fp32",
         (unsigned long long)params);
    jbuf j = json_begin(con, "model-info");
    jfmt(&j, ",\"model_id\":\"%s\",\"crc32\":\"%08x\",\"bytes\":%lu", id, m->crc32, (unsigned long)m->size);
    jfmt(&j, ",\"n_layers\":%u,\"d_model\":%u,\"n_heads\":%u,\"n_kv_heads\":%u,\"d_ff\":%u", c->n_layers, c->d_model,
         c->n_heads, c->n_kv_heads, c->d_ff);
    jfmt(&j, ",\"vocab_size\":%u,\"ctx_len\":%u,\"params\":%llu", c->vocab_size, c->ctx_len,
         (unsigned long long)params);
    jkey_str(&j, "mlp", c->mlp_type ? "swiglu" : "gelu");
    jkey_str(&j, "pos", c->pos_type ? "rope" : "learned");
    jkey_str(&j, "weights", c->weight_dtype ? "int8" : "fp32");
    jkey_str(&j, "activations", con->ctx->opt.act_mode == TLLM_ACT_I8 ? "int8" : "fp32");
    jkey_str(&j, "kv_cache", con->ctx->key_i8 != NULL ? "int8" : "fp32");
    json_end(con, &j);
}

static void cmd_memory(tllm_console *con) {
    const tllm_model *m = con->ctx->model;
    size_t hot = tllm_hot_arena_size(m), cold = tllm_cold_arena_size(m, con->ctx->key_i8 != NULL);
    outf(con, "model blob %lu B, hot arena %lu B, cold arena (KV cache) %lu B, console %lu B\n", (unsigned long)m->size,
         (unsigned long)hot, (unsigned long)cold, (unsigned long)sizeof(tllm_console));
    char extra[256] = "";
    if (con->memory != NULL && con->memory(con->user, extra, sizeof extra) > 0) outf(con, "%s\n", extra);
    jbuf j = json_begin(con, "memory");
    jfmt(&j, ",\"model_bytes\":%lu,\"hot_arena\":%lu,\"cold_arena\":%lu,\"console\":%lu", (unsigned long)m->size,
         (unsigned long)hot, (unsigned long)cold, (unsigned long)sizeof(tllm_console));
    jkey_str(&j, "platform", extra[0] ? extra : NULL);
    json_end(con, &j);
}

static void ok(tllm_console *con, const char *what) {
    outf(con, "ok: %s\n", what);
    jbuf j = json_begin(con, "ok");
    jkey_str(&j, "what", what);
    json_end(con, &j);
}

static void cmd_state(tllm_console *con) {
    char text[TLLM_STATE_TEXT_MAX];
    tllm_device_render(&con->state, text, sizeof text);
    outf(con, "%s\n", text);
    jbuf j = json_begin(con, "state");
    jkey_str(&j, "state", text);
    json_end(con, &j);
}

static void cmd_checksums(tllm_console *con) {
    const tllm_config *c = &con->ctx->model->cfg;
    jbuf j = json_begin(con, "checksums");
    jraw(&j, ",\"layers\":[");
    for (uint32_t l = 0; l <= c->n_layers; ++l) {
        outf(con, "%s %u: %.6f\n", l < c->n_layers ? "layer" : "final", l, con->ctx->checksums[l]);
        jfmt(&j, "%s%.6f", l ? "," : "", con->ctx->checksums[l]);
    }
    jraw(&j, "]");
    json_end(con, &j);
}

/* Token ids of TEXT plus a round-trip check (vision §7: same results in Python and on the chip). */
static void cmd_tokenize(tllm_console *con, const char *text) {
    const tllm_tokenizer *t = &con->ctx->model->tok;
    int32_t ids[TLLM_CONSOLE_LINE_MAX + 8];
    int n = tllm_tokenize(t, text, strlen(text), ids, (int)(sizeof ids / sizeof ids[0]));
    if (n < 0) {
        emit_error(con, "cannot tokenize");
        return;
    }
    char back[4 * TLLM_CONSOLE_LINE_MAX];
    size_t len = 0;
    for (int i = 0; i < n; ++i) {
        int k = tllm_token_bytes(t, ids[i], back + len, (int)(sizeof back - len - 1u));
        if (k < 0) break;
        len += (size_t)k;
    }
    back[len] = '\0';
    /* scored (llama2.c) tokenizers add a dummy-prefix space */
    const char *decoded = (t->kind == TLLM_TOK_SCORED && back[0] == ' ') ? back + 1 : back;
    int round_trip = strcmp(decoded, text) == 0;
    outf(con, "%d tokens, round trip %s\n", n, round_trip ? "ok" : "differs");
    jbuf j = json_begin(con, "tokens");
    jraw(&j, ",\"ids\":[");
    for (int i = 0; i < n; ++i) jfmt(&j, "%s%d", i ? "," : "", (int)ids[i]);
    jfmt(&j, "],\"round_trip\":%s", round_trip ? "true" : "false");
    json_end(con, &j);
}

static const struct {
    const char *name, *prompt;
} BENCHMARKS[] = {{"read", "what is the temperature"},
                  {"diagnose", "is something wrong"},
                  {"command", "turn on the fan"},
                  {"chat", "hello"}};

static void cmd_benchmark(tllm_console *con, const char *arg) {
    const char *prompt = NULL;
    for (size_t i = 0; i < sizeof BENCHMARKS / sizeof BENCHMARKS[0]; ++i)
        if (strcmp(arg, BENCHMARKS[i].name) == 0) prompt = BENCHMARKS[i].prompt;
    if (prompt == NULL) {
        emit_error(con, "unknown benchmark (read, diagnose, command, chat)");
        return;
    }
    /* deterministic conditions: fresh conversation, fixed state, greedy decoding */
    tllm_device_state saved = con->state;
    tllm_sampler_cfg saved_sampler = con->sampler;
    int saved_exec = con->execute, saved_prof = con->ctx->profiling;
    tllm_device_default(&con->state);
    con->state.t = 31.2f;
    con->state.h = 78;
    con->sampler.temperature = 0.0f;
    con->sampler.repeat_penalty = 1.0f;
    con->execute = 0;
    con->ctx->profiling = 1;
    con->hist_len = con->n_turns = 0;
    tllm_ctx_reset(con->ctx);
    if (con->chat)
        chat(con, prompt, 0, "benchmark", 0);
    else
        story(con, prompt, 0, "benchmark", 0);
    /* token latency statistics (decode tokens only) */
    uint32_t n = con->n_gen > 1u ? con->n_gen - 1u : 0u;
    uint32_t lat[TLLM_MAX_CTX];
    memcpy(lat, con->token_us + 1, sizeof(uint32_t) * n);
    for (uint32_t i = 1; i < n; ++i)
        for (uint32_t k = i; k > 0u && lat[k - 1u] > lat[k]; --k) {
            uint32_t tmp = lat[k];
            lat[k] = lat[k - 1u];
            lat[k - 1u] = tmp;
        }
    uint64_t sum = 0;
    for (uint32_t i = 0; i < n; ++i) sum += lat[i];
    /* append latency stats to the benchmark JSON */
    size_t len = strlen(con->json);
    if (con->json[0] == '{' && len > 0u && con->json[len - 1u] == '}' && strstr(con->json, "\"benchmark\"") != NULL) {
        jbuf j = {con->json, sizeof con->json, len - 1u};
        con->json[len - 1u] = '\0';
        jfmt(&j, ",\"case\":\"%s\",\"mean_token_ms\":%.3f,\"p95_token_ms\":%.3f}", arg, n ? (double)sum / n / 1e3 : 0.0,
             n ? (double)lat[(n * 95u) / 100u < n ? (n * 95u) / 100u : n - 1u] / 1e3 : 0.0);
        out(con, "@@");
        out(con, con->json);
        out(con, "\n");
    }
    con->state = saved;
    con->sampler = saved_sampler;
    con->execute = saved_exec;
    con->ctx->profiling = saved_prof;
    con->hist_len = con->n_turns = 0;
    tllm_ctx_reset(con->ctx);
}

static void cmd_help(tllm_console *con) {
    out(con, "commands: /help /model-info /memory /profile on|off /kv-reset /reset /seed N /temp X /topk N\n"
             "          /max-tokens N /state /set key=value... /execute on|off /checksums /benchmark <case>\n"
             "          /generate TEXT (continue a text, llama2.c style)  /tokenize TEXT\n");
    out(con, con->chat ? "anything else is a chat message.\n" : "anything else is a story prompt.\n");
    ok(con, "help");
}

static int on_off(const char *arg, int *flag) {
    if (strcmp(arg, "on") == 0)
        *flag = 1;
    else if (strcmp(arg, "off") == 0)
        *flag = 0;
    else
        return -1;
    return 0;
}

static void command(tllm_console *con, const char *line) {
    char buf[TLLM_CONSOLE_LINE_MAX + 1];
    snprintf(buf, sizeof buf, "%s", line);
    const char *cmd = buf;
    char *arg = strchr(buf, ' ');
    if (arg != NULL)
        *arg++ = '\0';
    else
        arg = buf + strlen(buf);
    arg = trim(arg);
    char *end = NULL;
    if (strcmp(cmd, "/help") == 0) {
        cmd_help(con);
    } else if (strcmp(cmd, "/model-info") == 0) {
        cmd_model_info(con);
    } else if (strcmp(cmd, "/memory") == 0) {
        cmd_memory(con);
    } else if (strcmp(cmd, "/profile") == 0) {
        if (on_off(arg, &con->ctx->profiling) != 0)
            emit_error(con, "usage: /profile on|off");
        else
            ok(con, con->ctx->profiling ? "profile on" : "profile off");
    } else if (strcmp(cmd, "/execute") == 0) {
        if (on_off(arg, &con->execute) != 0)
            emit_error(con, "usage: /execute on|off");
        else
            ok(con, con->execute ? "execute on" : "execute off");
    } else if (strcmp(cmd, "/kv-reset") == 0) {
        tllm_ctx_reset(con->ctx);
        con->hist_len = con->n_turns = 0;
        ok(con, "conversation cleared");
    } else if (strcmp(cmd, "/reset") == 0) {
        tllm_ctx_reset(con->ctx);
        con->hist_len = con->n_turns = 0;
        tllm_device_default(&con->state);
        ok(con, "conversation and device state reset");
    } else if (strcmp(cmd, "/seed") == 0) {
        unsigned long long v = strtoull(arg, &end, 10);
        if (end == arg || *end != '\0') {
            emit_error(con, "usage: /seed N");
        } else {
            con->seed = v;
            tllm_rng_seed(&con->rng, v);
            ok(con, "seed set");
        }
    } else if (strcmp(cmd, "/temp") == 0) {
        float v = strtof(arg, &end);
        if (end == arg || *end != '\0' || v < 0.0f || v > 5.0f)
            emit_error(con, "usage: /temp X (0 = greedy)");
        else {
            con->sampler.temperature = v;
            ok(con, "temperature set");
        }
    } else if (strcmp(cmd, "/topk") == 0 || strcmp(cmd, "/max-tokens") == 0) {
        long v = strtol(arg, &end, 10);
        int is_topk = strcmp(cmd, "/topk") == 0;
        long lo = is_topk ? 0 : 1, hi = is_topk ? (long)con->ctx->model->cfg.vocab_size : (long)TLLM_MAX_CTX;
        if (end == arg || *end != '\0' || v < lo || v > hi) {
            emit_error(con, is_topk ? "usage: /topk N" : "usage: /max-tokens N");
        } else if (is_topk) {
            con->sampler.top_k = (uint32_t)v;
            ok(con, "top-k set");
        } else {
            con->max_tokens = (uint32_t)v;
            ok(con, "max tokens set");
        }
    } else if (strcmp(cmd, "/state") == 0) {
        cmd_state(con);
    } else if (strcmp(cmd, "/set") == 0) {
        tllm_device_state next = con->state;
        int bad = *arg == '\0';
        for (const char *tok = strtok(arg, " "); tok != NULL && !bad; tok = strtok(NULL, " "))
            bad = tllm_device_set(&next, tok) != 0;
        if (bad) {
            emit_error(con, "usage: /set key=value ... (t h soil fan heat pump light win pa vib err)");
        } else {
            con->state = next;
            cmd_state(con);
        }
    } else if (strcmp(cmd, "/checksums") == 0) {
        cmd_checksums(con);
    } else if (strcmp(cmd, "/tokenize") == 0) {
        cmd_tokenize(con, arg);
    } else if (strcmp(cmd, "/generate") == 0) {
        story(con, arg, 1, "generate", 1);
    } else if (strcmp(cmd, "/benchmark") == 0) {
        cmd_benchmark(con, arg);
    } else if (con->platform_cmd == NULL || !con->platform_cmd(con, line)) {
        emit_error(con, "unknown command, try /help");
    }
}

void tllm_console_line(tllm_console *con, const char *line) {
    while (*line == ' ' || *line == '\t') ++line;
    if (*line == '/')
        command(con, line);
    else if (con->chat)
        chat(con, line, 1, "reply", 1);
    else
        story(con, line, 1, "generate", 1);
}
