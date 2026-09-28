/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Console tests with a *scripted* model: every transformer block is zeroed, token
 * embeddings are unit vectors, and the positional embedding at position p is set to
 * ALPHA * E[target(p)], so the logits' argmax is exactly the scripted token.
 */
#include <stdlib.h>

#include "builder.h"
#include "test.h"
#include "tinyllm/console.h"

void test_console(void);

#define ALPHA 100.0f

typedef struct {
    test_blob blob;
    tllm_model model;
    tllm_ctx ctx;
    tllm_console con;
    void *hot, *cold;
    char out[16384];
    size_t out_len;
} cfix;

static void capture(void *user, const char *text, size_t len) {
    cfix *f = (cfix *)user;
    if (f->out_len + len + 1u > sizeof f->out) len = sizeof f->out - f->out_len - 1u;
    memcpy(f->out + f->out_len, text, len);
    f->out_len += len;
    f->out[f->out_len] = '\0';
}

static int mem_report(void *user, char *out, size_t cap) {
    (void)user;
    return snprintf(out, cap, "psram free 123");
}

static uint64_t g_us;
static uint64_t tick(void) { return g_us += 7u; }

static float *emb(cfix *f) { return (float *)(void *)(f->blob.data + f->blob.tok_emb_offset); }
static float *pos(cfix *f) { return (float *)(void *)(f->blob.data + f->blob.pos_emb_offset); }

static void setup(cfix *f, uint32_t ctx_len) {
    memset(f, 0, sizeof *f);
    test_model_spec spec = test_default_spec();
    spec.zero_blocks = 1;
    spec.n_layers = 1;
    spec.d_model = 32;
    spec.n_heads = 2;
    spec.n_kv_heads = 2;
    spec.ctx_len = ctx_len;
    f->blob = test_build_model(&spec);
    CHECK_EQ_INT(tllm_model_load(&f->model, f->blob.data, f->blob.size), TLLM_OK);
    /* normalise embedding rows */
    const uint32_t V = f->model.cfg.vocab_size, d = f->model.cfg.d_model;
    for (uint32_t r = 0; r < V; ++r) {
        float *row = emb(f) + (size_t)r * d, n = 0.0f;
        for (uint32_t c = 0; c < d; ++c) n += row[c] * row[c];
        n = sqrtf(n);
        for (uint32_t c = 0; c < d; ++c) row[c] /= n;
    }
    size_t hs = tllm_hot_arena_size(&f->model), cs = tllm_cold_arena_size(&f->model, 0);
    f->hot = malloc(hs);
    f->cold = malloc(cs);
    tllm_ctx_options opt = {TLLM_ACT_F32, 0, tick};
    CHECK_EQ_INT(tllm_ctx_init(&f->ctx, &f->model, &opt, f->hot, hs, f->cold, cs), TLLM_OK);
    CHECK_EQ_INT(tllm_console_init(&f->con, &f->ctx, capture, f), TLLM_OK);
    f->con.memory = mem_report;
}

static void teardown(cfix *f) {
    free(f->hot);
    free(f->cold);
    test_free_blob(&f->blob);
}

static const char *run(cfix *f, const char *line) {
    f->out_len = 0;
    f->out[0] = '\0';
    tllm_console_line(&f->con, line);
    return f->out;
}

static long json_int(const char *json, const char *key) {
    char pat[64];
    snprintf(pat, sizeof pat, "\"%s\":", key);
    const char *p = strstr(json, pat);
    return p ? strtol(p + strlen(pat), NULL, 10) : -1;
}

/* Chat so that the model emits exactly `script` (ids). */
static const char *scripted(cfix *f, const char *line, const int32_t *script, int n) {
    tllm_console *saved = malloc(sizeof *saved);
    memcpy(saved, &f->con, sizeof *saved);
    run(f, line); /* dry run: learn the prompt length */
    long L = json_int(tllm_console_last_json(&f->con), "prompt_tokens");
    memcpy(&f->con, saved, sizeof *saved);
    free(saved);
    tllm_ctx_reset(&f->ctx);
    const uint32_t d = f->model.cfg.d_model, ctx_len = f->model.cfg.ctx_len;
    memset(pos(f), 0, sizeof(float) * d * ctx_len);
    for (int i = 0; i < n && L - 1 + i < (long)ctx_len; ++i) {
        float *p = pos(f) + (size_t)(L - 1 + i) * d;
        const float *e = emb(f) + (size_t)script[i] * d;
        for (uint32_t c = 0; c < d; ++c) p[c] = ALPHA * e[c];
    }
    return run(f, line);
}

#define B(ch) (TEST_N_SPECIAL + (int32_t)(ch))
enum { PAD, BOS, EOS, S_OPEN, S_CLOSE, U_OPEN, U_CLOSE, A_OPEN, A_CLOSE, ACT, ACT_CLOSE, CLARIFY, UNSUPPORTED };

void test_console(void) {
    cfix *f = malloc(sizeof *f);
    setup(f, 128);
    const char *out;

    /* approved action is executed */
    const int32_t ok_script[] = {B(' '), B('o'), B('k'), A_CLOSE, ACT,       B(' '), B('f'),
                                 B('a'), B('n'), B('='), B('2'),  ACT_CLOSE, EOS};
    out = scripted(f, "Turn ON the fan\r", ok_script, 13);
    CHECK_CONTAINS(out, " ok\n[action] fan=2 -> ok\n");
    CHECK_CONTAINS(out, "\"text\":\"ok\",\"fallback\":null,\"action\":\"fan=2\",\"verdict\":\"ok\"");
    CHECK_CONTAINS(out, "fan=2 heat=0");
    CHECK_EQ_INT(f->con.state.fan, 2);
    CHECK_EQ_INT(f->con.n_turns, 1);

    /* interlock rejection is reported and not executed */
    run(f, "/set t=36");
    const int32_t heat_script[] = {B('x'), A_CLOSE, ACT,    B('h'),    B('e'), B('a'),
                                   B('t'), B('='),  B('1'), ACT_CLOSE, EOS};
    out = scripted(f, "heat please", heat_script, 11);
    CHECK_CONTAINS(out, "[action] heat=1 -> overheat: the heater is not allowed above 35 degrees.");
    CHECK_EQ_INT(f->con.state.heat, 0);

    /* malformed action body */
    const int32_t bad_script[] = {B('x'), A_CLOSE, ACT, B('f'), B('a'), B('n'), B('='), B('9'), ACT_CLOSE, EOS};
    out = scripted(f, "fan nine", bad_script, 10);
    CHECK_CONTAINS(out, "\"verdict\":\"malformed\"");

    /* clarification fallback, quotes and control characters are JSON-escaped */
    const int32_t clarify_script[] = {CLARIFY, B(' '), B('"'), B('\\'), B('\t'), A_CLOSE, EOS};
    out = scripted(f, "turn it on", clarify_script, 7);
    CHECK_CONTAINS(out, "\"fallback\":\"clarify\"");
    CHECK_CONTAINS(out, "\\\"\\\\\\u0009");
    /* UTF-8: valid multi-byte passes through, invalid bytes become U+FFFD */
    const int32_t utf8_script[] = {B(0xC3), B(0xA9), B(0xFF), B(0xE2), B('x'), B(0xF0), A_CLOSE, EOS};
    out = scripted(f, "utf8", utf8_script, 8);
    CHECK_CONTAINS(out, "\"text\":\"\xc3\xa9\\ufffd\\ufffdx\\ufffd\"");
    const int32_t unsupported_script[] = {UNSUPPORTED, B(' '), B('n'), A_CLOSE, EOS};
    out = scripted(f, "sing", unsupported_script, 5);
    CHECK_CONTAINS(out, "\"fallback\":\"unsupported\"");

    /* malformed continuation after </A> stops generation */
    const int32_t junk_script[] = {B('a'), A_CLOSE, B('z'), B('z'), EOS};
    out = scripted(f, "junk", junk_script, 5);
    CHECK_EQ_INT(json_int(tllm_console_last_json(&f->con), "gen_tokens"), 3);

    /* an overlong action block is cut off */
    int32_t long_act[40];
    long_act[0] = A_CLOSE;
    long_act[1] = ACT;
    for (int i = 2; i < 40; ++i) long_act[i] = B('f');
    run(f, "/max-tokens 60");
    out = scripted(f, "long action", long_act, 40);
    CHECK_EQ_INT(json_int(tllm_console_last_json(&f->con), "gen_tokens"), 26); /* </A>, <ACT> + 23 more, then cut */

    /* reply without </A> hits max tokens; history still gets a closing tag */
    run(f, "/max-tokens 3");
    const int32_t open_script[] = {B('a'), B('b'), B('c'), B('d')};
    out = scripted(f, "no close", open_script, 4);
    CHECK_EQ_INT(f->con.hist[f->con.hist_len - 1u], A_CLOSE);
    run(f, "/max-tokens 64");

    /* commands */
    CHECK_CONTAINS(run(f, "/help"), "commands:");
    CHECK_CONTAINS(run(f, "/model-info"), "\"event\":\"model-info\"");
    CHECK_CONTAINS(f->out, "\"weights\":\"fp32\"");
    CHECK_CONTAINS(run(f, "/memory"), "psram free 123");
    CHECK_CONTAINS(run(f, "/profile on"), "profile on");
    CHECK_CONTAINS(scripted(f, "profiled", ok_script, 13), "\"profile_us\":{\"tokenizer\":");
    CHECK_CONTAINS(run(f, "/profile off"), "profile off");
    CHECK_CONTAINS(run(f, "/profile maybe"), "usage: /profile");
    CHECK_CONTAINS(run(f, "/execute off"), "execute off");
    CHECK_CONTAINS(run(f, "/execute sometimes"), "usage: /execute");
    run(f, "/execute on");
    CHECK_CONTAINS(run(f, "/seed 99"), "seed set");
    CHECK_CONTAINS(run(f, "/seed x"), "usage: /seed");
    CHECK_CONTAINS(run(f, "/temp 0.7"), "temperature set");
    CHECK_CONTAINS(run(f, "/temp 9"), "usage: /temp");
    CHECK_CONTAINS(run(f, "/topk 5"), "top-k set");
    CHECK_CONTAINS(run(f, "/topk -1"), "usage: /topk");
    CHECK_CONTAINS(run(f, "/max-tokens 0"), "usage: /max-tokens");
    CHECK_CONTAINS(run(f, "/max-tokens 32"), "max tokens set");
    CHECK_CONTAINS(run(f, "hello there"), "\"event\":\"reply\""); /* sampled path */
    run(f, "/temp 0");
    CHECK_CONTAINS(run(f, "/state"), "\"event\":\"state\"");
    CHECK_CONTAINS(run(f, "/set h=na soil=12"), "h=na soil=12");
    CHECK_CONTAINS(run(f, "/set fan=9"), "usage: /set");
    CHECK_CONTAINS(run(f, "/set"), "usage: /set");
    CHECK_CONTAINS(run(f, "/checksums"), "\"event\":\"checksums\",\"layers\":[");
    CHECK_CONTAINS(run(f, "/benchmark read"), "\"case\":\"read\",\"mean_token_ms\":");
    CHECK_CONTAINS(run(f, "/benchmark nope"), "unknown benchmark");
    CHECK_CONTAINS(run(f, "/frobnicate"), "unknown command");
    CHECK_CONTAINS(run(f, "   "), "empty input");
    CHECK_CONTAINS(run(f, "/kv-reset"), "conversation cleared");
    CHECK_EQ_INT(f->con.n_turns, 0);
    CHECK_CONTAINS(run(f, "/reset"), "device state reset");
    CHECK_EQ_INT(f->con.state.soil, 50);

    /* input longer than the context window */
    char longline[TLLM_CONSOLE_LINE_MAX + 1];
    for (int i = 0; i < TLLM_CONSOLE_LINE_MAX; ++i) longline[i] = (char)(i % 2 ? '1' : '2');
    longline[TLLM_CONSOLE_LINE_MAX] = '\0';
    CHECK_CONTAINS(run(f, longline), "input too long");

    /* history is trimmed to fit the context */
    run(f, "/max-tokens 4");
    for (int i = 0; i < 12; ++i) run(f, "what is the temperature now please");
    CHECK(f->con.n_turns < 12u);
    CHECK(json_int(tllm_console_last_json(&f->con), "prompt_tokens") <= 128);
    teardown(f);

    /* a big context hits the turn limit instead */
    setup(f, 1024);
    run(f, "/max-tokens 2");
    for (int i = 0; i < TLLM_CONSOLE_MAX_TURNS + 4; ++i) run(f, "hi");
    CHECK_EQ_INT(f->con.n_turns, TLLM_CONSOLE_MAX_TURNS);
    teardown(f);

    /* models without chat tokens cannot drive the console */
    test_model_spec spec = test_default_spec();
    spec.omit_chat_specials = 1;
    test_blob b = test_build_model(&spec);
    tllm_model m;
    CHECK_EQ_INT(tllm_model_load(&m, b.data, b.size), TLLM_OK);
    size_t hs = tllm_hot_arena_size(&m), cs = tllm_cold_arena_size(&m, 0);
    void *hot = malloc(hs), *cold = malloc(cs);
    tllm_ctx ctx;
    CHECK_EQ_INT(tllm_ctx_init(&ctx, &m, NULL, hot, hs, cold, cs), TLLM_OK);
    CHECK_EQ_INT(tllm_console_init(&f->con, &ctx, NULL, NULL), TLLM_ERR_TOKENIZER);
    CHECK_EQ_INT(tllm_console_init(NULL, &ctx, NULL, NULL), TLLM_ERR_ARG);
    free(hot);
    free(cold);
    test_free_blob(&b);
    free(f);
}
