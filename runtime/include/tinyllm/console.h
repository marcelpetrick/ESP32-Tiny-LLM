/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Line-oriented command layer shared by the desktop CLI, the web simulator and the
 * ESP32 UART console (docs/05-architecture.md §7). Plain lines are chat turns; lines
 * starting with '/' are commands. Every response ends with one machine-readable line
 * "@@{json}" for the serial runner and the web server.
 *
 * "/generate TEXT" continues TEXT like llama2.c's run.c (story mode).
 *
 * Prompt layout mirrors training/data/textformat.py:
 *   <bos>[<U> user</U><A> reply</A>[<ACT> body</ACT>]]*<S> state</S><U> user</U><A>
 */
#ifndef TINYLLM_CONSOLE_H
#define TINYLLM_CONSOLE_H

#include "tinyllm/device.h"
#include "tinyllm/tinyllm.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TLLM_CONSOLE_MAX_TURNS 16
#define TLLM_CONSOLE_LINE_MAX 256
#define TLLM_CONSOLE_JSON_MAX 8192 /* worst case: 1 KiB reply fully \u-escaped */

typedef void (*tllm_write_fn)(void *user, const char *text, size_t len);
/* Optional platform memory report appended to /memory (e.g. heap_caps statistics). */
typedef int (*tllm_memory_fn)(void *user, char *out, size_t cap);
struct tllm_console_s;
/* Optional platform commands (e.g. "/bandwidth" on the ESP32). Return 1 if handled. */
typedef int (*tllm_platform_cmd_fn)(struct tllm_console_s *con, const char *line);

typedef struct tllm_console_s {
    tllm_ctx *ctx;
    tllm_device_state state;
    tllm_sampler_cfg sampler;
    tllm_rng rng;
    uint64_t seed;
    uint32_t max_tokens;
    int profile;
    int execute; /* apply approved actions to the (simulated) device state */
    int chat;    /* model has the chat control tokens; otherwise plain lines run /generate */
    tllm_write_fn write;
    tllm_memory_fn memory;
    tllm_platform_cmd_fn platform_cmd;
    void *user;
    /* conversation history: completed exchanges as tokens */
    int32_t hist[TLLM_MAX_CTX];
    uint32_t hist_len;
    uint32_t turn_start[TLLM_CONSOLE_MAX_TURNS];
    uint32_t n_turns;
    /* scratch */
    int32_t prompt[TLLM_MAX_CTX];
    int32_t gen[TLLM_MAX_CTX];
    uint32_t token_us[TLLM_MAX_CTX];
    uint32_t n_gen;
    char json[TLLM_CONSOLE_JSON_MAX];
    /* special token ids */
    int32_t bos, eos, s_open, s_close, u_open, u_close, a_open, a_close, act_open, act_close, clarify, unsupported;
    /* optional fact retrieval (vision §24 D): -1 when the model has no <F>/</F> tokens */
    int32_t f_open, f_close;
    int fact; /* fact injected into the last prompt (tllm_fact_* index) or -1 */
} tllm_console;

/* Returns TLLM_OK, or TLLM_ERR_TOKENIZER if the model has no BOS/EOS tokens. Models
 * without the chat control tokens (e.g. llama2.c stories) run in story mode. */
tllm_status tllm_console_init(tllm_console *con, tllm_ctx *ctx, tllm_write_fn write, void *user);

/* Handle one input line (without the trailing newline). */
void tllm_console_line(tllm_console *con, const char *line);

/* Write text to the console output (for platform commands). */
void tllm_console_write(tllm_console *con, const char *text);

/* The last "@@" JSON payload (without the "@@" prefix), for hosts that want it directly. */
const char *tllm_console_last_json(const tllm_console *con);

#ifdef __cplusplus
}
#endif
#endif /* TINYLLM_CONSOLE_H */
