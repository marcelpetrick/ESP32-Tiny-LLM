/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * The (simulated) greenhouse device and the firmware-side authority over model output:
 * canonical state rendering, the strict action parser, validation rules, and execution.
 * Mirrors training/world/ exactly; tests/integration/test_device_parity.py checks both.
 *
 * The language model only proposes an action string. Nothing reaches the actuators
 * unless tllm_device_validate() approves it (vision §10, §20).
 */
#ifndef TINYLLM_DEVICE_H
#define TINYLLM_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TLLM_ACTION_MAX_PAIRS 4
#define TLLM_STATE_TEXT_MAX 128

typedef struct {
    float t;     /* air temperature in degrees C */
    int t_valid; /* 0 = sensor offline ("na") */
    int h;       /* relative humidity percent */
    int h_valid;
    int soil, fan, heat, pump, light, win;
    float pa; /* pump current in ampere */
    int vib, err;
} tllm_device_state;

typedef enum {
    TLLM_KEY_FAN = 0,
    TLLM_KEY_HEAT,
    TLLM_KEY_PUMP,
    TLLM_KEY_LIGHT,
    TLLM_KEY_WIN,
    TLLM_KEY_ACK,
    TLLM_KEY_DIAG,
    TLLM_KEY_COUNT
} tllm_action_key;

typedef struct {
    int n;
    tllm_action_key key[TLLM_ACTION_MAX_PAIRS];
    int value[TLLM_ACTION_MAX_PAIRS]; /* integer value, or diagnosis index for DIAG */
} tllm_action;

typedef enum {
    TLLM_VERDICT_OK = 0,
    TLLM_VERDICT_MALFORMED,
    TLLM_VERDICT_FAULT_HEATER,
    TLLM_VERDICT_FAULT_WINDOW,
    TLLM_VERDICT_FAULT_PUMP,
    TLLM_VERDICT_ACK_MISMATCH,
    TLLM_VERDICT_OVERHEAT,
    TLLM_VERDICT_HEATER_WINDOW
} tllm_verdict;

void tllm_device_default(tllm_device_state *s);

/* "<S> t=22.0 h=55 ... err=0</S>"; returns length or -1 if cap is too small. */
int tllm_device_render(const tllm_device_state *s, char *out, size_t cap);

/* Set one field from "key=value" (as used by the console's /set); returns 0 on success. */
int tllm_device_set(tllm_device_state *s, const char *assignment);

/* Parse an action body such as "fan=2 heat=0"; returns 0 on success, -1 if malformed. */
int tllm_action_parse(const char *body, size_t len, tllm_action *out);
int tllm_action_format(const tllm_action *a, char *out, size_t cap);

tllm_verdict tllm_device_validate(const tllm_device_state *s, const tllm_action *a);
void tllm_device_apply(tllm_device_state *s, const tllm_action *a);

const char *tllm_verdict_code(tllm_verdict v);    /* "ok", "overheat", ... (shared with Python) */
const char *tllm_verdict_message(tllm_verdict v); /* human-readable explanation */
const char *tllm_diagnosis_name(int index);       /* NULL if out of range */
int tllm_diagnosis_count(void);

#ifdef __cplusplus
}
#endif
#endif /* TINYLLM_DEVICE_H */
