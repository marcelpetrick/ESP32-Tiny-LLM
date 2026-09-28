/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Greenhouse device world — the deterministic authority layer. Rules, ordering and
 * messages match training/world/actions.py one-to-one.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinyllm/device.h"

#define OVERHEAT_LIMIT_C 35.0f
#define PUMP_NOMINAL_A 1.2f

static const char *const KEY_NAMES[TLLM_KEY_COUNT] = {"fan", "heat", "pump", "light", "win", "ack", "diag"};
/* {min, max, step} for integer keys (index = tllm_action_key) */
static const int RANGES[TLLM_KEY_DIAG][3] = {{0, 3, 1}, {0, 1, 1}, {0, 1, 1}, {0, 100, 10}, {0, 1, 1}, {1, 99, 1}};
static const char *const DIAGNOSES[] = {"pump_blocked", "pump_dry",       "heater_fault", "window_fault",
                                        "pump_fault",   "sensor_offline", "fan_bearing",  "too_hot",
                                        "too_cold",     "too_humid",      "soil_dry",     "soil_wet"};
#define N_DIAGNOSES ((int)(sizeof DIAGNOSES / sizeof DIAGNOSES[0]))

int tllm_diagnosis_count(void) { return N_DIAGNOSES; }

const char *tllm_diagnosis_name(int index) { return index >= 0 && index < N_DIAGNOSES ? DIAGNOSES[index] : NULL; }

void tllm_device_default(tllm_device_state *s) {
    memset(s, 0, sizeof *s);
    s->t = 22.0f;
    s->t_valid = 1;
    s->h = 55;
    s->h_valid = 1;
    s->soil = 50;
}

int tllm_device_render(const tllm_device_state *s, char *out, size_t cap) {
    char t[16], h[16];
    if (s->t_valid)
        snprintf(t, sizeof t, "%.1f", (double)s->t);
    else
        snprintf(t, sizeof t, "na");
    if (s->h_valid)
        snprintf(h, sizeof h, "%d", s->h);
    else
        snprintf(h, sizeof h, "na");
    int n = snprintf(out, cap, "<S> t=%s h=%s soil=%d fan=%d heat=%d pump=%d light=%d win=%d pa=%.1f vib=%d err=%d</S>",
                     t, h, s->soil, s->fan, s->heat, s->pump, s->light, s->win, (double)s->pa, s->vib, s->err);
    return (n < 0 || (size_t)n >= cap) ? -1 : n;
}

static int parse_int(const char *p, size_t len, int *out) {
    if (len == 0u || len > 9u) return -1;
    int v = 0;
    for (size_t i = 0; i < len; ++i) {
        if (p[i] < '0' || p[i] > '9') return -1;
        v = v * 10 + (p[i] - '0');
    }
    *out = v;
    return 0;
}

int tllm_device_set(tllm_device_state *s, const char *assignment) {
    const char *eq = strchr(assignment, '=');
    if (eq == NULL || eq[1] == '\0') return -1;
    size_t klen = (size_t)(eq - assignment);
    const char *val = eq + 1;
    char *end = NULL;
#define KEY_IS(name) (klen == strlen(name) && strncmp(assignment, name, klen) == 0)
    if (KEY_IS("t") || KEY_IS("pa")) {
        int is_t = KEY_IS("t");
        if (is_t && strcmp(val, "na") == 0) {
            s->t_valid = 0;
            return 0;
        }
        float f = strtof(val, &end);
        if (end == val || *end != '\0' || f < -40.0f || f > 80.0f) return -1;
        if (is_t) {
            s->t = f;
            s->t_valid = 1;
        } else {
            if (f < 0.0f || f > 10.0f) return -1;
            s->pa = f;
        }
        return 0;
    }
    if (KEY_IS("h") && strcmp(val, "na") == 0) {
        s->h_valid = 0;
        return 0;
    }
    int v;
    if (parse_int(val, strlen(val), &v) != 0) return -1;
    struct {
        const char *name;
        int *field, max;
    } fields[] = {{"h", &s->h, 100},     {"soil", &s->soil, 100}, {"fan", &s->fan, 3},
                  {"heat", &s->heat, 1}, {"pump", &s->pump, 1},   {"light", &s->light, 100},
                  {"win", &s->win, 1},   {"vib", &s->vib, 1},     {"err", &s->err, 99}};
    for (size_t i = 0; i < sizeof fields / sizeof fields[0]; ++i) {
        if (KEY_IS(fields[i].name)) {
            if (v > fields[i].max) return -1;
            *fields[i].field = v;
            if (fields[i].field == &s->h) s->h_valid = 1;
            return 0;
        }
    }
#undef KEY_IS
    return -1;
}

int tllm_action_parse(const char *body, size_t len, tllm_action *out) {
    memset(out, 0, sizeof *out);
    size_t i = 0;
    unsigned seen = 0;
    while (i < len) {
        while (i < len && (body[i] == ' ' || body[i] == '\t' || body[i] == '\n' || body[i] == '\r')) ++i;
        if (i >= len) break;
        size_t start = i;
        while (i < len && body[i] != ' ' && body[i] != '\t' && body[i] != '\n' && body[i] != '\r') ++i;
        if (out->n == TLLM_ACTION_MAX_PAIRS) return -1;
        const char *part = body + start;
        size_t plen = i - start;
        const char *eq = memchr(part, '=', plen);
        if (eq == NULL) return -1;
        size_t klen = (size_t)(eq - part);
        const char *raw = eq + 1;
        size_t rlen = plen - klen - 1u;
        int key = -1;
        for (int k = 0; k < TLLM_KEY_COUNT; ++k)
            if (strlen(KEY_NAMES[k]) == klen && strncmp(part, KEY_NAMES[k], klen) == 0) key = k;
        if (key < 0 || (seen & (1u << key))) return -1;
        seen |= 1u << key;
        int value = -1;
        if (key == TLLM_KEY_DIAG) {
            for (int d = 0; d < N_DIAGNOSES; ++d)
                if (strlen(DIAGNOSES[d]) == rlen && strncmp(raw, DIAGNOSES[d], rlen) == 0) value = d;
            if (value < 0) return -1;
        } else {
            if (parse_int(raw, rlen, &value) != 0) return -1;
            const int *r = RANGES[key];
            if (value < r[0] || value > r[1] || value % r[2] != 0) return -1;
        }
        out->key[out->n] = (tllm_action_key)key;
        out->value[out->n] = value;
        out->n++;
    }
    return out->n > 0 ? 0 : -1;
}

int tllm_action_format(const tllm_action *a, char *out, size_t cap) {
    size_t used = 0;
    if (cap == 0u) return -1;
    out[0] = '\0';
    for (int i = 0; i < a->n; ++i) {
        int n;
        if (a->key[i] == TLLM_KEY_DIAG)
            n = snprintf(out + used, cap - used, "%sdiag=%s", i ? " " : "", DIAGNOSES[a->value[i]]);
        else
            n = snprintf(out + used, cap - used, "%s%s=%d", i ? " " : "", KEY_NAMES[a->key[i]], a->value[i]);
        if (n < 0 || (size_t)n >= cap - used) return -1;
        used += (size_t)n;
    }
    return (int)used;
}

static int find(const tllm_action *a, tllm_action_key key, int *value) {
    for (int i = 0; i < a->n; ++i)
        if (a->key[i] == key) {
            *value = a->value[i];
            return 1;
        }
    return 0;
}

void tllm_device_apply(tllm_device_state *s, const tllm_action *a) {
    for (int i = 0; i < a->n; ++i) {
        int v = a->value[i];
        switch (a->key[i]) {
        case TLLM_KEY_FAN: s->fan = v; break;
        case TLLM_KEY_HEAT: s->heat = v; break;
        case TLLM_KEY_PUMP:
            if (v == 1 && s->pump != 1) s->pa = PUMP_NOMINAL_A;
            if (v == 0) s->pa = 0.0f;
            s->pump = v;
            break;
        case TLLM_KEY_LIGHT: s->light = v; break;
        case TLLM_KEY_WIN: s->win = v; break;
        case TLLM_KEY_ACK: s->err = 0; break;
        default: break; /* diag is informational */
        }
    }
}

tllm_verdict tllm_device_validate(const tllm_device_state *s, const tllm_action *a) {
    int v;
    if (a->n <= 0) return TLLM_VERDICT_MALFORMED;
    if (find(a, TLLM_KEY_HEAT, &v) && v == 1 && s->err == 3) return TLLM_VERDICT_FAULT_HEATER;
    if (find(a, TLLM_KEY_WIN, &v) && v != s->win && s->err == 5) return TLLM_VERDICT_FAULT_WINDOW;
    if (find(a, TLLM_KEY_PUMP, &v) && v == 1 && s->err == 7) return TLLM_VERDICT_FAULT_PUMP;
    if (find(a, TLLM_KEY_ACK, &v) && (s->err == 0 || v != s->err)) return TLLM_VERDICT_ACK_MISMATCH;
    if (find(a, TLLM_KEY_HEAT, &v) && v == 1 && s->t_valid && s->t >= OVERHEAT_LIMIT_C) return TLLM_VERDICT_OVERHEAT;
    tllm_device_state result = *s;
    tllm_device_apply(&result, a);
    int touches = find(a, TLLM_KEY_HEAT, &v) || find(a, TLLM_KEY_WIN, &v);
    if (touches && result.heat == 1 && result.win == 1) return TLLM_VERDICT_HEATER_WINDOW;
    return TLLM_VERDICT_OK;
}

const char *tllm_verdict_code(tllm_verdict v) {
    switch (v) {
    case TLLM_VERDICT_OK: return "ok";
    case TLLM_VERDICT_MALFORMED: return "malformed";
    case TLLM_VERDICT_FAULT_HEATER: return "fault_heater";
    case TLLM_VERDICT_FAULT_WINDOW: return "fault_window";
    case TLLM_VERDICT_FAULT_PUMP: return "fault_pump";
    case TLLM_VERDICT_ACK_MISMATCH: return "ack_mismatch";
    case TLLM_VERDICT_OVERHEAT: return "overheat";
    case TLLM_VERDICT_HEATER_WINDOW: return "heater_window";
    }
    return "unknown";
}

const char *tllm_verdict_message(tllm_verdict v) {
    switch (v) {
    case TLLM_VERDICT_OK: return "approved";
    case TLLM_VERDICT_MALFORMED: return "the proposed action is malformed and was ignored.";
    case TLLM_VERDICT_FAULT_HEATER: return "the heater has a fault (E3). it stays off until the fault is cleared.";
    case TLLM_VERDICT_FAULT_WINDOW: return "the window motor has a fault (E5). the window cannot move.";
    case TLLM_VERDICT_FAULT_PUMP: return "the pump tripped on overcurrent (E7). clear the fault first.";
    case TLLM_VERDICT_ACK_MISMATCH: return "there is no such active fault to acknowledge.";
    case TLLM_VERDICT_OVERHEAT: return "the heater is not allowed above 35 degrees.";
    case TLLM_VERDICT_HEATER_WINDOW: return "the heater may not run while the window is open.";
    }
    return "unknown";
}
