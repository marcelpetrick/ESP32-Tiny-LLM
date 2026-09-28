/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include "test.h"
#include "tinyllm/device.h"

void test_device(void);

static tllm_verdict check(const char *sets, const char *action) {
    tllm_device_state s;
    tllm_device_default(&s);
    char buf[128];
    snprintf(buf, sizeof buf, "%s", sets);
    for (char *t = strtok(buf, " "); t != NULL; t = strtok(NULL, " ")) CHECK_EQ_INT(tllm_device_set(&s, t), 0);
    tllm_action a;
    if (tllm_action_parse(action, strlen(action), &a) != 0) return TLLM_VERDICT_MALFORMED;
    return tllm_device_validate(&s, &a);
}

void test_device(void) {
    tllm_device_state s;
    char text[TLLM_STATE_TEXT_MAX];
    tllm_device_default(&s);
    CHECK(tllm_device_render(&s, text, sizeof text) > 0);
    CHECK_STR(text, "<S> t=22.0 h=55 soil=50 fan=0 heat=0 pump=0 light=0 win=0 pa=0.0 vib=0 err=0</S>");
    CHECK_EQ_INT(tllm_device_render(&s, text, 10), -1);
    CHECK_EQ_INT(tllm_device_set(&s, "t=na"), 0);
    CHECK_EQ_INT(tllm_device_set(&s, "h=na"), 0);
    tllm_device_render(&s, text, sizeof text);
    CHECK_CONTAINS(text, "t=na h=na");
    CHECK_EQ_INT(tllm_device_set(&s, "t=-3.25"), 0);
    CHECK_EQ_INT(tllm_device_set(&s, "h=80"), 0);
    CHECK_EQ_INT(tllm_device_set(&s, "pa=2.7"), 0);
    tllm_device_render(&s, text, sizeof text);
    CHECK_CONTAINS(text, "t=-3.2 h=80");
    CHECK_CONTAINS(text, "pa=2.7");
    const char *bad[] = {"t", "t=", "t=abc", "t=99", "pa=11", "pa=-1", "h=101", "fan=4", "nope=1", "fan=x", "err=100"};
    for (size_t i = 0; i < sizeof bad / sizeof bad[0]; ++i) CHECK_EQ_INT(tllm_device_set(&s, bad[i]), -1);
    const char *good[] = {"soil=10", "fan=3", "heat=1", "pump=1", "light=40", "win=1", "vib=1", "err=7"};
    for (size_t i = 0; i < sizeof good / sizeof good[0]; ++i) CHECK_EQ_INT(tllm_device_set(&s, good[i]), 0);

    /* parser */
    tllm_action a;
    char fmt[64];
    CHECK_EQ_INT(tllm_action_parse(" fan=2  heat=0 diag=too_humid ", 30, &a), 0);
    CHECK_EQ_INT(a.n, 3);
    CHECK(tllm_action_format(&a, fmt, sizeof fmt) > 0);
    CHECK_STR(fmt, "fan=2 heat=0 diag=too_humid");
    CHECK_EQ_INT(tllm_action_format(&a, fmt, 5), -1);
    CHECK_EQ_INT(tllm_action_format(&a, fmt, 0), -1);
    const char *malformed[] = {"",        "   ",           "fan=1 heat=0 pump=0 win=0 light=10",
                               "speed=2", "fan",           "fan=1 fan=2",
                               "fan=4",   "fan=-1",        "light=15",
                               "ack=0",   "diag=unknown",  "fan=x",
                               "fan=",    "fan=1234567890"};
    for (size_t i = 0; i < sizeof malformed / sizeof malformed[0]; ++i)
        CHECK_EQ_INT(tllm_action_parse(malformed[i], strlen(malformed[i]), &a), -1);

    /* validation (same table as tests/unit/world/test_actions.py) */
    CHECK_EQ_INT(check("", "fan=2"), TLLM_VERDICT_OK);
    CHECK_EQ_INT(check("err=3", "heat=1"), TLLM_VERDICT_FAULT_HEATER);
    CHECK_EQ_INT(check("err=3", "heat=0"), TLLM_VERDICT_OK);
    CHECK_EQ_INT(check("err=5", "win=1"), TLLM_VERDICT_FAULT_WINDOW);
    CHECK_EQ_INT(check("err=5 win=1", "win=1"), TLLM_VERDICT_OK);
    CHECK_EQ_INT(check("err=7", "pump=1"), TLLM_VERDICT_FAULT_PUMP);
    CHECK_EQ_INT(check("", "ack=3"), TLLM_VERDICT_ACK_MISMATCH);
    CHECK_EQ_INT(check("err=5", "ack=3"), TLLM_VERDICT_ACK_MISMATCH);
    CHECK_EQ_INT(check("err=5", "ack=5"), TLLM_VERDICT_OK);
    CHECK_EQ_INT(check("t=36.0", "heat=1"), TLLM_VERDICT_OVERHEAT);
    CHECK_EQ_INT(check("t=na", "heat=1"), TLLM_VERDICT_OK);
    CHECK_EQ_INT(check("win=1", "heat=1"), TLLM_VERDICT_HEATER_WINDOW);
    CHECK_EQ_INT(check("heat=1", "win=1"), TLLM_VERDICT_HEATER_WINDOW);
    CHECK_EQ_INT(check("heat=1 win=1", "fan=1"), TLLM_VERDICT_OK);
    CHECK_EQ_INT(check("win=1", "win=0 heat=1"), TLLM_VERDICT_OK);
    CHECK_EQ_INT(check("", "diag=too_hot"), TLLM_VERDICT_OK);
    tllm_action empty = {0, {TLLM_KEY_FAN}, {0}};
    CHECK_EQ_INT(tllm_device_validate(&s, &empty), TLLM_VERDICT_MALFORMED);

    /* apply */
    tllm_device_default(&s);
    s.err = 5;
    CHECK_EQ_INT(tllm_action_parse("pump=1 fan=3 ack=5 diag=too_hot", 31, &a), 0);
    tllm_device_apply(&s, &a);
    CHECK(s.pump == 1 && s.fan == 3 && s.err == 0);
    CHECK_NEAR(s.pa, 1.2, 1e-6);
    s.pa = 2.8f;
    CHECK_EQ_INT(tllm_action_parse("pump=1", 6, &a), 0);
    tllm_device_apply(&s, &a);
    CHECK_NEAR(s.pa, 2.8, 1e-6);
    CHECK_EQ_INT(tllm_action_parse("pump=0 light=30 win=1 heat=0", 28, &a), 0);
    tllm_device_apply(&s, &a);
    CHECK(s.pa == 0.0f && s.light == 30 && s.win == 1);

    for (int v = TLLM_VERDICT_OK; v <= TLLM_VERDICT_HEATER_WINDOW; ++v) {
        CHECK(strcmp(tllm_verdict_code((tllm_verdict)v), "unknown") != 0);
        CHECK(strcmp(tllm_verdict_message((tllm_verdict)v), "unknown") != 0);
    }
    CHECK_STR(tllm_verdict_code((tllm_verdict)42), "unknown");
    CHECK_STR(tllm_verdict_message((tllm_verdict)42), "unknown");
    CHECK_EQ_INT(tllm_diagnosis_count(), 12);
    CHECK_STR(tllm_diagnosis_name(0), "pump_blocked");
    CHECK(tllm_diagnosis_name(12) == NULL);
    CHECK(tllm_diagnosis_name(-1) == NULL);
}
