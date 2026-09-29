/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include "test.h"
#include "tinyllm/facts.h"

void test_facts(void);

static const char *retrieved(const char *text) { return tllm_fact_name(tllm_fact_retrieve(text)); }

void test_facts(void) {
    CHECK_EQ_INT(tllm_fact_count(), 16);
    CHECK_STR(tllm_fact_name(0), "e3");
    CHECK(tllm_fact_text(0) != NULL);
    CHECK(tllm_fact_text(-1) == NULL);
    CHECK(tllm_fact_text(16) == NULL);
    CHECK(tllm_fact_name(16) == NULL);
    for (int i = 0; i < tllm_fact_count(); ++i) CHECK(strlen(tllm_fact_text(i)) < TLLM_FACT_TEXT_MAX);
    CHECK_STR(retrieved("what does e5 mean"), "e5");
    CHECK_STR(retrieved("What Does ERROR 7 Mean?"), "e7"); /* case-insensitive */
    CHECK_STR(retrieved("e3"), "e3");                      /* whole text */
    CHECK(retrieved("e35 is not a code") == NULL);         /* whole words only */
    CHECK(retrieved("xe5") == NULL);
    CHECK_STR(retrieved("the e5, then e3"), "e3"); /* table order wins, not text order */
    CHECK_STR(retrieved("is there a warranty?"), "warranty");
    CHECK_STR(retrieved("tank tank"), "tank");
    CHECK(retrieved("hello there") == NULL);
    CHECK(retrieved("") == NULL);
    CHECK(retrieved("tanks") == NULL);
    CHECK_STR(retrieved("x-tank"), "tank"); /* punctuation delimits words */
    char big[1000];
    memset(big, 'a', sizeof big - 5u);
    memcpy(big + sizeof big - 5u, " e5", 4); /* keyword beyond the input limit is ignored */
    CHECK(retrieved(big) == NULL);
}
