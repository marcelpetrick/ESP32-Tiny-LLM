/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Device facts and deterministic keyword retrieval (vision §24 D). Mirrors
 * training/world/facts.py: same table, same order, same whole-word matching on lowercased
 * text. The console injects the retrieved fact as "<F> text</F>" for models that know
 * those tokens, so knowledge lives in a table instead of parameters.
 */
#ifndef TINYLLM_FACTS_H
#define TINYLLM_FACTS_H

#include <stddef.h>

#define TLLM_FACT_TEXT_MAX 256 /* longest fact text incl. NUL */

#ifdef __cplusplus
extern "C" {
#endif

/* Index of the first fact whose keyword occurs as a whole-word phrase, or -1. */
int tllm_fact_retrieve(const char *text);
const char *tllm_fact_text(int index); /* NULL if out of range */
const char *tllm_fact_name(int index);
int tllm_fact_count(void);

#ifdef __cplusplus
}
#endif
#endif /* TINYLLM_FACTS_H */
