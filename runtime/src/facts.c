/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Fact table and keyword retrieval — must stay identical to training/world/facts.py
 * (tests/integration/test_facts_parity.py compares both on many inputs).
 */
#include "tinyllm/facts.h"

#include <string.h>

#define MAX_KEYWORDS 4
#define INPUT_MAX 512 /* console lines are at most TLLM_CONSOLE_LINE_MAX (256) bytes */

typedef struct {
    const char *name;
    const char *keywords[MAX_KEYWORDS];
    const char *text;
} fact;

static const fact FACTS[] = {
    {"e3",
     {"e3", "error 3", "fault 3", NULL},
     "e3 means the heater overheated or its sensor failed. check the heater, then clear the fault."},
    {"e5",
     {"e5", "error 5", "fault 5", NULL},
     "e5 means the window motor is blocked. remove the obstacle, then clear the fault."},
    {"e7",
     {"e7", "error 7", "fault 7", NULL},
     "e7 means the pump drew too much current. check the pump for a blockage, then clear the fault."},
    {"humidity",
     {"ideal humidity", "good humidity", "best humidity", NULL},
     "the ideal humidity for most plants is 50 to 70 percent."},
    {"temperature",
     {"ideal temperature", "good temperature", "best temperature", NULL},
     "the ideal temperature is 18 to 26 degrees."},
    {"watering",
     {"how often", "when should i water", "watering", NULL},
     "water when the soil moisture drops below 25 percent, about once a day in summer."},
    {"fan",
     {"fan levels", "fan speeds", "how fast is the fan", NULL},
     "the fan has levels 0 to 3. level 3 moves the most air but is loud."},
    {"light", {"how long", "light hours", "hours of light", NULL}, "most plants need 12 to 16 hours of light a day."},
    {"bearing",
     {"bearing", "maintenance", "service the fan", NULL},
     "fan bearings wear after about two years. replace the fan when vibration stays high."},
    {"tank", {"water tank", "refill", "tank", NULL}, "the water tank holds 20 litres and lasts about a week."},
    {"heater",
     {"heater power", "how strong is the heater", "watts", NULL},
     "the heater has 500 watts and is not allowed above 35 degrees."},
    {"window",
     {"window open", "when to open", "ventilate", NULL},
     "open the window when it is warmer than 28 degrees and not raining."},
    {"frost", {"frost", "freezing night", NULL}, "below 5 degrees at night, close the window and run the heater."},
    {"soil", {"soil type", "which soil", "potting soil", NULL}, "use loose potting soil with good drainage."},
    {"pests", {"pests", "insects", "bugs", NULL}, "check the leaves for insects once a week and remove them by hand."},
    {"warranty", {"warranty", "guarantee", NULL}, "the controller has a two year warranty."},
};
#define N_FACTS ((int)(sizeof FACTS / sizeof FACTS[0]))

int tllm_fact_count(void) { return N_FACTS; }
const char *tllm_fact_text(int i) { return i >= 0 && i < N_FACTS ? FACTS[i].text : NULL; }
const char *tllm_fact_name(int i) { return i >= 0 && i < N_FACTS ? FACTS[i].name : NULL; }

static int word_char(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'); }

static int contains_phrase(const char *text, const char *phrase) {
    size_t n = strlen(phrase);
    for (const char *p = strstr(text, phrase); p != NULL; p = strstr(p + 1, phrase)) {
        int before = p == text || !word_char(p[-1]);
        int after = p[n] == '\0' || !word_char(p[n]);
        if (before && after) return 1;
    }
    return 0;
}

int tllm_fact_retrieve(const char *text) {
    char lower[INPUT_MAX];
    size_t i = 0;
    for (; text[i] != '\0' && i + 1u < sizeof lower; ++i)
        lower[i] = (text[i] >= 'A' && text[i] <= 'Z') ? (char)(text[i] - 'A' + 'a') : text[i];
    lower[i] = '\0';
    for (int f = 0; f < N_FACTS; ++f)
        for (int k = 0; k < MAX_KEYWORDS && FACTS[f].keywords[k] != NULL; ++k)
            if (contains_phrase(lower, FACTS[f].keywords[k])) return f;
    return -1;
}
