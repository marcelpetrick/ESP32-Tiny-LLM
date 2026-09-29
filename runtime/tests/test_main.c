/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 */
#include <stdio.h>

#include "test.h"

int g_failures;
int g_checks;

#define TESTS(X)                                                                                                       \
    X(test_kernels)                                                                                                    \
    X(test_loader)                                                                                                     \
    X(test_tokenizer) X(test_transformer) X(test_sampler) X(test_device) X(test_facts) X(test_console) X(test_host)

#define DECLARE(name) void name(void);
TESTS(DECLARE)

int main(void) {
#define RUN(name)                                                                                                      \
    do {                                                                                                               \
        int before = g_failures;                                                                                       \
        name();                                                                                                        \
        printf("%-18s %s\n", #name, g_failures == before ? "ok" : "FAILED");                                           \
    } while (0);
    TESTS(RUN)
    printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
