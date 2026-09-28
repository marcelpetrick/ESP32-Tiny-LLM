/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Minimal dependency-free unit-test harness for the runtime.
 */
#ifndef TINYLLM_TEST_H
#define TINYLLM_TEST_H

#include <math.h>
#include <stdio.h>
#include <string.h>

extern int g_failures;
extern int g_checks;

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        ++g_checks;                                                                                                    \
        if (!(cond)) {                                                                                                 \
            ++g_failures;                                                                                              \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);                                   \
        }                                                                                                              \
    } while (0)

#define CHECK_EQ_INT(a, b)                                                                                             \
    do {                                                                                                               \
        long long va_ = (long long)(a), vb_ = (long long)(b);                                                          \
        ++g_checks;                                                                                                    \
        if (va_ != vb_) {                                                                                              \
            ++g_failures;                                                                                              \
            fprintf(stderr, "%s:%d: %s == %s failed: %lld != %lld\n", __FILE__, __LINE__, #a, #b, va_, vb_);           \
        }                                                                                                              \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                                          \
    do {                                                                                                               \
        double va_ = (double)(a), vb_ = (double)(b);                                                                   \
        ++g_checks;                                                                                                    \
        if (!(fabs(va_ - vb_) <= (tol))) {                                                                             \
            ++g_failures;                                                                                              \
            fprintf(stderr, "%s:%d: %s ~ %s failed: %g vs %g\n", __FILE__, __LINE__, #a, #b, va_, vb_);                \
        }                                                                                                              \
    } while (0)

#define CHECK_STR(a, b)                                                                                                \
    do {                                                                                                               \
        const char *sa_ = (a), *sb_ = (b);                                                                             \
        ++g_checks;                                                                                                    \
        if (sa_ == NULL || sb_ == NULL || strcmp(sa_, sb_) != 0) {                                                     \
            ++g_failures;                                                                                              \
            fprintf(stderr, "%s:%d: %s == %s failed: \"%s\" vs \"%s\"\n", __FILE__, __LINE__, #a, #b,                  \
                    sa_ ? sa_ : "(null)", sb_ ? sb_ : "(null)");                                                       \
        }                                                                                                              \
    } while (0)

#define CHECK_CONTAINS(hay, needle)                                                                                    \
    do {                                                                                                               \
        const char *h_ = (hay), *n_ = (needle);                                                                        \
        ++g_checks;                                                                                                    \
        if (h_ == NULL || strstr(h_, n_) == NULL) {                                                                    \
            ++g_failures;                                                                                              \
            fprintf(stderr, "%s:%d: \"%s\" not found in: %s\n", __FILE__, __LINE__, n_, h_ ? h_ : "(null)");           \
        }                                                                                                              \
    } while (0)

#endif /* TINYLLM_TEST_H */
