/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * Board microbenchmarks (vision M0): memory read bandwidth per memory tier and kernel
 * throughput, exposed as console commands "/bandwidth" and "/gemv".
 */
#ifndef FIRMWARE_BENCH_H
#define FIRMWARE_BENCH_H

#include <stddef.h>
#include <stdint.h>

#include "tinyllm/console.h"

void bench_set_model(const uint8_t *blob, size_t size, const uint8_t *flash_blob, const tllm_model *model);
int bench_command(tllm_console *con, const char *line);

#endif /* FIRMWARE_BENCH_H */
