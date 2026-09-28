/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * tinyllm-cli — desktop REPL speaking exactly the ESP32 serial console protocol.
 *
 * Usage: tinyllm-cli MODEL.tllm [--act-int8] [--kv-int8] [-c LINE]...
 *   Without -c, lines are read from stdin (a "> " prompt is shown on a terminal).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "tinyllm/host.h"

static void usage(void) { fputs("usage: tinyllm-cli MODEL.tllm [--act-int8] [--kv-int8] [-c LINE]...\n", stderr); }

int main(int argc, char **argv) {
    const char *path = NULL;
    int act_int8 = 0, kv_int8 = 0, n_cmds = 0;
    const char *cmds[64];
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--act-int8") == 0)
            act_int8 = 1;
        else if (strcmp(argv[i], "--kv-int8") == 0)
            kv_int8 = 1;
        else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc && n_cmds < 64)
            cmds[n_cmds++] = argv[++i];
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage();
            return 0;
        } else if (path == NULL && argv[i][0] != '-')
            path = argv[i];
        else {
            usage();
            return 2;
        }
    }
    if (path == NULL) {
        usage();
        return 2;
    }
    char err[128];
    tllm_host *h = tllm_host_open(path, act_int8, kv_int8, err, sizeof err);
    if (h == NULL) {
        fprintf(stderr, "error: %s: %s\n", path, err);
        return 1;
    }
    if (n_cmds > 0) {
        for (int i = 0; i < n_cmds; ++i) fputs(tllm_host_submit(h, cmds[i]), stdout);
    } else {
        char line[512];
        int tty = isatty(STDIN_FILENO);
        for (;;) {
            if (tty) {
                fputs("> ", stdout);
                fflush(stdout);
            }
            if (fgets(line, sizeof line, stdin) == NULL) break;
            line[strcspn(line, "\r\n")] = '\0';
            if (line[0] == '\0') continue;
            fputs(tllm_host_submit(h, line), stdout);
            fflush(stdout);
        }
    }
    tllm_host_close(h);
    return 0;
}
