/*
 * SPDX-License-Identifier: GPL-3.0-or-later
 * Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
 *
 * ESP32-S3 firmware: loads a .tllm model (model partition if valid, else the one embedded
 * in the app), copies it to PSRAM, places hot buffers in internal SRAM and the KV cache in
 * PSRAM, then serves the tinyllm console over the serial port.
 *
 *   core 0: serial line reader -> queue          core 1: inference + console (pinned)
 *
 * The model only proposes actions; tllm_device_validate() in the runtime decides (the
 * simulated greenhouse here stands in for real actuators).
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "bench.h"
#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_chip_info.h"
#include "esp_flash.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_psram.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"
#include "sdkconfig.h"
#include "tinyllm/console.h"

#define MODEL_PARTITION_SUBTYPE 0x40
#define LINE_MAX_LEN TLLM_CONSOLE_LINE_MAX

extern const uint8_t model_start[] asm("_binary_greenhouse_m_int8_tllm_start");
extern const uint8_t model_end[] asm("_binary_greenhouse_m_int8_tllm_end");

static tllm_model s_model;
static tllm_ctx s_ctx;
static tllm_console *s_con;
static QueueHandle_t s_lines;
static const char *s_model_source = "embedded";

static uint64_t clock_us(void) { return (uint64_t)esp_timer_get_time(); }

/* Let lower-priority tasks (and the task watchdog's idle hook) run during long replies. */
static void yield_hook(void) {
    static int64_t last;
    int64_t now = esp_timer_get_time();
    if (now - last > 200000) {
        vTaskDelay(1);
        last = esp_timer_get_time();
    }
}

static void write_out(void *user, const char *text, size_t len) {
    (void)user;
    fwrite(text, 1, len, stdout);
    fflush(stdout);
}

static int memory_report(void *user, char *out, size_t cap) {
    (void)user;
    return snprintf(out, cap, "internal free %u B (min %u, largest %u); psram %u B total, free %u B (min %u)",
                    (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
                    (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL),
                    (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL), (unsigned)esp_psram_get_size(),
                    (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
                    (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_SPIRAM));
}

/* Returns a pointer to a valid model blob in flash (partition first, then embedded). */
static const uint8_t *find_model(size_t *size) {
    const esp_partition_t *part =
        esp_partition_find_first(ESP_PARTITION_TYPE_DATA, (esp_partition_subtype_t)MODEL_PARTITION_SUBTYPE, "model");
    if (part != NULL) {
        uint8_t header[TLLM_HEADER_SIZE];
        if (esp_partition_read(part, 0, header, sizeof header) == ESP_OK && memcmp(header, "TLLM", 4) == 0) {
            uint32_t payload;
            memcpy(&payload, header + 76, sizeof payload);
            size_t total = TLLM_HEADER_SIZE + (size_t)payload;
            const void *mapped = NULL;
            esp_partition_mmap_handle_t handle;
            if (total <= part->size &&
                esp_partition_mmap(part, 0, total, ESP_PARTITION_MMAP_DATA, &mapped, &handle) == ESP_OK) {
                tllm_model probe;
                if (tllm_model_load(&probe, mapped, total) == TLLM_OK) {
                    s_model_source = "partition";
                    *size = total;
                    return (const uint8_t *)mapped;
                }
                printf("model partition present but invalid; using the embedded model\n");
                esp_partition_munmap(handle);
            }
        }
    }
    *size = (size_t)(model_end - model_start);
    return model_start;
}

static void *alloc(size_t size, uint32_t caps) {
    void *p = heap_caps_aligned_alloc(16, size, caps);
    return p != NULL ? p : heap_caps_aligned_alloc(16, size, MALLOC_CAP_8BIT);
}

static void boot_report(size_t model_size, size_t hot, size_t cold, int psram_model) {
    esp_chip_info_t chip;
    esp_chip_info(&chip);
    uint32_t flash_size = 0;
    esp_flash_get_size(NULL, &flash_size);
#if CONFIG_SPIRAM_MODE_OCT
    const char *psram_mode = "octal";
#else
    const char *psram_mode = "quad";
#endif
#ifdef CONFIG_SPIRAM_SPEED
    const int psram_mhz = CONFIG_SPIRAM_SPEED;
#else
    const int psram_mhz = 0;
#endif
    char id[33];
    for (int i = 0; i < 16; ++i) snprintf(id + 2 * i, 3, "%02x", s_model.model_id[i]);
    printf("\nESP32 Tiny LLM — %d cores @ %d MHz, rev %d.%d, flash %" PRIu32 " MB, PSRAM %u MB %s @ %d MHz\n",
           chip.cores, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, chip.revision / 100, chip.revision % 100, flash_size >> 20,
           (unsigned)(esp_psram_get_size() >> 20), psram_mode, psram_mhz);
    printf("model %s (%s, %u bytes, crc %08" PRIx32 ", copied to %s); hot arena %u B, KV cache %u B\n", id,
           s_model_source, (unsigned)model_size, s_model.crc32, psram_model ? "PSRAM" : "flash (XIP)", (unsigned)hot,
           (unsigned)cold);
    printf("@@{\"event\":\"boot\",\"cpu_mhz\":%d,\"cores\":%d,\"flash_bytes\":%" PRIu32
           ",\"psram_bytes\":%u,\"psram_mode\":\"%s\",\"psram_mhz\":%d,\"model_id\":\"%s\",\"model_source\":\"%s\","
           "\"model_bytes\":%u,\"model_in\":\"%s\",\"hot_arena\":%u,\"cold_arena\":%u,\"idf\":\"%s\"}\n",
           CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, chip.cores, flash_size, (unsigned)esp_psram_get_size(), psram_mode,
           psram_mhz, id, s_model_source, (unsigned)model_size, psram_model ? "psram" : "flash", (unsigned)hot,
           (unsigned)cold, esp_get_idf_version());
    printf("type /help for commands; anything else is a chat message.\n");
}

static void reader_task(void *arg) {
    (void)arg;
    char line[LINE_MAX_LEN + 1];
    for (;;) {
        if (fgets(line, sizeof line, stdin) == NULL) {
            vTaskDelay(pdMS_TO_TICKS(20));
            continue;
        }
        line[strcspn(line, "\r\n")] = '\0';
        if (line[0] != '\0') xQueueSend(s_lines, line, portMAX_DELAY);
    }
}

static void inference_task(void *arg) {
    (void)arg;
    char line[LINE_MAX_LEN + 1];
    for (;;) {
        if (xQueueReceive(s_lines, line, portMAX_DELAY) == pdTRUE) tllm_console_line(s_con, line);
    }
}

static void console_io_init(void) {
    setvbuf(stdin, NULL, _IONBF, 0);
    ESP_ERROR_CHECK(uart_driver_install(CONFIG_ESP_CONSOLE_UART_NUM, 1024, 0, 0, NULL, 0));
    uart_vfs_dev_use_driver(CONFIG_ESP_CONSOLE_UART_NUM);
    uart_vfs_dev_port_set_rx_line_endings(CONFIG_ESP_CONSOLE_UART_NUM, ESP_LINE_ENDINGS_CR);
    uart_vfs_dev_port_set_tx_line_endings(CONFIG_ESP_CONSOLE_UART_NUM, ESP_LINE_ENDINGS_CRLF);
}

void app_main(void) {
    console_io_init();
    size_t size = 0;
    const uint8_t *flash_blob = find_model(&size);
    /* weights are re-read every token: PSRAM is faster than flash XIP (docs/01 §3) */
    uint8_t *blob = heap_caps_aligned_alloc(16, size, MALLOC_CAP_SPIRAM);
    int psram_model = blob != NULL; /* NULL when no PSRAM was detected */
    if (psram_model) memcpy(blob, flash_blob, size);
    tllm_status st = tllm_model_load(&s_model, psram_model ? blob : flash_blob, size);
    if (st != TLLM_OK) {
        printf("model load failed: %s\n", tllm_status_str(st));
        return;
    }
    /* without PSRAM the KV cache must fit into internal SRAM: use the int8 cache */
    const int kv_int8 = !psram_model;
    const size_t hot = tllm_hot_arena_size(&s_model), cold = tllm_cold_arena_size(&s_model, kv_int8);
    void *hot_mem = alloc(hot, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    void *cold_mem = alloc(cold, MALLOC_CAP_SPIRAM);
    s_con = heap_caps_calloc(1, sizeof *s_con, MALLOC_CAP_SPIRAM);
    if (s_con == NULL) s_con = heap_caps_calloc(1, sizeof *s_con, MALLOC_CAP_8BIT);
    tllm_ctx_options opt = {TLLM_ACT_F32, kv_int8, clock_us, yield_hook};
    if (hot_mem == NULL || cold_mem == NULL || s_con == NULL ||
        tllm_ctx_init(&s_ctx, &s_model, &opt, hot_mem, hot, cold_mem, cold) != TLLM_OK ||
        tllm_console_init(s_con, &s_ctx, write_out, NULL) != TLLM_OK) {
        printf("runtime init failed (out of memory?)\n");
        return;
    }
    s_con->memory = memory_report;
    s_con->platform_cmd = bench_command;
    bench_set_model(psram_model ? blob : flash_blob, size, flash_blob, &s_model);
    boot_report(size, hot, cold, psram_model);
    s_lines = xQueueCreate(4, LINE_MAX_LEN + 1);
    xTaskCreatePinnedToCore(reader_task, "tllm_reader", 4096, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(inference_task, "tllm_infer", 16384, NULL, 4, NULL, 1);
}
