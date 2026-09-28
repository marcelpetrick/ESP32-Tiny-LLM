# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
idf_component_register(
    SRCS "app_main.c" "bench.c"
    INCLUDE_DIRS "."
    REQUIRES tinyllm esp_partition spi_flash esp_psram esp_timer esp_driver_uart
    EMBED_FILES "${CMAKE_CURRENT_LIST_DIR}/../../models/greenhouse-m-int8.tllm")
target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)
