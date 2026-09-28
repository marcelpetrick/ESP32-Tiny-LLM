# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# The portable runtime compiled for the ESP32-S3 straight from runtime/ (no copies).
set(rt "${CMAKE_CURRENT_LIST_DIR}/../../../runtime")
idf_component_register(
    SRCS "${rt}/src/model_loader.c" "${rt}/src/kernels.c" "${rt}/src/tokenizer.c"
         "${rt}/src/transformer.c" "${rt}/src/sampler.c" "${rt}/src/device.c" "${rt}/src/console.c"
    INCLUDE_DIRS "${rt}/include")
target_compile_options(${COMPONENT_LIB} PRIVATE -O2 -Wall -Wextra -Werror -Wdouble-promotion)
