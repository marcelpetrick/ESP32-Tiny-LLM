# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# Host build of the tinyllm runtime: static core library, shared library for Python
# (ctypes), the tinyllm-cli REPL, and the C unit tests. The ESP-IDF firmware compiles
# the same sources through firmware/components/tinyllm.
cmake_minimum_required(VERSION 3.20)
project(tinyllm C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(TLLM_COVERAGE "Instrument for gcov coverage" OFF)
option(TLLM_SANITIZE "Build with address and undefined-behaviour sanitizers" OFF)

set(TLLM_WARNINGS -Wall -Wextra -Wpedantic -Werror -Wshadow -Wdouble-promotion -Wstrict-prototypes
                  -Wmissing-prototypes -Wformat=2 -Wcast-align -Wvla)

set(TLLM_CORE_SOURCES
    src/model_loader.c src/kernels.c src/tokenizer.c src/transformer.c src/sampler.c
    src/device.c src/facts.c src/console.c)

add_library(tinyllm STATIC ${TLLM_CORE_SOURCES})
target_include_directories(tinyllm PUBLIC include)
target_compile_options(tinyllm PRIVATE ${TLLM_WARNINGS})
target_link_libraries(tinyllm PUBLIC m)

add_library(tinyllm_host SHARED src/host.c ${TLLM_CORE_SOURCES})
set_target_properties(tinyllm_host PROPERTIES OUTPUT_NAME tinyllm)
target_include_directories(tinyllm_host PUBLIC include)
target_compile_options(tinyllm_host PRIVATE ${TLLM_WARNINGS})
target_link_libraries(tinyllm_host PUBLIC m)

add_executable(tinyllm-cli cli/main.c src/host.c)
target_compile_options(tinyllm-cli PRIVATE ${TLLM_WARNINGS})
target_link_libraries(tinyllm-cli PRIVATE tinyllm)

if(TLLM_COVERAGE)
  foreach(t tinyllm tinyllm_host tinyllm-cli)
    target_compile_options(${t} PRIVATE --coverage -O0)
    target_link_options(${t} PRIVATE --coverage)
  endforeach()
endif()
if(TLLM_SANITIZE)
  foreach(t tinyllm tinyllm_host tinyllm-cli)
    target_compile_options(${t} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
    target_link_options(${t} PRIVATE -fsanitize=address,undefined)
  endforeach()
endif()

enable_testing()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt)
  add_subdirectory(tests)
endif()
