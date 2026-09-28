# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
add_executable(tinyllm_tests
    test_main.c builder.c test_kernels.c test_loader.c test_tokenizer.c test_transformer.c
    test_sampler.c test_device.c test_console.c test_host.c ../src/host.c)
target_compile_options(tinyllm_tests PRIVATE -Wall -Wextra -Werror -Wno-missing-prototypes)
target_link_libraries(tinyllm_tests PRIVATE tinyllm)
if(TLLM_COVERAGE)
  target_compile_options(tinyllm_tests PRIVATE --coverage -O0)
  target_link_options(tinyllm_tests PRIVATE --coverage)
endif()
if(TLLM_SANITIZE)
  target_compile_options(tinyllm_tests PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(tinyllm_tests PRIVATE -fsanitize=address,undefined)
endif()
add_test(NAME tinyllm_unit_tests COMMAND tinyllm_tests)
