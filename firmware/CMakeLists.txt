# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# ESP-IDF project: the tinyllm runtime on an ESP32-S3 with a UART/USB console.
#   idf.py set-target esp32s3 && idf.py build flash monitor
cmake_minimum_required(VERSION 3.16)
include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(esp32_tiny_llm)
