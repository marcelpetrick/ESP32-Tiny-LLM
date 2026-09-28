# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Marcel Petrick <mail@marcelpetrick.it>
#
# Web simulator image: the C runtime (libtinyllm.so + tinyllm-cli), the standard-library
# web server, the trained greenhouse assistant and the llama2.c stories260K demo model.
#   docker run --rm -p 8080:8080 ghcr.io/marcelpetrick/esp32-tiny-llm
ARG PYTHON_IMAGE=python:3.14.7-slim@sha256:51dafde81dbdb6ebde285137a295cf18a47ca95234fe388a343719cb97305b3d

FROM ${PYTHON_IMAGE} AS build
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY runtime runtime
RUN cmake -S runtime -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --target tinyllm_host tinyllm-cli
COPY VERSION VERSION
COPY training training
COPY tools tools
COPY models models
RUN pip install --no-cache-dir --root-user-action=ignore numpy==2.5.3 \
    && python -m tools.convert_llama2c models/third_party/stories260K/stories260K.bin \
        models/third_party/stories260K/tok512.bin --out /out/stories260K.tllm

FROM ${PYTHON_IMAGE} AS runtime
LABEL org.opencontainers.image.title="esp32-tiny-llm" \
      org.opencontainers.image.description="Tiny transformer LLM for the ESP32-S3: desktop web simulator" \
      org.opencontainers.image.source="https://github.com/marcelpetrick/ESP32-Tiny-LLM" \
      org.opencontainers.image.licenses="GPL-3.0-or-later"
RUN useradd --create-home --uid 10001 tinyllm
WORKDIR /app
COPY --from=build /src/build/libtinyllm.so /app/lib/libtinyllm.so
COPY --from=build /src/build/tinyllm-cli /usr/local/bin/tinyllm-cli
COPY --from=build /out/stories260K.tllm /app/models/stories260K.tllm
COPY VERSION LICENSE THIRD_PARTY_NOTICES.md /app/
COPY training/__init__.py /app/training/__init__.py
COPY training/world /app/training/world
COPY tools/__init__.py tools/runtime.py /app/tools/
COPY web /app/web
COPY models/greenhouse-m-int8.tllm /app/models/greenhouse-m-int8.tllm
ENV TINYLLM_LIB=/app/lib/libtinyllm.so \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER tinyllm
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8080/healthz', timeout=3)"]
CMD ["python", "-m", "web.server", "--host", "0.0.0.0", "--port", "8080", \
     "--model", "greenhouse=/app/models/greenhouse-m-int8.tllm", "--model", "stories=/app/models/stories260K.tllm"]
