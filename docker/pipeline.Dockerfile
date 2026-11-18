# FAA CxER pipeline image: runs every stage, the tests and the demo.
#
# Build from the REPO ROOT (the build context must be the repo, not docker/):
#
#   GPU (default, for AWS GPU servers and local NVIDIA GPUs):
#     docker build -f docker/pipeline.Dockerfile -t faa-cxer-pipeline:gpu .
#
#   CPU-only (laptops / no NVIDIA GPU; runs everything except stage 2, the judge):
#     docker build -f docker/pipeline.Dockerfile -t faa-cxer-pipeline:cpu \
#       --build-arg BASE_IMAGE=python:3.12-slim --build-arg CPU_ONLY=1 .
#
# The GPU base is the official vLLM image at the exact version
# requirements-judge.txt pins (vllm 0.11.2 + torch 2.9.0, CUDA 12.x).
# Hosts need an NVIDIA driver and the NVIDIA Container Toolkit; nothing
# else (no Python, no CUDA toolkit) is needed on the host.
#
# This file lives next to docker/pipeline.Dockerfile.dockerignore, which
# BuildKit uses instead of the repo's top-level .dockerignore (that one is
# for the demo image and excludes scripts/ and tests/).

ARG BASE_IMAGE=vllm/vllm-openai:v0.11.2
FROM ${BASE_IMAGE}

ARG CPU_ONLY=0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HOME=/models/hf \
    MPLBACKEND=Agg \
    APP_DIR=/app \
    RESULTS_DIR=/results

WORKDIR /app

# CPU flavor only: a CPU torch wheel (the GPU base already ships CUDA torch).
RUN if [ "$CPU_ONLY" = "1" ]; then \
      python3 -m pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu ; \
    fi

# Install the pipeline's Python deps WITHOUT touching anything the base
# image already has (torch, transformers, numpy... stay at the versions
# vLLM was built against): every installed package is pinned as a constraint.
# Extras not listed in requirements.txt: scipy,
# transformers<5 (what vLLM 0.11.2 supports), and the judge/test helpers.
COPY requirements.txt requirements-demo.txt ./
RUN python3 -m pip freeze | grep -E '^[A-Za-z0-9_.-]+==' > /tmp/base-constraints.txt ; \
    python3 -m pip install -c /tmp/base-constraints.txt \
        -r requirements.txt -r requirements-demo.txt \
        "transformers>=4.56,<5" scipy python-dotenv huggingface_hub pytest
RUN if ! command -v python >/dev/null; then ln -s "$(command -v python3)" /usr/local/bin/python; fi

# Code + shipped data (predictions, judge labels, demo audio).
COPY . /app
RUN sed -i 's/\r$//' /app/docker/entrypoint.sh && chmod +x /app/docker/entrypoint.sh \
    && mkdir -p /results /models/hf

VOLUME ["/results", "/models"]
EXPOSE 8501

# The vLLM base image's own entrypoint starts an API server; replace it.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["help"]
