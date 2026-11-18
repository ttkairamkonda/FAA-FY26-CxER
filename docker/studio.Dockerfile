# FAA CxER Studio (app/cxer_studio.py): upload audio, live ASR, WER, and
# CxER verdicts from the GPU judge server when one is running.
#
# Build from the REPO ROOT:
#   docker build -f docker/studio.Dockerfile -t faa-cxer-studio .
#
# The default torch wheel from PyPI is CUDA-enabled and also runs on
# machines without a GPU, so this one image covers both. For a smaller,
# CPU-only image:
#   docker build -f docker/studio.Dockerfile -t faa-cxer-studio:cpu \
#     --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
#
# The 70B judge is NOT in this image; it runs as the "judge" service in
# docker-compose.studio.yml (official vLLM image) and the app calls it over HTTP.

FROM python:3.12-slim

ARG TORCH_INDEX_URL=""

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/hf \
    MPLBACKEND=Agg

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "$TORCH_INDEX_URL" ]; then \
      pip install torch==2.9.0 --index-url "$TORCH_INDEX_URL" ; \
    else \
      pip install torch==2.9.0 ; \
    fi

RUN pip install "transformers>=4.56,<5" "streamlit>=1.50,<2" librosa soundfile jiwer \
        pandas numpy plotly requests bert-score tqdm matplotlib openpyxl imageio-ffmpeg

WORKDIR /app
COPY faa_cxer/ ./faa_cxer/
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/
COPY data/curated_samples.json ./data/curated_samples.json
COPY data/icao_telephony.json ./data/icao_telephony.json
COPY data/atc_benchmarks/ ./data/atc_benchmarks/
COPY data/audio_samples/ ./data/audio_samples/
COPY data/cxer_labels/ ./data/cxer_labels/
COPY data/predictions/FAA_metrics_summary.csv ./data/predictions/FAA_metrics_summary.csv
COPY outputs/examples/ ./outputs/examples/

VOLUME ["/models"]
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app/cxer_studio.py", "--server.maxUploadSize=200", "--client.toolbarMode=minimal"]
