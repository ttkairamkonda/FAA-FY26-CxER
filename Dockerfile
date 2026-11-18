# Self-contained image for the FAA_CxER listen-and-compare demo
# (demo_app.py). Just Streamlit + a handful of small audio files and a
# JSON of pre-computed results - no GPU, no host Python, no PyTorch, no
# internet access needed once this image is built/loaded. See
# README.md#demos for build/ship/run steps.

FROM python:3.10-slim

WORKDIR /app

COPY requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements-demo.txt

COPY demo_app.py ./
COPY .streamlit/ ./.streamlit/
COPY data/curated_samples.json ./data/curated_samples.json
COPY data/audio_samples/ ./data/audio_samples/

EXPOSE 8501

CMD ["streamlit", "run", "demo_app.py"]
