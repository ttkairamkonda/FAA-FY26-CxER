# Self-contained image for the FAA_CxER interactive demo (demo_app.py).
# Runs entirely on CPU — no GPU, no host Python, no internet access needed
# once this image is built/loaded. See DEMO_README.md for build/ship/run steps.

FROM python:3.10-slim

WORKDIR /app

# CPU-only torch wheel: a fraction of the size of the default CUDA build,
# and the demo never touches a GPU (the GPU-only judge stage isn't part of
# this image — its output is pre-computed and shipped under data/).
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-demo.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-demo.txt

COPY pyproject.toml ./
COPY faa_cxer/ ./faa_cxer/
COPY demo_app.py ./
COPY .streamlit/ ./.streamlit/
COPY data/ ./data/

EXPOSE 8501

CMD ["streamlit", "run", "demo_app.py"]
