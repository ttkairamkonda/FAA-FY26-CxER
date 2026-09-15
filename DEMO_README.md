# FAA CxER — Live Demo Packaging

An interactive dashboard (`demo_app.py`, built with Streamlit) that lets a
viewer click through the WER-CxER paradox analysis live — same numbers and
plots the project's headline results are built from, generated on the fly
from the data already shipped in `data/`. No GPU, no Python, no internet
needed on the machine it's demoed on.

It does **not** re-run the LLM-judge stage (that needs a multi-GPU host
running Llama-3.3-70B under vLLM) — those labels are pre-computed and ship
in `data/cxer_labels/`. Everything the demo shows is downstream of that,
which is also true of every plot in the paper/README.

## 1. Build the image (on your machine)

```bash
docker build -t faa-cxer-demo .
```

Test it locally before packaging for the client:

```bash
docker run --rm -p 8501:8501 faa-cxer-demo
```

Open http://localhost:8501 — click through all five tabs once, including
moving the paradox-threshold slider, to make sure everything renders before
you hand it off.

## 2. Package it for a machine with no Docker Hub / internet access

```bash
docker save faa-cxer-demo | gzip > faa-cxer-demo.tar.gz
```

This produces one file (a few GB, mostly the CPU-only PyTorch runtime) you
can put on a USB drive or shared folder. Nothing else from the repo needs
to go with it — the data and code are baked into the image.

## 3. On the client's computer

One-time setup: install **Docker Desktop** (Windows/Mac/Linux installer,
no coding involved) — https://www.docker.com/products/docker-desktop/

Then, with `faa-cxer-demo.tar.gz` copied onto that machine:

```bash
docker load -i faa-cxer-demo.tar.gz
docker run --rm -p 8501:8501 faa-cxer-demo
```

Open a browser to **http://localhost:8501**. That's the whole demo — closing
the terminal window (Ctrl+C) stops it; running the same `docker run`
command again starts it fresh.

## Notes

- Everything runs on CPU; a laptop with no GPU is fine.
- No data leaves the machine — the container makes no network calls.
- To update the demo after changing `faa_cxer/` or `demo_app.py`, re-run
  step 1 (`docker build`) and re-package (step 2) — Docker will only
  re-download/rebuild the layers that changed.
