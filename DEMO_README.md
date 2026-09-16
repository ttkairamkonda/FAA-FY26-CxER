# FAA CxER — Demo Packaging

Two different demos live in this repo, for two different situations.

## For most demos: the static site

[https://ttkairamkonda.github.io/FAA-Project/](https://ttkairamkonda.github.io/FAA-Project/)
is the primary way to show this — pick a sample, listen, see the
transcription and the judge's verdict. No install, no laptop setup, works
anywhere with a browser. Use this unless someone specifically needs it
running locally on their own machine.

## When someone wants it running on their own computer, live

`Dockerfile.live_asr` + `live_asr_app.py` is for that: two real Whisper
Medium checkpoints (pretrained vs. the fine-tuned production model) run a
genuine live transcription, on that computer's CPU, on either a bundled
reference clip or audio they upload themselves. It's real inference, not
a lookup — watch it actually run.

The CxER judge verdict is NOT reproduced live (that needs Llama-3.3-70B
under vLLM on a multi-GPU host — not realistic on a laptop). When the
audio happens to be one of the two bundled reference clips already
evaluated offline, the app shows that clip's real, pre-computed verdict
alongside the fresh transcription. For anything else you upload, you get
the live transcription only, with a note explaining why there's no
verdict.

### Build (on your machine — needs internet, ~6GB download the first time)

```bash
docker build -f Dockerfile.live_asr -t faa-live-asr-demo .
```

Both model checkpoints get baked into the image during this build, so the
*run* step below needs no internet at all.

### Test locally

```bash
docker run --rm -p 8501:8501 faa-live-asr-demo
```

Open http://localhost:8501, try both reference clips, and try uploading a
random short audio file to confirm the "no verdict available" path also
looks right.

### Package for the client's computer

```bash
docker save faa-live-asr-demo | gzip > faa-live-asr-demo.tar.gz
```

This file is large (multiple GB — it's carrying two real model
checkpoints) so a USB drive or direct transfer works better than email.
On the client's machine: install **Docker Desktop**
(https://www.docker.com/products/docker-desktop/), make sure it's
actually running (whale icon in the menu bar/tray), then:

```bash
docker load -i faa-live-asr-demo.tar.gz
docker run --rm -p 8501:8501 faa-live-asr-demo
```

Open http://localhost:8501. No GPU needed — CPU inference on a short ATC
clip takes a few seconds per model.

Or, if that machine has internet, skip the transfer and `docker build`
directly from a clone/download of this repo instead.

---

## The other Docker image (listen-and-compare, superseded by the static site)

`Dockerfile` + `demo_app.py` do the same thing as the static site — pick
one of eight pre-computed samples, listen, see the transcription and
verdict — just packaged as a tiny (~200MB, no PyTorch) Docker image
instead of a webpage. Kept here as a fallback for a machine that somehow
can't reach the static site at all; otherwise the static site above is
simpler and needs nothing installed.

```bash
docker build -t faa-cxer-demo .
docker run --rm -p 8501:8501 faa-cxer-demo
```

## Notes

- To change which samples the live-ASR reference dropdown offers, edit
  `data/curated_samples.json` and `data/audio_samples/` — the app only
  shows clips whose `"family"` matches the two loaded models (Whisper
  Medium). Swapping in a different family means changing `PRETRAINED_ID`
  / `FINETUNED_ID` in `live_asr_app.py` and rebuilding.
- No data leaves the machine in either image — no network calls at
  runtime.
