# Run FAA CxER Studio from this repository

This setup uses only this checkout, Python and downloaded public packages/models.
No Docker, sudo, sibling repositories or pre-existing project environments are used.
The saved FAA and ATC results are bundled in `data/`.

## 1. Requirements

- Linux x86-64 with Python 3.10–3.12 and Python's `venv` module available.
- Internet access for the initial package and model downloads.
- For GPU use: an NVIDIA driver compatible with CUDA 12.4 and GPUs allocated to you.
- For a 70B/72B FP16 judge: roughly 140–150 GB for weights, additional GPU runtime
  memory and enough disk/RAM. At least 160 GB free disk for one model's weights
  is a practical starting point; packages and additional models need more space.

CPU operation supports Studio, ASR and saved results. Local 70B judges require
multiple supported GPUs; these large models are not intended for CPU execution.
Hardware compatibility cannot be supplied by a Python environment.

## 2. Install inside the project

Open a terminal in the cloned repository. For Studio with CUDA 12.4 PyTorch:

```bash
bash scripts/setup_local.sh studio
```

For Studio **and** local judge dependencies:

```bash
bash scripts/setup_local.sh all
```

Or install just the judge later:

```bash
bash scripts/setup_local.sh judge
```

For a machine without NVIDIA GPUs, use:

```bash
bash scripts/setup_local.sh cpu
```

These commands create `.venv-studio/` and, when requested, `.venv-judge/` inside
this checkout. They are separate to prevent vLLM's torch/transformers requirements
from conflicting with ASR. Neither environment uses system site packages. Both
are ignored by Git. Downloads/cache, model weights and run outputs are also
ignored; do not commit environment directories or tokens.

The setup script uses `python3` by default. Select another supported base Python with:

```bash
PYTHON_BIN=python3.10 bash scripts/setup_local.sh all
```

The local judge profile pins vLLM 0.8.5.post1 with its CUDA 12.4-era dependency
stack for older Turing hardware; it is independent of the newer Docker/batch
profile in `requirements-judge.txt`. Upstream documents this version's supported
hardware and recommends a fresh environment:
https://docs.vllm.ai/en/v0.8.5.post1/getting_started/installation/gpu.html
The chosen PyTorch CUDA wheel is documented at:
https://docs.pytorch.org/get-started/previous-versions/

Top-level versions are pinned in the requirements files. For the tested Linux / Python
3.10 profile, setup also applies the resolved transitive versions in `constraints/`.
Python 3.11–3.12 resolve their transitive packages separately; Python 3.10 is the
recommended version for reproducing the verified environment.

### Equivalent manual commands (Python 3.10, Linux NVIDIA)

If you prefer to see every installation step instead of using the setup script:

```bash
python3.10 -m venv .venv-studio
.venv-studio/bin/python -I -m pip install --upgrade pip
.venv-studio/bin/python -I -m pip install 'torch==2.6.0+cu124' --index-url https://download.pytorch.org/whl/cu124
.venv-studio/bin/python -I -m pip install -c constraints/studio-linux-py310.txt -r requirements-studio.txt

python3.10 -m venv .venv-judge
.venv-judge/bin/python -I -m pip install --upgrade pip
.venv-judge/bin/python -I -m pip install -c constraints/judge-linux-py310.txt -r requirements-local-judge.txt
```

## 3. Start Studio

```bash
.venv-studio/bin/python -I -m streamlit run app/cxer_studio.py \
  --server.address 127.0.0.1 --server.port 8501
```

No environment activation is required. `-I` ignores external Python paths and
user site packages. The app adds only its own source directories to Python's
import path. Audio decoding uses the included FFmpeg Python package as a fallback,
so installing system FFmpeg with sudo is not required.

From your laptop, forward the port using your actual Lambda SSH address:

```bash
ssh -L 8501:localhost:8501 YOUR_USER@YOUR_LAMBDA_HOST
```

Leave SSH connected and open http://localhost:8501. On a local machine, open that
URL directly. Stop Streamlit with Ctrl+C after stopping any local judge.

## 4. Devices and judge models

Choose **ASR device** in the sidebar. The GPU list respects `CUDA_VISIBLE_DEVICES`
if you restrict GPUs before launch. Use only GPUs allocated to you. Changing the
ASR device may leave older models cached; use **Release cached ASR models** before
reallocating their GPUs to a judge.

Choose **Run on this machine**, then Llama 3.3 70B, DeepSeek R1 Distill Llama 70B or
Qwen 2.5 72B. Select judge GPUs separately from the ASR GPU and click **Load judge**.
The launcher always uses **this repository's `.venv-judge/bin/python`**. Selecting
a model does not start it until you click Load. GPU counts must support the model's
tensor-parallel layout; sufficient aggregate GPU memory alone is not enough.

For gated models, obtain model access on Hugging Face and export your token
before starting Studio (the app does not automatically read `.env`):

```bash
read -rsp 'Hugging Face token: ' HF_TOKEN; echo
export HF_TOKEN
```

Model caches default to `models/hf/` in this repository. First launch downloads
the chosen model. Inspect **Local judge startup logs** and click **Re-check judge**
to refresh readiness. **Stop local judge** stops the process owned by Studio.
The judge is shared by browser sessions connected to this app; stop it before
exiting Studio. Stop and load again to change judge models or GPUs. Startup logs
are written to `results/studio_judge.log` and replaced on the next launch.

For an existing AWS or other remote server, choose **Connect to a server** and
enter its endpoint including `/v1`. That server controls model loading and GPU
assignment. A remote endpoint is optional; local operation never needs one.

## 5. Saved results

- **FAA results**: original FAA corpus scores, selectable by judge, ASR model and metric.
- **ATC benchmark results**: ATCOSIM test, ATCO2 OOD and UWB ATCC test, selectable
  by dataset, judge, model/variant and metric. All source files needed at runtime
  are in `data/atc_benchmarks/`; see `SOURCE.md` there.

ATC cache `CCER` is a fraction converted to CxER % for display. The raw JSON has
reference/hypothesis pairs, not per-utterance judge verdicts. Switching the live
judge does not recompute these saved benchmark scores.

## 6. Batch uploads and exports

Upload up to 500 files / 1 GiB total. Each file is also subject to Streamlit's
upload limit. All accepted files are processed with the selected models; there
are no batch-duration or computation-time cutoffs.

CSV/Excel has one row per audio/model, saved Lambda audio path, prediction, ASR
runtime, audio duration, callsigns, extraction status/errors and CxER/judge timing
when available. Failed/skipped model predictions are retained with an error.
Browser uploads cannot expose the original laptop path. Decoded uploads are
saved in `results/uploads/<batch-id>/`; delete those folders when no longer needed.

Enable **Extract callsigns from predictions using the judge** for callsigns.
Extraction reads ASR text, so recognition errors can affect it. It requires a
ready judge but no reference. CxER evaluation requires both a judge and a reference.
The ordinary CxER error list alone does not extract every callsign.

## 7. Verify the installation

```bash
.venv-studio/bin/python -I -m pip check
.venv-judge/bin/python -I -m pip check  # if the judge was installed
```

Install the test runner and run tests in the project environment:

```bash
.venv-studio/bin/python -I -m pip install -r requirements-dev.txt
.venv-studio/bin/python -m pytest tests -q
```

The test suite uses mocked judge responses and model inference; GPU model loading
requires a separate hardware smoke test on the target server.

## Transcript postprocessing

Live predictions and references use the same Whisper English number normalizer
(from the installed Transformers package), including aviation “niner” → “nine”.
Raw prediction/reference text is retained in exports and the raw prediction is
available in an expander. Saved FAA/ATC scores are unchanged; live WER uses the
new normalization and is not directly comparable to historic scores without recomputation.

ICAO callsigns use the bundled FAA telephony table in `data/icao_telephony.json`,
with source URL and retrieval date. For example EXECJET 401 → EJA401. Unmapped or
missing callsigns are Unknown. Registration-style callsigns such as N123AB are
also supported. This does not restore callsigns missed by ASR.

Speaker is Pilot, Air Traffic Controller or Unknown. Judge assistance uses the
transcript's instruction/request/readback meaning; callsign position alone cannot
reliably distinguish the roles. Without a judge only explicit pilot cues are
labelled, and ambiguous text remains Unknown. These are text estimates, not
acoustic diarization; multiple-speaker clips should remain Unknown.
CSV, Excel and downloaded JSON omit asr_error, callsign_error and callsign_seconds;
failures still appear in the analysis UI.

### Quiet gaps and repeated Whisper output

**Skip long quiet gaps** is enabled by default in the sidebar and applies equally
to all selected ASR models. It retains energy-active regions with 250 ms margins
and merges short pauses, avoiding decoding long near-silent chunks. This is an
energy heuristic, not a trained voice-activity detector: disable it when quieter
speakers might be lost. Original audio files and reported duration are preserved.
Continuous segments longer than 30 seconds use Whisper's native timestamp-based
long-form decoding with silence/repetition fallback rather than fixed pipeline
chunks. Short segments use ordinary short-form decoding. English-only checkpoints
are detected from model metadata rather than repository naming.

On the supplied 57.93-second EJA429 clip, the original decoder reproduced the long
“you/ju” loop. Separating the two active regions removed that loop. The final region
still decoded “and departure” instead of the supplied reference “to parking”;
this is a remaining recognition error, not something number normalization repairs.
