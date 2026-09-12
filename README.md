# FAA_CxER

**Does a lower Word Error Rate actually mean a *safer* air-traffic-control transcription?**

This project measures, judges, and analyzes **operationally critical**
transcription errors in ASR (speech-to-text) output for air-traffic-control
(ATC) radio communications — on top of the standard Word Error Rate (WER)
everyone already reports. It introduces **Contextual Error Rate (CxER)**: the
fraction of transcriptions that contain an error an air-traffic controller
would actually care about (a wrong callsign, a flipped left/right, a wrong
altitude or runway), as judged by a large language model acting as an expert
evaluator.

The headline finding this codebase produces evidence for — the **WER-CxER
paradox** — is that fine-tuning an ASR model on ATC data drives WER down
dramatically (often 80-90%), but CxER drops by much less, and some
individual utterances get *more* dangerous even as their WER improves. WER
alone is not a safe proxy for "is this transcription trustworthy."

If you're new to this project, read this file top to bottom once — it
explains the concepts, the pipeline, and exactly which script to run for
each step, in order.

---

## 1. The problem, in plain terms

An ASR model transcribes a pilot or controller's radio call. Two kinds of
mistakes it can make are very different in consequence:

| Example | WER impact | Operationally |
|---|---|---|
| Reference: `descend to altitude` → Hypothesis: `descend altitude` | 1 word "wrong" | **Fine** — same instruction |
| Reference: `runway two seven **left**` → Hypothesis: `runway two seven **right**` | 1 word "wrong" | **Critical** — sends the aircraft to the wrong runway |
| Reference: `american twenty one **ninety six**` → Hypothesis: `american twenty one **nine six**` | 1-2 words "wrong" | **Critical** — wrong callsign, could apply an instruction to the wrong aircraft |

WER treats all three the same: one wrong word out of N. It cannot tell "safe
paraphrase" from "safety-critical error." **CxER** is built specifically to
tell them apart, using ICAO phonetic-alphabet and ATC-phraseology rules (e.g.
"tree" = 3, "niner" = 9, "FL350" = "flight level three five zero" are
*equivalent*, but a changed callsign digit, direction, altitude, heading,
runway, frequency, clearance, or waypoint is *critical*).

## 2. How CxER is measured

There's no large hand-labeled dataset of "which ATC transcription errors are
dangerous," so this project uses an **LLM as judge**: a large instruction-following
model (default: Llama-3.3-70B-Instruct, run locally under
[vLLM](https://github.com/vllm-project/vllm)) is given the reference
transcript and the ASR hypothesis, told the ICAO/ATC equivalence rules, and
asked to classify the pair as:

- **`Equivalent`** — no operationally critical error (wording differs, meaning doesn't)
- **`Critical_Errors`** — a critical/contextual error was found
- **`Parse_Failure`** — the judge's output was empty, malformed, or otherwise untrustworthy (excluded from the rate)

For one (judge, ASR model) pair:

```
CxER = Critical_Errors / (Critical_Errors + Equivalent)
```

(`Parse_Failure` samples are excluded from both the numerator and
denominator — a judge that fails to answer isn't evidence either way.)

Several judge **prompt variants** exist (`base`, `cot`, `cot_v3`,
`compressed`, `no_examples`, `lenient` — see
[`faa_cxer/judge/prompts/`](faa_cxer/judge/prompts/)) as an ablation: does a
more detailed chain-of-thought prompt change the measured CxER? `cot_v3` is
the default used for the headline results.

## 3. The pipeline, end to end

```
 ┌─────────────────────┐
 │ normalized           │  one row per audio sample:
 │ predictions JSON     │  {reference, audio_path, <model_1>: hyp, <model_2>: hyp, ...}
 └──────────┬───────────┘
            │  stage 1 — scripts/01_compute_benchmark_scores.py
            │  (faa_cxer.metrics.BenchmarkScorer)
            ▼
 ┌─────────────────────┐
 │ predictions +        │  adds <model>_WER / _BERTDist / _SemDist per sample,
 │ WER/BERTDist/SemDist │  + corpus_metrics per model
 └──────────┬───────────┘
            │  stage 2 — scripts/02_run_cxer_judge.py   (needs a GPU + vLLM)
            │  (faa_cxer.judge.CxERJudgePipeline)
            ▼
 ┌─────────────────────┐
 │ cxer_labels/         │  one JSON per (judge, asr_model):
 │  <judge>/<model>.json│  per-sample classification + corpus-level CxER
 └──────────┬───────────┘
            │  stage 3 — scripts/03_merge_cxer_labels.py
            │  (faa_cxer.analysis.CxERMetricsMerger)
            ▼
 ┌─────────────────────┐
 │ merged JSON:          │  every sample now also has
 │ metrics + CxER labels │  sample["cxer"][judge][asr_model] = {classification, ...}
 └──────────┬───────────┘
            │
            ├── scripts/04_analyze_token_frequency.py  → do rare/domain words improve less than common ones?
            ├── scripts/05_analyze_transitions.py      → per-sample safety drift (Equivalent↔Critical) + Sankey
            ├── scripts/06_analyze_paradox.py          → low-WER-but-critical concentration/trust-violation table + scatter plots
            └── scripts/07_make_comparison_plots.py    → the headline WER vs BERTDist vs SemDist vs CxER bar chart
```

Each stage's script is a thin CLI wrapper (argument parsing + wiring) around
a class in the `faa_cxer` package that does the actual work — see
[§6](#6-package-layout) for what lives where.

## 4. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Stage 2 (the LLM judge) needs a CUDA GPU host and vLLM, which pins its own
torch/CUDA build — install it separately, only on the machine that will run
the judge:

```bash
pip install -r requirements-judge.txt
```

You do **not** need a GPU or vLLM to run stages 1 (partially — BERTScore/SemDist
use a GPU if available but fall back to CPU), 3, 4, 5, 6, or 7 against the
data already included in `data/` — see [§7](#7-quickstart).

## 5. Data

`data/` ships with the canonical data this project's results are built from
(see [§9](#9-what-was-carried-over-from-faa_emnlp) for what was intentionally
left out as superseded/debug artifacts):

```
data/
├── predictions/
│   ├── FAA_multi_model_predictions_normalized.json          # stage 1 input  (6,364 samples × 10 ASR models)
│   ├── FAA_multi_model_predictions_with_metrics.json         # stage 1 output (+ WER/BERTDist/SemDist)
│   ├── FAA_multi_model_predictions_with_metrics_and_cxer.json# stage 3 output (+ CxER labels) — most analysis/plot scripts read THIS file
│   └── FAA_metrics_summary.csv                                # flat corpus-level WER/BERTDist/SemDist summary
├── cxer_labels/
│   └── Llama-3.3-70B-Instruct/
│       ├── FAA_Model.json                                     # one file per ASR model judged
│       ├── FAA_Model_prediction_whisper-medium.en.json
│       └── ...
└── eval_cache/                                                 # small cached corpus-level CSVs from earlier exploratory runs
```

**Predictions JSON schema** (wide format — one row per audio sample, one
column per ASR model):

```json
{
  "audio_path": "/path/to/LAX_0010__20180102T221956740Z.wav",
  "reference": "ground i believe original for skywest thirty three fifteen were charlie six charlie to the gate",
  "FAA_Model_prediction_whisper-large-v3": "ground, i believe original instructions for skyler 3315 were charlie 6, charlie to the gate.",
  "FAA_Model_prediction_FAA_Whisper_L_TL_500K": "...",
  "FAA_Model_prediction_FAA_Whisper_L_TL_500K_WER": 0.083,          // added by stage 1
  "FAA_Model_prediction_FAA_Whisper_L_TL_500K_BERTDist": 0.012,      // added by stage 1
  "FAA_Model_prediction_FAA_Whisper_L_TL_500K_SemDist": 0.0004,      // added by stage 1
  "cxer": {                                                          // added by stage 3
    "Llama-3.3-70B-Instruct": {
      "FAA_Model_prediction_FAA_Whisper_L_TL_500K": {
        "classification": "Equivalent",
        "meaning_preserved": true,
        "operational_safety": "safe",
        "explanation": "...",
        "transcription_errors": []
      }
    }
  }
}
```

**CxER label file schema** (`data/cxer_labels/<judge>/<asr_model>.json`, one
per model the judge evaluated):

```json
{
  "asr_model": "FAA_Model_prediction_FAA_Whisper_L_TL_500K",
  "judge_model": "meta-llama/Llama-3.3-70B-Instruct",
  "prompt_type": "cot_v3",
  "status": "complete",
  "num_samples_total": 6364,
  "contextual_error_rate": 0.225,
  "records": [ {"sample_index": 0, "classification": "Equivalent", "...": "..."} ]
}
```

### Model families compared

Three ASR architectures, each pretrained vs. fine-tuned on 500K FAA/ATC
utterances (see `faa_cxer.config.DEFAULT_MODEL_PAIRS`):

| Family | Pretrained checkpoint | Fine-tuned checkpoint |
|---|---|---|
| Whisper Medium | `openai/whisper-medium.en` | `FAA_Whisper_M_TL_500K` |
| Whisper Large | `openai/whisper-large-v3` | `FAA_Whisper_L_TL_500K` |
| Parakeet | `nvidia/parakeet-tdt-0.6b-v2` | `FAA_parakeet_TL_500K` |

`FAA_Model` is the standalone production model (no pretrained counterpart in
this comparison).

## 6. Package layout

```
faa_cxer/
├── config.py            # Config dataclass — every file path & default in one place
├── io_utils.py           # TextNormalizer, atomic JSON save/load — shared by every stage
├── metrics/               # Stage 1: WER / BERTScore / SemDist, + token-frequency analysis
│   ├── wer.py              WERScorer          — corpus/sample WER, word-level alignment
│   ├── semantic.py          SemanticDistanceScorer — BERTDist + SemDist (shared RoBERTa-large)
│   ├── benchmark.py          BenchmarkScorer    — orchestrates stage 1 end to end
│   └── token_frequency.py    TokenFrequencyAnalyzer — error rate by reference-word frequency bucket
├── judge/                 # Stage 2: the LLM-as-judge CxER classifier
│   ├── prompts/              PromptRegistry + one module per prompt variant
│   ├── json_extractor.py     JudgeResponseParser — robust JSON recovery from raw LLM text
│   ├── evaluator.py           CxEREvaluator      — batched classification + GPU-OOM backoff
│   └── pipeline.py            CxERJudgePipeline  — resumable, crash-safe run orchestration
├── analysis/               # Stage 3 + downstream analysis
│   ├── merge.py               CxERMetricsMerger  — stage 3: combine metrics + CxER labels
│   ├── transitions.py         TransitionAnalyzer — per-sample safety drift (Eq↔Critical)
│   └── paradox.py             ParadoxAnalyzer    — low-WER-but-critical concentration/trust-violation
└── viz/                    # Plotting — one class per figure type, all reusable
    ├── comparison_bars.py     ComparisonBarChart    — the 2x2 WER/BERTDist/SemDist/CxER bar chart
    ├── frequency_plots.py     TokenFrequencyPlotter — rare-vs-frequent-word plots
    ├── paradox_scatter.py     ParadoxScatterPlotter — WER-jitter scatter with tau threshold
    └── sankey.py              SafetyDriftSankeyPlotter — Sankey diagram of classification transitions
```

Every class takes plain data in (paths, lists of dicts, a `Config`) and
returns plain data out (dicts, `pandas.DataFrame`s, or a saved file) — there
is no hidden global state, so any piece can be imported and used from a
notebook, a test, or a different script without dragging in the rest of the
pipeline.

## 7. Quickstart

All commands assume you're in the `FAA_CxER/` directory with the venv from
[§4](#4-setup) active.

```bash
# Stage 1 — score every ASR model's predictions (WER, BERTDist, SemDist).
# Already-computed output ships in data/predictions/, so this is only
# needed if you add a new model column or change the normalization.
python scripts/01_compute_benchmark_scores.py

# Stage 2 — run the LLM judge (needs a GPU host with vLLM; see requirements-judge.txt).
# CUDA_VISIBLE_DEVICES must be set BEFORE this runs (vLLM reads it at import time).
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_TOKEN=hf_...                 # or put HF_TOKEN=... in a .env file
python scripts/02_run_cxer_judge.py --max-samples 10       # sanity check first
python scripts/02_run_cxer_judge.py                        # full run — safe to Ctrl-C and resume

# Stage 3 — merge CxER labels into the metrics file. Already-computed
# output ships in data/predictions/FAA_multi_model_predictions_with_metrics_and_cxer.json.
python scripts/03_merge_cxer_labels.py

# Everything below reads the merged file from stage 3 and needs no GPU:
python scripts/04_analyze_token_frequency.py
python scripts/05_analyze_transitions.py
python scripts/06_analyze_paradox.py
python scripts/07_make_comparison_plots.py
```

Every script accepts `--help` for its full option list (input/output paths,
which model pairs, which judge, threshold settings, etc.) — defaults come
from `faa_cxer.config.default_config()` and point at the paths above.

There's also [`notebooks/exploration.ipynb`](notebooks/exploration.ipynb),
a short interactive companion for poking at the merged dataset and
rendering one figure inline — the pipeline stages themselves are meant to
be run as the scripts above, not copy-pasted into notebook cells (that's
what the old FAA_EMNLP notebooks did, and it's exactly the duplication this
restructuring removes).

Example figures from a full run live in
[`outputs/examples/`](outputs/examples/) so you can see what each script
produces without running anything.

## 8. Tests

A small pytest suite covers the pure-logic pieces that don't need a GPU or
model weights — the judge's JSON-recovery logic, the prompt registry, and
the transition/bucket analysis:

```bash
pip install pytest
pytest tests/ -v
```

## 9. What was carried over from FAA_EMNLP

This package is a restructuring of `../FAA_EMNLP` into a proper Python
package (nothing in `FAA_EMNLP` was modified). Concretely:

- **~15 near-duplicate notebook cells** in `FAA_graphs.ipynb` / `Helper.ipynb`
  (the same pretrained-vs-finetuned bar chart, the same WER-paradox scatter
  plot, etc. — each copy-pasted with different hardcoded paths/labels)
  became **one parameterized class each** (`ComparisonBarChart`,
  `ParadoxScatterPlotter`, ...) driven by `faa_cxer.config.DEFAULT_MODEL_PAIRS`.
- `token_frequency_analysis.py` originally read a separate narrow-format
  data source (`output_transcriptions_faa_Claude/`, outside `FAA_EMNLP`)
  with one file per model. It's been rewritten to work off the same wide
  merged-JSON schema every other analysis class uses — one fewer file
  format for the whole pipeline to carry around.
- **Superseded/debug data was left behind, not copied**: `cxer-lables`
  (a 1,000-sample partial run, superseded by the full 6,364-sample
  `cxer-lables-all`), `cxer-lables-all_v0` (an identical duplicate backup),
  `cxer-lables_sanity` (a 10-sample smoke test), and `parakeet_debug.csv`.
  If you need any of those, they're untouched in `../FAA_EMNLP/`.
- `judge/prompts/*.py` template files are copied verbatim — the wording of
  a judge prompt is experimental content, not implementation detail, so it
  wasn't rewritten.

## 10. Before pushing this to GitHub

`data/` (~150MB) and `outputs/examples/` (~13MB) contain ATC-transcript-derived
content copied in at your request. Before making this repository public (or
even just pushing it), decide how to handle that:

- Keep the repo private, or
- Move `data/`/`outputs/` out from under git and reference them via
  `faa_cxer.config.Config(data_dir=..., outputs_dir=...)` instead, or
- Use [Git LFS](https://git-lfs.github.com/) for the JSON/PNG files, or
- Confirm the transcript content is already de-identified/cleared for public release.

Nothing in `.gitignore` excludes `data/`/`outputs/` by default — that's
deliberate, so `git status` shows you exactly what you're about to commit
rather than silently hiding it.
