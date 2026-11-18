#!/usr/bin/env python3
"""FAA CxER Studio: upload ATC audio, transcribe it with pretrained and
fine-tuned ASR models, score WER, and (when a GPU judge is available) run
the LLM CxER judge on every transcription.

    streamlit run app/cxer_studio.py

ASR runs on a GPU when one is visible and on the CPU otherwise. The CxER
judge (Llama-3.3-70B-Instruct under vLLM) needs a multi-GPU host; it runs
as a project-local subprocess or a remote server, and the
app talks to it over HTTP. Without it the app still transcribes and scores
WER, and says what is needed for CxER.

Settings (environment variables):
    JUDGE_URL        Optional OpenAI-compatible vLLM endpoint; empty by default
    Local judge uses the repository-owned .venv-judge environment.
    ASR_MODELS       comma-separated HF ids offered in the model picker
    CXER_ASR_BACKEND "hf" (default) or "mock" (UI testing without model downloads)
"""

from __future__ import annotations

import io
import uuid
import hashlib
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_HOME"] = str(ROOT / "models" / "hf")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from faa_cxer.io_utils import TextNormalizer  # noqa: E402
from faa_cxer.judge.prompts import PromptRegistry  # noqa: E402

from judge_client import RemoteCxEREvaluator, probe_judge  # noqa: E402
from local_judge import JUDGE_MODELS, LocalJudge, gpu_choices  # noqa: E402
from audio_utils import load_audio, active_audio_regions  # noqa: E402
from transcript_processing import normalize_transcript, extract_icao_callsigns, infer_speaker  # noqa: E402

DATA = ROOT / "data"
AUDIO_DIR = DATA / "audio_samples"
CURATED = DATA / "curated_samples.json"
LABELS_DIR = DATA / "cxer_labels"
METRICS_CSV = DATA / "predictions" / "FAA_metrics_summary.csv"
FIGURES_DIR = ROOT / "outputs" / "examples"

JUDGE_URL = os.environ.get("JUDGE_URL", "")
ASR_BACKEND = os.environ.get("CXER_ASR_BACKEND", "hf")

# label -> HF model id. The first two are the pair the live demo already uses.
DEFAULT_ASR_MODELS = {
    "Whisper Medium (pretrained)": "openai/whisper-medium.en",
    "Whisper Medium (FAA fine-tuned 500K)": "ttkairamkonda/FAA_Whisper_M_TL_500K",
    "Whisper Large v3 (pretrained)": "openai/whisper-large-v3",
    "Whisper Large (FAA fine-tuned 500K)": "ttkairamkonda/FAA_Whisper_L_TL_500K",
}
if os.environ.get("ASR_MODELS"):
    DEFAULT_ASR_MODELS = {m.strip(): m.strip() for m in os.environ["ASR_MODELS"].split(",") if m.strip()}

VERDICT_STYLE = {
    "Equivalent": ("🟢", "Equivalent", "No operationally critical error"),
    "Critical_Errors": ("🔴", "Critical error", "An air-traffic controller would care about this"),
    "Parse_Failure": ("⚪", "No verdict", "The judge's answer could not be parsed"),
}
RISK_ICON = {"high": "🔴 high", "medium": "🟠 medium", "low": "🟡 low"}

st.set_page_config(page_title="FAA CxER Studio", page_icon="🛫", layout="wide")

st.markdown(
    """
<style>
.block-container {padding-top: 1.6rem; max-width: 1300px;}
.cx-hero h1 {font-size: 1.9rem; margin-bottom: .2rem;}
.cx-hero p {color: #6b7280; margin-top: 0;}
.cx-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px; padding: 14px 16px; margin-bottom: 10px;}
.cx-hyp {font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .95rem; line-height: 1.45;}
.cx-pill {display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;}
.cx-ok {background: rgba(16,185,129,.15); color: #059669;}
.cx-bad {background: rgba(239,68,68,.15); color: #dc2626;}
.cx-na {background: rgba(107,114,128,.15); color: #6b7280;}
</style>
""",
    unsafe_allow_html=True,
)

# ── system detection ─────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def gpu_info() -> Dict:
    try:
        import torch

        if torch.cuda.is_available():
            gpus = [torch.cuda.get_device_properties(i) for i in range(torch.cuda.device_count())]
            return {"cuda": True, "gpus": [(g.name, g.total_memory / 1e9) for g in gpus]}
    except Exception:
        pass
    return {"cuda": False, "gpus": []}


def judge_status(url: str):
    # Re-probed on each rerun (cheap) so the badge flips as soon as the judge finishes loading.
    return probe_judge(url)


# ── ASR ──────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner=False)
def load_asr(model_id: str, device: int = -1):
    """Load an ASR checkpoint. Whisper models are loaded explicitly (model +
    processor) instead of through pipeline(model=<id>), so a failure shows
    the real cause instead of the generic "could not load with any class"."""
    if ASR_BACKEND == "mock":
        return None
    import torch
    from transformers import AutoProcessor, pipeline

    cuda = device >= 0 and torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    token = os.environ.get("HF_TOKEN") or None

    if "whisper" in model_id.lower():
        from transformers import WhisperForConditionalGeneration

        model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype, low_cpu_mem_usage=True, token=token)
        processor = AutoProcessor.from_pretrained(model_id, token=token)
        return pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device if cuda else -1,
            torch_dtype=dtype,
        )
    return pipeline("automatic-speech-recognition", model=model_id, device=device if cuda else -1, torch_dtype=dtype, token=token)



def transcribe(model_id: str, array, sr, device: int = -1, skip_quiet: bool = True) -> str:
    if ASR_BACKEND == "mock":
        return f"mock transcription from {model_id.split('/')[-1]}"
    pipe = load_asr(model_id, device)
    regions = active_audio_regions(array, sr) if skip_quiet else [[0, len(array)]]
    texts = []
    for start, end in regions:
        segment = array[start:end]
        if not len(segment):
            continue
        if "whisper" in model_id.lower():
            generation = {"do_sample": False}
            # Inspect the checkpoint, not its repo name: FAA medium is English-only.
            if getattr(pipe.model.generation_config, "is_multilingual", False):
                generation.update(language="en", task="transcribe")
            call_options = {}
            if len(segment) > 30 * sr:
                # Whisper's native long-form decoder handles timestamps/silence.
                call_options["return_timestamps"] = True
                generation.update(condition_on_prev_tokens=False,
                                  compression_ratio_threshold=2.4, logprob_threshold=-1.0,
                                  no_speech_threshold=0.6,
                                  temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
            out = pipe({"array": segment, "sampling_rate": sr},
                       generate_kwargs=generation, **call_options)
        else:
            out = pipe({"array": segment, "sampling_rate": sr}, chunk_length_s=30, batch_size=4)
        texts.append(out["text"].strip())
    return " ".join(texts).strip()


# ── metrics ──────────────────────────────────────────────────────────────

_norm = TextNormalizer()


def wer(reference: str, hypothesis: str) -> Optional[float]:
    import jiwer

    ref, hyp = normalize_transcript(reference), normalize_transcript(hypothesis)
    if not ref:
        return None
    return jiwer.wer(ref, hyp if hyp else "<empty>")


# ── reference data ───────────────────────────────────────────────────────


@st.cache_data(show_spinner=False)
def curated_samples() -> Dict:
    if not CURATED.exists():
        return {"judge": "", "samples": []}
    return json.loads(CURATED.read_text())


@st.cache_data(show_spinner=False)
def curated_by_hash() -> Dict[str, Dict]:
    out = {}
    for s in curated_samples()["samples"]:
        p = AUDIO_DIR / s["audio_file"]
        if p.exists():
            out[hashlib.sha256(p.read_bytes()).hexdigest()] = s
    return out


@st.cache_data(show_spinner="Loading benchmark results…")
def benchmark_table() -> pd.DataFrame:
    rows = []
    wer_df = pd.read_csv(METRICS_CSV) if METRICS_CSV.exists() else pd.DataFrame(columns=["model", "WER"])
    wer_map = dict(zip(wer_df["model"], wer_df["WER"]))
    for f in sorted(LABELS_DIR.glob("*/*.json")):
        d = json.loads(f.read_text())
        counts = {"Equivalent": 0, "Critical_Errors": 0, "Parse_Failure": 0}
        for r in d.get("records", []):
            counts[r.get("classification", "Parse_Failure")] = counts.get(r.get("classification", "Parse_Failure"), 0) + 1
        judged = counts["Equivalent"] + counts["Critical_Errors"]
        key = d.get("asr_model", f.stem)
        rows.append({
            "ASR model": key.replace("FAA_Model_prediction_", ""),
            "Fine-tuned": "FAA_" in key.replace("FAA_Model_prediction_", "") or key == "FAA_Model",
            "WER %": round(float(wer_map[key]), 2) if key in wer_map else None,
            "CxER %": round(100 * counts["Critical_Errors"] / judged, 2) if judged else None,
            "Critical": counts["Critical_Errors"],
            "Equivalent": counts["Equivalent"],
            "Parse failures": counts["Parse_Failure"],
            "Judge": f.parent.name,
        })
    return pd.DataFrame(rows).sort_values("CxER %") if rows else pd.DataFrame()


# ── rendering helpers ────────────────────────────────────────────────────


def pill(classification: Optional[str]) -> str:
    if classification is None:
        return '<span class="cx-pill cx-na">CxER: not run</span>'
    icon, label, _ = VERDICT_STYLE.get(classification, ("⚪", classification, ""))
    cls = {"Equivalent": "cx-ok", "Critical_Errors": "cx-bad"}.get(classification, "cx-na")
    return f'<span class="cx-pill {cls}">{icon} {label}</span>'


def errors_frame(errors: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Entity": e.get("entity_type", ""),
            "Error": e.get("error_name", ""),
            "Expected": e.get("reference_value", ""),
            "Heard": e.get("transcribed_value", ""),
            "Risk": RISK_ICON.get(str(e.get("safety_risk", "")).lower(), e.get("safety_risk", "")),
            "Why": e.get("explanation", ""),
        }
        for e in errors if isinstance(e, dict)
    ])


# ── sidebar ──────────────────────────────────────────────────────────────

@st.cache_resource
def local_judge_manager():
    return LocalJudge(ROOT / "results" / "studio_judge.log")


info = gpu_info()
gpus = gpu_choices()
manager = local_judge_manager()
with st.sidebar:
    st.markdown("### 🛫 FAA CxER Studio")
    st.caption("Contextual Error Rate for ATC speech recognition")

    st.markdown("#### System")
    if info["cuda"]:
        for name, mem in info["gpus"]:
            st.success(f"GPU: {name} · {mem:.0f} GB", icon="⚡")
        st.caption("Choose an ASR device below.")
    else:
        st.info("No GPU detected. ASR runs on the CPU (slower, still works).", icon="🖥️")

    asr_device = st.selectbox(
        "ASR device", [-1] + list(gpus), index=1 if gpus else 0,
        format_func=lambda i: "CPU" if i == -1 else gpus[i][1],
    )
    skip_quiet = st.checkbox(
        "Skip long quiet gaps", value=True,
        help="Apply the same energy-based segmentation to all models. Disable for very quiet speech. "
             "Short pauses and 250 ms margins are retained; the original file is unchanged.",
    )
    if st.button("Release cached ASR models"):
        load_asr.clear()
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        st.success("Cached ASR models released.")

    st.markdown("#### CxER judge")
    location = st.radio("Judge location", ["Connect to a server", "Run on this machine"])
    if location == "Run on this machine":
        label = st.selectbox("Judge model", list(JUDGE_MODELS))
        selected_model = JUDGE_MODELS[label]
        judge_gpus = st.multiselect(
            "GPUs for judge", list(gpus), format_func=lambda i: gpus[i][1],
            help="Choose GPUs allocated to you. Use different GPUs for ASR and the judge.",
        )
        st.caption("Local judge environment: .venv-judge inside this repository. "
                   "Create it with: bash scripts/setup_local.sh judge")
        port = st.number_input("Local judge port", 1024, 65535, 8000)
        memory = st.slider("Judge GPU memory fraction", 0.50, 0.95, 0.80, 0.05)
        judge_url = f"http://127.0.0.1:{port}/v1"
        st.caption("Each judge needs roughly 140–150 GB for FP16 weights, plus runtime memory. "
                   "GPU architecture, free memory and tensor parallel support also matter.")
        if asr_device in judge_gpus:
            st.warning("ASR and judge share a GPU. Choose separate GPUs to avoid running out of memory.")
        if st.button("Load judge", disabled=manager.running() or not judge_gpus):
            try:
                if probe_judge(judge_url).ok:
                    st.error("A server already uses this endpoint. Connect to it or choose another port.")
                else:
                    manager.start(selected_model, [gpus[i][0] for i in judge_gpus], port, memory)
            except Exception as exc:
                st.error(str(exc))
        if st.button("Stop local judge", disabled=not manager.running()):
            manager.stop()
        if manager.description:
            state = "Running / loading" if manager.running() else "Stopped"
            st.caption(f"{state}: {manager.description}")
        with st.expander("Local judge startup logs"):
            st.code(manager.logs(), language="text")
        st.caption("The local judge is shared by browser sessions connected to this app. "
                   "Stop it explicitly before closing Studio. "
                   "vLLM is installed in this project with bash scripts/setup_local.sh judge. "
                   "Loading can take several minutes; use Re-check judge to refresh.")
    else:
        judge_url = st.text_input("CxER judge endpoint", value=JUDGE_URL,
                                  help="OpenAI-compatible vLLM server, including /v1")
        st.caption("The server controls its GPUs and which models are loaded.")

    js = judge_status(judge_url)
    selected_served_model = js.served_model
    if js.ok:
        try:
            import requests
            response = requests.get(judge_url.rstrip("/") + "/models", timeout=3)
            response.raise_for_status()
            served = [m["id"] for m in response.json().get("data", [])]
            if served:
                selected_served_model = st.selectbox("Served judge model", served)
        except (requests.RequestException, ValueError, KeyError):
            pass
        st.success(f"CxER judge ready: {selected_served_model}", icon="⚖️")
    else:
        st.warning(f"CxER judge unavailable. {js.message}.", icon="⚖️")
    if st.button("Re-check judge", width="stretch"):
        st.rerun()

    st.markdown("#### Models")
    model_labels = st.multiselect(
        "ASR models to compare",
        list(DEFAULT_ASR_MODELS),
        default=list(DEFAULT_ASR_MODELS)[:2],
        help="Checkpoints download from Hugging Face on first use and are cached.",
    )
    prompt_type = st.selectbox("Judge prompt", PromptRegistry().available(), index=PromptRegistry().available().index("cot_v3"))
    run_judge_opt = st.toggle("Run CxER judge", value=js.ok, disabled=not js.ok,
                              help="Needs the GPU judge server" if not js.ok else "Judge every transcription that has a reference")

# ── header ───────────────────────────────────────────────────────────────

st.markdown(
    '<div class="cx-hero"><h1>Does a lower Word Error Rate mean a safer transcription?</h1>'
    "<p>Transcribe air-traffic-control audio with pretrained and FAA fine-tuned models, then let an "
    "LLM judge flag the errors a controller would actually care about: callsigns, altitudes, headings, "
    "runways, clearances.</p></div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("ASR device", gpus[asr_device][1] if asr_device >= 0 else "CPU")
c2.metric("CxER judge", "Ready" if js.ok else "Needs GPU")
c3.metric("Models selected", len(model_labels))

if not js.ok:
    st.warning(
        "**No judge is connected.** Transcription and WER remain available. "
        "Use the sidebar to load a local judge on allocated GPUs, or connect to a running server. "
        "CxER and judge-assisted speaker labels require a ready judge; CxER also requires a reference transcript.",
        icon="⚠️",
    )

tab_analyze, tab_bench, tab_saved, tab_about = st.tabs(["🎧 Analyze audio", "📊 FAA results", "📚 ATC benchmark results", "ℹ️ How it works"])

# ── tab: analyze ─────────────────────────────────────────────────────────

with tab_analyze:
    source = st.radio("Audio source", ["Upload files", "Use a bundled ATC clip"], horizontal=True)
    items: List[Dict] = []  # {name, bytes, reference, known}

    if source == "Upload files":
        files = st.file_uploader(
            "Upload ATC audio (wav, mp3, flac, m4a, ogg)",
            type=["wav", "mp3", "flac", "m4a", "ogg"],
            accept_multiple_files=True,
        )
        known = curated_by_hash()
        for f in files or []:
            raw = f.getvalue()
            match = known.get(hashlib.sha256(raw).hexdigest())
            with st.expander(f"📄 {f.name}", expanded=len(files) <= 3):
                st.audio(raw)
                ref = st.text_area(
                    "Reference transcript (ground truth)",
                    value=match["reference"] if match else "",
                    key=f"ref_{f.name}_{len(raw)}",
                    placeholder="e.g. delta four five two turn left heading two seven zero",
                    help="Needed for WER and for the CxER judge, which compares each transcription against it.",
                    height=70,
                )
            items.append({"name": f.name, "bytes": raw, "reference": ref, "known": match})
    else:
        samples = curated_samples()["samples"]
        if not samples:
            st.info("No bundled clips found in data/audio_samples.")
        else:
            labels = [f"{s['family']}: {s['title']}" for s in samples]
            picks = st.multiselect("Clips", labels, default=labels[:1])
            for lab in picks:
                s = samples[labels.index(lab)]
                raw = (AUDIO_DIR / s["audio_file"]).read_bytes()
                with st.expander(f"🎙️ {lab}", expanded=True):
                    st.audio(raw)
                    ref = st.text_area("Reference transcript", value=s["reference"], key=f"ref_s_{s['id']}", height=70)
                items.append({"name": s["audio_file"], "bytes": raw, "reference": ref, "known": s})

    st.caption("Batch limits: 500 files, 1 GiB uploaded bytes. "
               "Browser uploads do not expose original local paths; exports use the saved Lambda paths.")
    extract_callsigns = st.checkbox("Refine callsigns and infer speaker using the judge", value=js.ok, disabled=not js.ok)
    st.caption("Predictions and references use Whisper English number normalization. ICAO callsigns are extracted locally; "
               "judge-assisted speaker labels are estimates, not acoustic speaker identification. Missing values are Unknown.")
    size_ok = len(items) <= 500 and sum(len(i["bytes"]) for i in items) <= 1024**3
    if not size_ok:
        st.error("Batch exceeds 500 files or 1 GiB. Reduce the upload.")
    can_run = bool(items) and bool(model_labels) and size_ok
    go = st.button("▶ Run analysis", type="primary", disabled=not can_run, width="stretch")
    if not model_labels:
        st.caption("Pick at least one ASR model in the sidebar.")

    if go:
        results = []
        do_judge = run_judge_opt and js.ok
        upload_dir = ROOT / "results" / "uploads" / uuid.uuid4().hex
        upload_dir.mkdir(parents=True, exist_ok=True)
        extractor = RemoteCxEREvaluator(judge_url, selected_served_model) if extract_callsigns and js.ok else None
        with st.status("Running analysis…", expanded=True) as status:
            for item_index, item in enumerate(items):
                st.write(f"**{item['name']}**: loading audio")
                saved_path = upload_dir / f"{item_index:04d}_{Path(item['name']).name}"
                saved_path.write_bytes(item["bytes"])
                try:
                    array, sr = load_audio(item["bytes"], item["name"])
                except Exception as e:  # unreadable / unsupported file
                    st.error(f"Could not read {item['name']}: {e}")
                    for label in model_labels:
                        results.append({"file": item["name"], "audio_path": str(saved_path), "model": label,
                                        "model_id": DEFAULT_ASR_MODELS[label], "reference": item["reference"],
                                        "hypothesis": "", "WER": None, "asr_error": f"Audio decode failed: {e}",
                                        "cxer": None, "asr_seconds": 0, "audio_seconds": None,
                                        "callsigns": [], "callsign_status": "not run", "callsign_error": None})
                    continue
                for label in model_labels:
                    mid = DEFAULT_ASR_MODELS[label]
                    st.write(f"Transcribing with {label} on {gpus[asr_device][1] if asr_device >= 0 else 'CPU'}…")
                    t0 = time.time()
                    try:
                        raw_hyp = transcribe(mid, array, sr, asr_device, skip_quiet)
                        hyp = normalize_transcript(raw_hyp)
                        err = None
                    except Exception as e:
                        import traceback

                        traceback.print_exc()  # full trace in the Streamlit terminal
                        raw_hyp, hyp, err = "", "", f"{type(e).__name__}: {e}"
                    callsigns, callsign_error = extract_icao_callsigns(hyp), None
                    speaker = infer_speaker(hyp)
                    asr_elapsed = round(time.time() - t0, 2)
                    extraction_started = time.monotonic()
                    if extractor and hyp:
                        try:
                            metadata = extractor.extract_metadata(hyp)
                            speaker = metadata["speaker"]
                            for mention in metadata["callsigns"]:
                                if normalize_transcript(mention) in hyp:
                                    callsigns.extend(extract_icao_callsigns(mention))
                            callsigns = list(dict.fromkeys(callsigns))
                        except Exception as exc:
                            callsign_error = str(exc)
                    results.append({
                        "audio_path": str(saved_path), "callsigns": callsigns, "speaker": speaker,
                        "raw_hypothesis": raw_hyp, "raw_reference": item["reference"].strip(),
                        "callsign_error": callsign_error,
                        "callsign_status": "failed" if callsign_error else ("extracted" if extractor and hyp else "not run"),
                        "callsign_seconds": round(time.monotonic() - extraction_started, 2),
                        "file": item["name"], "model": label, "model_id": mid,
                        "reference": normalize_transcript(item["reference"]), "hypothesis": hyp,
                        "asr_seconds": asr_elapsed,
                        "audio_seconds": round(len(array) / sr, 1),
                        "WER": wer(item["reference"], hyp) if item["reference"].strip() and not err else None,
                        "asr_error": err, "cxer": None,
                    })

            if do_judge:
                todo = [r for r in results if r["reference"] and r["hypothesis"]]
                if todo:
                    st.write(f"Running the CxER judge on {len(todo)} transcription(s)…")
                    try:
                        ev = RemoteCxEREvaluator(judge_url, selected_served_model, selected_served_model, prompt_type=prompt_type)
                        verdicts = ev.judge_pairs([(normalize_transcript(r["reference"]), r["hypothesis"]) for r in todo])
                        for r, v in zip(todo, verdicts):
                            r["cxer"] = v
                    except Exception as e:
                        st.error(f"CxER judge failed: {e}")
            status.update(label="Analysis complete", state="complete", expanded=False)
        st.session_state["results"] = results
        st.session_state["judged"] = do_judge

    results = st.session_state.get("results")
    if results:
        st.divider()
        judged = st.session_state.get("judged", False)
        with_ref = [r for r in results if r["WER"] is not None]
        crit = [r for r in results if r["cxer"] and r["cxer"]["classification"] == "Critical_Errors"]
        evald = [r for r in results if r["cxer"] and r["cxer"]["classification"] in ("Critical_Errors", "Equivalent")]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Transcriptions", len(results))
        m2.metric("Mean WER", f"{100 * sum(r['WER'] for r in with_ref) / len(with_ref):.1f}%" if with_ref else "–")
        m3.metric("CxER", f"{100 * len(crit) / len(evald):.0f}%" if evald else ("–" if judged else "Needs GPU"))
        m4.metric("Critical errors", len(crit) if judged else "–")

        if not any(r["reference"] for r in results):
            st.info("Add a reference transcript to get WER and a CxER verdict. Without one, only the transcriptions are shown.")

        for fname in dict.fromkeys(r["file"] for r in results):
            rows = [r for r in results if r["file"] == fname]
            st.markdown(f"#### {fname}")
            if rows[0]["reference"]:
                st.markdown(f"**Reference:** {rows[0]['reference']}")
            cols = st.columns(len(rows))
            for col, r in zip(cols, rows):
                with col:
                    v = r["cxer"]
                    wer_txt = "–" if r["WER"] is None else f"{r['WER']:.1%}"
                    hyp_html = html.escape(r["hypothesis"]) if r["hypothesis"] else "<i>(no output)</i>"
                    st.markdown(
                        f'<div class="cx-card"><b>{html.escape(r["model"])}</b><br>'
                        f'<span class="cx-hyp">{hyp_html}</span><br><br>'
                        f'{pill(v["classification"] if v else None)} &nbsp; '
                        f'<span class="cx-pill cx-na">WER {wer_txt}</span> &nbsp; '
                        f'<span class="cx-pill cx-na">{r["asr_seconds"]}s</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"Callsign: {'; '.join(r.get('callsigns', [])) or 'Unknown'} · Speaker: {r.get('speaker', 'Unknown')}")
                    if r.get("raw_hypothesis") and r["raw_hypothesis"] != r["hypothesis"]:
                        with st.expander("Raw ASR output"):
                            st.write(r["raw_hypothesis"])
                    if r["asr_error"]:
                        st.error("ASR failed for this model.")
                        with st.expander("Error details"):
                            st.code(r["asr_error"])
                    if v:
                        if v.get("explanation"):
                            st.caption(v["explanation"])
                        if v.get("transcription_errors"):
                            st.dataframe(errors_frame(v["transcription_errors"]), hide_index=True, width="stretch")
            known = next((i["known"] for i in items if i["name"] == fname and i.get("known")), None)
            if known:
                with st.expander("Stored verdicts from the offline benchmark run for this clip"):
                    for var in known["variants"]:
                        st.markdown(f"{pill(var['classification'])} **{var['label']}**: `{var['hypothesis']}`", unsafe_allow_html=True)
                        if var.get("explanation"):
                            st.caption(var["explanation"])

        flat = pd.DataFrame([{
            "audio_path": r.get("audio_path", ""), "callsigns": "; ".join(r.get("callsigns", [])) or "Unknown",
            "speaker": r.get("speaker", "Unknown"),
            "raw_hypothesis": r.get("raw_hypothesis", r["hypothesis"]),
            "raw_reference": r.get("raw_reference", r["reference"]),
            "judge_seconds": r["cxer"].get("inference_time") if r["cxer"] else None,
            "file": r["file"], "model": r["model"], "model_id": r["model_id"], "reference": r["reference"],
            "hypothesis": r["hypothesis"], "WER": r["WER"],
            "cxer_classification": r["cxer"]["classification"] if r["cxer"] else None,
            "operational_safety": r["cxer"].get("operational_safety") if r["cxer"] else None,
            "n_critical_errors": len(r["cxer"]["transcription_errors"]) if r["cxer"] else None,
            "asr_seconds": r["asr_seconds"], "audio_seconds": r["audio_seconds"],
        } for r in results])
        st.markdown("#### Summary")
        st.dataframe(flat, hide_index=True, width="stretch")
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download CSV", flat.to_csv(index=False), "cxer_results.csv", "text/csv", width="stretch")
        try:
            excel = io.BytesIO()
            flat.to_excel(excel, index=False, engine="openpyxl")
            st.download_button("Download Excel", excel.getvalue(), "cxer_results.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except ImportError:
            st.caption("Install openpyxl to enable Excel downloads; CSV is available.")
        d2.download_button("⬇ Download full JSON", json.dumps([{k: v for k, v in r.items() if k not in ("asr_error", "callsign_error", "callsign_seconds")} for r in results], indent=2, default=str), "cxer_results.json", "application/json", width="stretch")

# ── tab: benchmark ───────────────────────────────────────────────────────

with tab_bench:
    st.markdown("Explore saved FAA corpus results. These are offline results; loading a different live judge does not recompute them.")
    df = benchmark_table()
    if df.empty:
        st.info("No judge label files found under data/cxer_labels.")
    else:
        import plotly.graph_objects as go_

        benchmark_judge = st.selectbox("Benchmark judge", sorted(df["Judge"].unique()))
        df = df[df["Judge"] == benchmark_judge]
        available = df["ASR model"].tolist()
        selected = st.multiselect("Benchmark ASR models", available, default=available)
        metrics = st.multiselect("Metrics to visualize", ["WER %", "CxER %"], default=["WER %", "CxER %"])
        df = df[df["ASR model"].isin(selected)]
        if df.empty:
            st.info("Select at least one benchmark ASR model.")
        best = df.iloc[0] if not df.empty else None
        b1, b2, b3 = st.columns(3)
        if best is not None:
            b1.metric("Lowest CxER in selection", f"{best['CxER %']:.1f}%")
            b1.caption(f"Best model: {best['ASR model']}")
        pre = df[~df["Fine-tuned"]]
        ft = df[df["Fine-tuned"]]
        if not pre.empty and not ft.empty:
            b2.metric("Mean CxER, pretrained", f"{pre['CxER %'].mean():.1f}%")
            b3.metric("Mean CxER, fine-tuned", f"{ft['CxER %'].mean():.1f}%")

        plot_df = df.sort_values("WER %", na_position="last")
        fig = go_.Figure()
        for metric in metrics:
            fig.add_bar(name=metric, x=plot_df["ASR model"], y=plot_df[metric],
                        marker_color="#94a3b8" if metric == "WER %" else "#ef4444")
        fig.update_layout(barmode="group", height=420, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(orientation="h", y=1.08), yaxis_title="%")
        st.plotly_chart(fig, width="stretch")
        st.caption("WER measures word errors; CxER measures transcriptions with an operationally critical error. "
                   "Chart, table and summary statistics reflect your selection.")
        st.download_button("Download selected benchmarks", df.to_csv(index=False),
                           "selected_benchmarks.csv", "text/csv")
        st.dataframe(df, hide_index=True, width="stretch")

        figs = sorted(FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []
        if figs:
            st.markdown("#### Figures from the pipeline")
            pick = st.selectbox("Figure", [p.stem.replace("_", " ") for p in figs])
            st.image(str(figs[[p.stem.replace("_", " ") for p in figs].index(pick)]), width="stretch")

# Saved ATC benchmarks are independent of live inference settings.
with tab_saved:
    st.markdown("### Precomputed ATC benchmark results")
    st.caption("ATCOSIM, ATCO2 OOD and UWB ATCC. Saved scores are not recomputed by changing the live judge.")
    base = DATA / "atc_benchmarks"
    if not (base / "all_results.csv").exists():
        st.info("No ATC benchmark cache found.")
    else:
        import plotly.express as px
        cached = pd.read_csv(base / "all_results.csv")
        dataset = st.selectbox("Dataset", sorted(cached["dataset"].unique()))
        subset = cached[cached["dataset"] == dataset].copy()
        judges = st.multiselect("Benchmark judges", sorted(subset["judge"].unique()), default=sorted(subset["judge"].unique()))
        subset["ASR model"] = subset["model"] + " / " + subset["variant"]
        available_models = sorted(subset["ASR model"].unique())
        models = st.multiselect("ATC ASR models", available_models, default=available_models)
        subset = subset[subset["judge"].isin(judges) & subset["ASR model"].isin(models)].copy()
        subset["CxER %"] = pd.to_numeric(subset["CCER"]) * 100
        subset = subset.rename(columns={"WER": "WER %", "WeightedWER": "Weighted WER %"})
        metric = st.selectbox("ATC metric", ["CxER %", "WER %", "Weighted WER %", "BERTDist", "SemDist"])
        if subset.empty:
            st.info("Select at least one judge and ASR model.")
        else:
            st.plotly_chart(px.bar(subset, x="ASR model", y=metric, color="judge", barmode="group"), width="stretch")
            st.dataframe(subset, hide_index=True, width="stretch")
            st.download_button("Download ATC scores", subset.to_csv(index=False), f"{dataset}_scores.csv", "text/csv")
        st.markdown("#### Saved transcript pairs")
        raw_files = sorted((base / "raw").glob(f"*_{dataset}_predictions.json"))
        if raw_files:
            prediction_file = st.selectbox("Prediction file", raw_files, format_func=lambda p: p.name)
            records = pd.DataFrame(json.loads(prediction_file.read_text()))
            st.caption(f"{len(records):,} utterances. Showing the first 200; the download contains all rows. "
                       "These files contain reference/prediction pairs, not per-utterance judge verdicts.")
            st.dataframe(records.head(200), hide_index=True, width="stretch")
            st.download_button("Download transcript pairs", records.to_csv(index=False), prediction_file.stem + ".csv", "text/csv")

# ── tab: about ───────────────────────────────────────────────────────────

with tab_about:
    st.markdown(
        """
**What happens when you press Run**

1. Each audio file is resampled to 16 kHz mono and transcribed by every selected ASR model
   (Hugging Face `transformers`, GPU if available, otherwise CPU).
2. If you gave a reference transcript, WER is computed after the pipeline's standard
   normalisation (lowercase, strip punctuation).
3. If the judge server is up, each (reference, transcription) pair goes to the LLM judge with the
   same prompt, chat template and parser as the batch pipeline (`faa_cxer.judge`). It answers
   **Equivalent** (wording differs, meaning doesn't) or **Critical error** (a wrong callsign,
   altitude, heading, runway, frequency, clearance or waypoint), with each error listed.

**CxER** = critical / (critical + equivalent). Unparseable judge answers are excluded.

**Hardware.** ASR works anywhere. The judge, Llama-3.3-70B-Instruct in fp16, needs about 150 GB of
GPU memory for weights plus runtime overhead. Use the sidebar to load a local judge on GPUs
allocated to you, or connect to an existing server. A local judge requires vLLM; its first start
downloads model weights. GPU architecture compatibility must also be supported by vLLM.
"""
    )
