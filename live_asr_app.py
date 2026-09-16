#!/usr/bin/env python3
"""Live ASR demo: pick a reference clip or upload your own audio, and get a
REAL transcription from two Whisper Medium checkpoints (pretrained vs.
fine-tuned) running on CPU, right now - not a pre-computed lookup.

The CxER judge verdict (Equivalent / Critical_Errors) is a different
story: that needs Llama-3.3-70B under vLLM on a multi-GPU host, which
isn't something to spin up for a demo. So when the audio in front of you
happens to be one of the two clips already evaluated by the offline
judge, this shows that clip's real, pre-computed verdict alongside the
transcription you just watched run live. For anything else (a genuinely
new upload), there's no verdict to show - only the live transcription.

Run directly:
    streamlit run live_asr_app.py

Packaged in Dockerfile.live_asr, which bakes both model checkpoints in at
build time so no internet is needed at demo time.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import time
from pathlib import Path

import librosa
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
CURATED_JSON = DATA_DIR / "curated_samples.json"
AUDIO_DIR = DATA_DIR / "audio_samples"

PRETRAINED_ID = "openai/whisper-medium.en"
FINETUNED_ID = "ttkairamkonda/FAA_Whisper_M_TL_500K"
PRETRAINED_LABEL = "Pretrained (whisper-medium.en)"
FINETUNED_LABEL = "Fine-tuned (FAA_Whisper_M_TL_500K)"

# The two curated clips whose stored CxER data is for THIS SAME model
# family (Whisper Medium) - the only ones we can honestly show a verdict
# for alongside a live transcription from these two models.
KNOWN_FAMILY = "Whisper Medium"

CLASSIFICATION_BADGE = {
    "Equivalent": "\U0001F7E2 Equivalent — no operationally critical error",
    "Critical_Errors": "\U0001F534 Critical error — an air-traffic controller would care about this",
}

st.set_page_config(page_title="FAA CxER — Live ASR", layout="centered")


@st.cache_resource(show_spinner="Loading both Whisper Medium models onto CPU (one-time, ~10-30s)...")
def load_pipelines():
    from transformers import pipeline

    print(f"[live_asr] loading pretrained ({PRETRAINED_ID})...", flush=True)
    t0 = time.time()
    pretrained = pipeline("automatic-speech-recognition", model=PRETRAINED_ID, device=-1)
    print(f"[live_asr] pretrained loaded in {time.time() - t0:.1f}s", flush=True)

    print(f"[live_asr] loading fine-tuned ({FINETUNED_ID})...", flush=True)
    t0 = time.time()
    finetuned = pipeline("automatic-speech-recognition", model=FINETUNED_ID, device=-1)
    print(f"[live_asr] fine-tuned loaded in {time.time() - t0:.1f}s", flush=True)

    return pretrained, finetuned


@st.cache_data
def load_known_samples():
    data = json.loads(CURATED_JSON.read_text())
    known = [s for s in data["samples"] if s["family"] == KNOWN_FAMILY]
    hashes = {}
    for s in known:
        audio_bytes = (AUDIO_DIR / s["audio_file"]).read_bytes()
        hashes[hashlib.sha256(audio_bytes).hexdigest()] = s
    return data["judge"], known, hashes


def transcribe(pipe, audio_array, sampling_rate):
    print(f"[live_asr] transcribing {len(audio_array) / sampling_rate:.1f}s of audio...", flush=True)
    t0 = time.time()
    result = pipe({"array": audio_array, "sampling_rate": sampling_rate})
    elapsed = time.time() - t0
    print(f"[live_asr] done in {elapsed:.1f}s", flush=True)
    return result["text"].strip(), elapsed


def render_known_verdict(judge, sample):
    st.success(f"This exact clip was evaluated by the offline LLM judge (`{judge}`) — here's what it found:")
    st.caption(
        "The verdict below is for the model's stored prediction from that offline run. Your live transcription "
        "above may differ slightly in wording (decoding isn't perfectly deterministic) — if so, treat this as "
        "context for the same clip, not a live judgment of the exact text just produced."
    )
    cols = st.columns(len(sample["variants"]))
    for col, variant in zip(cols, sample["variants"]):
        with col:
            st.markdown(f"**{variant['label']}**")
            st.markdown(f"> {variant['hypothesis']}")
            if variant["wer"] is not None:
                st.caption(f"WER: {variant['wer']:.3f}")
            classification = variant["classification"]
            badge = CLASSIFICATION_BADGE.get(classification, classification)
            (st.error if classification == "Critical_Errors" else st.success)(badge)
            if variant.get("explanation"):
                st.caption(variant["explanation"])
            for err in variant.get("errors") or []:
                st.markdown(
                    f"- **{err.get('entity_type', 'error')}**: expected *“{err.get('reference_value')}”*, "
                    f"got *“{err.get('transcribed_value')}”*  \n"
                    f"  risk: `{err.get('safety_risk')}` — {err.get('explanation', '')}"
                )


st.title("Live ASR: Watch the Model Transcribe in Real Time")
st.markdown(
    "This runs two real Whisper checkpoints on this machine's CPU right now — a "
    "**pretrained** model and the **fine-tuned** production model — on whatever audio you give it below."
)

judge, known_samples, known_hashes = load_known_samples()

mode = st.radio("Audio source", ["Use a reference clip", "Upload your own audio"], horizontal=True)

audio_bytes = None
audio_path_for_display = None
matched_known = None

if mode == "Use a reference clip":
    labels = [s["title"] for s in known_samples]
    choice = st.selectbox("Reference clip", labels)
    matched_known = known_samples[labels.index(choice)]
    audio_path_for_display = AUDIO_DIR / matched_known["audio_file"]
    audio_bytes = audio_path_for_display.read_bytes()
    st.markdown(f"**Reference (ground truth):** {matched_known['reference']}")
else:
    uploaded = st.file_uploader("Upload a .wav / .mp3 / .flac file", type=["wav", "mp3", "flac", "m4a", "ogg"])
    if uploaded is not None:
        audio_bytes = uploaded.getvalue()
        matched_known = known_hashes.get(hashlib.sha256(audio_bytes).hexdigest())

if audio_bytes is not None:
    st.audio(audio_bytes)

    if st.button("Transcribe live", type="primary"):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            array, sr = librosa.load(tmp.name, sr=16000, mono=True)

        pretrained_pipe, finetuned_pipe = load_pipelines()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{PRETRAINED_LABEL}**")
            with st.spinner("Transcribing..."):
                text, elapsed = transcribe(pretrained_pipe, array, sr)
            st.markdown(f"> {text}")
            st.caption(f"{elapsed:.1f}s on CPU")
        with col2:
            st.markdown(f"**{FINETUNED_LABEL}**")
            with st.spinner("Transcribing..."):
                text, elapsed = transcribe(finetuned_pipe, array, sr)
            st.markdown(f"> {text}")
            st.caption(f"{elapsed:.1f}s on CPU")

        st.divider()
        if matched_known is not None:
            render_known_verdict(judge, matched_known)
        else:
            st.info(
                "No pre-computed judge verdict is available for this clip — it wasn't part of the offline "
                "evaluation. This demo shows live transcription only; CxER classification needs the LLM judge "
                "(Llama-3.3-70B under vLLM on a GPU host), which isn't run here."
            )

with st.sidebar:
    st.header("About")
    st.markdown(
        """
Both models here are real Hugging Face checkpoints (`openai/whisper-medium.en`
and `ttkairamkonda/FAA_Whisper_M_TL_500K`), baked into this Docker image at
build time. Transcription happens live, on this machine's CPU, when you
click **Transcribe live** — nothing is pre-computed.

The **CxER judge verdict** (whether a transcription contains an
operationally critical error) is a separate, much heavier step that needs
a 70-billion-parameter model on a multi-GPU host. It's only shown here for
the two reference clips that were already evaluated offline — see
`demo_app.py` for the full listen-and-compare set of 8 such clips.
"""
    )
