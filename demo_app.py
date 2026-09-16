#!/usr/bin/env python3
"""Listen-and-compare demo: pick a real ATC radio call, hear it, and see
what each model transcribed plus whether the LLM judge flagged an
operationally critical error. Reads only data/curated_samples.json (a
small, pre-built set of illustrative cases) and the matching audio files
in data/audio_samples/ - no model weights, no GPU, no faa_cxer pipeline
imports at runtime.

Run directly:
    streamlit run demo_app.py

For a live model-inference walkthrough instead of these pre-computed
results, see notebooks/live_inference_walkthrough.ipynb.
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

DATA_PATH = Path(__file__).parent / "data" / "curated_samples.json"
AUDIO_DIR = Path(__file__).parent / "data" / "audio_samples"

CLASSIFICATION_BADGE = {
    "Equivalent": "\U0001F7E2 Equivalent — no operationally critical error",
    "Critical_Errors": "\U0001F534 Critical error — an air-traffic controller would care about this",
}

st.set_page_config(page_title="FAA CxER — Listen & Compare", layout="centered")


@st.cache_data
def load_samples():
    return json.loads(DATA_PATH.read_text())


data = load_samples()
samples = data["samples"]

st.title("Does a Lower Word Error Rate Mean a Safer Transcription?")
st.markdown(
    "Pick a real air-traffic-control radio call below, listen to it, and compare what each model "
    "transcribed. An LLM judge (acting as an ATC-phraseology expert) flags whether each transcription "
    "contains an **operationally critical error** — a wrong callsign, altitude, runway, or clearance — "
    "on top of the plain Word Error Rate (WER)."
)

labels = [f"{s['family']} — {s['title']}" for s in samples]
choice = st.selectbox("Choose a sample", labels)
sample = samples[labels.index(choice)]

audio_path = AUDIO_DIR / sample["audio_file"]
st.audio(str(audio_path))

st.markdown(f"**Reference (ground truth):** {sample['reference']}")

st.divider()

cols = st.columns(len(sample["variants"]))
for col, variant in zip(cols, sample["variants"]):
    with col:
        st.markdown(f"**{variant['label']}**")
        st.markdown(f"> {variant['hypothesis']}")
        if variant["wer"] is not None:
            st.caption(f"WER: {variant['wer']:.3f}")

        classification = variant["classification"]
        badge = CLASSIFICATION_BADGE.get(classification, classification)
        if classification == "Critical_Errors":
            st.error(badge)
        else:
            st.success(badge)

        if variant.get("explanation"):
            st.caption(variant["explanation"])

        for err in variant.get("errors") or []:
            st.markdown(
                f"- **{err.get('entity_type', 'error')}**: expected *“{err.get('reference_value')}”*, "
                f"got *“{err.get('transcribed_value')}”*  \n"
                f"  risk: `{err.get('safety_risk')}` — {err.get('explanation', '')}"
            )

with st.sidebar:
    st.header("About")
    st.markdown(
        """
**CxER (Contextual Error Rate)** measures the fraction of transcriptions
that contain an error an air-traffic controller would actually care about
— as opposed to plain Word Error Rate, which treats every wrong word the
same whether it changes an instruction or not.

**The headline finding:** fine-tuning an ASR model on ATC data drives WER
down 80-90%, but CxER drops by much less — and some individual
transcriptions get *more* dangerous even as their WER improves. Several
of the samples above show exactly that.
"""
    )
    st.markdown(f"**Judge model:** `{data['judge']}`")
