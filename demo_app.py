#!/usr/bin/env python3
"""Interactive demo of the FAA_CxER pipeline's analysis/visualization stages
(3-8) for a live, in-person walkthrough — no GPU, no notebook, no code
visible. Every number and plot on this page is produced by calling the same
`faa_cxer` classes the CLI scripts in `scripts/` use, against the same data
already shipped in `data/`.

Run directly:
    streamlit run demo_app.py

Or via Docker (see Dockerfile / DEMO_README.md) for a machine with no
Python installed at all.

Note: stage 2 (the LLM-as-judge CxER classification, which needs a
multi-GPU host running Llama-3.3-70B under vLLM) is NOT re-run here — its
output ships pre-computed in `data/cxer_labels/`. This demo shows exactly
the same numbers the project's headline results are built from; it just
lets a viewer explore them live instead of looking at static PNGs.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from faa_cxer.analysis import CriticalErrorTokenAnalyzer, ParadoxAnalyzer, TransitionAnalyzer
from faa_cxer.config import DEFAULT_MODEL_PAIRS, default_config
from faa_cxer.io_utils import load_json, load_records
from faa_cxer.metrics import TokenFrequencyAnalyzer
from faa_cxer.viz import (
    ComparisonBarChart,
    ErrorTokenFrequencyPlotter,
    ParadoxScatterPlotter,
    SafetyDriftSankeyPlotter,
    TokenFrequencyPlotter,
)

STANDALONE_MODEL_KEY = "FAA_Model"
PLOT_DPI = 150  # lower than the library's print-quality default (600) — fast enough for live clicking

st.set_page_config(page_title="FAA CxER — WER-CxER Paradox Demo", layout="wide")


@st.cache_resource
def get_scratch_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="faa_cxer_demo_"))
    return d


@st.cache_data(show_spinner="Loading merged predictions + CxER labels...")
def get_samples(path: str):
    return load_records(path)


@st.cache_data(show_spinner=False)
def get_json(path: str):
    return load_json(path)


def cache_key(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


cfg = default_config()
scratch = get_scratch_dir()

st.title("FAA CxER: Does a Lower Word Error Rate Mean a *Safer* Transcription?")
st.caption(
    "Live walkthrough of the Contextual Error Rate (CxER) analysis — every chart below is generated on this "
    "machine, from the same predictions + judge-labeled data the project's results are built from."
)

with st.sidebar:
    st.header("About this demo")
    st.markdown(
        """
This explores **CxER**: the fraction of ASR transcriptions that contain an
operationally critical error (wrong callsign, flipped left/right, wrong
altitude or runway) — as judged by an LLM acting as an ATC-phraseology
expert.

**Headline finding (the "WER-CxER paradox"):** fine-tuning drives Word
Error Rate down 80-90%, but CxER drops by much less — and some individual
transcriptions get *more* dangerous even as their WER improves.

The judge labels themselves (produced by Llama-3.3-70B-Instruct under
vLLM, on a multi-GPU host) are pre-computed and shipped with this demo —
re-running that stage live isn't part of this walkthrough.
"""
    )
    judge = cfg.judge_short_name()
    st.markdown(f"**Judge model:** `{judge}`")

tab_overview, tab_paradox, tab_drift, tab_freq, tab_rarity = st.tabs(
    ["Overview", "WER vs CxER Paradox", "Safety Drift", "Token Frequency", "Critical-Error Word Rarity"]
)

# ── Overview ────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("Pretrained vs. Fine-tuned, all four metrics")
    st.markdown(
        "WER, BERTDist and SemDist all improve sharply after fine-tuning on 500K ATC utterances — "
        "but watch the CxER panel."
    )
    merged = get_json(str(cfg.predictions_with_metrics_and_cxer))
    chart = ComparisonBarChart(
        DEFAULT_MODEL_PAIRS,
        corpus_metrics=merged["corpus_metrics"],
        corpus_cxer_metrics=merged.get("corpus_cxer_metrics", {}),
        judge=judge,
    )
    out_path = scratch / "overview_comparison.png"
    chart.plot(str(out_path), dpi=PLOT_DPI)
    st.image(str(out_path), width="stretch")

# ── Paradox ─────────────────────────────────────────────────────────────
with tab_paradox:
    st.subheader("Low WER doesn't mean safe")
    st.markdown(
        "**Concentration** — of all critical-error samples, what % have low WER? "
        "**Trust-violation** — of all low-WER samples, what % are actually critical? "
        "A **PARADOX** flag means fine-tuning made the trust-violation rate *worse*."
    )
    col1, col2, col3 = st.columns(3)
    threshold_mode = col1.radio("Threshold mode", ["percentile", "fixed"], horizontal=True)
    if threshold_mode == "fixed":
        threshold_fixed = col2.slider("WER threshold (τ)", 0.0, 1.0, 0.2, 0.01)
        threshold_percentile = 25.0
    else:
        threshold_percentile = col2.slider("Percentile of pretrained WER distribution", 1.0, 50.0, 25.0, 1.0)
        threshold_fixed = 0.2

    samples = get_samples(str(cfg.predictions_with_metrics_and_cxer))
    paradox = ParadoxAnalyzer(samples, judge=judge)
    table = paradox.build_paradox_table(
        DEFAULT_MODEL_PAIRS, standalone_key=STANDALONE_MODEL_KEY,
        threshold_mode=threshold_mode, threshold_fixed=threshold_fixed, threshold_percentile=threshold_percentile,
    )
    st.dataframe(table, width="stretch")

    st.markdown("**Delta (fine-tuned − pretrained)**")
    st.dataframe(paradox.delta_table(table, DEFAULT_MODEL_PAIRS), width="stretch")

    st.markdown("**Per-sample WER, colored by CxER classification** (dots left of the dashed line = low WER)")
    plotter = ParadoxScatterPlotter(paradox)
    key = cache_key("paradox", threshold_mode, threshold_fixed, threshold_percentile)
    cols = st.columns(2)
    for i, pair in enumerate(DEFAULT_MODEL_PAIRS):
        out_name = scratch / f"wer_paradox_{pair.family.lower().replace(' ', '_')}_{key}.png"
        if not out_name.exists():
            plotter.plot_comparison(
                pair.family, pair.pretrained_key, pair.finetuned_key, str(out_name),
                threshold_mode=threshold_mode, threshold_fixed=threshold_fixed, threshold_percentile=threshold_percentile,
                dpi=PLOT_DPI,
            )
        cols[i % 2].image(str(out_name), caption=pair.family, width="stretch")

    standalone_out = scratch / f"wer_paradox_faa_model_{key}.png"
    if not standalone_out.exists():
        plotter.plot_standalone(
            STANDALONE_MODEL_KEY, STANDALONE_MODEL_KEY, str(standalone_out),
            threshold_mode=threshold_mode, threshold_fixed=threshold_fixed, threshold_percentile=threshold_percentile,
            dpi=PLOT_DPI,
        )
    cols[len(DEFAULT_MODEL_PAIRS) % 2].image(str(standalone_out), caption=STANDALONE_MODEL_KEY, width="stretch")

# ── Safety drift (Sankey) ───────────────────────────────────────────────
with tab_drift:
    st.subheader("Contextual safety drift after fine-tuning")
    st.markdown(
        "Does a sample that was `Equivalent` before fine-tuning ever become `Critical` after? "
        "That's fine-tuning trading a lexical error for a genuinely dangerous one."
    )
    samples = get_samples(str(cfg.predictions_with_metrics_and_cxer))
    transitions = TransitionAnalyzer(samples, judge=judge)
    results = transitions.analyze_families(DEFAULT_MODEL_PAIRS)

    st.markdown("**Summary across all model families**")
    st.dataframe(transitions.summary_across_families(results), width="stretch")

    family_choice = st.selectbox("Inspect one family's transition matrix", [p.family for p in DEFAULT_MODEL_PAIRS])
    r = results[family_choice]
    col1, col2 = st.columns(2)
    col1.markdown("**Counts**")
    col1.dataframe(r["counts"], width="stretch")
    col2.markdown("**Transition matrix** (rows = pretrained, cols = fine-tuned)")
    col2.dataframe(r["matrix"], width="stretch")
    if not r["wer_deltas"].empty:
        st.markdown("**WER change by transition**")
        st.dataframe(r["wer_deltas"], width="stretch")

    st.markdown("**Sankey diagram** (all families)")
    sankey_fig = SafetyDriftSankeyPlotter(transitions).plot(
        DEFAULT_MODEL_PAIRS, str(scratch / "safety_drift_sankey.html")
    )
    st.plotly_chart(sankey_fig, width="stretch")

# ── Token frequency ─────────────────────────────────────────────────────
with tab_freq:
    st.subheader("Does fine-tuning help common words more than rare ones?")
    st.markdown(
        "Reference words are bucketed by how often they appear in the corpus. Callsigns, altitudes and "
        "waypoints tend to be rare — if fine-tuning helps those buckets less, errors concentrate exactly "
        "where they're most dangerous."
    )
    samples = get_samples(str(cfg.predictions_with_metrics_and_cxer))
    freq_analyzer = TokenFrequencyAnalyzer()
    freq_analyzer.build_token_frequency(samples)
    freq_results = freq_analyzer.analyze_model_pairs(samples, DEFAULT_MODEL_PAIRS)

    family_choice2 = st.selectbox("Model family", [p.family for p in DEFAULT_MODEL_PAIRS], key="freq_family")
    variant_choice = st.selectbox("Variant", list(freq_results[family_choice2].keys()), key="freq_variant")
    stats = freq_results[family_choice2][variant_choice]
    rows = [
        {"Bucket": lbl.replace("\n", " "), "Errors": stats[lbl].errors, "Total": stats[lbl].total, "Error %": round(stats[lbl].error_rate_pct, 1)}
        for lbl in freq_analyzer.bucket_labels_ordered()[::-1]
    ]
    st.dataframe(rows, width="stretch")

    freq_plotter = TokenFrequencyPlotter(freq_analyzer)
    abs_path = scratch / "finetuning_rare_vs_frequent.png"
    rel_path = scratch / "finetuning_relative_improvement.png"
    if not abs_path.exists():
        freq_plotter.plot_absolute_rates(freq_results, str(abs_path))
    if not rel_path.exists():
        freq_plotter.plot_relative_improvement(freq_results, str(rel_path))
    col1, col2 = st.columns(2)
    col1.image(str(abs_path), width="stretch")
    col2.image(str(rel_path), width="stretch")

# ── Critical-error word rarity ───────────────────────────────────────────
with tab_rarity:
    st.subheader("Are critical-error words disproportionately rare?")
    st.markdown(
        "Pulls the exact words the LLM judge flagged inside each `Critical_Errors` sample and checks whether "
        "they're rarer than the corpus average — and whether fine-tuning's *residual* critical errors skew "
        "even rarer."
    )
    mode = st.radio(
        "Mode", ["all", "rarest"], horizontal=True,
        help="'all' counts every word in each flagged error; 'rarest' isolates the least-frequent word per error "
             "(usually the callsign/waypoint).",
    )
    samples = get_samples(str(cfg.predictions_with_metrics_and_cxer))
    rarity_analyzer = CriticalErrorTokenAnalyzer(samples, judge=judge)

    family_choice3 = st.selectbox("Model family", [p.family for p in DEFAULT_MODEL_PAIRS], key="rarity_family")
    pair3 = next(p for p in DEFAULT_MODEL_PAIRS if p.family == family_choice3)
    tables = rarity_analyzer.compare_pretrained_vs_finetuned(pair3, mode=mode)

    for variant, df in tables.items():
        model_key = pair3.pretrained_key if variant == "pretrained" else pair3.finetuned_key
        summary = rarity_analyzer.rarity_summary(model_key, mode=mode)
        chi2 = rarity_analyzer.chi_square_test(model_key, mode=mode)
        st.markdown(
            f"**{variant}** — {df.attrs.get('n_critical_errors_scanned', '?')} critical errors, "
            f"{df.attrs.get('n_error_token_instances', '?')} word instances"
        )
        st.dataframe(df, width="stretch")
        st.caption(
            f"mean corpus freq: {summary.get('mean_corpus_freq')} · median: {summary.get('median_corpus_freq')} · "
            f"% rare (≤4 occurrences): {summary.get('pct_rare_le4')}% · % hapax: {summary.get('pct_hapax')}%"
        )
        if chi2:
            sig = "***" if chi2["p_value"] < 0.001 else "**" if chi2["p_value"] < 0.01 else "*" if chi2["p_value"] < 0.05 else "n.s."
            st.caption(f"chi-square vs. corpus distribution: χ²={chi2['chi2']}, p={chi2['p_value']:.2e} ({sig}), n={chi2['n']}")

    rarity_results = {family_choice3: tables}
    rarity_plotter = ErrorTokenFrequencyPlotter()
    rarity_out = scratch / f"critical_error_word_rarity_{family_choice3}_{mode}.png".replace(" ", "_")
    if not rarity_out.exists():
        rarity_plotter.plot_enrichment(rarity_results, str(rarity_out))
    st.image(str(rarity_out), width="stretch")
