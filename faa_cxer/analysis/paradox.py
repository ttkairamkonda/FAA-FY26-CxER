"""The core "WER-CxER paradox" analysis: does a low-WER transcription
actually mean a *safe* one?

For each ASR model, this splits samples into "low-WER" (WER <= tau) and
the rest, then asks two questions about the critical (operationally
unsafe) samples:

  - Concentration:     of all critical-error samples, what % have low WER?
                        (i.e. how much does WER hide critical errors?)
  - Trust-violation:    of all low-WER samples, what % are actually critical?
                        (i.e. if you trusted low WER as "safe", how often
                        would you be wrong?)

A PARADOX is flagged when fine-tuning *increases* the trust-violation rate
even though (or because) it drove WER down — the headline finding this
project's plots (`wer_paradox_*.png`) visualize.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import ModelPair

VALID_CLASSIFICATIONS = ("Critical_Errors", "Equivalent")


class ParadoxAnalyzer:
    """Operates on the merged `samples` list produced by `CxERMetricsMerger`."""

    def __init__(self, samples: List[dict], judge: str):
        self.samples = samples
        self.judge = judge

    # ── record extraction (shared with the scatter-plot viz) ────────────

    def get_model_records(
        self,
        model_key: str,
        pretrained_key: Optional[str] = None,
        finetuned_key: Optional[str] = None,
        transition_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """One row per sample scored for `model_key`: sample_index,
        reference, hypothesis, WER, classification.

        If `transition_filter` is set (one of "Equivalent->Critical",
        "Critical->Equivalent", "Critical->Critical", "Equivalent->Equivalent")
        together with `pretrained_key`/`finetuned_key`, only samples whose
        pretrained->finetuned classification transition matches are kept —
        e.g. isolate exactly the samples fine-tuning made *newly unsafe*.
        """
        rows = []
        for s in self.samples:
            wer_key = f"{model_key}_WER"
            if wer_key not in s or model_key not in s:
                continue

            cxer_entry = s.get("cxer", {}).get(self.judge, {}).get(model_key)
            if cxer_entry is None:
                continue
            classification = cxer_entry["classification"]

            if transition_filter is not None and pretrained_key is not None and finetuned_key is not None:
                try:
                    pre_class = s["cxer"][self.judge][pretrained_key]["classification"]
                    fin_class = s["cxer"][self.judge][finetuned_key]["classification"]
                except KeyError:
                    continue

                if pre_class == "Equivalent" and fin_class == "Critical_Errors":
                    transition = "Equivalent->Critical"
                elif pre_class == "Critical_Errors" and fin_class == "Equivalent":
                    transition = "Critical->Equivalent"
                elif pre_class == "Critical_Errors" and fin_class == "Critical_Errors":
                    transition = "Critical->Critical"
                else:
                    transition = "Equivalent->Equivalent"

                if transition != transition_filter:
                    continue

            rows.append({
                "sample_index": s["sample_index"],
                "reference": s["reference"],
                "hypothesis": s[model_key],
                "WER": s[wer_key],
                "classification": classification,
            })
        return rows

    @staticmethod
    def get_valid(rows: List[dict], wer_min: float = 0.0, wer_max: float = 1.0) -> List[dict]:
        return [
            r for r in rows
            if r["classification"] in VALID_CLASSIFICATIONS
            and r["WER"] is not None
            and wer_min <= r["WER"] <= wer_max
        ]

    # ── threshold (tau) computation ──────────────────────────────────────

    def compute_threshold(self, rows: List[dict], mode: str = "fixed", fixed: float = 0.2, percentile: float = 25) -> float:
        """`rows` should be the PRETRAINED model's valid records — tau is
        derived from the pretrained WER distribution so it represents "what
        counted as a good transcription before fine-tuning"."""
        if mode == "fixed":
            return fixed
        wers = np.array([r["WER"] for r in rows if r["WER"] is not None])
        return float(np.percentile(wers, percentile))

    def compute_pooled_threshold(self, pretrained_keys: List[str], mode: str = "percentile", fixed: float = 0.2, percentile: float = 25, wer_min: float = 0.0) -> float:
        """Same as `compute_threshold`, but pooling the pretrained WER
        distribution across every model family — one tau applied uniformly
        everywhere, so families are compared on the same bar."""
        if mode == "fixed":
            return fixed
        pooled = []
        for key in pretrained_keys:
            pooled.extend(r["WER"] for r in self.get_valid(self.get_model_records(key), wer_min=wer_min, wer_max=float("inf")))
        return float(np.percentile(pooled, percentile))

    # ── concentration / trust-violation table ────────────────────────────

    def compute_metrics(self, model_key: str, threshold: float, include_zero_wer: bool = True) -> Dict[str, Any]:
        wer_min = 0.0 if include_zero_wer else 1e-8
        valid = self.get_valid(self.get_model_records(model_key), wer_min=wer_min, wer_max=float("inf"))
        n_total = len(valid)
        n_low = sum(1 for r in valid if r["WER"] < threshold)

        critical = [r for r in valid if r["classification"] == "Critical_Errors"]
        n_crit_total = len(critical)
        n_crit_low = sum(1 for r in critical if r["WER"] < threshold)

        concentration_pct = 100 * n_crit_low / n_crit_total if n_crit_total else float("nan")
        trust_violation_pct = 100 * n_crit_low / n_low if n_low else float("nan")

        return {
            "N valid": n_total,
            "N low-WER": n_low,
            "Total Critical": n_crit_total,
            "Critical in low-WER": n_crit_low,
            "Concentration %": round(concentration_pct, 2),
            "Trust-violation %": round(trust_violation_pct, 2),
        }

    def build_paradox_table(
        self,
        model_pairs: List[ModelPair],
        standalone_key: Optional[str] = None,
        threshold_mode: str = "percentile",
        threshold_fixed: float = 0.2,
        threshold_percentile: float = 25,
        include_zero_wer: bool = True,
    ) -> pd.DataFrame:
        """One row per (family, variant) with N valid / N low-WER / critical
        counts / concentration % / trust-violation %, using one threshold
        pooled across every family's pretrained WER distribution."""
        threshold = self.compute_pooled_threshold(
            [p.pretrained_key for p in model_pairs], mode=threshold_mode, fixed=threshold_fixed, percentile=threshold_percentile,
        )
        print(f"WER threshold (tau): {threshold:.4f}  [mode={threshold_mode}]")

        rows = []
        for pair in model_pairs:
            for variant, key in (("pretrained", pair.pretrained_key), ("finetuned", pair.finetuned_key)):
                rows.append({"Model": pair.family, "Variant": variant, **self.compute_metrics(key, threshold, include_zero_wer)})

        if standalone_key is not None:
            rows.append({"Model": standalone_key, "Variant": "standalone", **self.compute_metrics(standalone_key, threshold, include_zero_wer)})

        return pd.DataFrame(rows)

    def delta_table(self, df: pd.DataFrame, model_pairs: List[ModelPair]) -> pd.DataFrame:
        """Pretrained -> fine-tuned change in concentration/trust-violation
        per family, flagging PARADOX when trust-violation % *increases*
        despite (or because of) fine-tuning."""
        rows = []
        for pair in model_pairs:
            pre = df[(df.Model == pair.family) & (df.Variant == "pretrained")]
            ft = df[(df.Model == pair.family) & (df.Variant == "finetuned")]
            if pre.empty or ft.empty:
                continue

            p_conc, f_conc = pre["Concentration %"].values[0], ft["Concentration %"].values[0]
            p_trust, f_trust = pre["Trust-violation %"].values[0], ft["Trust-violation %"].values[0]
            p_crit, f_crit = pre["Total Critical"].values[0], ft["Total Critical"].values[0]
            d_trust = f_trust - p_trust

            rows.append({
                "Model": pair.family,
                "Total Critical (pre -> ft)": f"{p_crit} -> {f_crit}",
                "Concentration % (pre -> ft)": f"{p_conc:.1f} -> {f_conc:.1f}",
                "Trust-violation % (pre -> ft)": f"{p_trust:.2f} -> {f_trust:.2f}",
                "Δ Trust-violation %": round(d_trust, 2),
                "Flag": "PARADOX" if d_trust > 0 else "IMPROVED",
            })
        return pd.DataFrame(rows)
