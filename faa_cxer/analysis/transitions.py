"""Contextual "safety drift" analysis: for each ASR model family, how do
per-sample CxER classifications change between the pretrained and
fine-tuned checkpoint?

A sample can move Equivalent -> Critical_Errors after fine-tuning even
while its WER drops (fine-tuning can trade a purely lexical error for a
different, operationally critical one) — this is the core evidence for the
WER-CxER paradox this project studies. This module reproduces the
transition-matrix and WER-delta-by-transition analyses from Helper.ipynb.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from ..config import ModelPair

TRANSITIONS = [
    "Equivalent → Equivalent",
    "Equivalent → Critical",
    "Critical → Equivalent",
    "Critical → Critical",
]


class TransitionAnalyzer:
    """Operates on the merged `samples` list produced by `CxERMetricsMerger`
    (each sample has `sample["cxer"][judge][asr_model_key]["classification"]`
    and `sample["<asr_model_key>_WER"]`)."""

    def __init__(self, samples: List[dict], judge: str):
        self.samples = samples
        self.judge = judge

    @staticmethod
    def _transition_key(pre_cls: str, fin_cls: str) -> str:
        if pre_cls == "Equivalent" and fin_cls == "Equivalent":
            return TRANSITIONS[0]
        if pre_cls == "Equivalent" and fin_cls == "Critical_Errors":
            return TRANSITIONS[1]
        if pre_cls == "Critical_Errors" and fin_cls == "Equivalent":
            return TRANSITIONS[2]
        if pre_cls == "Critical_Errors" and fin_cls == "Critical_Errors":
            return TRANSITIONS[3]
        return "other"

    def group_by_transition(self, pair: ModelPair) -> Dict[str, List[dict]]:
        """One list of samples per transition category, each carrying its
        pretrained/fine-tuned WER for downstream delta analysis."""
        groups: Dict[str, List[dict]] = {t: [] for t in TRANSITIONS}

        for sample in self.samples:
            try:
                cxer = sample["cxer"][self.judge]
                pre_cls = cxer[pair.pretrained_key]["classification"]
                fin_cls = cxer[pair.finetuned_key]["classification"]
            except KeyError:
                continue

            key = self._transition_key(pre_cls, fin_cls)
            if key == "other":
                continue

            entry = {"sample": sample}
            pre_wer_key, fin_wer_key = f"{pair.pretrained_key}_WER", f"{pair.finetuned_key}_WER"
            if pre_wer_key in sample and fin_wer_key in sample:
                entry["pre_wer"] = sample[pre_wer_key]
                entry["fin_wer"] = sample[fin_wer_key]
                entry["delta_wer"] = sample[pre_wer_key] - sample[fin_wer_key]

            groups[key].append(entry)

        return groups

    def counts_table(self, groups: Dict[str, List[dict]]) -> pd.DataFrame:
        total = sum(len(v) for v in groups.values())
        counts = [len(groups[t]) for t in TRANSITIONS]
        return pd.DataFrame({
            "Transition": TRANSITIONS,
            "Count": counts,
            "Percent": [round(100 * c / total, 2) if total else float("nan") for c in counts],
        })

    def transition_matrix(self, groups: Dict[str, List[dict]]) -> pd.DataFrame:
        ee, ec, ce, cc = (len(groups[t]) for t in TRANSITIONS)
        return pd.DataFrame([[ee, ec], [ce, cc]], index=["Equivalent", "Critical"], columns=["Equivalent", "Critical"])

    def wer_delta_stats(self, groups: Dict[str, List[dict]]) -> pd.DataFrame:
        """Mean/median/std/min/max WER improvement (pre - finetuned) within
        each transition category — which transition drives the biggest WER
        gains, and does it line up with getting *safer* or just *quieter*?"""
        rows = []
        for transition in TRANSITIONS:
            vals = [e["delta_wer"] for e in groups[transition] if "delta_wer" in e]
            if not vals:
                continue
            delta = np.array(vals)
            pre = np.array([e["pre_wer"] for e in groups[transition] if "delta_wer" in e])
            fin = np.array([e["fin_wer"] for e in groups[transition] if "delta_wer" in e])
            rows.append({
                "Transition": transition,
                "N": len(vals),
                "Mean Pre WER": pre.mean(),
                "Mean FT WER": fin.mean(),
                "Mean ΔWER": delta.mean(),
                "Median ΔWER": np.median(delta),
                "Std ΔWER": delta.std(),
                "Min ΔWER": delta.min(),
                "Q1 ΔWER": np.percentile(delta, 25),
                "Q3 ΔWER": np.percentile(delta, 75),
                "Max ΔWER": delta.max(),
            })
        return pd.DataFrame(rows)

    def analyze_families(self, model_pairs: List[ModelPair]) -> Dict[str, Dict[str, Any]]:
        """Run counts + transition matrix + WER-delta stats for every model
        family in `model_pairs`. Returns {family: {"groups", "counts", "matrix", "wer_deltas"}}."""
        results = {}
        for pair in model_pairs:
            groups = self.group_by_transition(pair)
            results[pair.family] = {
                "groups": groups,
                "counts": self.counts_table(groups),
                "matrix": self.transition_matrix(groups),
                "wer_deltas": self.wer_delta_stats(groups),
            }
        return results

    def summary_across_families(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        rows = []
        for family, r in results.items():
            counts_df = r["counts"].set_index("Transition")
            total = counts_df["Count"].sum()
            rows.append({
                "Model": family,
                "Equivalent→Equivalent": int(counts_df.loc[TRANSITIONS[0], "Count"]),
                "Equivalent→Critical": int(counts_df.loc[TRANSITIONS[1], "Count"]),
                "Critical→Equivalent": int(counts_df.loc[TRANSITIONS[2], "Count"]),
                "Critical→Critical": int(counts_df.loc[TRANSITIONS[3], "Count"]),
                "Equivalent→Equivalent (%)": round(100 * counts_df.loc[TRANSITIONS[0], "Count"] / total, 2) if total else float("nan"),
                "Equivalent→Critical (%)": round(100 * counts_df.loc[TRANSITIONS[1], "Count"] / total, 2) if total else float("nan"),
                "Critical→Equivalent (%)": round(100 * counts_df.loc[TRANSITIONS[2], "Count"] / total, 2) if total else float("nan"),
                "Critical→Critical (%)": round(100 * counts_df.loc[TRANSITIONS[3], "Count"] / total, 2) if total else float("nan"),
            })
        return pd.DataFrame(rows)
