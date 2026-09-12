"""Pretrained-vs-fine-tuned grouped bar chart: WER / BERTDist / SemDist,
plus a 4th CxER panel once every model in the comparison has a CxER score.

This single class replaces three near-identical scripts that used to exist
in the notebooks (one per judge: none/Claude/CxER) — the only real
difference between them was which corpus_metrics keys were available.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from ..config import ModelPair
from .style import FAMILY_VARIANT_COLORS

_FALLBACK_CYCLE = ["#4C72B0", "#76A0D4", "#2E8B57", "#7BC8A4", "#CC6633", "#F0A070", "#8172B2", "#A594D1"]


def _family_key(family_name: str) -> str:
    return family_name.lower().replace(" ", "-")


class ComparisonBarChart:
    """Builds a 2x2 grouped bar chart from a merged JSON's `corpus_metrics`
    (+ optional `corpus_cxer_metrics` for the judge-label lookup).

    A model's `corpus_metrics[key]["cxer"]` may be a single number (one
    judge) or `{judge_short: rate}` (several judges evaluated it) — pass
    `judge` to pick one when multiple are present; otherwise a model with
    multiple judges raises rather than silently averaging them.
    """

    METRIC_LABELS = {"WER": "Word Error Rate", "BERTDist": "BERT Distance", "SemDist": "Semantic Distance"}

    def __init__(
        self,
        model_pairs: List[ModelPair],
        corpus_metrics: Dict[str, Any],
        corpus_cxer_metrics: Optional[Dict[str, Any]] = None,
        judge: Optional[str] = None,
    ):
        self.model_pairs = model_pairs
        self.corpus_metrics = corpus_metrics
        self.corpus_cxer_metrics = corpus_cxer_metrics or {}
        self.judge = judge
        self.colors = self._build_colors()

    def _build_colors(self) -> Dict[tuple, str]:
        colors = {}
        for i, pair in enumerate(self.model_pairs):
            fk = _family_key(pair.family)
            pre_default = _FALLBACK_CYCLE[(2 * i) % len(_FALLBACK_CYCLE)]
            ft_default = _FALLBACK_CYCLE[(2 * i + 1) % len(_FALLBACK_CYCLE)]
            colors[(pair.family, "pretrained")] = FAMILY_VARIANT_COLORS.get((fk, "pretrained"), pre_default)
            colors[(pair.family, "finetuned")] = FAMILY_VARIANT_COLORS.get((fk, "finetuned"), ft_default)
        return colors

    def _resolve_cxer(self, model_key: str) -> Optional[float]:
        m = self.corpus_metrics.get(model_key, {})
        cxer = m.get("cxer")
        if not isinstance(cxer, dict):
            return cxer
        if self.judge is not None:
            return cxer.get(self.judge)
        if len(cxer) == 1:
            return next(iter(cxer.values()))
        raise ValueError(
            f"'{model_key}' has CxER from multiple judges {list(cxer)} — pass `judge=` to ComparisonBarChart "
            f"to pick one (this class doesn't average across judges)."
        )

    def _judges_used(self) -> set:
        judges = set()
        for pair in self.model_pairs:
            for key in (pair.pretrained_key, pair.finetuned_key):
                for judge_short, models in self.corpus_cxer_metrics.items():
                    if key in models:
                        judges.add(judge_short)
        return judges

    def _collect_rows(self):
        rows, missing_cxer = {}, []
        for pair in self.model_pairs:
            for variant, key in (("pretrained", pair.pretrained_key), ("finetuned", pair.finetuned_key)):
                if key not in self.corpus_metrics:
                    raise KeyError(f"'{key}' not found in corpus_metrics — check your ModelPair keys against the merged JSON.")
                m = self.corpus_metrics[key]
                cxer = self._resolve_cxer(key)
                if cxer is None:
                    missing_cxer.append(key)
                rows[(pair.family, variant)] = {"WER": m["WER"], "BERTDist": m["BERTDist"], "SemDist": m["SemDist"], "CxER": cxer}
        return rows, missing_cxer

    @staticmethod
    def _annotate_improvement(ax, x_pre, x_ft, y_pre, y_ft, pct, is_cxer=False):
        x_mid = (x_pre + x_ft) / 2
        y_top = max(y_pre, y_ft)
        cap = y_top * 0.07
        y_line = y_top + cap * 0.6
        color = "#B22222" if is_cxer else "#333333"

        ax.plot([x_pre, x_pre], [y_pre, y_line], color=color, lw=1.1, clip_on=False)
        ax.plot([x_ft, x_ft], [y_ft, y_line], color=color, lw=1.1, clip_on=False)
        ax.plot([x_pre, x_ft], [y_line, y_line], color=color, lw=1.1, clip_on=False)
        ax.text(x_mid, y_line + cap * 0.3, f"{abs(pct):.1f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=color, clip_on=False)

    def plot(self, output_path: str, dpi: int = 600, show: bool = False):
        rows, missing_cxer = self._collect_rows()
        plot_cxer = not missing_cxer
        if missing_cxer:
            print("⚠ CxER missing for:")
            for mk in missing_cxer:
                print(f"    {mk}")
            print("  → plotting WER / BERTDist / SemDist only (3 panels). Run the CxER judge pipeline + merge on the "
                  "missing models, then re-run this plot to get the 4th panel.")

        cxer_title = "Contextual Error Rate (CxER)"
        if plot_cxer:
            judges_used = self._judges_used()
            if len(judges_used) == 1:
                cxer_title += f" [{next(iter(judges_used))}]"
            elif len(judges_used) > 1:
                cxer_title += f" [{len(judges_used)} judges]"

        metrics = dict(self.METRIC_LABELS)
        if plot_cxer:
            metrics["CxER"] = cxer_title
        n_panels = len(metrics)

        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        fig.patch.set_facecolor("#FAFAFA")
        axes = axes.flatten()

        bar_width, gap = 0.15, 0.05
        x = np.arange(len(self.model_pairs)) * 0.7
        offsets = {"pretrained": -bar_width / 2 - gap / 2, "finetuned": bar_width / 2 + gap / 2}

        for ax, (metric, ylabel) in zip(axes, metrics.items()):
            is_cxer = metric == "CxER"
            bar_map = {}

            for fi, pair in enumerate(self.model_pairs):
                for variant in ("pretrained", "finetuned"):
                    val = rows[(pair.family, variant)][metric]
                    xi = x[fi] + offsets[variant]
                    ax.bar(xi, val, width=bar_width, color=self.colors[(pair.family, variant)], edgecolor="black", linewidth=0.5, zorder=2)
                    bar_map[(pair.family, variant)] = (xi, val)

            for pair in self.model_pairs:
                x_pre, h_pre = bar_map[(pair.family, "pretrained")]
                x_ft, h_ft = bar_map[(pair.family, "finetuned")]
                pre_val, ft_val = rows[(pair.family, "pretrained")][metric], rows[(pair.family, "finetuned")][metric]
                pct = (pre_val - ft_val) / pre_val * 100.0
                self._annotate_improvement(ax, x_pre, x_ft, h_pre, h_ft, pct, is_cxer=is_cxer)

            ax.set_xticks(x)
            ax.set_xticklabels([pair.family.replace(" ", "\n") for pair in self.model_pairs], fontsize=9)
            ax.set_title(ylabel, fontsize=10, fontweight="bold", color="#B22222" if is_cxer else "#1A1A2E", pad=8)
            ax.grid(axis="y", linestyle="--", alpha=0.4)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_facecolor("#F7F9FC")
            ax.set_ylim(0, ax.get_ylim()[1] * 1.3)

        for ax in axes[n_panels:]:
            ax.axis("off")

        legend_elements = []
        for pair in self.model_pairs:
            legend_elements.append(Patch(facecolor=self.colors[(pair.family, "pretrained")], edgecolor="black", label=f"{pair.family} Pretrained"))
            legend_elements.append(Patch(facecolor=self.colors[(pair.family, "finetuned")], edgecolor="black", label=f"{pair.family} Fine-tuned"))

        fig.legend(handles=legend_elements, loc="upper center", ncol=min(3, len(legend_elements)), fontsize=8.5, bbox_to_anchor=(0.5, 1.04), framealpha=0.9, edgecolor="#CCC")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"✓ Saved: {output_path}")
        if show:
            plt.show()
        plt.close(fig)
