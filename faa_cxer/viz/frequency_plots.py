"""Plots for `faa_cxer.metrics.TokenFrequencyAnalyzer`: does fine-tuning
help common words more than rare ones?"""

from __future__ import annotations

from typing import Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..metrics.token_frequency import BucketStats, TokenFrequencyAnalyzer
from .style import VARIANT_COLORS


class TokenFrequencyPlotter:
    def __init__(self, analyzer: TokenFrequencyAnalyzer):
        self.analyzer = analyzer

    def plot_absolute_rates(self, results: Dict[str, Dict[str, Dict[str, BucketStats]]], output_path: str, dpi: int = 150, show: bool = False):
        """Per model family: bar chart of pretrained vs. finetuned error
        rate per frequency bucket, plus the improvement (pp reduction)."""
        x_labels = self.analyzer.bucket_labels_ordered()
        x = np.arange(len(x_labels))
        bar_w = 0.35

        fig, axes = plt.subplots(len(results), 2, figsize=(16, 5 * len(results)))
        if len(results) == 1:
            axes = axes.reshape(1, -1)
        fig.suptitle(
            "Effect of Finetuning on Token Recognition by Frequency Bucket\n(Pretrained vs Finetuned — word-level error rate)",
            fontsize=14, fontweight="bold", y=1.01,
        )

        for row_idx, (family, variants) in enumerate(results.items()):
            ax_bar, ax_diff = axes[row_idx, 0], axes[row_idx, 1]

            pre_rates = [variants["pretrained"][lbl].error_rate_pct for lbl in x_labels]
            ft_rates = [variants["finetuned"][lbl].error_rate_pct for lbl in x_labels]
            deltas = [(p - f) if not (np.isnan(p) or np.isnan(f)) else np.nan for p, f in zip(pre_rates, ft_rates)]

            bars_pre = ax_bar.bar(x - bar_w / 2, pre_rates, bar_w, color=VARIANT_COLORS["pretrained"], label="Pretrained", alpha=0.85)
            bars_ft = ax_bar.bar(x + bar_w / 2, ft_rates, bar_w, color=VARIANT_COLORS["finetuned"], label="Finetuned", alpha=0.85)

            ax_bar.set_title(f"{family} — Error Rate per Frequency Bucket", fontsize=12)
            ax_bar.set_xlabel("Token Frequency Bucket  (<- rare    common ->)")
            ax_bar.set_ylabel("Word Error Rate (%)")
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels(x_labels, fontsize=8)
            ax_bar.legend(fontsize=9)
            ax_bar.yaxis.grid(True, alpha=0.3)
            ax_bar.set_axisbelow(True)

            for bars, color in ((bars_pre, VARIANT_COLORS["pretrained"]), (bars_ft, VARIANT_COLORS["finetuned"])):
                for bar in bars:
                    h = bar.get_height()
                    if not np.isnan(h):
                        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7, color=color)

            colors_delta = ["#2ca02c" if d > 0 else "#d62728" if d < 0 else "#aaa" for d in [v if not np.isnan(v) else 0 for v in deltas]]
            ax_diff.bar(x, deltas, color=colors_delta, alpha=0.85)
            ax_diff.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax_diff.set_title(f"{family} — Improvement (pretrained - finetuned)", fontsize=12)
            ax_diff.set_xlabel("Token Frequency Bucket  (<- rare    common ->)")
            ax_diff.set_ylabel("Error Rate Reduction (pp)")
            ax_diff.set_xticks(x)
            ax_diff.set_xticklabels(x_labels, fontsize=8)
            ax_diff.yaxis.grid(True, alpha=0.3)
            ax_diff.set_axisbelow(True)
            ax_diff.legend(handles=[mpatches.Patch(color="#2ca02c", label="Improved"), mpatches.Patch(color="#d62728", label="Degraded")], fontsize=9)

            for xi, d in zip(x, deltas):
                if not np.isnan(d):
                    va, yoff = ("bottom", 0.3) if d >= 0 else ("top", -0.3)
                    ax_diff.text(xi, d + yoff, f"{d:+.1f}pp", ha="center", va=va, fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        plt.close(fig)

    def plot_relative_improvement(self, results: Dict[str, Dict[str, Dict[str, BucketStats]]], output_path: str, dpi: int = 150, show: bool = False):
        """What fraction of the pretrained model's errors did fine-tuning
        fix, per frequency bucket? (Normalizes away the fact that rare
        buckets start with different baseline error rates.)"""
        x_labels = self.analyzer.bucket_labels_ordered()
        x = np.arange(len(x_labels))

        fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5), sharey=False)
        if len(results) == 1:
            axes = [axes]
        fig.suptitle("Relative Error Reduction by Token Frequency Bucket\n(What fraction of pretrained errors did finetuning fix?)", fontsize=13, fontweight="bold")

        for col_idx, (family, variants) in enumerate(results.items()):
            ax = axes[col_idx]
            pre_rates = np.array([variants["pretrained"][lbl].error_rate_pct for lbl in x_labels])
            ft_rates = np.array([variants["finetuned"][lbl].error_rate_pct for lbl in x_labels])
            rel = np.where(pre_rates > 0, (pre_rates - ft_rates) / pre_rates * 100, np.nan)

            colors_rel = ["#2ca02c" if v >= 0 else "#d62728" for v in np.nan_to_num(rel)]
            ax.bar(x, rel, color=colors_rel, alpha=0.85, edgecolor="white")
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")

            ax.set_title(family, fontsize=12, fontweight="bold")
            ax.set_xlabel("Token Frequency Bucket\n(<- rare    common ->)")
            ax.set_ylabel("Relative Error Reduction (%)" if col_idx == 0 else "")
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            for xi, v in zip(x, rel):
                if not np.isnan(v):
                    va, yoff = ("bottom", 1) if v >= 0 else ("top", -1)
                    ax.text(xi, v + yoff, f"{v:.0f}%", ha="center", va=va, fontsize=8, fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        plt.close(fig)
