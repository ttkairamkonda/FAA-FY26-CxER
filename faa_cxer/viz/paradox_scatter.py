"""Jitter-scatter visualization of the WER-CxER paradox: plot every
sample's WER on the x-axis, colored by CxER classification, with a
threshold line at tau. The visual claim: a meaningful chunk of
"Critical_Errors" points sit to the LEFT of tau (low WER, but unsafe).

Reuses `ParadoxAnalyzer` for all data extraction so the numbers behind the
plot and the numbers in the paradox table (`faa_cxer.analysis.paradox`)
never drift apart.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.paradox import ParadoxAnalyzer
from .style import CLASSIFICATION_COLORS


class ParadoxScatterPlotter:
    def __init__(self, analyzer: ParadoxAnalyzer, wer_max: float = 1.0, seed: int = 42):
        self.analyzer = analyzer
        self.wer_max = wer_max
        self.rng = np.random.default_rng(seed)

    def _plot_panel(self, ax, model_key: str, title: str, threshold: float, pretrained_key=None, finetuned_key=None, transition_filter=None, n_each: Optional[int] = None):
        rows = self.analyzer.get_model_records(model_key, pretrained_key, finetuned_key, transition_filter=transition_filter)
        valid = self.analyzer.get_valid(rows, wer_max=self.wer_max)

        critical = [r for r in valid if r["classification"] == "Critical_Errors"]
        equivalent = [r for r in valid if r["classification"] == "Equivalent"]

        n_crit = len(critical) if n_each is None else min(n_each, len(critical))
        n_eq = len(equivalent) if n_each is None else min(n_each, len(equivalent))
        critical_plot = list(self.rng.choice(critical, size=n_crit, replace=False)) if n_crit else []
        equivalent_plot = list(self.rng.choice(equivalent, size=n_eq, replace=False)) if n_eq else []

        plot_rows = critical_plot + equivalent_plot
        wers = np.array([r["WER"] for r in plot_rows])
        statuses = [r["classification"] for r in plot_rows]
        jitter = self.rng.uniform(-0.4, 0.4, size=len(wers))

        ax.axvspan(0, threshold, color="#efefef", alpha=0.8, zorder=0)
        ax.axvline(threshold, color="#888888", linewidth=1.2, linestyle="--", zorder=2)
        ax.text(threshold / 2, 0.96, r"($WER \leq \tau$)", transform=ax.get_xaxis_transform(), fontsize=12, color="#999999", va="top", ha="center", style="italic")

        for status in ("Equivalent", "Critical_Errors"):
            mask = np.array([s == status for s in statuses])
            ax.scatter(
                wers[mask], jitter[mask], c=CLASSIFICATION_COLORS[status], marker="o",
                alpha=0.4 if status == "Equivalent" else 0.7, s=14 if status == "Equivalent" else 20, linewidths=0,
            )

        ax.set_title(title, fontsize=10, fontweight="bold", pad=3)
        ax.set_xlim(-0.01, self.wer_max + 0.02)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.grid(False)
        ax.spines[["top", "right", "left"]].set_visible(False)

    @staticmethod
    def _legend_handles():
        return [
            mpatches.Patch(color=CLASSIFICATION_COLORS["Critical_Errors"], label="Contextual Error"),
            mpatches.Patch(color=CLASSIFICATION_COLORS["Equivalent"], label="Equivalent"),
        ]

    def plot_comparison(self, family_name: str, pretrained_key: str, finetuned_key: str, output_path: str, threshold_mode: str = "fixed", threshold_fixed: float = 0.2, threshold_percentile: float = 25, dpi: int = 600, show: bool = False):
        """Two stacked panels: pretrained vs. fine-tuned, sharing one tau
        derived from the pretrained model's WER distribution."""
        fig, axes = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)

        pretrained_rows = self.analyzer.get_valid(self.analyzer.get_model_records(pretrained_key), wer_max=self.wer_max)
        threshold = self.analyzer.compute_threshold(pretrained_rows, mode=threshold_mode, fixed=threshold_fixed, percentile=threshold_percentile)

        self._plot_panel(axes[0], pretrained_key, f"Pretrained ({family_name})", threshold, pretrained_key, finetuned_key)
        self._plot_panel(axes[1], finetuned_key, f"Finetuned ({family_name})", threshold, pretrained_key, finetuned_key)

        axes[1].set_xlabel("Word Error Rate (WER)", fontsize=10)
        axes[1].legend(handles=self._legend_handles(), loc="lower right", frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}  (tau={threshold:.3f})")
        if show:
            plt.show()
        plt.close(fig)

    def plot_standalone(self, model_key: str, title: str, output_path: str, threshold_mode: str = "fixed", threshold_fixed: float = 0.2, threshold_percentile: float = 25, dpi: int = 600, show: bool = False):
        """Single panel for a model with no pretrained counterpart (e.g. the
        production "FAA_Model")."""
        fig, ax = plt.subplots(1, 1, figsize=(7, 3.2))

        rows = self.analyzer.get_valid(self.analyzer.get_model_records(model_key), wer_max=self.wer_max)
        threshold = self.analyzer.compute_threshold(rows, mode=threshold_mode, fixed=threshold_fixed, percentile=threshold_percentile)

        self._plot_panel(ax, model_key, title, threshold)
        ax.set_xlabel("Word Error Rate (WER)", fontsize=10)
        ax.legend(handles=self._legend_handles(), loc="lower right", frameon=True, framealpha=0.9, edgecolor="#cccccc", fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}  (tau={threshold:.3f})")
        if show:
            plt.show()
        plt.close(fig)
