"""Visualizes `CriticalErrorTokenAnalyzer`'s enrichment tables: are
critical-error words concentrated in rare frequency buckets, and does that
concentration get WORSE after fine-tuning?"""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .style import VARIANT_COLORS


class ErrorTokenFrequencyPlotter:
    def plot_enrichment(self, results: Dict[str, Dict[str, pd.DataFrame]], output_path: str, dpi: int = 150, show: bool = False):
        """One panel per model family: enrichment ratio by frequency
        bucket, pretrained vs. finetuned, with a dashed line at 1.0 (the
        "no enrichment / matches the corpus" baseline). Bars above the line
        are overrepresented among critical errors relative to the corpus."""
        families = list(results.keys())
        fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), sharey=True)
        if len(families) == 1:
            axes = [axes]

        fig.suptitle(
            "Are Critical-Error Words Rarer Than the Corpus Average?\n(Enrichment > 1 = overrepresented among critical errors)",
            fontsize=13, fontweight="bold",
        )

        for ax, family in zip(axes, families):
            pre_df, ft_df = results[family]["pretrained"], results[family]["finetuned"]
            labels = pre_df["Bucket"].tolist()
            x = np.arange(len(labels))
            bar_w = 0.35

            ax.bar(x - bar_w / 2, pre_df["Enrichment"], bar_w, color=VARIANT_COLORS["pretrained"], label="Pretrained", alpha=0.85)
            ax.bar(x + bar_w / 2, ft_df["Enrichment"], bar_w, color=VARIANT_COLORS["finetuned"], label="Finetuned", alpha=0.85)
            ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")

            ax.set_title(family, fontsize=12, fontweight="bold")
            ax.set_xlabel("Token Frequency Bucket\n(<- rare    common ->)")
            ax.set_ylabel("Enrichment (Critical-Error % / Corpus %)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            ax.legend(fontsize=9)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {output_path}")
        if show:
            plt.show()
        plt.close(fig)
