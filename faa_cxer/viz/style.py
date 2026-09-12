"""Shared color palette and small matplotlib style helpers, so every plot
in this project looks consistent without copy-pasting hex codes."""

from __future__ import annotations

# Classification colors used throughout (scatter plots, sankey, bars).
CLASSIFICATION_COLORS = {
    "Critical_Errors": "#d62728",
    "Equivalent": "#2ca02c",
}

# Pretrained vs. fine-tuned pair, per model family — used by ComparisonBarChart.
FAMILY_VARIANT_COLORS = {
    ("whisper-medium", "pretrained"): "#4C72B0",
    ("whisper-medium", "finetuned"): "#76A0D4",
    ("whisper-large", "pretrained"): "#2E8B57",
    ("whisper-large", "finetuned"): "#7BC8A4",
    ("parakeet", "pretrained"): "#CC6633",
    ("parakeet", "finetuned"): "#F0A070",
}

# Generic pretrained/finetuned pair for plots that don't split by family.
VARIANT_COLORS = {"pretrained": "#4C72B0", "finetuned": "#DD8452"}

IMPROVED_COLOR = "#2ca02c"
DEGRADED_COLOR = "#d62728"


def style_axes(ax, hide_spines=("top", "right")) -> None:
    """Small shared cleanup: drop the top/right spines and add a faint
    y-grid, the look every figure in this project uses."""
    for spine in hide_spines:
        ax.spines[spine].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
