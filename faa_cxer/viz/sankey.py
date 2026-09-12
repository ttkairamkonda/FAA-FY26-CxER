"""Sankey diagram of "contextual safety drift": for each model family, how
many samples flow Equivalent->Equivalent / Equivalent->Critical /
Critical->Equivalent / Critical->Critical between the pretrained and
fine-tuned checkpoint. One subplot per family, side by side.
"""

from __future__ import annotations

from typing import List

from ..analysis.transitions import TransitionAnalyzer
from ..config import ModelPair
from .style import CLASSIFICATION_COLORS


class SafetyDriftSankeyPlotter:
    def __init__(self, analyzer: TransitionAnalyzer):
        self.analyzer = analyzer

    def _counts_binary(self, pair: ModelPair):
        groups = self.analyzer.group_by_transition(pair)
        return {
            ("Equivalent", "Equivalent"): len(groups["Equivalent → Equivalent"]),
            ("Equivalent", "Critical"): len(groups["Equivalent → Critical"]),
            ("Critical", "Equivalent"): len(groups["Critical → Equivalent"]),
            ("Critical", "Critical"): len(groups["Critical → Critical"]),
        }

    def plot(self, model_pairs: List[ModelPair], output_html_path: str, title: str = "Contextual Safety Drift After Fine-tuning"):
        import plotly.graph_objects as go  # deferred: keep this an optional dependency
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=1, cols=len(model_pairs),
            specs=[[{"type": "domain"}] * len(model_pairs)],
            subplot_titles=[p.family for p in model_pairs],
        )

        node_map = {("pre", "Equivalent"): 0, ("pre", "Critical"): 1, ("ft", "Equivalent"): 2, ("ft", "Critical"): 3}
        labels = ["Pre\nEquivalent", "Pre\nCritical", "FT\nEquivalent", "FT\nCritical"]
        node_colors = [CLASSIFICATION_COLORS["Equivalent"], CLASSIFICATION_COLORS["Critical_Errors"], CLASSIFICATION_COLORS["Equivalent"], CLASSIFICATION_COLORS["Critical_Errors"]]

        for col, pair in enumerate(model_pairs, start=1):
            counts = self._counts_binary(pair)

            sources, targets, values, link_colors = [], [], [], []
            for (pre, fin), count in counts.items():
                sources.append(node_map[("pre", pre)])
                targets.append(node_map[("ft", fin)])
                values.append(count)
                link_colors.append(CLASSIFICATION_COLORS["Equivalent"] if fin == "Equivalent" else CLASSIFICATION_COLORS["Critical_Errors"])

            sankey = go.Sankey(
                arrangement="snap",
                node=dict(pad=25, thickness=25, line=dict(color="black", width=0.5), label=labels, color=node_colors),
                link=dict(source=sources, target=targets, value=values, color=link_colors),
            )
            fig.add_trace(sankey, row=1, col=col)

        fig.update_layout(title_text=title, font=dict(size=14), height=500, width=500 * len(model_pairs))
        fig.write_html(output_html_path)
        print(f"Saved: {output_html_path}")
        return fig
