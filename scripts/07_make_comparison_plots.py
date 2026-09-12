#!/usr/bin/env python3
"""2x2 grouped bar chart comparing every model family's pretrained vs.
fine-tuned WER / BERTDist / SemDist, plus a 4th CxER panel once every
model in the comparison has a CxER score merged in.

Example:
    python scripts/07_make_comparison_plots.py
"""

import argparse

import _bootstrap  # noqa: F401
from faa_cxer.config import DEFAULT_MODEL_PAIRS, default_config
from faa_cxer.io_utils import load_json
from faa_cxer.viz import ComparisonBarChart


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.predictions_with_metrics_and_cxer), help="Merged JSON from stage 3 (falls back to plotting without CxER if it only has WER/BERTDist/SemDist)")
    parser.add_argument("--output", default=f"{cfg.outputs_dir}/faa_pretrained_vs_finetuned.png")
    parser.add_argument("--judge", default=None, help="Pick one judge's CxER score when a model has results from more than one (default: auto, errors if ambiguous)")
    args = parser.parse_args()

    data = load_json(args.input)
    chart = ComparisonBarChart(
        DEFAULT_MODEL_PAIRS,
        corpus_metrics=data["corpus_metrics"],
        corpus_cxer_metrics=data.get("corpus_cxer_metrics", {}),
        judge=args.judge,
    )
    chart.plot(args.output)


if __name__ == "__main__":
    main()
