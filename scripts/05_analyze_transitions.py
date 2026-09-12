#!/usr/bin/env python3
"""Contextual "safety drift": for each ASR model family, how many samples
move between Equivalent/Critical_Errors CxER classifications from
pretrained to fine-tuned? Prints a transition matrix + WER-delta breakdown
per family, and (unless --no-sankey) renders a Sankey diagram.

Requires the merged JSON from stage 3 (`03_merge_cxer_labels.py`).

Example:
    python scripts/05_analyze_transitions.py
"""

import argparse

import _bootstrap  # noqa: F401
from faa_cxer.analysis import TransitionAnalyzer
from faa_cxer.config import DEFAULT_MODEL_PAIRS, default_config
from faa_cxer.io_utils import load_records


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.predictions_with_metrics_and_cxer))
    parser.add_argument("--judge", default=cfg.judge_short_name())
    parser.add_argument("--output-dir", default=str(cfg.outputs_dir))
    parser.add_argument("--no-sankey", action="store_true", help="Skip rendering the Sankey HTML (requires plotly)")
    args = parser.parse_args()

    samples = load_records(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    analyzer = TransitionAnalyzer(samples, judge=args.judge)
    results = analyzer.analyze_families(DEFAULT_MODEL_PAIRS)

    for family, r in results.items():
        print(f"\n{'=' * 100}\n{family}\n{'=' * 100}")
        print(r["counts"].to_string(index=False))
        print("\nTransition Matrix\n-----------------")
        print(r["matrix"])
        if not r["wer_deltas"].empty:
            print("\nWER change by transition\n-------------------------")
            print(r["wer_deltas"].to_string(index=False))

    print(f"\n{'=' * 100}\nSUMMARY ACROSS ALL MODEL FAMILIES\n{'=' * 100}")
    print(analyzer.summary_across_families(results).to_string(index=False))

    if not args.no_sankey:
        from faa_cxer.viz import SafetyDriftSankeyPlotter

        SafetyDriftSankeyPlotter(analyzer).plot(DEFAULT_MODEL_PAIRS, f"{args.output_dir}/contextual_safety_drift_sankey.html")


if __name__ == "__main__":
    main()
