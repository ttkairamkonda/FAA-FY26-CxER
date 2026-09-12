#!/usr/bin/env python3
"""The core WER-CxER paradox analysis: for each model, what fraction of
low-WER ("looks fine") samples actually contain a critical error, and does
that fraction get WORSE after fine-tuning even as WER drops? Prints the
concentration/trust-violation table, then renders the per-family jitter
scatter plots (`wer_paradox_*.png`).

Requires the merged JSON from stage 3 (`03_merge_cxer_labels.py`).

Example:
    python scripts/06_analyze_paradox.py
    python scripts/06_analyze_paradox.py --threshold-mode fixed --threshold-fixed 0.2
"""

import argparse

import _bootstrap  # noqa: F401
from faa_cxer.analysis import ParadoxAnalyzer
from faa_cxer.config import DEFAULT_MODEL_PAIRS, default_config
from faa_cxer.io_utils import load_records
from faa_cxer.viz import ParadoxScatterPlotter

STANDALONE_MODEL_KEY = "FAA_Model"  # the production model, no pretrained counterpart


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.predictions_with_metrics_and_cxer))
    parser.add_argument("--judge", default=cfg.judge_short_name())
    parser.add_argument("--output-dir", default=str(cfg.outputs_dir))
    parser.add_argument("--threshold-mode", choices=["fixed", "percentile"], default="percentile")
    parser.add_argument("--threshold-fixed", type=float, default=0.2)
    parser.add_argument("--threshold-percentile", type=float, default=25)
    parser.add_argument("--no-plots", action="store_true", help="Print the table only, skip the scatter plots")
    args = parser.parse_args()

    samples = load_records(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    analyzer = ParadoxAnalyzer(samples, judge=args.judge)

    table = analyzer.build_paradox_table(
        DEFAULT_MODEL_PAIRS, standalone_key=STANDALONE_MODEL_KEY,
        threshold_mode=args.threshold_mode, threshold_fixed=args.threshold_fixed, threshold_percentile=args.threshold_percentile,
    )
    print(f"\n{'=' * 100}\nPARADOX TABLE\n{'=' * 100}")
    print(table.to_string(index=False))

    print(f"\n{'=' * 100}\nDELTA (Finetuned - Pretrained)\n{'=' * 100}")
    print(analyzer.delta_table(table, DEFAULT_MODEL_PAIRS).to_string(index=False))

    if args.no_plots:
        return

    plotter = ParadoxScatterPlotter(analyzer)
    for pair in DEFAULT_MODEL_PAIRS:
        out_name = f"wer_paradox_{pair.family.lower().replace(' ', '_')}.png"
        plotter.plot_comparison(
            pair.family, pair.pretrained_key, pair.finetuned_key, f"{args.output_dir}/{out_name}",
            threshold_mode=args.threshold_mode, threshold_fixed=args.threshold_fixed, threshold_percentile=args.threshold_percentile,
        )
    plotter.plot_standalone(
        STANDALONE_MODEL_KEY, STANDALONE_MODEL_KEY, f"{args.output_dir}/wer_paradox_faa_model.png",
        threshold_mode=args.threshold_mode, threshold_fixed=args.threshold_fixed, threshold_percentile=args.threshold_percentile,
    )


if __name__ == "__main__":
    main()
