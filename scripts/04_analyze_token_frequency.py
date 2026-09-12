#!/usr/bin/env python3
"""Does fine-tuning help common words more than rare, domain-specific ones
(callsigns, altitudes, headings, frequencies)? Buckets reference tokens by
corpus frequency and compares each model family's word-level error rate
per bucket, pretrained vs. fine-tuned.

Reads the same wide-format predictions file stage 1 produces (needs
`reference` + one column per ASR model — the un-normalized-yet `with_metrics`
or the normalized-predictions file both work, since WER/BERTDist columns
are ignored here).

Example:
    python scripts/04_analyze_token_frequency.py
"""

import argparse

import _bootstrap  # noqa: F401
from faa_cxer.config import DEFAULT_MODEL_PAIRS, default_config
from faa_cxer.io_utils import load_records
from faa_cxer.metrics import TokenFrequencyAnalyzer
from faa_cxer.viz import TokenFrequencyPlotter


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.normalized_predictions))
    parser.add_argument("--output-dir", default=str(cfg.outputs_dir))
    args = parser.parse_args()

    samples = load_records(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    analyzer = TokenFrequencyAnalyzer()
    analyzer.build_token_frequency(samples)
    print("\nToken frequency distribution:")
    analyzer.summarize()

    results = analyzer.analyze_model_pairs(samples, DEFAULT_MODEL_PAIRS)

    for family, variants in results.items():
        for variant, stats in variants.items():
            print(f"\n[{family} / {variant}]")
            print(f"  {'Bucket':22s}  {'Errors':>8}  {'Total':>8}  {'Error%':>8}")
            for lbl in analyzer.bucket_labels_ordered()[::-1]:
                s = stats[lbl]
                print(f"  {lbl.replace(chr(10), ' '):22s}  {s.errors:8d}  {s.total:8d}  {s.error_rate_pct:7.1f}%")

    plotter = TokenFrequencyPlotter(analyzer)
    plotter.plot_absolute_rates(results, f"{args.output_dir}/finetuning_rare_vs_frequent.png")
    plotter.plot_relative_improvement(results, f"{args.output_dir}/finetuning_relative_improvement.png")


if __name__ == "__main__":
    main()
