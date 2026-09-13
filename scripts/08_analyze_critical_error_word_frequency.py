#!/usr/bin/env python3
"""Does the WER-CxER paradox have a word-frequency explanation?

Pulls the exact words the LLM judge flagged inside each Critical_Errors
sample (transcription_errors[].reference_value) and checks whether those
words are disproportionately RARE compared to the corpus overall — and
whether fine-tuning's residual critical errors skew even rarer than the
pretrained model's did. A chi-square goodness-of-fit test backs the
"this isn't just noise" claim.

Requires the merged JSON from stage 3 (03_merge_cxer_labels.py).

Example:
    python scripts/08_analyze_critical_error_word_frequency.py
    python scripts/08_analyze_critical_error_word_frequency.py --mode rarest
"""

import argparse

import _bootstrap  # noqa: F401
from faa_cxer.analysis import CriticalErrorTokenAnalyzer
from faa_cxer.config import DEFAULT_MODEL_PAIRS, default_config
from faa_cxer.io_utils import load_records
from faa_cxer.viz import ErrorTokenFrequencyPlotter


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.predictions_with_metrics_and_cxer))
    parser.add_argument("--judge", default=cfg.judge_short_name())
    parser.add_argument("--output-dir", default=str(cfg.outputs_dir))
    parser.add_argument("--mode", choices=["all", "rarest"], default="all",
                         help="'all' counts every word in each flagged error snippet; "
                              "'rarest' counts only the least-frequent word per error "
                              "(isolates the likely callsign/waypoint from connective words)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    samples = load_records(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    analyzer = CriticalErrorTokenAnalyzer(samples, judge=args.judge)
    print("\nCorpus token-frequency table built from the shared reference column.")

    results = {}
    for pair in DEFAULT_MODEL_PAIRS:
        print(f"\n{'=' * 100}\n{pair.family}\n{'=' * 100}")
        tables = analyzer.compare_pretrained_vs_finetuned(pair, mode=args.mode)
        results[pair.family] = tables

        for variant, df in tables.items():
            summary = analyzer.rarity_summary(pair.pretrained_key if variant == "pretrained" else pair.finetuned_key, mode=args.mode)
            chi2_result = analyzer.chi_square_test(pair.pretrained_key if variant == "pretrained" else pair.finetuned_key, mode=args.mode)

            print(f"\n[{variant}]  ({df.attrs['n_critical_errors_scanned']} critical errors, "
                  f"{df.attrs['n_error_token_instances']} word instances)")
            print(df.to_string(index=False))
            print(f"  mean corpus freq of error words: {summary.get('mean_corpus_freq')}   "
                  f"median: {summary.get('median_corpus_freq')}   "
                  f"% rare (<=4 occurrences): {summary.get('pct_rare_le4')}%   "
                  f"% hapax: {summary.get('pct_hapax')}%")
            if chi2_result:
                sig = "***" if chi2_result["p_value"] < 0.001 else "**" if chi2_result["p_value"] < 0.01 else "*" if chi2_result["p_value"] < 0.05 else "n.s."
                print(f"  chi-square goodness-of-fit vs. corpus distribution: "
                      f"chi2={chi2_result['chi2']}, p={chi2_result['p_value']:.2e} ({sig}), n={chi2_result['n']}")

    if args.no_plots:
        return

    plotter = ErrorTokenFrequencyPlotter()
    plotter.plot_enrichment(results, f"{args.output_dir}/critical_error_word_rarity_enrichment.png")


if __name__ == "__main__":
    main()
