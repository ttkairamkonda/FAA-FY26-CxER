#!/usr/bin/env python3
"""Stage 1: score every ASR model's predictions against the reference
transcript — WER, BERTDist, SemDist — and write a metrics JSON + CSV.

Example:
    python scripts/01_compute_benchmark_scores.py
    python scripts/01_compute_benchmark_scores.py --input data/predictions/FAA_multi_model_predictions_normalized.json
"""

import argparse

import _bootstrap  # noqa: F401
from faa_cxer.config import default_config
from faa_cxer.metrics import BenchmarkScorer


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.normalized_predictions), help="Normalized predictions JSON (list of {audio_path, reference, <model>: hypothesis, ...})")
    parser.add_argument("--output-json", default=str(cfg.predictions_with_metrics), help="Where to write predictions + per-sample/per-model metrics")
    parser.add_argument("--output-csv", default=str(cfg.metrics_summary_csv), help="Where to write the flat corpus-metrics summary CSV")
    parser.add_argument("--models", nargs="*", default=None, help="Subset of model columns to score (default: every column except audio_path/reference)")
    parser.add_argument("--semantic-batch-size", type=int, default=32)
    args = parser.parse_args()

    scorer = BenchmarkScorer(semantic_batch_size=args.semantic_batch_size)
    scorer.run(args.input, args.output_json, args.output_csv, model_keys=args.models)


if __name__ == "__main__":
    main()
