#!/usr/bin/env python3
"""Stage 3: merge the CxER label tree from stage 2 into the metrics JSON
from stage 1, producing the single merged file every analysis/plot script
downstream reads.

Example:
    python scripts/03_merge_cxer_labels.py
"""

import argparse

import _bootstrap  # noqa: F401
from faa_cxer.analysis import CxERMetricsMerger
from faa_cxer.config import default_config


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-json", default=str(cfg.predictions_with_metrics))
    parser.add_argument("--cxer-labels-root", default=str(cfg.cxer_labels_root))
    parser.add_argument("--output-json", default=str(cfg.predictions_with_metrics_and_cxer))
    parser.add_argument("--keep-uncovered-samples", action="store_true", help="Keep samples with zero CxER coverage instead of dropping them")
    parser.add_argument("--strict-reference-check", action="store_true", help="Raise on the first reference mismatch instead of skipping + reporting it")
    args = parser.parse_args()

    merger = CxERMetricsMerger(
        strict_reference_check=args.strict_reference_check,
        filter_to_cxer_coverage=not args.keep_uncovered_samples,
    )
    merger.run(args.metrics_json, args.cxer_labels_root, args.output_json)


if __name__ == "__main__":
    main()
