"""Stage 3 of the pipeline: merge per-(judge, asr_model) CxER label files
produced by `faa_cxer.judge` into the WER/BERTDist/SemDist metrics JSON
produced by `faa_cxer.metrics.BenchmarkScorer`.

Equivalent to the original Helper.ipynb "merge CxER labels" cell, as a
reusable class instead of a top-to-bottom script. Neither input file is
ever modified — this always produces a new merged JSON.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from ..io_utils import PathLike, load_json, save_json_atomic


class CxERMetricsMerger:
    """Combines:

      - a metrics JSON (per-sample WER/BERTDist/SemDist + corpus_metrics), and
      - a directory tree of CxER label files at
        `cxer_labels_root/<judge_short_name>/<asr_model_key>.json`
        (as produced by `CxERJudgePipeline`)

    into one JSON where every sample gains a `sample["cxer"][judge][asr_model]`
    block, and `corpus_metrics[asr_model]["cxer"]` gains the headline rate.

    Every merged CxER record's `reference` is checked against the metrics
    file's `reference` at the same `sample_index` — a mismatch means the two
    runs are misaligned, and that sample is skipped (or raised on, if
    `strict=True`) rather than silently attaching the wrong label.
    """

    def __init__(self, strict_reference_check: bool = False, filter_to_cxer_coverage: bool = True):
        self.strict = strict_reference_check
        self.filter_to_cxer_coverage = filter_to_cxer_coverage

    @staticmethod
    def discover_cxer_files(root: PathLike) -> List[Dict[str, str]]:
        """Walk `root/<judge_short_name>/<asr_model_key>.json` and return
        `[{"judge_short", "asr_model_key", "path"}, ...]`."""
        root = str(root)
        found = []
        if not os.path.isdir(root):
            print(f"  ⚠ CxER labels root does not exist: {root}")
            return found

        for judge_short in sorted(os.listdir(root)):
            judge_dir = os.path.join(root, judge_short)
            if not os.path.isdir(judge_dir):
                continue
            for fname in sorted(os.listdir(judge_dir)):
                if not fname.endswith(".json"):
                    continue
                found.append({
                    "judge_short": judge_short,
                    "asr_model_key": fname[: -len(".json")],
                    "path": os.path.join(judge_dir, fname),
                })
        return found

    def merge(self, metrics_data: Any, cxer_files: List[Dict[str, str]]) -> Any:
        is_wrapped = isinstance(metrics_data, dict)
        samples = metrics_data["samples"] if is_wrapped else metrics_data
        if not isinstance(samples, list):
            raise ValueError("metrics data must contain a list under 'samples'")

        original_total = len(samples)

        # Stamp original position BEFORE any filtering — once samples are
        # dropped, list position no longer equals position in the master file.
        for idx, sample in enumerate(samples):
            sample["sample_index"] = idx

        corpus_cxer_metrics: Dict[str, Dict[str, Any]] = {}
        mismatches = []

        for entry in cxer_files:
            judge_short, asr_model_key, path = entry["judge_short"], entry["asr_model_key"], entry["path"]
            cxer_data = load_json(path)

            corpus_cxer_metrics.setdefault(judge_short, {})[asr_model_key] = {
                "judge_model": cxer_data.get("judge_model"),
                "prompt_type": cxer_data.get("prompt_type"),
                "status": cxer_data.get("status"),
                "num_samples_total": cxer_data.get("num_samples_total"),
                "num_samples_done": cxer_data.get("num_samples_done"),
                "num_equivalent": cxer_data.get("num_equivalent"),
                "num_critical_errors": cxer_data.get("num_critical_errors"),
                "num_parse_failures": cxer_data.get("num_parse_failures"),
                "num_evaluated_successfully": cxer_data.get("num_evaluated_successfully"),
                "contextual_error_rate": cxer_data.get("contextual_error_rate"),
                "total_inference_time": cxer_data.get("total_inference_time"),
            }

            records = cxer_data.get("records", [])
            n_merged = 0
            for rec in records:
                idx = rec.get("sample_index")
                if idx is None or idx < 0 or idx >= len(samples):
                    print(f"  ⚠ {judge_short}/{asr_model_key}: sample_index {idx!r} out of range, skipping")
                    continue

                sample = samples[idx]
                master_ref = (sample.get("reference") or "").strip()
                cxer_ref = (rec.get("reference") or "").strip()
                if master_ref != cxer_ref:
                    mismatches.append((judge_short, asr_model_key, idx, master_ref, cxer_ref))
                    if self.strict:
                        raise ValueError(
                            f"Reference mismatch at sample_index {idx} for {judge_short}/{asr_model_key}:\n"
                            f"  master: {master_ref!r}\n  cxer:   {cxer_ref!r}"
                        )
                    continue

                cxer_entry = {
                    "classification": rec.get("classification"),
                    "meaning_preserved": rec.get("meaning_preserved"),
                    "operational_safety": rec.get("operational_safety"),
                    "explanation": rec.get("explanation"),
                    "transcription_errors": rec.get("transcription_errors"),
                    "inference_time": rec.get("inference_time"),
                }
                sample.setdefault("cxer", {}).setdefault(judge_short, {})[asr_model_key] = cxer_entry
                n_merged += 1

            print(f"  ✓ {judge_short}/{asr_model_key}: merged {n_merged}/{len(records)} records")

        if mismatches:
            print(f"\n  ⚠ {len(mismatches)} reference mismatch(es) were SKIPPED (not merged). First few:")
            for judge_short, asr_model_key, idx, master_ref, cxer_ref in mismatches[:5]:
                print(f"    [{judge_short}/{asr_model_key}] idx={idx}\n      master: {master_ref!r}\n      cxer:   {cxer_ref!r}")

        corpus_metrics = metrics_data.get("corpus_metrics", {}) if is_wrapped else {}

        rates_by_model: Dict[str, Dict[str, Any]] = {}
        for judge_short, models in corpus_cxer_metrics.items():
            for asr_model_key, summary in models.items():
                rates_by_model.setdefault(asr_model_key, {})[judge_short] = summary.get("contextual_error_rate")

        for asr_model_key, judge_rates in rates_by_model.items():
            if asr_model_key not in corpus_metrics:
                print(f"  ⚠ '{asr_model_key}' has CxER results but no entry in corpus_metrics — adding a new entry with just 'cxer'")
                corpus_metrics[asr_model_key] = {}
            if len(judge_rates) == 1:
                corpus_metrics[asr_model_key]["cxer"] = next(iter(judge_rates.values()))
            else:
                corpus_metrics[asr_model_key]["cxer"] = judge_rates
                print(f"  ⚠ '{asr_model_key}' has CxER results from {len(judge_rates)} judges — storing 'cxer' as {{judge: rate}} instead of a single number")

        kept_samples = [s for s in samples if s.get("cxer")] if self.filter_to_cxer_coverage else samples

        if is_wrapped:
            metrics_data["corpus_metrics"] = corpus_metrics
            metrics_data["corpus_cxer_metrics"] = corpus_cxer_metrics
            metrics_data["samples"] = kept_samples
            metadata = metrics_data.get("_metadata", {})
            metadata["total_samples"] = len(kept_samples)
            metadata["total_samples_before_cxer_filter"] = original_total
            metadata["filtered_to_cxer_coverage"] = self.filter_to_cxer_coverage
            metrics_data["_metadata"] = metadata
            return metrics_data

        return {"samples": kept_samples, "corpus_metrics": corpus_metrics, "corpus_cxer_metrics": corpus_cxer_metrics}

    def run(self, metrics_json_path: PathLike, cxer_labels_root: PathLike, output_json_path: PathLike) -> Any:
        print(f"Loading metrics file: {metrics_json_path}")
        metrics_data = load_json(metrics_json_path)
        n_before = len(metrics_data["samples"]) if isinstance(metrics_data, dict) else len(metrics_data)
        print(f"  {n_before} samples loaded")

        print(f"\nDiscovering CxER label files under: {cxer_labels_root}")
        cxer_files = self.discover_cxer_files(cxer_labels_root)
        print(f"  Found {len(cxer_files)} (judge, asr_model) label file(s):")
        for entry in cxer_files:
            print(f"    {entry['judge_short']} / {entry['asr_model_key']}")
        if not cxer_files:
            print("\n  ⚠ No CxER label files found. With filter_to_cxer_coverage=True this would produce an EMPTY samples list.")

        print("\nMerging...")
        merged = self.merge(metrics_data, cxer_files)

        n_after = len(merged["samples"]) if isinstance(merged, dict) else len(merged)
        print(f"\n  Samples: {n_before} → {n_after} ({'filtered to CxER coverage' if self.filter_to_cxer_coverage else 'no filtering applied'})")

        save_json_atomic(output_json_path, merged)
        print(f"\nSaved merged file → {output_json_path}")
        return merged
