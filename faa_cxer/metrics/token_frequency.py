"""Token-frequency analysis: does fine-tuning help common words more than
rare, domain-specific ones (callsigns, frequencies, altitudes, headings)?

This is the analysis side of the original `token_frequency_analysis.py`;
plotting lives in `faa_cxer.viz.frequency_plots.TokenFrequencyPlotter` so
this class can be reused (e.g. in tests, or with a different plot style)
without importing matplotlib.

Unlike the original script, this operates on the same wide-format sample
records (`{"reference": ..., "<model_key>": hypothesis, ...}`) that
`ParadoxAnalyzer` and `TransitionAnalyzer` use, instead of a separate
per-model narrow-format file — one less data shape for the pipeline to
carry around.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..config import ModelPair
from ..io_utils import TextNormalizer
from .wer import WERScorer

# (low, high, label) frequency buckets, evaluated in order — a token's
# bucket is the first range its corpus-wide count falls into.
DEFAULT_FREQ_BUCKETS: List[Tuple[int, float, str]] = [
    (100, float("inf"), ">=100\n(very common)"),
    (50, 99, "50-99"),
    (20, 49, "20-49"),
    (10, 19, "10-19"),
    (5, 9, "5-9"),
    (2, 4, "2-4"),
    (1, 1, "1\n(hapax)"),
]


@dataclass
class BucketStats:
    label: str
    errors: int
    total: int

    @property
    def error_rate_pct(self) -> float:
        return self.errors / self.total * 100 if self.total > 0 else float("nan")


class TokenFrequencyAnalyzer:
    """Builds a token-frequency distribution from a reference corpus, then
    measures each model's word-level error rate within each frequency
    bucket."""

    def __init__(self, buckets: List[Tuple[int, float, str]] = None, wer_scorer: WERScorer = None):
        self.buckets = buckets or DEFAULT_FREQ_BUCKETS
        # Matches the original script's normalizer, which also expands
        # contractions ("don't" -> "do not") before tokenizing — a slightly
        # different normalization than the benchmark scorer uses, so token
        # counts line up with how words are split for bucketing.
        self.wer_scorer = wer_scorer or WERScorer(normalizer=TextNormalizer(expand_contractions=True))
        self.token_freq: Counter = Counter()

    def bucket_label(self, freq: int) -> str:
        for lo, hi, label in self.buckets:
            if lo <= freq <= hi:
                return label
        return "other"

    def bucket_labels_ordered(self) -> List[str]:
        """Labels from rarest to most common — the x-axis order plots use."""
        return [lbl for _, _, lbl in self.buckets][::-1]

    def build_token_frequency(self, samples: List[Dict[str, Any]]) -> Counter:
        """Build the corpus-wide token-frequency table from the shared
        "reference" column (references are the same for every model)."""
        self.token_freq = Counter()
        for rec in samples:
            ref_norm = self.wer_scorer.normalize(rec["reference"])
            if ref_norm:
                self.token_freq.update(ref_norm.split())
        return self.token_freq

    def error_rates_by_bucket(self, samples: List[Dict[str, Any]], model_key: str) -> Dict[str, BucketStats]:
        """Word-level error rate per frequency bucket for one ASR model's
        column. Requires `build_token_frequency()` to have been called first."""
        if not self.token_freq:
            raise RuntimeError("call build_token_frequency() before error_rates_by_bucket()")

        stats = {lbl: BucketStats(lbl, errors=0, total=0) for _, _, lbl in self.buckets}

        for rec in samples:
            ref_norm = self.wer_scorer.normalize(rec["reference"])
            hyp_norm = self.wer_scorer.normalize(rec.get(model_key, ""))
            for word, is_error in self.wer_scorer.word_results(ref_norm, hyp_norm):
                label = self.bucket_label(self.token_freq.get(word, 0))
                if label == "other":
                    continue
                stats[label].total += 1
                stats[label].errors += int(is_error)

        return stats

    def analyze_model_pairs(self, samples: List[Dict[str, Any]], model_pairs: List[ModelPair]) -> Dict[str, Dict[str, Dict[str, BucketStats]]]:
        """Returns {family: {"pretrained"|"finetuned": {bucket_label: BucketStats}}}.

        Builds the token-frequency table once (if not already built) from
        `samples`'s shared "reference" column.
        """
        if not self.token_freq:
            self.build_token_frequency(samples)

        results: Dict[str, Dict[str, Dict[str, BucketStats]]] = {}
        for pair in model_pairs:
            results[pair.family] = {
                "pretrained": self.error_rates_by_bucket(samples, pair.pretrained_key),
                "finetuned": self.error_rates_by_bucket(samples, pair.finetuned_key),
            }
        return results

    def summarize(self) -> None:
        """Print the corpus-wide token-frequency breakdown (unique types,
        hapax count, top tokens, per-bucket type/instance counts)."""
        print(f"  Unique token types    : {len(self.token_freq)}")
        print(f"  Total token instances : {sum(self.token_freq.values())}")
        print(f"  Hapax legomena (freq=1): {sum(1 for v in self.token_freq.values() if v == 1)}")
        print("\n  Top 20 tokens:")
        for tok, cnt in self.token_freq.most_common(20):
            print(f"    {tok!r:20s}  {cnt}")
        print("\n  Token types per bucket:")
        for lo, hi, lbl in self.buckets:
            count = sum(1 for v in self.token_freq.values() if lo <= v <= hi)
            instances = sum(v for v in self.token_freq.values() if lo <= v <= hi)
            print(f"    {lbl.replace(chr(10), ' '):22s}  {count:4d} types  {instances:6d} instances")
