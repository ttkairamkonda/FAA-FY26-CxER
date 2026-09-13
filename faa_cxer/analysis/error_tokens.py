"""Does the WER-CxER paradox have a word-frequency explanation?

`faa_cxer.metrics.TokenFrequencyAnalyzer` shows fine-tuning fixes common
words far more than rare ones. `faa_cxer.analysis.paradox.ParadoxAnalyzer`
shows CxER barely improves even as WER collapses. This module connects the
two: it pulls the EXACT words the LLM judge flagged inside each
`Critical_Errors` sample (`transcription_errors[].reference_value` — e.g.
"southwest" in a wrong-callsign error, "mattr" in a wrong-waypoint error)
and checks whether those words are disproportionately rare compared to the
corpus as a whole — and whether that skew gets WORSE after fine-tuning,
i.e. the errors that survive fine-tuning are increasingly concentrated on
the words the model has seen the least.

This turns two side-by-side plots into one causal claim: fine-tuning
doesn't just help common words more (already shown) — the critical errors
that remain are specifically clustered on the rare words that didn't
improve, which is why WER (dominated by common words) drops so much more
than CxER (driven by whichever words are actually in the errors).
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import ModelPair
from ..metrics.token_frequency import TokenFrequencyAnalyzer


class CriticalErrorTokenAnalyzer:
    """Operates on the merged `samples` list (`CxERMetricsMerger` output) —
    reuses a `TokenFrequencyAnalyzer` for corpus-wide word frequencies and
    bucket definitions, so bucket boundaries and normalization never drift
    from the plain frequency analysis this builds on.
    """

    def __init__(self, samples: List[dict], judge: str, token_analyzer: Optional[TokenFrequencyAnalyzer] = None):
        self.samples = samples
        self.judge = judge
        self.token_analyzer = token_analyzer or TokenFrequencyAnalyzer()
        if not self.token_analyzer.token_freq:
            self.token_analyzer.build_token_frequency(samples)

    def _normalize_words(self, text: str) -> List[str]:
        norm = self.token_analyzer.wer_scorer.normalize(text or "")
        return norm.split() if norm else []

    def extract_error_tokens(self, model_key: str, mode: str = "all") -> Tuple[Counter, int]:
        """Counter of words implicated in `model_key`'s Critical_Errors
        samples, plus how many flagged errors were scanned.

        `mode="all"` counts every word in every flagged error's
        `reference_value` snippet (same granularity the plain word-error
        analysis uses). `mode="rarest"` counts only the single
        least-frequent word per flagged error — isolating the "signature"
        operational word (a callsign, a waypoint) from connective words
        ("from", "to", "the") that often ride along in the same snippet.
        """
        counter: Counter = Counter()
        n_errors = 0
        for s in self.samples:
            entry = s.get("cxer", {}).get(self.judge, {}).get(model_key)
            if entry is None or entry.get("classification") != "Critical_Errors":
                continue
            for err in entry.get("transcription_errors") or []:
                words = self._normalize_words(err.get("reference_value", ""))
                if not words:
                    continue
                n_errors += 1
                if mode == "rarest":
                    rarest = min(words, key=lambda w: self.token_analyzer.token_freq.get(w, 0))
                    counter[rarest] += 1
                else:
                    counter.update(words)
        return counter, n_errors

    def bucket_instance_counts(self, token_counter: Counter) -> Dict[str, int]:
        """Sum `token_counter`'s occurrence counts into frequency buckets,
        bucketing each word by its CORPUS-WIDE frequency (not its count
        within `token_counter`)."""
        counts = {lbl: 0 for _, _, lbl in self.token_analyzer.buckets}
        for word, n in token_counter.items():
            label = self.token_analyzer.bucket_label(self.token_analyzer.token_freq.get(word, 0))
            if label == "other":
                continue
            counts[label] += n
        return counts

    def corpus_baseline_distribution(self) -> Dict[str, float]:
        """% of all corpus token INSTANCES falling in each bucket — the
        "if critical errors were spread like ordinary text" null hypothesis."""
        total = sum(self.token_analyzer.token_freq.values())
        dist = {}
        for lo, hi, lbl in self.token_analyzer.buckets:
            instances = sum(v for v in self.token_analyzer.token_freq.values() if lo <= v <= hi)
            dist[lbl] = 100 * instances / total if total else float("nan")
        return dist

    def concentration_table(self, model_key: str, mode: str = "all") -> pd.DataFrame:
        """Per bucket: % of the corpus vs. % of critical-error tokens vs.
        the enrichment ratio (>1 = this bucket is overrepresented among
        critical errors relative to how common it is in the corpus)."""
        counter, n_errors = self.extract_error_tokens(model_key, mode=mode)
        bucket_counts = self.bucket_instance_counts(counter)
        total_error_tokens = sum(bucket_counts.values())
        baseline = self.corpus_baseline_distribution()

        rows = []
        for lbl in self.token_analyzer.bucket_labels_ordered():
            corpus_pct = baseline[lbl]
            error_pct = 100 * bucket_counts[lbl] / total_error_tokens if total_error_tokens else float("nan")
            enrichment = error_pct / corpus_pct if corpus_pct else float("nan")
            rows.append({
                "Bucket": lbl.replace("\n", " "),
                "Corpus %": round(corpus_pct, 2),
                "Critical-Error %": round(error_pct, 2),
                "Enrichment": round(enrichment, 2),
            })

        df = pd.DataFrame(rows)
        df.attrs["n_critical_errors_scanned"] = n_errors
        df.attrs["n_error_token_instances"] = total_error_tokens
        return df

    def compare_pretrained_vs_finetuned(self, pair: ModelPair, mode: str = "all") -> Dict[str, pd.DataFrame]:
        return {
            "pretrained": self.concentration_table(pair.pretrained_key, mode=mode),
            "finetuned": self.concentration_table(pair.finetuned_key, mode=mode),
        }

    def rarity_summary(self, model_key: str, mode: str = "all") -> Dict[str, float]:
        """Single-number headline stats: how rare, on average, are the
        words behind this model's critical errors?"""
        counter, n_errors = self.extract_error_tokens(model_key, mode=mode)
        freqs: List[int] = []
        for word, n in counter.items():
            freqs.extend([self.token_analyzer.token_freq.get(word, 0)] * n)

        if not freqs:
            return {"n_critical_errors_scanned": n_errors, "n_error_token_instances": 0}

        arr = np.array(freqs)
        return {
            "n_critical_errors_scanned": n_errors,
            "n_error_token_instances": int(arr.size),
            "mean_corpus_freq": round(float(arr.mean()), 2),
            "median_corpus_freq": float(np.median(arr)),
            "pct_rare_le4": round(100 * float((arr <= 4).mean()), 2),
            "pct_hapax": round(100 * float((arr == 1).mean()), 2),
        }

    def chi_square_test(self, model_key: str, mode: str = "all") -> Optional[Dict[str, float]]:
        """Chi-square goodness-of-fit: is the bucket distribution of
        critical-error tokens consistent with the corpus-wide distribution
        (null hypothesis), or is the skew toward rare buckets too large to
        be explained by chance?"""
        from scipy.stats import chisquare

        counter, _ = self.extract_error_tokens(model_key, mode=mode)
        observed_counts = self.bucket_instance_counts(counter)
        total_observed = sum(observed_counts.values())
        if total_observed == 0:
            return None

        total_corpus = sum(self.token_analyzer.token_freq.values())
        observed, expected = [], []
        for lo, hi, lbl in self.token_analyzer.buckets:
            corpus_instances = sum(v for v in self.token_analyzer.token_freq.values() if lo <= v <= hi)
            observed.append(observed_counts[lbl])
            expected.append(total_observed * corpus_instances / total_corpus)

        chi2, p_value = chisquare(f_obs=observed, f_exp=expected)
        return {"chi2": round(float(chi2), 2), "p_value": float(p_value), "n": total_observed}
