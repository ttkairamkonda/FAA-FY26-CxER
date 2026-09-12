"""Stage 1 of the pipeline: turn raw multi-model predictions into WER /
BERTDist / SemDist scores, per sample and per corpus.

Equivalent to the original `faa_benchmark_scores.py`, restructured as a
class so it can be reused (e.g. scored on a subset, or with a different
semantic scorer) instead of being a top-to-bottom script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..io_utils import PathLike, TextNormalizer, load_json, save_json_atomic
from .semantic import SemanticDistanceScorer
from .wer import WERScorer


class BenchmarkScorer:
    """Scores every ASR model column found in a normalized predictions file
    against the shared "reference" column, writing:

      - a JSON with per-sample `<model>_WER` / `_BERTDist` / `_SemDist` keys
        plus a top-level `corpus_metrics` block, and
      - a flat CSV summary of just the corpus-level numbers.

    A sample is only scored for a model if BOTH its normalized reference
    and normalized hypothesis are non-empty; empty pairs are silently
    excluded from that model's corpus WER (jiwer errors on empty strings).
    """

    IGNORE_COLUMNS = {"audio_path", "reference"}

    def __init__(
        self,
        semantic_scorer: Optional[SemanticDistanceScorer] = None,
        wer_scorer: Optional[WERScorer] = None,
        semantic_batch_size: int = 32,
    ):
        self.wer_scorer = wer_scorer or WERScorer()
        # SemanticDistanceScorer loads roberta-large onto the GPU — defer
        # construction until first needed so `BenchmarkScorer()` alone
        # doesn't require a GPU (e.g. in tests).
        self._semantic_scorer = semantic_scorer
        self.semantic_batch_size = semantic_batch_size

    @property
    def semantic_scorer(self) -> SemanticDistanceScorer:
        if self._semantic_scorer is None:
            self._semantic_scorer = SemanticDistanceScorer()
        return self._semantic_scorer

    def _model_keys(self, samples: List[Dict[str, Any]]) -> List[str]:
        return [k for k in samples[0].keys() if k not in self.IGNORE_COLUMNS]

    def score_model(self, samples: List[Dict[str, Any]], model_key: str) -> Dict[str, Any]:
        """Score one ASR model column against `reference` for every sample.
        Mutates `samples` in place (adds `<model_key>_WER/_BERTDist/_SemDist`)
        and returns that model's corpus-level metrics dict."""
        refs, hyps, valid_idx = [], [], []
        for i, s in enumerate(samples):
            ref = self.wer_scorer.normalize(s.get("reference", ""))
            hyp = self.wer_scorer.normalize(s.get(model_key, ""))
            if ref and hyp:
                refs.append(ref)
                hyps.append(hyp)
                valid_idx.append(i)

        sample_wers = [self.wer_scorer.sample_wer(r, h) for r, h in zip(refs, hyps)]
        bert_dists, sem_dists = self.semantic_scorer.score(refs, hyps, batch_size=self.semantic_batch_size)

        for j, idx in enumerate(valid_idx):
            samples[idx][f"{model_key}_WER"] = float(sample_wers[j])
            samples[idx][f"{model_key}_BERTDist"] = float(bert_dists[j])
            samples[idx][f"{model_key}_SemDist"] = float(sem_dists[j])

        return {
            "WER": round(self.wer_scorer.corpus_wer(refs, hyps) * 100, 4),
            "BERTDist": round(float(bert_dists.mean()), 6),
            "SemDist": round(float(sem_dists.mean()), 6),
        }

    def score_all(
        self, samples: List[Dict[str, Any]], model_keys: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        model_keys = model_keys or self._model_keys(samples)
        corpus_metrics: Dict[str, Dict[str, Any]] = {}
        for model_key in model_keys:
            print(f"\n{'=' * 80}\n{model_key}\n{'=' * 80}")
            corpus_metrics[model_key] = self.score_model(samples, model_key)
            print(corpus_metrics[model_key])
        return corpus_metrics

    def run(
        self,
        input_json: PathLike,
        output_json: PathLike,
        output_csv: PathLike,
        model_keys: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Full stage-1 pipeline: load -> score every model -> save JSON + CSV."""
        samples = load_json(input_json)
        print(f"Loaded {len(samples)} samples from {input_json}")

        corpus_metrics = self.score_all(samples, model_keys=model_keys)

        output = {
            "_metadata": {"total_samples": len(samples), "num_models": len(corpus_metrics)},
            "corpus_metrics": corpus_metrics,
            "samples": samples,
        }
        save_json_atomic(output_json, output)
        print(f"\nSaved JSON: {output_json}")

        rows = [{"model": m, **v} for m, v in corpus_metrics.items()]
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"Saved CSV: {output_csv}")

        return corpus_metrics
