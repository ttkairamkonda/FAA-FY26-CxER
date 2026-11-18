"""Word error rate scoring and word-level alignment."""

from __future__ import annotations

from typing import List, Tuple

import jiwer

from ..io_utils import TextNormalizer


class WERScorer:
    """Thin, testable wrapper around jiwer for both corpus-level and
    per-sample WER, plus word-level correct/incorrect alignment."""

    def __init__(self, normalizer: TextNormalizer = None):
        self.normalizer = normalizer or TextNormalizer()

    def normalize(self, text: str) -> str:
        return self.normalizer(text)

    def sample_wer(self, reference: str, hypothesis: str) -> float:
        """WER for a single (reference, hypothesis) pair. Assumes both are
        already normalized (call `.normalize()` first) — jiwer.wer([r],[h])
        on a single pair is what the corpus-level scripts in this pipeline
        use for per-sample numbers."""
        return jiwer.wer(reference, hypothesis)

    def corpus_wer(self, references: List[str], hypotheses: List[str]) -> float:
        return jiwer.wer(references, hypotheses)

    def word_results(self, ref_norm: str, hyp_norm: str) -> List[Tuple[str, bool]]:
        """Align reference vs. hypothesis at the word level and return one
        (word, is_error) tuple per reference word — used to bucket errors
        by how frequent each reference word is in the corpus."""
        ref_words = ref_norm.split()
        if not ref_words:
            return []
        if not hyp_norm.strip():
            return [(w, True) for w in ref_words]

        output = jiwer.process_words(ref_norm, hyp_norm)
        results: List[Tuple[str, bool]] = []
        for chunk in output.alignments[0]:
            if chunk.type == "equal":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    results.append((ref_words[i], False))
            elif chunk.type in ("substitute", "delete"):
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    results.append((ref_words[i], True))
        return results
