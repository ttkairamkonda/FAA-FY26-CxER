"""Embedding-based distance metrics: BERTScore distance and SemDist
(1 - cosine similarity of mean-pooled RoBERTa embeddings).

Both metrics need the same RoBERTa-large model loaded on the GPU, so they
are grouped behind one class that loads the model once and reuses it.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from bert_score import score as bert_score_fn
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer


class SemanticDistanceScorer:
    """Computes BERTDist (1 - BERTScore F1) and SemDist (1 - cosine
    similarity of mean-pooled embeddings), both against `roberta-large`.

    Loading the model is expensive, so construct one instance and reuse it
    across every ASR model you're scoring in a run.
    """

    def __init__(self, model_name: str = "roberta-large", device: str = None, max_length: int = 256):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        print(f"[SemanticDistanceScorer] loading {model_name} on {self.device}")
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 64, show_progress: bool = False) -> torch.Tensor:
        """Mean-pool the last hidden state (masked by attention) into one
        vector per text."""
        all_emb = []
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, leave=False)

        for i in iterator:
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
            ).to(self.device)
            out = self.model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            all_emb.append(emb.cpu())

        return torch.cat(all_emb, dim=0)

    def bertscore_distances(self, references: List[str], hypotheses: List[str], batch_size: int = 32) -> np.ndarray:
        """Per-sample BERTDist = 1 - BERTScore F1. Lower is better."""
        _, _, f1 = bert_score_fn(
            hypotheses,
            references,
            lang="en",
            model_type=self.model_name,
            batch_size=batch_size,
            device=self.device,
            verbose=False,
            rescale_with_baseline=False,
        )
        return 1.0 - f1.cpu().numpy()

    def semdist_distances(self, references: List[str], hypotheses: List[str], batch_size: int = 64) -> np.ndarray:
        """Per-sample SemDist = 1 - cosine(embed(ref), embed(hyp)). Lower is better."""
        ref_emb = self.embed(references, batch_size=batch_size)
        hyp_emb = self.embed(hypotheses, batch_size=batch_size)
        sims = torch.nn.functional.cosine_similarity(ref_emb, hyp_emb, dim=1)
        return (1.0 - sims).numpy()

    def score(
        self, references: List[str], hypotheses: List[str], batch_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience: compute both distances in one call."""
        bert_dists = self.bertscore_distances(references, hypotheses, batch_size=batch_size)
        sem_dists = self.semdist_distances(references, hypotheses, batch_size=batch_size)
        return bert_dists, sem_dists
