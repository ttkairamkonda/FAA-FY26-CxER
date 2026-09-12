"""
Central configuration for the FAA_CxER pipeline.

Every script and class in this package reads its default file locations
from a single `Config` object built by `default_config()`. Nothing else
in the codebase hardcodes a path — override what you need either by
editing the values here, passing CLI flags to a script (see `scripts/`),
or constructing your own `Config(...)` in code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


# Root of the FAA_CxER project (the directory containing this package).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class ModelPair:
    """A (pretrained, finetuned) checkpoint pair for one ASR model family."""

    family: str
    pretrained_key: str
    finetuned_key: str


# The three ASR model families this study compares pretrained vs. fine-tuned
# (on 500K FAA/ATC utterances). Keys must match the prediction column names
# used throughout data/predictions/*.json.
DEFAULT_MODEL_PAIRS: List[ModelPair] = [
    ModelPair(
        family="Whisper Medium",
        pretrained_key="FAA_Model_prediction_whisper-medium.en",
        finetuned_key="FAA_Model_prediction_FAA_Whisper_M_TL_500K",
    ),
    ModelPair(
        family="Whisper Large",
        pretrained_key="FAA_Model_prediction_whisper-large-v3",
        finetuned_key="FAA_Model_prediction_FAA_Whisper_L_TL_500K",
    ),
    ModelPair(
        family="Parakeet",
        pretrained_key="FAA_Model_prediction_parakeet-tdt-0.6b-v2",
        finetuned_key="FAA_Model_prediction_FAA_parakeet_TL_500K",
    ),
]

# Every prediction column present in the master JSON (pretrained + all
# fine-tuning checkpoints + the standalone "FAA_Model" production model).
ALL_ASR_MODEL_KEYS: List[str] = [
    "FAA_Model_prediction_whisper-medium.en",
    "FAA_Model_prediction_FAA_Whisper_M_TL_100K",
    "FAA_Model_prediction_FAA_Whisper_M_TL_500K",
    "FAA_Model_prediction_whisper-large-v3",
    "FAA_Model_prediction_FAA_Whisper_L_TL_100K",
    "FAA_Model_prediction_FAA_Whisper_L_TL_500K",
    "FAA_Model",
    "FAA_Model_prediction_parakeet-tdt-0.6b-v2",
    "FAA_Model_prediction_FAA_parakeet_TL_100K",
    "FAA_Model_prediction_FAA_parakeet_TL_500K",
]

DEFAULT_JUDGE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"


@dataclass
class Config:
    """Filesystem layout for one pipeline run. All paths default under
    DATA_DIR / OUTPUTS_DIR but can be overridden per-field."""

    data_dir: Path = DATA_DIR
    outputs_dir: Path = OUTPUTS_DIR

    # Stage 1 input: raw normalized predictions, one row per audio sample,
    # one column per ASR model + "reference" + "audio_path".
    normalized_predictions: Path = field(default=None)  # type: ignore[assignment]

    # Stage 1 output (BenchmarkScorer): predictions + per-sample/per-model
    # WER, BERTDist, SemDist.
    predictions_with_metrics: Path = field(default=None)  # type: ignore[assignment]
    metrics_summary_csv: Path = field(default=None)  # type: ignore[assignment]

    # Stage 2 output (CxER judge pipeline): OUTPUT_ROOT/<judge>/<asr_model>.json
    cxer_labels_root: Path = field(default=None)  # type: ignore[assignment]

    # Stage 3 output (merge): predictions_with_metrics + cxer labels combined.
    predictions_with_metrics_and_cxer: Path = field(default=None)  # type: ignore[assignment]

    judge_model: str = DEFAULT_JUDGE_MODEL

    def __post_init__(self):
        if self.normalized_predictions is None:
            self.normalized_predictions = self.data_dir / "predictions" / "FAA_multi_model_predictions_normalized.json"
        if self.predictions_with_metrics is None:
            self.predictions_with_metrics = self.data_dir / "predictions" / "FAA_multi_model_predictions_with_metrics.json"
        if self.metrics_summary_csv is None:
            self.metrics_summary_csv = self.data_dir / "predictions" / "FAA_metrics_summary.csv"
        if self.cxer_labels_root is None:
            self.cxer_labels_root = self.data_dir / "cxer_labels"
        if self.predictions_with_metrics_and_cxer is None:
            self.predictions_with_metrics_and_cxer = self.data_dir / "predictions" / "FAA_multi_model_predictions_with_metrics_and_cxer.json"

    def judge_short_name(self) -> str:
        return self.judge_model.split("/")[-1]


def default_config() -> Config:
    return Config()
