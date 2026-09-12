"""Shared I/O and text-normalization helpers used across every pipeline stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import jiwer

PathLike = Union[str, os.PathLike]


class TextNormalizer:
    """Wraps a `jiwer.Compose` transform so every pipeline stage normalizes
    reference/hypothesis text the same way before computing metrics.

    A single normalization bug (e.g. forgetting to strip punctuation before
    WER) silently invalidates every downstream number, so this is
    intentionally the *only* place normalization rules are defined.
    """

    def __init__(self, expand_contractions: bool = False):
        steps = [
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ]
        if expand_contractions:
            steps += [jiwer.ExpandCommonEnglishContractions(), jiwer.RemoveEmptyStrings()]
        self._transform = jiwer.Compose(steps)

    def __call__(self, text: str) -> str:
        result = self._transform([str(text)])
        return result[0].strip() if result else ""


def load_json(path: PathLike) -> Any:
    with open(path) as f:
        return json.load(f)


def save_json_atomic(path: PathLike, obj: Any, indent: int = 2) -> None:
    """Write JSON via temp-file + os.replace so a crash mid-write never
    leaves a truncated/corrupt file behind."""
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, indent=indent)
    os.replace(tmp_path, path)


def load_records(path: PathLike) -> List[Dict[str, Any]]:
    """Load a predictions/metrics JSON that is either a bare list of sample
    dicts, or a wrapper dict with a "samples" or "records" key."""
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("samples", "records"):
            if key in data:
                return data[key]
    raise ValueError(f"{path}: expected a list, or a dict with a 'samples'/'records' key")
