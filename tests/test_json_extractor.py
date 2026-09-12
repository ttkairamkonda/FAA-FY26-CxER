"""Unit tests for the trickiest piece of logic in the pipeline: recovering
a JSON verdict from messy LLM output. No GPU/model required."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faa_cxer.judge.json_extractor import JudgeResponseParser

parser = JudgeResponseParser()


def test_clean_json():
    raw = '{"transcription_quality": "Equivalent", "meaning_preserved": true}'
    assert parser.extract(raw) == {"transcription_quality": "Equivalent", "meaning_preserved": True}


def test_reasoning_block_is_stripped():
    raw = '<reasoning>thinking about it...</reasoning>{"transcription_quality": "Critical_Errors"}'
    assert parser.extract(raw) == {"transcription_quality": "Critical_Errors"}


def test_markdown_fence():
    raw = 'Here is my answer:\n```json\n{"transcription_quality": "Equivalent"}\n```'
    assert parser.extract(raw) == {"transcription_quality": "Equivalent"}


def test_duplicated_json_prefers_last():
    raw = '{"transcription_quality": "Critical_Errors"} then I reconsidered {"transcription_quality": "Equivalent"}'
    assert parser.extract(raw)["transcription_quality"] == "Equivalent"


def test_trailing_comma_tolerated():
    raw = '{"transcription_quality": "Equivalent", "transcription_errors": [],}'
    assert parser.extract(raw) == {"transcription_quality": "Equivalent", "transcription_errors": []}


def test_truncated_json_is_repaired():
    raw = '{"transcription_quality": "Equivalent", "explanation": "looks fine'
    parsed = parser.extract(raw)
    assert parsed is not None
    assert parsed["transcription_quality"] == "Equivalent"


def test_unparseable_returns_none():
    assert parser.extract("I refuse to output JSON today.") is None


def test_safe_get():
    assert parser.safe_get({"a": 1}, "a") == 1
    assert parser.safe_get({"a": 1}, "b", "default") == "default"
    assert parser.safe_get("not a dict", "a", "default") == "default"
