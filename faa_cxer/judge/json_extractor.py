"""Robust extraction of the judge's JSON verdict from raw LLM text output.

LLM judges reliably follow "output only this JSON" instructions maybe 95%
of the time. The other 5% include: reasoning text before/after the JSON,
markdown code fences, duplicated JSON objects (the model re-emits its
answer), and truncated output that got cut off mid-object because it hit
max_tokens. `JudgeResponseParser` exists to recover a valid classification
from as many of those cases as possible instead of throwing them all away
as parse failures.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional


class JudgeResponseParser:
    """Extracts a JSON object from a raw LLM completion, in order of
    preference:

      1. Strip <reasoning>...</reasoning> blocks, try a clean parse.
      2. Try a markdown ```json fence.
      3. Scan for every balanced top-level {...} span (string-literal
         aware, so braces inside quoted values don't confuse the depth
         count) and try them LAST-to-FIRST — the model's final answer
         typically comes after any intermediate/duplicated JSON emitted
         during chain-of-thought.
      4. Every parse attempt also tries a trailing-comma-stripped variant.
      5. Truncation repair: append closing-brace suffixes to every
         candidate (and a final greedy regex match) before giving up.
    """

    _REASONING_RE = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL)
    _MARKDOWN_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
    _GREEDY_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)
    _TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
    _TRUNCATION_REPAIR_SUFFIXES = ["}", "\"}", "\"]}", "}]}", "}]}\n}"]

    @classmethod
    def _strip_trailing_commas(cls, s: str) -> str:
        return cls._TRAILING_COMMA_RE.sub(r"\1", s)

    @classmethod
    def _find_balanced_json_candidates(cls, text: str) -> List[str]:
        candidates = []
        depth = 0
        start = None
        in_string = False
        escape = False

        for i, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        candidates.append(text[start : i + 1])
                        start = None

        return candidates

    def _try_parse(self, s: str) -> Optional[dict]:
        for candidate in (s, self._strip_trailing_commas(s)):
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None

    def extract(self, raw_text: str) -> Optional[dict]:
        raw_text = self._REASONING_RE.sub("", raw_text).strip()

        parsed = self._try_parse(raw_text.strip())
        if parsed is not None:
            return parsed

        md_match = self._MARKDOWN_FENCE_RE.search(raw_text)
        if md_match:
            parsed = self._try_parse(md_match.group(1))
            if parsed is not None:
                return parsed

        candidates = self._find_balanced_json_candidates(raw_text)
        for candidate in reversed(candidates):
            parsed = self._try_parse(candidate)
            if parsed is not None:
                return parsed

        greedy_match = self._GREEDY_OBJECT_RE.search(raw_text)
        repair_targets = list(candidates)
        if greedy_match:
            repair_targets.append(greedy_match.group(0))
        repair_targets.append(raw_text)

        for target in repair_targets:
            if not target:
                continue
            for suffix in self._TRUNCATION_REPAIR_SUFFIXES:
                parsed = self._try_parse(target + suffix)
                if parsed is not None:
                    return parsed

        return None

    @staticmethod
    def safe_get(d: dict, key: str, default=None):
        return d.get(key, default) if isinstance(d, dict) else default
