"""CxER classification: given (reference, hypothesis) pairs, ask an LLM
judge whether the transcription contains an operationally critical error.

This is the core of stage 2 of the pipeline. `CxEREvaluator` wraps a vLLM
`LLM` instance and turns a batch of (reference, hypothesis) pairs into
classification results, handling GPU-OOM backoff and malformed judge
output along the way. Orchestration (which ASR models to run, resuming a
partial run, incremental saving) lives one layer up in `pipeline.py`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

from .json_extractor import JudgeResponseParser
from .prompts import DEFAULT_PROMPT_TYPE, PromptRegistry


class CxEREvaluator:
    """Chunk-based CxER classifier. Sends every prompt in a chunk to a
    single `llm.generate()` call so vLLM can parallelize across GPUs.

    Classification taxonomy:
        STATUS_EQUIVALENT : no critical error found
        STATUS_CRITICAL   : critical/contextual error found
        STATUS_PARSE_FAIL : empty input / unparseable output / unrecognized
                             transcription_quality value / generation failed
                             even at the minimum batch size — excluded from
                             the CxER numerator AND denominator

    `self.batch_size` is a hint the caller (`pipeline.CxERJudgePipeline`)
    re-reads before slicing each chunk. On a GPU OOM, `_generate_with_backoff`
    permanently lowers it so later chunks start smaller too, instead of
    re-discovering the same ceiling every time.
    """

    VALID_ENTITY_TYPES = {
        "callsign", "altitude", "heading", "runway",
        "frequency", "clearance", "navigation", "weather", "other",
    }

    STATUS_EQUIVALENT = "Equivalent"
    STATUS_CRITICAL = "Critical_Errors"
    STATUS_PARSE_FAIL = "Parse_Failure"

    def __init__(
        self,
        llm,
        model_name: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        batch_size: int = 200,
        min_batch_size: int = 25,
        prompt_type: str = DEFAULT_PROMPT_TYPE,
        prompt_registry: Optional[PromptRegistry] = None,
        parser: Optional[JudgeResponseParser] = None,
    ):
        self.model_name = model_name
        self.llm = llm
        self.batch_size = batch_size
        self.min_batch_size = min_batch_size
        self.prompt_type = prompt_type
        self.prompts = prompt_registry or PromptRegistry()
        self.parser = parser or JudgeResponseParser()

        from vllm import SamplingParams  # deferred: only needed when actually judging

        self.sampling_params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        print(f"CxEREvaluator ready (judge={model_name}, batch_size={batch_size}, min_batch_size={min_batch_size})")

    # ── OOM-aware generation ────────────────────────────────────────────

    def _generate_with_backoff(self, prompts: List[str]) -> List[str]:
        """Generate completions for a list of prompts. On GPU OOM, halve the
        list and retry each half recursively until it succeeds or hits
        `min_batch_size`. The shrink is written back to `self.batch_size` so
        the caller's NEXT chunk starts at the smaller size."""
        if not prompts:
            return []
        try:
            outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
            return [o.outputs[0].text if o.outputs else "" for o in outputs]
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            import torch

            torch.cuda.empty_cache()

            if len(prompts) <= self.min_batch_size:
                print(f"  ⚠ OOM at floor batch size ({len(prompts)}) — marking as Parse_Failure, not retrying further")
                return ["" for _ in prompts]

            mid = max(len(prompts) // 2, self.min_batch_size)
            if mid < self.batch_size:
                print(f"  ⚠ OOM — shrinking batch_size {self.batch_size} → {mid} for the rest of this run")
                self.batch_size = mid
            print(f"  ⚠ Retrying {len(prompts)} prompts as {mid} + {len(prompts) - mid}")
            return self._generate_with_backoff(prompts[:mid]) + self._generate_with_backoff(prompts[mid:])

    # ── evaluate one chunk of (sample_index, reference, hypothesis) ─────

    def evaluate_chunk(self, chunk: List[Tuple[int, str, str]]) -> List[Dict[str, Any]]:
        """Returns one result dict per item, in the same order as `chunk`."""
        prompts: List[Optional[str]] = []
        valid_mask: List[bool] = []
        for _, ref, hyp in chunk:
            if not ref.strip() or not hyp.strip():
                prompts.append(None)
                valid_mask.append(False)
            else:
                prompts.append(self.prompts.build(ref.strip(), hyp.strip(), self.model_name, prompt_type=self.prompt_type))
                valid_mask.append(True)

        valid_prompts = [p for p in prompts if p is not None]
        t0 = time.time()
        raw_outputs = self._generate_with_backoff(valid_prompts) if valid_prompts else []
        batch_time = time.time() - t0
        per_sample_time = batch_time / max(len(valid_prompts), 1)

        results = []
        output_iter = iter(raw_outputs)
        for is_valid, (sample_index, ref, hyp) in zip(valid_mask, chunk):
            if not is_valid:
                results.append(self._parse_failure_result("Empty input (missing reference or hypothesis)", 0.0, sample_index))
                continue

            raw = next(output_iter)
            if not raw:
                results.append(self._parse_failure_result("Generation failed (GPU OOM at floor batch size)", per_sample_time, sample_index))
                continue

            parsed = self.parser.extract(raw)
            if parsed is None:
                results.append(self._parse_failure_result(f"JSON parse failed. Raw:\n{raw[:300]}", per_sample_time, sample_index))
            else:
                results.append(self._build_result(parsed, per_sample_time, sample_index))

        return results

    def evaluate_single(self, reference: str, transcription: str, sample_index: int = 0) -> Dict[str, Any]:
        """Convenience shim for smoke-testing a judge on one example."""
        return self.evaluate_chunk([(sample_index, reference, transcription)])[0]

    # ── result builders ─────────────────────────────────────────────────

    def _build_result(self, parsed: dict, inference_time: float, sample_index) -> Dict[str, Any]:
        errors = self.parser.safe_get(parsed, "transcription_errors", []) or []
        for err in errors:
            if isinstance(err, dict) and err.get("entity_type") not in self.VALID_ENTITY_TYPES:
                err["entity_type"] = "other"

        quality_raw = self.parser.safe_get(parsed, "transcription_quality", None)
        if quality_raw == "Equivalent":
            classification = self.STATUS_EQUIVALENT
        elif quality_raw == "Critical_Errors":
            classification = self.STATUS_CRITICAL
        else:
            # Parseable JSON, but not in the expected vocabulary — treat as
            # rubbish output, not a trustworthy classification.
            return self._parse_failure_result(f"Unrecognized transcription_quality: {quality_raw!r}", inference_time, sample_index)

        return {
            "classification": classification,
            "meaning_preserved": self.parser.safe_get(parsed, "meaning_preserved"),
            "operational_safety": self.parser.safe_get(parsed, "operational_safety"),
            "explanation": self.parser.safe_get(parsed, "explanation", ""),
            "transcription_errors": errors,
            "inference_time": round(inference_time, 4),
            "sample_index": sample_index,
        }

    @classmethod
    def _parse_failure_result(cls, msg: str, inference_time: float, sample_index) -> Dict[str, Any]:
        return {
            "classification": cls.STATUS_PARSE_FAIL,
            "meaning_preserved": None,
            "operational_safety": None,
            "explanation": msg,
            "transcription_errors": [],
            "inference_time": round(inference_time, 4),
            "sample_index": sample_index,
        }
