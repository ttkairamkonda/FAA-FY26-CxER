"""Orchestrates running a `CxEREvaluator` over one or more ASR models, with
sample-level resume and incremental, crash-safe saving.

This is the class-based equivalent of the original `llm_judge_new.py`
script's `evaluate_asr_model` / `_save_progress` / `_load_existing_output`
functions plus its `load_model` helper.
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

from ..io_utils import PathLike, load_json, save_json_atomic
from .evaluator import CxEREvaluator


class CxERJudgePipeline:
    """Runs one judge model over a list of ASR-model prediction columns,
    writing one output file per (judge, asr_model) pair to
    `output_root/<judge_short_name>/<asr_model_key>.json`.

    Reliability features (carried over from the original script):
      - Sample-level resume: on restart, only samples missing from the
        existing output file are re-evaluated.
      - Incremental, atomic saving after every chunk — a crash loses at
        most one chunk's worth of work.
      - Raising `MAX_SAMPLES` between runs is safe: the skip-existing check
        compares against the CURRENT target sample count, not just whether
        a prior (possibly smaller) run finished.
    """

    def __init__(self, evaluator: CxEREvaluator, output_root: PathLike):
        self.evaluator = evaluator
        self.output_root = str(output_root)

    # ── static-ish helper: build the vLLM judge model ───────────────────

    @staticmethod
    def _no_compilation_config() -> dict:
        """`{"level": 0}` disabled torch.compile under vLLM < 0.9; newer
        vLLM renamed that field to `mode` (an enum where 0/NONE means the
        same "no compilation"). Inspect whichever `CompilationConfig` is
        actually installed instead of hardcoding one API generation, so
        this keeps working across vLLM upgrades."""
        import dataclasses

        from vllm.config import CompilationConfig

        field_names = {f.name for f in dataclasses.fields(CompilationConfig)}
        key = "mode" if "mode" in field_names else "level"
        return {key: 0}

    @staticmethod
    def load_model(model_name: str, tensor_parallel_size: int = 4, gpu_memory_utilization: float = 0.8, max_model_len: int = 8000):
        """Load a judge LLM under vLLM. Requires `CUDA_VISIBLE_DEVICES` to
        already be set in the environment before this call (vLLM/torch read
        it at import time)."""
        import torch
        from vllm import LLM

        torch.cuda.empty_cache()
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="float16",
            disable_log_stats=True,
            compilation_config=CxERJudgePipeline._no_compilation_config(),
            max_model_len=max_model_len,
        )
        print("Model loaded successfully!")
        return llm

    def output_path_for(self, asr_model_key: str) -> str:
        judge_dir = os.path.join(self.output_root, self.evaluator.model_name.split("/")[-1])
        return os.path.join(judge_dir, f"{asr_model_key}.json")

    def _load_existing_output(self, output_path: str) -> Optional[dict]:
        if not os.path.exists(output_path):
            return None
        return load_json(output_path)

    def _save_progress(
        self,
        output_path: str,
        asr_model_key: str,
        records_by_index: Dict[str, dict],
        num_samples_total: int,
        status: str,
        elapsed_time: float,
    ) -> None:
        n_equiv = sum(1 for r in records_by_index.values() if r["classification"] == CxEREvaluator.STATUS_EQUIVALENT)
        n_crit = sum(1 for r in records_by_index.values() if r["classification"] == CxEREvaluator.STATUS_CRITICAL)
        n_fail = sum(1 for r in records_by_index.values() if r["classification"] == CxEREvaluator.STATUS_PARSE_FAIL)
        n_evaluated = n_equiv + n_crit
        cxer = round(n_crit / n_evaluated, 4) if n_evaluated > 0 else None

        records_sorted = sorted(records_by_index.values(), key=lambda r: r["sample_index"])

        output = {
            "asr_model": asr_model_key,
            "judge_model": self.evaluator.model_name,
            "prompt_type": self.evaluator.prompt_type,
            "status": status,
            "num_samples_total": num_samples_total,
            "num_samples_done": len(records_by_index),
            "num_equivalent": n_equiv,
            "num_critical_errors": n_crit,
            "num_parse_failures": n_fail,
            "num_evaluated_successfully": n_evaluated,
            "contextual_error_rate": cxer,
            "total_inference_time": round(elapsed_time, 2),
            "records": records_sorted,
        }
        save_json_atomic(output_path, output)

    def evaluate_asr_model(
        self,
        samples: List[dict],
        asr_model_key: str,
        skip_existing: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        if max_samples is not None:
            samples = samples[:max_samples]
        num_samples_total = len(samples)

        output_path = self.output_path_for(asr_model_key)
        existing = self._load_existing_output(output_path)

        # Only skip outright if a prior run already completed AT LEAST this
        # many samples — a smaller prior run (e.g. a sanity check) won't
        # block a bigger one; it continues from where it left off instead.
        if existing is not None and skip_existing:
            if existing.get("status") == "complete" and existing.get("num_samples_total", 0) >= num_samples_total:
                print(f"  ⏭  Skipping (already complete for {num_samples_total} samples): {output_path}")
                return

        records_by_index: Dict[str, dict] = {}
        if existing is not None:
            for rec in existing.get("records", []):
                records_by_index[str(rec["sample_index"])] = rec
            print(f"  ↻ Resuming {asr_model_key}: {len(records_by_index)} samples already done")

        pending_items = []
        for idx, s in enumerate(samples):
            if str(idx) in records_by_index:
                continue
            ref = s.get("reference", "") or ""
            hyp = s.get(asr_model_key, "") or ""
            pending_items.append((idx, ref, hyp, s.get("audio_path")))

        if not pending_items:
            print(f"  ✓ Nothing pending for {asr_model_key} — already fully covered.")
            self._save_progress(
                output_path, asr_model_key, records_by_index, num_samples_total,
                status="complete", elapsed_time=existing.get("total_inference_time", 0.0) if existing else 0.0,
            )
            return

        label = f"{self.evaluator.model_name.split('/')[-1]} | {asr_model_key}"
        t0_total = time.time()
        prior_elapsed = existing.get("total_inference_time", 0.0) if existing else 0.0

        pbar = tqdm(total=len(pending_items), desc=label, unit="sample") if HAS_TQDM else None

        pos = 0
        while pos < len(pending_items):
            cur_bs = self.evaluator.batch_size  # re-read every iteration — may have shrunk from an OOM
            chunk_items = pending_items[pos : pos + cur_bs]
            chunk_for_eval = [(idx, ref, hyp) for idx, ref, hyp, _ in chunk_items]

            chunk_results = self.evaluator.evaluate_chunk(chunk_for_eval)

            for (idx, ref, hyp, audio_path), result in zip(chunk_items, chunk_results):
                records_by_index[str(idx)] = {
                    "sample_index": idx,
                    "audio_path": audio_path,
                    "reference": ref,
                    "hypothesis": hyp,
                    "classification": result["classification"],
                    "meaning_preserved": result["meaning_preserved"],
                    "operational_safety": result["operational_safety"],
                    "explanation": result["explanation"],
                    "transcription_errors": result["transcription_errors"],
                    "inference_time": result["inference_time"],
                }

            elapsed_so_far = prior_elapsed + (time.time() - t0_total)
            self._save_progress(output_path, asr_model_key, records_by_index, num_samples_total, status="in_progress", elapsed_time=elapsed_so_far)

            if pbar:
                pbar.update(len(chunk_items))
            pos += len(chunk_items)

        if pbar:
            pbar.close()

        total_elapsed = prior_elapsed + (time.time() - t0_total)
        self._save_progress(output_path, asr_model_key, records_by_index, num_samples_total, status="complete", elapsed_time=total_elapsed)

        n_equiv = sum(1 for r in records_by_index.values() if r["classification"] == CxEREvaluator.STATUS_EQUIVALENT)
        n_crit = sum(1 for r in records_by_index.values() if r["classification"] == CxEREvaluator.STATUS_CRITICAL)
        n_fail = sum(1 for r in records_by_index.values() if r["classification"] == CxEREvaluator.STATUS_PARSE_FAIL)
        n_evaluated = n_equiv + n_crit
        cxer = round(n_crit / n_evaluated, 4) if n_evaluated > 0 else None

        print(f"  ⏱  {asr_model_key}: {num_samples_total} samples, total elapsed {total_elapsed:.1f}s")
        print(f"  CxER = {n_crit}/{n_evaluated} = {cxer}  (equivalent={n_equiv}, critical={n_crit}, parse_failures={n_fail})")
        print(f"  Saved → {output_path}")

    def run(
        self,
        samples: List[dict],
        asr_model_keys: List[str],
        skip_existing: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        for asr_model_key in asr_model_keys:
            print(f"\n▶ {self.evaluator.model_name.split('/')[-1]} → {asr_model_key}")
            self.evaluate_asr_model(samples, asr_model_key, skip_existing=skip_existing, max_samples=max_samples)
