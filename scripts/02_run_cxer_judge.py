#!/usr/bin/env python3
"""Stage 2: run an LLM judge over one or more ASR models' predictions and
classify each sample as Equivalent / Critical_Errors / Parse_Failure.

Requires a GPU host with vLLM installed and `CUDA_VISIBLE_DEVICES` set
BEFORE this script imports torch/vllm (set it in your shell, not via
--tensor-parallel-size — vLLM reads the env var at import time).

Reliability: this is safe to Ctrl-C and re-run. Progress is saved after
every chunk to output_root/<judge_short_name>/<asr_model>.json; a re-run
resumes from whatever's already there instead of starting over.

Example:
    export CUDA_VISIBLE_DEVICES=4,5,6,7
    export HF_TOKEN=hf_...            # or put it in a .env file
    python scripts/02_run_cxer_judge.py \\
        --asr-models FAA_Model_prediction_whisper-medium.en FAA_Model_prediction_FAA_Whisper_M_TL_500K \\
        --max-samples 10   # sanity check before a full run
"""

import argparse
import os

import _bootstrap  # noqa: F401
from faa_cxer.config import ALL_ASR_MODEL_KEYS, DEFAULT_JUDGE_MODEL, default_config
from faa_cxer.io_utils import load_records


def main():
    cfg = default_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(cfg.predictions_with_metrics), help="Metrics JSON from stage 1 (predictions + WER/BERTDist/SemDist)")
    parser.add_argument("--output-root", default=str(cfg.cxer_labels_root), help="Root dir for CxER label output (OUTPUT_ROOT/<judge>/<asr_model>.json)")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="HF model id of the judge LLM")
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--prompt-type", default="cot_v3", help="See faa_cxer.judge.prompts.PromptRegistry.available()")
    parser.add_argument("--asr-models", nargs="*", default=ALL_ASR_MODEL_KEYS, help="Which prediction columns to judge")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap samples per model (sanity-check runs); omit for the full set")
    parser.add_argument("--batch-size", type=int, default=200, help="Starting vLLM generate() batch size; shrinks automatically on GPU OOM")
    parser.add_argument("--min-batch-size", type=int, default=25)
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-judge samples even if a completed output already covers them")
    args = parser.parse_args()

    from dotenv import load_dotenv
    from huggingface_hub import login

    load_dotenv()
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))

    from faa_cxer.judge import CxEREvaluator, CxERJudgePipeline

    samples = load_records(args.input)
    print(f"Loaded {len(samples)} samples from {args.input}")

    llm = CxERJudgePipeline.load_model(args.judge_model, tensor_parallel_size=args.tensor_parallel_size)
    evaluator = CxEREvaluator(
        llm, model_name=args.judge_model, batch_size=args.batch_size,
        min_batch_size=args.min_batch_size, prompt_type=args.prompt_type,
    )
    pipeline = CxERJudgePipeline(evaluator, output_root=args.output_root)
    pipeline.run(samples, args.asr_models, skip_existing=not args.no_skip_existing, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
