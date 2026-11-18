#!/usr/bin/env bash
# Entrypoint for the FAA CxER pipeline image (docker/pipeline.Dockerfile).
#
# It only calls the existing scripts/0*.py with explicit paths. It never
# changes the code, and it never writes into the shipped data/ folder:
# everything a run produces goes under $RESULTS_DIR (default /results,
# meant to be a mounted volume).
#
# Usage:  docker run ... <image> <command> [extra args for that stage]
# Run with "help" to see the command list.

set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
DATA_DIR="${DATA_DIR:-$APP_DIR/data}"          # shipped inputs (read-only is fine)
RESULTS_DIR="${RESULTS_DIR:-/results}"         # everything this run writes
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
PROMPT_TYPE="${PROMPT_TYPE:-cot_v3}"

# Shipped inputs
SHIPPED_NORMALIZED="$DATA_DIR/predictions/FAA_multi_model_predictions_normalized.json"
SHIPPED_METRICS="$DATA_DIR/predictions/FAA_multi_model_predictions_with_metrics.json"
SHIPPED_LABELS="$DATA_DIR/cxer_labels"
SHIPPED_MERGED="$DATA_DIR/predictions/FAA_multi_model_predictions_with_metrics_and_cxer.json"

# Run outputs
R_PRED="$RESULTS_DIR/predictions"
R_METRICS="$R_PRED/FAA_multi_model_predictions_with_metrics.json"
R_SUMMARY="$R_PRED/FAA_metrics_summary.csv"
R_MERGED="$R_PRED/FAA_multi_model_predictions_with_metrics_and_cxer.json"
R_LABELS="$RESULTS_DIR/cxer_labels"
R_OUT="$RESULTS_DIR/outputs"
R_LOGS="$RESULTS_DIR/logs"

cd "$APP_DIR"
mkdir -p "$R_PRED" "$R_LABELS" "$R_OUT" "$R_LOGS"

log() { printf '\n\033[1;34m[cxer]\033[0m %s\n' "$*"; }

# Use this run's file if it exists, otherwise fall back to the shipped one.
pick() { if [[ -f "$1" ]]; then echo "$1"; else echo "$2"; fi; }

run_logged() {  # run_logged <name> <cmd...>
  local name="$1"; shift
  log "$name"
  "$@" 2>&1 | tee "$R_LOGS/$name.log"
}

gpu_count() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l
  else
    echo 0
  fi
}

# The judge needs the Llama weights; it is gated on Hugging Face.
require_hf_token() {
  if [[ -z "${HF_TOKEN:-}" && ! -f "$APP_DIR/.env" ]]; then
    echo "ERROR: HF_TOKEN is not set. ${JUDGE_MODEL} is a gated model:" >&2
    echo "  accept its license on huggingface.co, then pass -e HF_TOKEN=hf_... (or set it in .env)." >&2
    exit 2
  fi
}

# Copy the shipped judge labels into the results folder so the judge
# resumes from them instead of re-judging 63,640 samples (the pipeline
# skips anything already complete). Never overwrites existing results.
seed_labels() {
  if [[ -d "$SHIPPED_LABELS" ]]; then
    log "Seeding $R_LABELS from shipped labels (existing result files are kept)"
    cp -rn "$SHIPPED_LABELS/." "$R_LABELS/"
  fi
}

stage1() {
  run_logged stage1_benchmark_scores python scripts/01_compute_benchmark_scores.py \
    --input "$SHIPPED_NORMALIZED" --output-json "$R_METRICS" --output-csv "$R_SUMMARY" "$@"
}

stage2() {
  require_hf_token
  local n tp
  n="$(gpu_count)"
  tp="${TENSOR_PARALLEL_SIZE:-$n}"
  if [[ "$n" -eq 0 ]]; then
    echo "ERROR: no GPU visible in the container. Run with --gpus all (docker) or the compose file." >&2
    exit 3
  fi
  log "Judge: $JUDGE_MODEL on $n GPU(s), tensor-parallel-size=$tp, prompt=$PROMPT_TYPE"
  run_logged stage2_cxer_judge python scripts/02_run_cxer_judge.py \
    --input "$(pick "$R_METRICS" "$SHIPPED_METRICS")" \
    --output-root "$R_LABELS" \
    --judge-model "$JUDGE_MODEL" \
    --prompt-type "$PROMPT_TYPE" \
    --tensor-parallel-size "$tp" "$@"
}

stage3() {
  # Use this run's labels if any exist, otherwise the shipped ones.
  local labels="$SHIPPED_LABELS"
  if compgen -G "$R_LABELS/*/*.json" >/dev/null; then labels="$R_LABELS"; fi
  run_logged stage3_merge python scripts/03_merge_cxer_labels.py \
    --metrics-json "$(pick "$R_METRICS" "$SHIPPED_METRICS")" \
    --cxer-labels-root "$labels" --output-json "$R_MERGED" "$@"
}

stage5() { run_logged stage5_transitions python scripts/05_analyze_transitions.py --input "$(pick "$R_MERGED" "$SHIPPED_MERGED")" --output-dir "$R_OUT" "$@"; }
stage6() { run_logged stage6_paradox python scripts/06_analyze_paradox.py --input "$(pick "$R_MERGED" "$SHIPPED_MERGED")" --output-dir "$R_OUT" "$@"; }
stage7() { run_logged stage7_comparison_plots python scripts/07_make_comparison_plots.py --input "$(pick "$R_MERGED" "$SHIPPED_MERGED")" --output "$R_OUT/faa_pretrained_vs_finetuned.png" "$@"; }

analysis() {  # stages 3 and 5–7 from the shipped labels (or this run's); no GPU needed
  stage3; stage5; stage6; stage7
}

usage() {
  cat <<EOF
FAA CxER pipeline container. Results go to $RESULTS_DIR (mount a volume there).

Commands:
  all              Stages 1–3 and 5–7 end to end. Judge resumes from shipped labels
                   unless FRESH_JUDGE=1 (then all 10 models are re-judged).
  analysis         Stages 3 and 5–7 from the shipped judge labels. No GPU needed.
  judge-smoke      Stage 2 on 10 samples per model into a scratch folder, to
                   check GPUs, HF_TOKEN and vLLM before a long run.
  stage{1,2,3,5,6,7} One stage. Extra args go straight to that script,
                   e.g.  stage2 --asr-models FAA_Model --max-samples 100
  test             Run the pytest suite.
  gpu-check        Show visible GPUs, torch CUDA and vLLM versions.
  demo             Start the listen-and-compare Streamlit demo on port 8501.
  bash | <cmd>     Shell, or any other command.

Environment:
  HF_TOKEN               Hugging Face token (needed for the judge; Llama is gated)
  JUDGE_MODEL            default meta-llama/Llama-3.3-70B-Instruct
  PROMPT_TYPE            default cot_v3
  TENSOR_PARALLEL_SIZE   default = number of visible GPUs
  CUDA_VISIBLE_DEVICES   which GPUs to use (default all)
  FRESH_JUDGE=1          with "all": don't seed from shipped labels
EOF
}

cmd="${1:-help}"
[[ $# -gt 0 ]] && shift

case "$cmd" in
  help|-h|--help) usage ;;
  all)
    stage1
    if [[ "${FRESH_JUDGE:-0}" != "1" ]]; then seed_labels; fi
    stage2
    analysis
    log "Done. Results in $RESULTS_DIR"
    ;;
  analysis) analysis; log "Done. Results in $RESULTS_DIR" ;;
  judge-smoke)
    R_LABELS="$RESULTS_DIR/smoke/cxer_labels"; mkdir -p "$R_LABELS"
    stage2 --max-samples "${SMOKE_SAMPLES:-10}" --no-skip-existing "$@"
    ;;
  stage1) stage1 "$@" ;;
  stage2) stage2 "$@" ;;
  stage3) stage3 "$@" ;;
  stage5) stage5 "$@" ;;
  stage6) stage6 "$@" ;;
  stage7) stage7 "$@" ;;
  test) python -m pytest -q -p no:cacheprovider tests "$@" ;;
  gpu-check)
    nvidia-smi || echo "nvidia-smi not available (no GPU passed to the container?)"
    python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.0f} GB")
try:
    import vllm; print("vllm", vllm.__version__)
except Exception as e:
    print("vllm not importable:", e)
PY
    ;;
  demo) exec streamlit run demo_app.py "$@" ;;
  *) exec "$cmd" "$@" ;;
esac
