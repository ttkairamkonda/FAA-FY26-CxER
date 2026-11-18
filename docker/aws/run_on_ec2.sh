#!/usr/bin/env bash
# Run the pipeline on an AWS GPU instance (or any Linux GPU box).
#
# Checks the host (driver, Docker, NVIDIA Container Toolkit), picks a big
# disk for model weights, pulls the image from ECR if CXER_IMAGE points
# there (otherwise builds it from this repo), then runs the command.
#
#   bash docker/aws/run_on_ec2.sh gpu-check
#   bash docker/aws/run_on_ec2.sh judge-smoke
#   bash docker/aws/run_on_ec2.sh all
#   bash docker/aws/run_on_ec2.sh stage2 --asr-models FAA_Model --max-samples 200
#
# Env: HF_TOKEN (or a .env file in the repo root), CXER_IMAGE, MODELS_DIR,
#      RESULTS_DIR_HOST, CUDA_VISIBLE_DEVICES, TENSOR_PARALLEL_SIZE.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
if [[ -f .env ]]; then set -a; . ./.env; set +a; fi

IMAGE="${CXER_IMAGE:-faa-cxer-pipeline:gpu}"

# 1. Host checks
command -v nvidia-smi >/dev/null || { echo "ERROR: no NVIDIA driver. Use an AWS Deep Learning AMI (GPU) or install the driver." >&2; exit 1; }
command -v docker >/dev/null     || { echo "ERROR: Docker is not installed." >&2; exit 1; }
if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "WARNING: Docker does not list the nvidia runtime. If --gpus fails, install the NVIDIA Container Toolkit:" >&2
  echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html" >&2
fi
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# 2. Disk for the judge weights (~140 GB for Llama-3.3-70B). Prefer the
#    instance-store NVMe that Deep Learning AMIs mount at /opt/dlami/nvme.
if [[ -z "${MODELS_DIR:-}" ]]; then
  if [[ -d /opt/dlami/nvme && -w /opt/dlami/nvme ]]; then MODELS_DIR=/opt/dlami/nvme/models; else MODELS_DIR="$ROOT/models"; fi
fi
RESULTS_DIR_HOST="${RESULTS_DIR_HOST:-$ROOT/results}"
mkdir -p "$MODELS_DIR" "$RESULTS_DIR_HOST"
echo "Models cache: $MODELS_DIR ($(df -h "$MODELS_DIR" | awk 'NR==2{print $4" free"}'))"
echo "Results:      $RESULTS_DIR_HOST"

# 3. Get the image
if [[ "$IMAGE" == *.dkr.ecr.*.amazonaws.com/* ]]; then
  REGISTRY="${IMAGE%%/*}"
  REGION="$(echo "$REGISTRY" | cut -d. -f4)"
  aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
  docker pull "$IMAGE"
elif ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Building $IMAGE from $ROOT (first time takes a while; the base image is ~14 GB)"
  docker build -f docker/pipeline.Dockerfile -t "$IMAGE" .
fi

# 4. Run
# Pass variables by name only, so the token never shows up in `ps`.
ENV_ARGS=()
for v in HF_TOKEN CUDA_VISIBLE_DEVICES TENSOR_PARALLEL_SIZE JUDGE_MODEL PROMPT_TYPE FRESH_JUDGE SMOKE_SAMPLES; do
  if [[ -n "${!v:-}" ]]; then export "$v"; ENV_ARGS+=(-e "$v"); fi
done

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then TTY_ARGS=(-it); fi

exec docker run --rm ${TTY_ARGS[@]+"${TTY_ARGS[@]}"} --gpus all --ipc=host --shm-size=16g \
  ${ENV_ARGS[@]+"${ENV_ARGS[@]}"} \
  -v "$RESULTS_DIR_HOST:/results" -v "$MODELS_DIR:/models" \
  "$IMAGE" "${@:-help}"
