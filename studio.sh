#!/usr/bin/env bash
# FAA CxER Studio launcher: one command, no configuration.
#
#   bash studio.sh start     check the machine, build, and start the app
#   bash studio.sh status    is it running? is the CxER judge ready?
#   bash studio.sh logs      follow the app and judge logs (Ctrl+C to leave)
#   bash studio.sh stop      stop everything
#
# It looks at the machine and picks the setup by itself:
#   * GPUs with >= 150 GB total memory -> app on GPU + the CxER judge (Llama-3.3-70B)
#   * a smaller GPU                    -> app on GPU, no CxER judge
#   * no GPU                           -> app on CPU, no CxER judge
# The app always does speech-to-text and WER; it shows CxER verdicts only
# when the judge is running.

set -euo pipefail

cd "$(dirname "$0")"
REPO="$(pwd)"

APP_IMAGE="faa-cxer-studio"
JUDGE_IMAGE="vllm/vllm-openai:v0.11.2"
JUDGE_MODEL="${JUDGE_MODEL:-meta-llama/Llama-3.3-70B-Instruct}"
NET="cxer-net"
APP_NAME="cxer-studio"
JUDGE_NAME="cxer-judge"
MIN_JUDGE_VRAM_GB=150

say()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '    \033[1;32mOK\033[0m %s\n' "$*"; }
warn() { printf '    \033[1;33m!!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

DOCKER="docker"
pick_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    say "Installing Docker (one time, ~1 minute)"
    sudo apt-get update -y >/dev/null
    sudo apt-get install -y docker.io >/dev/null
    sudo systemctl enable --now docker >/dev/null 2>&1 || true
    sudo usermod -aG docker "$USER" || true
  fi
  command -v docker >/dev/null 2>&1 || die "Docker install failed. Use the AWS Deep Learning AMI (START_HERE.md, step 3)."
  if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi
  $DOCKER info >/dev/null 2>&1 || die "Docker is installed but not running. Try: sudo systemctl start docker"
  # The app's image needs BuildKit (buildx) for its per-image .dockerignore.
  if ! $DOCKER buildx version >/dev/null 2>&1; then
    say "Installing Docker's build plugin (one time)"
    sudo apt-get update -y >/dev/null
    sudo apt-get install -y docker-buildx >/dev/null 2>&1 || sudo apt-get install -y docker-buildx-plugin >/dev/null 2>&1 \
      || die "Could not install docker buildx. Use the AWS Deep Learning AMI (START_HERE.md, step 3)."
  fi
  ok "Docker ready"
}

# ── machine checks ──────────────────────────────────────────────────────
GPU_COUNT=0; VRAM_GB=0; GPU_NAME=""
detect_gpu() {
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
    VRAM_GB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s+=$1} END {printf "%d", s/1024}')"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  fi
}

gpu_works_in_docker() {
  # Docker needs the NVIDIA Container Toolkit to hand GPUs to containers.
  $DOCKER info 2>/dev/null | grep -qi nvidia && return 0
  command -v nvidia-ctk >/dev/null 2>&1 && return 0
  return 1
}

free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }

ensure_memory() {
  local mem_gb swap_gb
  mem_gb="$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
  swap_gb="$(awk '/SwapTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)"
  if (( mem_gb + swap_gb >= 7 )); then ok "Memory: ${mem_gb} GB RAM + ${swap_gb} GB swap"; return; fi
  if (( $(free_gb /) < 14 )); then
    warn "Only ${mem_gb} GB RAM and too little free disk to add swap. Speech models may fail to load."
    warn "Use an instance with 8 GB+ RAM (e.g. m7i.large) for reliable results."
    return
  fi
  say "Only ${mem_gb} GB RAM: adding an 8 GB swap file so the speech models fit"
  sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  grep -q '^/swapfile ' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab >/dev/null
  ok "Swap added"
}

models_dir() {
  # Big, fast local disk on AWS GPU instances (Deep Learning AMIs mount it here).
  if [[ -d /opt/dlami/nvme && -w /opt/dlami/nvme ]] || sudo test -d /opt/dlami/nvme 2>/dev/null; then
    sudo mkdir -p /opt/dlami/nvme/models && sudo chmod 777 /opt/dlami/nvme/models
    echo /opt/dlami/nvme/models
  else
    mkdir -p "$REPO/models"; echo "$REPO/models"
  fi
}

get_hf_token() {
  if [[ -f "$REPO/.env" ]]; then set -a; . "$REPO/.env"; set +a; fi
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo
    echo "    The CxER judge uses Llama-3.3-70B, which needs a free Hugging Face token."
    echo "    (See START_HERE.md, step 2. Press Enter to skip the judge for now.)"
    read -r -s -p "    Paste your Hugging Face token (hf_...): " HF_TOKEN; echo
    if [[ -n "$HF_TOKEN" ]]; then
      umask 077; echo "HF_TOKEN=$HF_TOKEN" > "$REPO/.env"; ok "Token saved to .env (only you can read it)"
    fi
  fi
}

public_ip() {
  local t
  t="$(curl -s -m 2 -X PUT http://169.254.169.254/latest/api/token -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' 2>/dev/null || true)"
  curl -s -m 2 -H "X-aws-ec2-metadata-token: $t" http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true
}

# ── commands ────────────────────────────────────────────────────────────
cmd_start() {
  say "Checking this machine"
  pick_docker
  detect_gpu
  local mode="cpu"
  if (( GPU_COUNT > 0 )); then
    ok "GPU: ${GPU_COUNT} x ${GPU_NAME} (${VRAM_GB} GB total)"
    if gpu_works_in_docker; then
      if (( VRAM_GB >= MIN_JUDGE_VRAM_GB )); then mode="judge"; else mode="gpu"; fi
    else
      warn "GPUs found but Docker can't use them (NVIDIA Container Toolkit missing). Using CPU."
      warn "Use the AWS 'Deep Learning' AMI, which includes it (START_HERE.md, step 3)."
    fi
  else
    ok "No GPU: the app will run on the CPU"
  fi
  ensure_memory
  local free_root; free_root="$(free_gb /)"
  (( free_root >= 15 )) || die "Only ${free_root} GB free disk. The app needs ~15 GB. Increase the instance's disk size (START_HERE.md, troubleshooting)."
  ok "Disk: ${free_root} GB free"

  local models; models="$(models_dir)"
  local tp=1
  if [[ "$mode" == "judge" ]]; then
    get_hf_token
    if [[ -z "${HF_TOKEN:-}" ]]; then
      warn "No token: starting without the CxER judge."; mode="gpu"
    elif (( $(free_gb "$models") < 160 )); then
      warn "The judge needs ~160 GB free disk for its weights; ${models} has $(free_gb "$models") GB. Starting without the judge."
      mode="gpu"
    fi
    if   (( GPU_COUNT >= 8 )); then tp=8
    elif (( GPU_COUNT >= 4 )); then tp=4
    elif (( GPU_COUNT >= 2 )); then tp=2
    fi
  elif (( GPU_COUNT > 0 )) && gpu_works_in_docker; then
    warn "The CxER judge needs >= ${MIN_JUDGE_VRAM_GB} GB of GPU memory (e.g. AWS g6e.12xlarge). This machine has ${VRAM_GB} GB."
  fi

  case "$mode" in
    judge) say "Setup: speech-to-text on GPU + CxER judge on ${GPU_COUNT} GPUs" ;;
    gpu)   say "Setup: speech-to-text on GPU (no CxER judge on this machine)" ;;
    cpu)   say "Setup: speech-to-text on CPU (no CxER judge on this machine)" ;;
  esac

  say "Building the app (first time: 5-10 minutes)"
  local build_args=()
  [[ "$mode" == "cpu" ]] && build_args=(--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu)
  DOCKER_BUILDKIT=1 $DOCKER buildx build --load --progress=plain -f docker/studio.Dockerfile -t "$APP_IMAGE" \
    ${build_args[@]+"${build_args[@]}"} . 2>&1 | grep -E '^#[0-9]+ \[[0-9]+/[0-9]+\]|ERROR|error:' || true
  $DOCKER image inspect "$APP_IMAGE" >/dev/null 2>&1 || die "The app image failed to build (see the lines above)."
  ok "App image built"

  $DOCKER network inspect "$NET" >/dev/null 2>&1 || $DOCKER network create "$NET" >/dev/null
  $DOCKER rm -f "$APP_NAME" "$JUDGE_NAME" >/dev/null 2>&1 || true

  local gpu_flag=()
  [[ "$mode" != "cpu" ]] && gpu_flag=(--gpus all)

  if [[ "$mode" == "judge" ]]; then
    say "Starting the CxER judge (first time: downloads ~140 GB, 15-40 minutes)"
    local envf; envf="$(mktemp)"; chmod 600 "$envf"
    printf 'HF_TOKEN=%s\nHUGGING_FACE_HUB_TOKEN=%s\nHF_HOME=/models/hf\n' "$HF_TOKEN" "$HF_TOKEN" > "$envf"
    $DOCKER run -d --name "$JUDGE_NAME" --network "$NET" --gpus all \
      --ipc=host --shm-size=16g --restart unless-stopped --env-file "$envf" \
      -v "$models:/models" \
      "$JUDGE_IMAGE" \
      --model "$JUDGE_MODEL" --tensor-parallel-size "$tp" --dtype float16 \
      --gpu-memory-utilization 0.8 --max-model-len 8000 --host 0.0.0.0 --port 8000 >/dev/null
    rm -f "$envf"
    ok "Judge container started (it keeps loading in the background)"
  fi

  say "Starting the app"
  $DOCKER run -d --name "$APP_NAME" --network "$NET" ${gpu_flag[@]+"${gpu_flag[@]}"} \
    -p 127.0.0.1:8501:8501 --restart unless-stopped \
    -e JUDGE_URL="http://${JUDGE_NAME}:8000/v1" -e JUDGE_MODEL="$JUDGE_MODEL" \
    -v "$models:/models" "$APP_IMAGE" >/dev/null
  for _ in $(seq 1 30); do
    curl -fsS -m 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 && break
    sleep 2
  done
  curl -fsS -m 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1 || die "The app did not start. Run: bash studio.sh logs"
  ok "App is running"

  local ip; ip="$(public_ip)"; ip="${ip:-<server-ip>}"
  cat <<EOF

  ┌──────────────────────────────────────────────────────────────────────┐
    The app is running. To open it, on YOUR computer:

    1. Open a NEW PowerShell window and run (leave it open):
         ssh -i Downloads\\YOUR-KEY.pem -L 8501:localhost:8501 ubuntu@${ip}

    2. Open your web browser at:   http://localhost:8501
  └──────────────────────────────────────────────────────────────────────┘
EOF
  if [[ "$mode" == "judge" ]]; then
    echo "  The CxER judge is still loading. Check progress with:  bash studio.sh status"
    echo "  In the app, press 'Re-check judge' once it says READY."
  fi
}

cmd_status() {
  pick_docker >/dev/null
  echo
  if $DOCKER ps --format '{{.Names}}' | grep -qx "$APP_NAME"; then
    if curl -fsS -m 2 http://127.0.0.1:8501/_stcore/health >/dev/null 2>&1; then ok "App: running"; else warn "App: starting"; fi
  else
    warn "App: not running (start it with: bash studio.sh start)"
  fi
  if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$JUDGE_NAME"; then
    if $DOCKER exec "$JUDGE_NAME" curl -fsS -m 2 http://localhost:8000/health >/dev/null 2>&1; then
      ok "CxER judge: READY"
    elif $DOCKER ps --format '{{.Names}}' | grep -qx "$JUDGE_NAME"; then
      warn "CxER judge: still loading. Last log line:"
      $DOCKER logs --tail 1 "$JUDGE_NAME" 2>&1 | sed 's/^/      /'
    else
      warn "CxER judge: stopped. See: bash studio.sh logs"
    fi
  else
    echo "    CxER judge: not used on this machine (needs >= ${MIN_JUDGE_VRAM_GB} GB GPU memory)"
  fi
}

cmd_logs() {
  pick_docker >/dev/null
  if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$JUDGE_NAME"; then
    echo "--- judge (last 20 lines) ---"; $DOCKER logs --tail 20 "$JUDGE_NAME" 2>&1
  fi
  echo "--- app (following; Ctrl+C to leave) ---"
  $DOCKER logs -f --tail 50 "$APP_NAME"
}

cmd_stop() {
  pick_docker >/dev/null
  $DOCKER rm -f "$APP_NAME" "$JUDGE_NAME" >/dev/null 2>&1 || true
  ok "Stopped. Start again with: bash studio.sh start"
  echo "    Remember to also STOP the server in the AWS console when you're done (it is billed per hour)."
}

case "${1:-start}" in
  start)  cmd_start ;;
  status) cmd_status ;;
  logs)   cmd_logs ;;
  stop)   cmd_stop ;;
  *) echo "Usage: bash studio.sh [start|status|logs|stop]"; exit 1 ;;
esac
