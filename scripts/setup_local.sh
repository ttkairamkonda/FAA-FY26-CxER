#!/usr/bin/env bash
# No sudo, Docker or other project's environments. Run from any working directory.
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODE="${1:-studio}"
STUDIO_CONSTRAINTS=()
JUDGE_CONSTRAINTS=()
case "$MODE" in studio|cpu|judge|all|dev) ;; *) echo "Usage: bash scripts/setup_local.sh [studio|cpu|judge|all|dev]" >&2; exit 2 ;; esac
"$PYTHON_BIN" -I -c 'import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12), "Use Python 3.10, 3.11 or 3.12 (set PYTHON_BIN)."'
if [[ "$("$PYTHON_BIN" -I -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" == 3.10 ]]; then
    STUDIO_CONSTRAINTS=(-c constraints/studio-linux-py310.txt)
    JUDGE_CONSTRAINTS=(-c constraints/judge-linux-py310.txt)
fi
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$PROJECT_DIR/.cache/pip"
export HF_HOME="$PROJECT_DIR/models/hf"
prepare() {
    local target="$1"
    "$PYTHON_BIN" -I -m venv "$target"
    "$target/bin/python" -I -m pip install --upgrade pip
}
if [[ "$MODE" == studio || "$MODE" == cpu || "$MODE" == all || "$MODE" == dev ]]; then
    prepare .venv-studio
    torch_index="https://download.pytorch.org/whl/cu124"
    torch_version="2.6.0+cu124"
    if [[ "$MODE" == cpu ]]; then
        torch_index="https://download.pytorch.org/whl/cpu"
        torch_version="2.6.0+cpu"
    fi
    .venv-studio/bin/python -I -m pip install "torch==$torch_version" --index-url "$torch_index"
    .venv-studio/bin/python -I -m pip install "${STUDIO_CONSTRAINTS[@]}" -r requirements-studio.txt
    .venv-studio/bin/python -I -m pip check
fi
if [[ "$MODE" == judge || "$MODE" == all ]]; then
    prepare .venv-judge
    .venv-judge/bin/python -I -m pip install "${JUDGE_CONSTRAINTS[@]}" -r requirements-local-judge.txt
    .venv-judge/bin/python -I -m pip check
    .venv-judge/bin/python -I -c 'import vllm; print("Project vLLM:", vllm.__version__)'
fi
if [[ "$MODE" == dev ]]; then
    .venv-studio/bin/python -I -m pip install -r requirements-dev.txt
fi
printf '\nSetup complete. Start Studio:\n  .venv-studio/bin/python -I -m streamlit run app/cxer_studio.py --server.address 127.0.0.1 --server.port 8501\n'
