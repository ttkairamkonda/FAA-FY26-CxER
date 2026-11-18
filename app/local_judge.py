"""Manage only the judge subprocess started by this Studio instance."""
import os
from pathlib import Path
import signal
import subprocess

JUDGE_MODELS = {
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct",
    "DeepSeek R1 Distill Llama 70B": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "Qwen 2.5 72B": "Qwen/Qwen2.5-72B-Instruct",
}


def gpu_choices():
    import torch
    if not torch.cuda.is_available():
        return {}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    choices = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        physical = visible[i].strip() if visible != [""] and i < len(visible) else str(i)
        choices[i] = (physical, f"GPU {physical}: {props.name} ({props.total_memory / 1024**3:.0f} GiB)")
    return choices


class LocalJudge:
    def __init__(self, log_path):
        self.process = None
        self.log_path = Path(log_path)
        self.description = ""

    def running(self):
        return self.process is not None and self.process.poll() is None

    def start(self, model, devices, port, memory):
        if self.running():
            raise RuntimeError("Stop the current local judge before loading another model.")
        if not devices:
            raise ValueError("Select at least one GPU for the judge.")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
        project = Path(__file__).resolve().parents[1]
        executable = str(project / ".venv-judge" / "bin" / "python")
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        env["HF_HOME"] = str(project / "models" / "hf")
        env["VLLM_USE_V1"] = "0"
        try:
            check = subprocess.run(
                [executable, "-I", "-c", "import vllm"], env=env,
                capture_output=True, text=True, timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                "Project judge environment is unavailable. Run bash scripts/setup_local.sh judge "
                "from this repository. " + str(exc)
            ) from exc
        if check.returncode:
            raise RuntimeError(
                "Project judge environment cannot import vLLM. Run bash scripts/setup_local.sh judge. "
                + check.stderr[-1000:]
            )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [executable, "-I", "-m", "vllm.entrypoints.openai.api_server",
                   "--model", model, "--tensor-parallel-size", str(len(devices)),
                   "--dtype", "float16", "--gpu-memory-utilization", str(memory),
                   "--max-model-len", "8000", "--enforce-eager",
                   "--host", "127.0.0.1", "--port", str(port)]
        with self.log_path.open("w") as log:
            self.process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        self.description = f"{model} · GPUs {', '.join(devices)} · port {port}"

    def stop(self):
        if self.running():
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=5)

    def logs(self):
        if not self.log_path.exists():
            return "No local judge has been started."
        with self.log_path.open("rb") as log:
            log.seek(0, 2)
            log.seek(max(0, log.tell() - 12000))
            return log.read().decode(errors="replace")
