# Running the FAA CxER pipeline in Docker (local GPUs and AWS)

One image runs every stage of the pipeline, the test
suite and the listen-and-compare demo. It runs the same way on a local
NVIDIA workstation and on an AWS GPU instance. You don't need Python,
CUDA or vLLM on the host. You only need an NVIDIA driver, Docker and the
NVIDIA Container Toolkit.

The image adds new files and calls the existing scripts unchanged:

| File | Purpose |
|---|---|
| `docker/pipeline.Dockerfile` | The image (GPU by default, CPU-only with a build arg) |
| `docker/pipeline.Dockerfile.dockerignore` | Build-context filter for this image only (the top-level `.dockerignore` still belongs to the demo image) |
| `docker/entrypoint.sh` | The commands (`all`, `analysis`, individual stages, `test`, ...) |
| `docker-compose.pipeline.yml` | Compose file for local machines and AWS |
| `docker/.env.example` | Template for your `.env` (Hugging Face token, GPU selection) |
| `docker/aws/push_to_ecr.sh` | Build the image once and publish it to Amazon ECR for the team |
| `docker/aws/run_on_ec2.sh` | Check the host, pull or build the image, and run a command on an EC2 GPU box |
| `app/cxer_studio.py`, `app/judge_client.py` | **CxER Studio**, the interactive app (section 0) |
| `docker/studio.Dockerfile`, `docker-compose.studio.yml` | Image and compose file for the Studio app + GPU judge |

**Your data stays untouched.** The shipped `data/` folder is only read.
Every run writes to `./results/` (predictions, judge labels, figures,
and one log per stage). Downloaded model weights go to `./models/`, so a
second run doesn't download them again.

---

## 0. CxER Studio (the interactive app)

A Streamlit app where you upload ATC audio (wav, mp3, flac, m4a, ogg), or pick a bundled
clip, and get:

- **Live ASR** from the models you pick (pretrained vs FAA fine-tuned Whisper), on the GPU if
  there is one, otherwise on the CPU.
- **WER** for each transcription, when you type the reference transcript.
- **A CxER verdict** from the LLM judge for each transcription (Equivalent or Critical error, with
  each error's entity, expected vs heard value and risk level). This needs the GPU judge server.
  Without it, the app says the judge needs a GPU and still does ASR and WER.
- A **Benchmark results** tab with the corpus-level WER vs CxER chart and table from the
  offline run, plus the pipeline's figures.
- CSV and JSON export of everything.

The judge runs as its own service (the official vLLM image serving Llama-3.3-70B-Instruct).
The app sends it the same prompt and uses the same parser as the batch pipeline
(`faa_cxer.judge`), so its verdicts mean the same thing as the ones in `data/cxer_labels/`.

**Without a GPU** (laptop, small AWS instance): ASR and WER only
```bash
docker compose -f docker-compose.studio.yml up --build studio
```
On a machine with little RAM or disk, build the smaller CPU-only image first:
`docker build -f docker/studio.Dockerfile -t faa-cxer-studio --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .`
Whisper Medium needs about 4 GB of RAM per model on CPU, so use an instance with 8 GB+.

**With a GPU host** (~150 GB total VRAM, e.g. AWS g6e.12xlarge): ASR on GPU + the CxER judge
```bash
cp docker/.env.example .env      # set HF_TOKEN (Llama-3.3 license accepted)
docker compose -f docker-compose.studio.yml --profile gpu up --build -d studio-gpu judge
docker compose -f docker-compose.studio.yml logs -f judge     # wait for "Application startup complete"
```
The first judge start downloads ~140 GB and takes 10 to 30 minutes. Until it's done, the
sidebar shows "CxER judge unavailable". Press **Re-check judge** once the log says it's ready.

**Opening the app** at `http://localhost:8501`. On AWS, the safest way is an SSH tunnel from
your PC, so nothing is exposed to the internet:
```
ssh -i Downloads/FAA_Demo.pem -L 8501:localhost:8501 ubuntu@<server-ip>
```
Leave that window open and browse to http://localhost:8501 on your PC. To share the app with
others, add an inbound rule for TCP 8501 in the instance's security group, limited to their IPs,
and use `http://<server-ip>:8501`.

The judge can also live on a different machine: set `JUDGE_URL=http://<judge-host>:8000/v1` in
`.env` (or type it in the sidebar), and the app on a CPU box will use it.

## 1. What you need

| | Analysis only (stages 1, 3 and 5–7, tests, demo) | Full pipeline incl. stage 2 (LLM judge) |
|---|---|---|
| GPU | none (CPU image works) | NVIDIA GPUs with **~150 GB+ total VRAM** for Llama-3.3-70B in fp16 |
| Host | Docker | Linux (or Windows with Docker Desktop + WSL2), NVIDIA driver **≥ 575** (the vLLM base image is built on CUDA 12.9), Docker, NVIDIA Container Toolkit |
| Disk | ~5 GB | ~20 GB for the image + **~150 GB** for the judge weights |
| Hugging Face | not needed (stage 1 downloads `roberta-large`, which is public) | `HF_TOKEN` with access to `meta-llama/Llama-3.3-70B-Instruct` (accept the license on its model page first) |

The pipeline code loads the judge with `dtype=float16` and
`gpu_memory_utilization=0.8`. So the usable VRAM is 80% of the total, and
it has to hold the ~141 GB of weights plus the KV cache.
`TENSOR_PARALLEL_SIZE` defaults to the number of visible GPUs (use 4 or 8).

## 2. Quick start (any machine with Docker)

From the repo root:

```bash
cp docker/.env.example .env              # then put your HF_TOKEN in .env

# GPU image (first build pulls the ~14 GB vLLM base image)
docker compose -f docker-compose.pipeline.yml build pipeline

docker compose -f docker-compose.pipeline.yml run --rm pipeline gpu-check     # GPUs, torch, vLLM visible?
docker compose -f docker-compose.pipeline.yml run --rm pipeline test          # 24 unit tests
docker compose -f docker-compose.pipeline.yml run --rm pipeline analysis      # stages 3 and 5–7 from the shipped judge labels (~1 min)
docker compose -f docker-compose.pipeline.yml run --rm pipeline judge-smoke   # stage 2 on 10 samples/model: checks token, GPUs, vLLM
docker compose -f docker-compose.pipeline.yml run --rm pipeline all           # stages 1–3 and 5–7 end to end
```

No NVIDIA GPU (a laptop, for example)? Use the CPU image. It runs
everything except stage 2:

```bash
docker compose -f docker-compose.pipeline.yml --profile cpu build pipeline-cpu
docker compose -f docker-compose.pipeline.yml --profile cpu run --rm pipeline-cpu analysis
```

The same thing without compose:

```bash
docker build -f docker/pipeline.Dockerfile -t faa-cxer-pipeline:gpu .
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e HF_TOKEN -v "$PWD/results:/results" -v "$PWD/models:/models" \
  faa-cxer-pipeline:gpu all
```

## 3. Commands

Run `docker compose -f docker-compose.pipeline.yml run --rm pipeline help` to see these:

| Command | What it does |
|---|---|
| `all` | Stages 1–3 and 5–7. Before the judge runs, the shipped labels are copied into `results/cxer_labels`, so stage 2 only judges what's missing. Set `FRESH_JUDGE=1` to re-judge all 10 models (63,640 judge calls). |
| `analysis` | Stages 3 and 5–7 from the shipped labels. Needs no GPU and no token. |
| `judge-smoke` | Stage 2 on `SMOKE_SAMPLES` (default 10) samples per model, written to `results/smoke/`. Run this first on any new machine. |
| `stage1/2/3/5/6/7` | One stage. Extra arguments go straight to that script, for example `stage2 --asr-models FAA_Model --max-samples 200` or `stage6 --threshold-mode fixed --threshold-fixed 0.1`. |
| `test` | The pytest suite |
| `gpu-check` | `nvidia-smi`, torch CUDA status and vLLM version |
| `demo` | The listen-and-compare Streamlit app on port 8501 (`--profile demo` service, or `-p 8501:8501`) |
| `bash` | A shell inside the container |

Each stage reads the previous stage's output from `results/` when it
exists, and falls back to the shipped file in `data/` otherwise. So
`stage6` works on its own right after a fresh clone.

Settings (in `.env` or with `-e`): `HF_TOKEN`, `JUDGE_MODEL`
(default `meta-llama/Llama-3.3-70B-Instruct`), `PROMPT_TYPE` (default `cot_v3`),
`TENSOR_PARALLEL_SIZE`, `CUDA_VISIBLE_DEVICES=0,1,2,3` to use only some GPUs,
`MODELS_DIR` to keep weights on a different disk.

Output layout:

```
results/
├── predictions/   FAA_multi_model_predictions_with_metrics.json, FAA_metrics_summary.csv,
│                  FAA_multi_model_predictions_with_metrics_and_cxer.json
├── cxer_labels/   Llama-3.3-70B-Instruct/<asr_model>.json   (stage 2)
├── outputs/       all figures + contextual_safety_drift_sankey.html
└── logs/          one log per stage (stage 5 and 6 tables are printed there)
```

Stage 2 saves progress after every chunk and resumes where it stopped.
If an AWS run is interrupted (spot reclaim, SSH drop), run the same command again.

## 4. Running on AWS

### Pick an instance

| Instance | GPUs | Total VRAM | Notes |
|---|---|---|---|
| `g6e.12xlarge` | 4 × L40S | 192 GB | Smallest that fits the 70B judge at the code's 0.8 memory setting. Tight KV cache. |
| `g6e.48xlarge` | 8 × L40S | 384 GB | Comfortable. `TENSOR_PARALLEL_SIZE=8` |
| `p4d.24xlarge` | 8 × A100 40 GB | 320 GB | Comfortable. `TENSOR_PARALLEL_SIZE=8` |
| `p5.48xlarge` | 8 × H100 80 GB | 640 GB | Fastest, most expensive |
| `g5.12xlarge` / `g6.12xlarge` | 4 × 24 GB | 96 GB | **Too small** for the 70B judge. Fine for stages 1 and 3 and 5–7. |

Use the **AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)**.
It comes with the driver, Docker and the NVIDIA Container Toolkit, and it
mounts the instance's local NVMe at `/opt/dlami/nvme`, which
`run_on_ec2.sh` uses for the model weights automatically. Give the root
EBS volume at least 100 GB. If the instance has no local NVMe, use 300 GB.
You'll need GPU instance quota in the region (Service Quotas, "Running On-Demand G and VT instances" or "P instances").

### Option A: one shared GPU server for the team

1. Launch the instance and SSH in.
2. `git clone https://github.com/ttkairamkonda/FAA-FY26-CxER.git && cd FAA-FY26-CxER`
3. `cp docker/.env.example .env` and set `HF_TOKEN`
4. `bash docker/aws/run_on_ec2.sh gpu-check`, then `judge-smoke`, then `all`

The first run builds the image on the instance. Teammates SSH into the
same box and run the same script. To keep a long run alive after you
disconnect, wrap it: `nohup bash docker/aws/run_on_ec2.sh all > run.log 2>&1 &`
(or use `tmux`).

### Option B: publish the image to ECR so anyone can launch their own box

Build and push once, from any machine with Docker and the AWS CLI:

```bash
AWS_REGION=us-east-1 bash docker/aws/push_to_ecr.sh
# -> Pushed: <account>.dkr.ecr.us-east-1.amazonaws.com/faa-cxer-pipeline:gpu
```

Then anyone with ECR pull rights (`AmazonEC2ContainerRegistryReadOnly`, or
an instance role that has it) can run it on any GPU instance:

```bash
git clone https://github.com/ttkairamkonda/FAA-FY26-CxER.git && cd FAA-FY26-CxER
cp docker/.env.example .env      # set HF_TOKEN and CXER_IMAGE=<the URI above>
bash docker/aws/run_on_ec2.sh all
```

With an ECR `CXER_IMAGE`, the script logs in and pulls the image instead
of building it. Results land in `./results` on that instance. Copy them
off with `aws s3 cp --recursive results/ s3://<bucket>/cxer-runs/$(date +%F)/`
or `scp`.

### Cost tips

- The full judge run (10 models × 6,364 samples) is the only long step.
  Run `judge-smoke` first and check its timing before you commit to a big instance.
- Stop the instance when you're done. The NVMe cache is wiped on stop, so
  weights download again next time. To keep them, set `MODELS_DIR` to a
  path on EBS.
- Stages 1 and 3 and 5–7 don't need the big instance. Run `analysis` on a
  laptop with the CPU image, or on any small GPU box.

## 5. Local GPUs

- **Linux workstation:** install the NVIDIA driver and the
  [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
  then follow section 2. `bash docker/aws/run_on_ec2.sh` works here too; nothing in it is AWS-specific
  unless `CXER_IMAGE` points at ECR.
- **Windows:** Docker Desktop with the WSL2 backend passes NVIDIA GPUs
  through (current NVIDIA Windows driver needed). Run the compose
  commands from PowerShell or a WSL shell in the repo folder.
- **Not enough VRAM for the 70B judge:** everything except stage 2 still
  works. To try stage 2 end to end on a smaller GPU, set a smaller judge
  such as `JUDGE_MODEL=meta-llama/Llama-3.1–3 and 5–7B-Instruct`. Those labels are
  **not comparable** to the published Llama-3.3-70B results, and they go to
  their own `results/cxer_labels/<judge>/` folder.

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `could not select device driver "nvidia"` / `--gpus` error | The NVIDIA Container Toolkit isn't installed or Docker wasn't restarted after installing it |
| `gpu-check` shows `CUDA available: False` | Driver too old for CUDA 12.9 (needs ≥ 575), or `CUDA_VISIBLE_DEVICES` set to something invalid |
| `HF_TOKEN is not set` / 401 / 403 from Hugging Face | Set the token, and accept the Llama-3.3 license on huggingface.co with the same account |
| CUDA out of memory while loading the judge | Not enough total VRAM. Use a bigger instance (section 4) |
| `No space left on device` during download | Point `MODELS_DIR` at a disk with 150 GB+ free |
| vLLM hangs or crashes with NCCL / shared-memory errors | Keep `--ipc=host` and `--shm-size=16g` (the compose file and `run_on_ec2.sh` set both) |
| `entrypoint.sh: not found` or `bad interpreter` after a Windows checkout | Line endings. The Dockerfile already strips CRLF; rebuild the image |

## 7. What was tested

- CxER Studio: the app runs headless under Streamlit's test harness and in a browser. Checked: the
  bundled-clip flow, WER, the "judge needs GPU" path, and the judge path against a mock vLLM server
  (prompt format, temperature 0, out-of-order answers, verdict parsing). Real Whisper downloads and
  the real 70B judge could not be run from the test machine.

- In a Linux container (CPU torch 2.9.0, transformers 4.57): the image
  builds, all 24 tests pass, `analysis` runs stages 3 and 5–7 without errors and
  writes every figure, and the merged stage-3 file matches the shipped one
  exactly.
- `stage2` stops with the token message when `HF_TOKEN` is missing.
- **Not run:** stage 1's model download, the real stage 2 on GPUs, and a
  build on the 14 GB vLLM base image. The test machine had no GPU, couldn't
  reach Hugging Face, and didn't have the disk for that image. Do the
  first full build and `judge-smoke` on the target GPU box.
