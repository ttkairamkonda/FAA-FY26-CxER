# FAA CxER Studio

Studio compares speech recognition models on air traffic control recordings. Upload audio to see what each model heard, enter a reference to measure word error rate (WER), or connect a judge to flag operationally critical errors. The FAA and ATC benchmark tabs also work with the results bundled in the repository.

| Where you want to run it | Start here |
|---|---|
| Lambda GPU server, shared Linux machine, or Linux workstation without Docker | [Linux and Lambda](#linux-and-lambda) |
| Windows or macOS laptop with Docker Desktop, or Linux with Docker Engine | [Docker Studio](#docker-studio) |
| AWS EC2 | [AWS and remote browser access](#aws-and-remote-browser-access) |
| Browser-only walkthrough or portable offline demo | [Demos](#demos) |
| Batch scoring, analysis, or figures | [Batch pipeline](#batch-pipeline) |

For a local or server installation, clone the repository **on the machine that will run the app**. Run subsequent commands from its root.

```sh
git clone https://github.com/ttkairamkonda/FAA-FY26-CxER.git
cd FAA-FY26-CxER
```

## Linux and Lambda

Use Linux x86-64 with Python 3.10–3.12 and the Python `venv` module. Python 3.10 uses the repository's tested dependency constraints. The NVIDIA profile uses a CUDA 12.4 PyTorch build. Initial installation and first-time model downloads need internet access.

**Install once.** Pick one command for your machine. The installer creates environments inside this checkout and requires no sudo.

| What you need | Command |
|---|---|
| Studio with CPU transcription | `bash scripts/setup_local.sh cpu` |
| Studio with GPU transcription, optionally using a remote judge | `bash scripts/setup_local.sh studio` |
| Studio with GPU transcription and a judge on the same host | `bash scripts/setup_local.sh all` |
| Add a local judge to an existing Studio installation | `bash scripts/setup_local.sh judge` |

The [installer](scripts/setup_local.sh) uses pip to install Studio into `.venv-studio`. Only the native local judge needs `.venv-judge`. Its vLLM version requires a different Transformers version from Studio. The app launches that environment for you. Set `PYTHON_BIN=python3.10` before the setup command to select that interpreter.

**Start Studio** from the repository root. On shared GPU hosts, set `CUDA_VISIBLE_DEVICES` to the devices allocated to you before launch. For gated model access, export `HF_TOKEN` in the same shell.

```sh
.venv-studio/bin/python -I -m streamlit run app/cxer_studio.py --server.address 127.0.0.1 --server.port 8501
```

On the same machine, open [localhost:8501](http://localhost:8501). For Lambda or EC2, use the [SSH tunnel](#aws-and-remote-browser-access). The browser runs on your laptop and inference runs on the machine hosting Studio. Stop a local judge with **Stop local judge**, then press Ctrl+C in the Studio terminal.

## Docker Studio

Start Docker Desktop (Windows/macOS, Linux containers) or Docker Engine (Linux). Run `docker info` and confirm it shows a **Server** section. If a shared server denies access to the Docker socket, use the Linux instructions above.

**CPU Studio** transcribes audio and displays saved benchmarks. The [Dockerfile](docker/studio.Dockerfile) specifies what gets installed in the image. The [Compose file](docker-compose.studio.yml) tells Docker which services to start, which GPUs they can use, and which port opens Studio.

```sh
docker compose -f docker-compose.studio.yml up --build -d studio
```

**GPU Studio** needs NVIDIA GPU access inside Docker and drivers compatible with the selected images. Choose one of these commands in place of the CPU command.

```sh
# Transcription on GPU
docker compose -f docker-compose.studio.yml --profile gpu up --build -d studio-gpu

# Transcription and a separate GPU judge service
docker compose -f docker-compose.studio.yml --profile gpu up --build -d studio-gpu judge
```

For the judge service, set the variables in [Judge settings](#judge-settings) before starting Compose. In Studio select **Connect to a server** and use `http://judge:8000/v1`. Open [localhost:8501](http://localhost:8501) on the host or use the SSH tunnel. Use the first command below to inspect logs and the second to stop the stack.

```sh
docker compose -f docker-compose.studio.yml --profile gpu logs --tail 50
docker compose -f docker-compose.studio.yml --profile gpu down
```

Download results you need before removing the Studio container. The supplied Studio Compose file mounts the judge model cache at `/models`. Studio's upload directory is inside the app container.

## AWS and remote browser access

For EC2, launch an Ubuntu instance with an SSH key, allow inbound SSH from your IP, and attach enough disk for the selected models. CPU Studio can use a general-purpose instance. For a local 70B judge, select a multi-GPU instance, check your regional GPU quota, and use a GPU image with NVIDIA drivers. Allow room for the judge weights, the cache and container images. A 300 GiB disk is a starting point for one judge.

Connect from your laptop. The example opens a shell on EC2 and forwards the browser port in the same connection. Substitute your key and login address. On Lambda, use your usual SSH host and omit `-i` if your SSH configuration supplies the key.

```sh
ssh -i /path/to/key.pem -L 8501:127.0.0.1:8501 USER@HOST
```

At the remote shell, follow the clone instructions at the top of this guide, then follow **Linux and Lambda** or **Docker Studio** above. On a dedicated Docker-based Ubuntu EC2 host, `bash studio.sh start` is an alternative that detects hardware and builds the appropriate services. It may use sudo to install system components. The same launcher accepts `status`, `logs`, and `stop`.

Leave SSH connected and open [localhost:8501](http://localhost:8501) on your laptop. If local port 8501 is occupied, use `-L 8502:127.0.0.1:8501` and open local port 8502. When finished, stop Studio, save any results you need, and stop the EC2 instance to end compute charges. Use persistent storage for caches you want to keep across stops.

## Judge settings

Studio supports Llama 3.3 70B, DeepSeek R1 Distill Llama 70B, and Qwen 2.5 72B. FP16 weights alone need roughly 140–150 GB, plus GPU memory for inference. Allow at least 160 GB of available model storage for one judge. Usable GPU memory, supported architecture, and the number of compatible GPUs all matter.

| Judge location | How to configure it |
|---|---|
| Native Linux installation | Choose **Run on this machine**, the judge model, and allocated GPUs. Choose a separate **ASR device** when possible. Click **Load judge** and check the startup logs. Stop the judge before switching its model or devices. Use **Release cached ASR models** to free memory before reassigning a GPU. |
| Docker Compose | Create a local `.env` using [the example](docker/.env.example). Set `JUDGE_MODEL`, `TENSOR_PARALLEL_SIZE` (default 4), and `HF_TOKEN` for gated models. `MODELS_DIR` changes the host directory mounted for the judge cache. |
| Existing vLLM server | Choose **Connect to a server**, enter the reachable API address ending in `/v1`, and pick a model the server actually serves. The server operator assigns its GPUs. `JUDGE_URL` sets the initial endpoint. |

For gated weights, accept the model license with the Hugging Face account that owns your read token. Native Studio reads exported environment variables. **Re-check judge** refreshes readiness after startup. GPU memory use alone does not tell you whether the HTTP server is ready. For a native judge, inspect `results/studio_judge.log` and run `curl -i http://127.0.0.1:8000/health` on the server, substituting the port selected in Studio. HTTP 200 indicates readiness.

## Analyze audio

1. In **Analyze audio**, upload WAV, MP3, FLAC, M4A or OGG files, or select a bundled clip. A batch can contain up to 500 files and 1 GiB total, subject to Streamlit's per-file limit.
2. Select the ASR models in the sidebar. Studio offers pretrained and FAA fine-tuned Whisper checkpoints. To supply your own choices, set `ASR_MODELS` to comma-separated Hugging Face model IDs before launch. Add a reference transcript for each clip you want scored.
3. Click **Run analysis**. Predictions and references use Whisper English number normalization, so “four zero one” and “401” compare consistently. **Skip long quiet gaps** reduces repetition on radio recordings. Turn it off if a quieter speaker is being missed.
4. Download CSV or Excel for one row per audio file and model, or JSON for detailed verdicts. Exports include raw and normalized predictions, runtime, audio duration, callsigns and estimated speaker role. The audio path points to the saved upload on the Studio host under `results/uploads/`.

Callsigns use the bundled [FAA telephony mapping](data/icao_telephony.json), for example `EXECJET 401 → EJA401`. Studio labels the speaker as Pilot, Air Traffic Controller, or Unknown from the transcript. Judge assistance can refine these estimates. Unrecognized callsigns and ambiguous speaker roles appear as Unknown.

**WER** counts word edits relative to the reference. **CxER** measures how often the judge finds an operationally critical error, such as changing “runway left” to “runway right.” For a given model and judge, `CxER = Critical_Errors / (Critical_Errors + Equivalent)`. Parse failures are excluded. These metrics can move differently after fine-tuning.

## Saved results and data

| Studio tab | Data and interpretation |
|---|---|
| **FAA results** | [Predictions and metrics](data/predictions/) and [per-model judge labels](data/cxer_labels/) for 6,364 FAA utterances and ten ASR outputs. The comparison includes pretrained and FAA fine-tuned Whisper and Parakeet models. Select models and metrics to update the chart and download the selection. |
| **ATC benchmark results** | [Bundled benchmark snapshot](data/atc_benchmarks/) for ATCOSIM test, ATCO2 OOD, and UWB ATCC test. Select the dataset, ASR variants, judge and metric. The snapshot came from the `cxer-release` exports `data/raw` (reference/prediction pairs) and `eval_cache/all_results.csv` (aggregate scores for Llama, DeepSeek, and Qwen). WER and WeightedWER are percentages. Cached CCER is a fraction displayed as CxER %. |

The ATC snapshot contains reference and hypothesis text pairs and aggregate scores. Audio uploads are handled separately in **Analyze audio**. The tabs read saved evaluations. Switching the live judge affects new analyses. Compare live WER with historical scores using the same normalization.

## Batch pipeline

The batch pipeline evaluates saved predictions and regenerates figures. For example, to print the table of low-WER transcripts with critical errors on Linux without Docker, use the shipped FAA results:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/06_analyze_paradox.py --no-plots
```

To run all the analysis stages on the shipped labels with Docker Compose, use the CPU service.

```sh
docker compose -f docker-compose.pipeline.yml --profile cpu run --build --rm pipeline-cpu analysis
```

For a new GPU evaluation, configure the judge credentials described above and run the GPU service.

```sh
docker compose -f docker-compose.pipeline.yml build pipeline
docker compose -f docker-compose.pipeline.yml run --rm pipeline gpu-check
docker compose -f docker-compose.pipeline.yml run --rm pipeline judge-smoke
docker compose -f docker-compose.pipeline.yml run --rm pipeline all
```

The smoke run uses ten samples per model by default. The full run seeds from shipped labels and resumes completed work. Set `FRESH_JUDGE=1` to skip seeding a fresh results directory. Container outputs appear under `results/{predictions,cxer_labels,outputs,logs}` on the host.

Individual stages accept extra arguments. For example, `stage2 --asr-models FAA_Model --max-samples 10` limits evaluation. The corresponding [Python scripts](scripts/) also accept `--help`. Stages 1, 2 and 3 compute WER and semantic metrics, judge the predictions, and merge the labels. Stages 5, 6 and 7 analyze classification transitions, compare WER with critical errors, and plot model comparisons. Batch defaults live in [config.py](faa_cxer/config.py), and judge templates live in [prompts](faa_cxer/judge/prompts/). The default prompt is `cot_v3`.

On EC2, [run_on_ec2.sh](docker/aws/run_on_ec2.sh) builds or pulls the pipeline image and accepts the same commands. To distribute a prebuilt pipeline image through Amazon ECR, run `bash docker/aws/push_to_ecr.sh` with AWS credentials and ECR push permissions. Set `CXER_IMAGE` to the returned URI on the GPU host and run `bash docker/aws/run_on_ec2.sh judge-smoke` with ECR pull access.

## Demos

The [hosted demo](https://ttkairamkonda.github.io/FAA-Project/) lets visitors listen to samples and compare saved results. For offline use, build one of these images while connected, then run it on a machine with a compatible Docker platform.

| Demo | Build | Run |
|---|---|---|
| Eight bundled clips with saved predictions and verdicts | `docker build -t faa-cxer-demo .` | `docker run --rm -p 127.0.0.1:8501:8501 faa-cxer-demo` |
| Live CPU transcription with two bundled Whisper Medium checkpoints | `docker build -f Dockerfile.live_asr -t faa-live-asr-demo .` | `docker run --rm -p 127.0.0.1:8501:8501 faa-live-asr-demo` |

Open the browser on port 8501. The live demo shows stored verdicts for matching reference clips. To move either image to another machine, use `docker save -o demo.tar IMAGE` on the build machine, transfer the file, and run `docker load -i demo.tar` on the destination.

## Code and configuration

Live model choices, Whisper decoding and the Streamlit interface are in [cxer_studio.py](app/cxer_studio.py). Transcript normalization, callsigns and speaker roles are in [transcript_processing.py](app/transcript_processing.py). For judge startup or connection issues, see the [local process launcher](app/local_judge.py) and the [HTTP client](app/judge_client.py).

## Development checks

Run the tests from the native Studio installation.

```sh
.venv-studio/bin/python -I -m pip install -r requirements-dev.txt
.venv-studio/bin/python -I -m pip check
.venv-studio/bin/python -m pytest tests -q
```

The CPU suite exercises the UI, exports, audio handling, and judge integration with mocked inference. Use a bundled clip to check actual ASR on the target device and `judge-smoke` for a batch GPU judge check.
