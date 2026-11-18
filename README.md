# ICT Pipeline

The ICT Pipeline CLI is a Python entry point for the AI Hub SMT manufacturing defect detection project.

## Install

Use Python 3.11 or newer. From the repository root, create a virtual environment and activate it in your shell.

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux, WSL2, or macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install the CLI in editable mode and check the entry point.

```text
python -m pip install -e .
ict --help
ict --version
```

The CLI runs without PyTorch, GPU libraries, or dataset files.

## Available commands

`ict --help` lists the registered commands, and `ict --version` prints the installed package version. Both exit with status `0`.

`train`, `evaluate`, `compare`, `report`, and `clean` are placeholders and exit with status `1`. Existing audit and conversion tools can be invoked directly from `scripts/` with their own dependencies. The rollout is described in [PLANS.md](PLANS.md).

## Acquire and verify dataset 71900

```text
ict data status
ict data verify
ict data download
ict data download --approve-download
ict data extract
```

The packaged [SMT dataset selection](src/ict_pipeline/data/aihub_smt.yaml) records the 60 file keys from [the original download script](scripts/download_data.sh). It covers Training and Validation for normal and defect cold solder, misalignment, short, missing component, and tombstone groups, with JPG images, JSON labels, and XLSX sensors. Use `ict data --raw-dir <archive-root> --data-path <dataset-root> <command>` to override the default archive and dataset directories. The dataset root can also come from setup.

`ict data download` checks local data and previews any missing split transfers. The approval flag authorizes the listed transfers. The provider checks free space, write permissions, API reachability, and the local `AIHUB_API_KEY` supplied through `.env` or the environment. AI Hub checks the key and dataset access on the authorized request. Linux, WSL, and macOS use curl. On Windows the provider discovers an installed WSL distribution with curl and streams files into the Windows archive root. Set `ICT_WSL_DISTRIBUTION` to select a distribution explicitly. Existing English-layout data and local ZIPs work independently of WSL, curl, and AI Hub credentials. The transport uses the v0.6 request format verified in [AI Hub's aihubshell workflow](https://www.aihub.or.kr/devsport/apishell/list.do).

Validated bundles are kept under `data/raw/.ict-aihub/` by selection and split. Interrupted transfers remain as `.partial` files. A retry starts a fresh transfer for the affected split and reuses any verified completed split. The downloader checks bundle integrity and ZIP CRCs before publishing a checkpoint. The file keys and size caps are based on the repository's historical XLSX selection. If AI Hub changes the selection or format, ingestion reports a failed validation.

`ict data extract` publishes a new English-layout dataset directory with `.ict-ingestion.json`, which records the selection fingerprint, selected file-tree hash, and counts per split. Source archives are retained. `ict data verify` checks selected folder coverage, matched sample stems, basic file signatures, and the completion hash when present. Run the existing audit scripts for annotation and sensor-content validation.

## Plan, inspect, and resume a pipeline run

```text
ict run --dry-run
ict run --dry-run --until validate_detection
ict status
ict resume --run-id <id>
```

The production DAG declares ten stages from download through report. With `dataset_source: aihub` or `dataset_source: existing` configured, `ict run --until extract` executes the two ingestion stages. Their artifacts are `archives/` and `dataset/` under the corresponding published attempt directories. When an existing dataset is selected, the download stage records its reuse and the extract stage copies the selected files into the run artifact. For missing transfers, `ict run --until extract --approve-download` authorizes acquisition. Pass the same approval flag to `ict resume` if a resumed run still needs transfers. Adapters from audit through report are planned.

With executable stages registered, `ict run` selects the entire DAG and creates a run ID. Pass `--run-id <id>` to choose a name or reuse that run's cache. `--from` and `--until` select an inclusive range in dependency order. Repeat `--only` to select individual stages. Any omitted prerequisite must have a valid result in the same run. `--force` recomputes the selection in new artifact directories.

`ict status` shows the latest recorded attempt per stage. `ict resume` replays the saved selection and rechecks settings, inputs, stage versions, and artifact bytes before reuse. Both commands default to the run with the most recently updated state file. If you ran with `--config`, pass the same file to `resume` to use that profile again.

Each run stores an atomic snapshot in `runs/<id>/state.json` and a structured event log in `events.jsonl`. A stage writes to `work/<stage>/<attempt>/`. After output verification, the runner publishes the directory under `artifacts/<stage>/<attempt>/` and commits the successful attempt to state. Failed and abandoned attempts remain available for inspection. Only an attempt recorded as completed and verified against its fingerprints is eligible for cache reuse. A lock allows one writer per run, and the OS releases it when the process exits. Cancellation returns status `130`. Progress includes elapsed time and an ETA when the stage reports measurable work.

The Python adapter interface is `Stage`, `StageInput`, `StageContext`, and `StageResult` in `ict_pipeline.pipeline`. An adapter should read the declared inputs, write artifacts exclusively to `StageContext.output_dir`, and report progress as completed and total units. `StageContext.artifact_dir` is the future published path for files that must refer to it. Adapters own data-format validation and must keep secrets out of outputs and console messages. Bump `Stage.version` when changes to an adapter affect the validity of its results.

## Environment diagnostics

`ict doctor` displays the local report. `ict doctor --json` writes one JSON document to stdout using the [version 1 schema](docs/doctor.schema.json). A completed report exits with status `0` even when optional tools or backends are unavailable. Missing measurements use `null`. Capabilities use `available`, `unavailable`, or `unknown`. Memory and disk measurements are in bytes, and free disk refers to the filesystem containing the current directory. RAM and logical CPU counts describe the environment as reported by the OS and may exceed container limits.

GPU inventory is best effort. PyTorch availability comes from PyTorch itself, and readiness requires a small tensor operation and a successful scalar readback. CUDA and ROCm test the first visible device. Apple unified memory and unreported dedicated VRAM appear as unknown. Tool statuses reflect executable discovery and, for the repository's AI Hub helper, file presence. Docker daemon access, WSL distributions, AI Hub credentials, and remote hosts are outside the probe. Container detection relies on visible runtime markers.

`local_cpu` covers CLI diagnostics and CPU preprocessing. The `torch_*` modes describe the tested device operations. The WSL and container modes show whether the current process is already running in those environments or their readiness is unverified. External hardware commands have individual timeouts of at most eight seconds, and the isolated PyTorch process has a 20-second timeout.

## Configure the pipeline

Run `ict setup` from the workspace root in a terminal. Press Enter to accept a default or Ctrl+C to cancel. The wizard previews the settings and asks before saving. Setup records runtime preferences for later commands. Dataset and device readiness checks, downloads, installs, training, and remote jobs are separate operations.

```text
ict setup --dry-run
ict setup --non-interactive --dataset-source skip
ict setup --non-interactive --config profile.yaml --model rt_detr --dry-run
```

Headless setup requires an explicit dataset source from a flag, environment variable, or YAML. It exits with status `2` on invalid or missing settings and uses `--replace` to authorize changing an existing `configs/local.yaml`. Dry runs validate and preview the effective settings without writing files. An interrupted wizard exits with status `130`.

| YAML key | Accepted values and defaults |
|---|---|
| `execution_mode` | `local` (default), `wsl`, `docker`, `remote` |
| `dataset_source` | `aihub`, `existing`, `skip` (wizard default) |
| `data_path` | Required for `aihub` and `existing`. The wizard suggests `data/extracted`. |
| `output_path` | Filesystem path, defaults to `outputs` |
| `model` | `yolo` (default), `faster_rcnn`, `rt_detr` |
| `device` | `cpu` (default), `auto`, `cuda`, `rocm`, `mps` |
| `training_profile` | `smoke` (default), `standard`, `full` |
| `tracking` | `"off"` (default), `local`, `mlflow` |

CLI flags take precedence over process environment variables such as `ICT_MODEL` and `ICT_OUTPUT_PATH`. Environment variables take precedence over the selected YAML, which takes precedence over the packaged defaults in `src/ict_pipeline/config/default.yaml`. The wizard uses the resolved settings as prompt defaults and accepts answers for fields without explicit CLI flags. `--config` selects an input YAML in place of `configs/local.yaml`. Successful saves write the effective settings to `configs/local.yaml`. Relative paths use the workspace root. Setup validates path syntax and leaves filesystem and device availability checks to execution.

Use the process environment or copy `.env.example` to `.env` for credentials. A nonblank environment value takes precedence over the corresponding value in `.env`. The AI Hub source requires `AIHUB_API_KEY`. MLflow tracking requires `MLFLOW_TRACKING_URI`. Remote mode requires `REMOTE_HOST` and `REMOTE_USER` and accepts `REMOTE_SSH_KEY_PATH` for key-based authentication. The wizard uses hidden input for missing credentials and saves those entries to `.env` after confirmation. Headless setup reads credentials and leaves `.env` intact.

## Run tests

```text
python -m pip install -e ".[test]"
python -m pytest
```

The CLI tests use a temporary directory and run without dataset files, credentials, or a GPU.
