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

`run`, `status`, `resume`, `data`, `train`, `evaluate`, `compare`, `report`, and `clean` are placeholders. Invoking one prints a message to stderr and exits with status `1`. Workflow options and subcommands are planned in [PLANS.md](PLANS.md). Existing dataset tools can still be invoked directly from `scripts/` with their own dependencies.

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
