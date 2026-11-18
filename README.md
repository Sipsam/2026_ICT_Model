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

The CLI requires Typer and Rich and runs without PyTorch, GPU libraries, or dataset files.

## Available commands

`ict --help` lists the registered commands, and `ict --version` prints the installed package version. Both exit with status `0`.

`setup`, `run`, `status`, `resume`, `data`, `train`, `evaluate`, `compare`, `report`, and `clean` are placeholders. Invoking one prints a message to stderr and exits with status `1`. Workflow options and subcommands are planned in [PLANS.md](PLANS.md). Existing dataset tools can still be invoked directly from `scripts/` with their own dependencies.

## Environment diagnostics

`ict doctor` displays the local report. `ict doctor --json` writes one JSON document to stdout using the [version 1 schema](docs/doctor.schema.json). A completed report exits with status `0` even when optional tools or backends are unavailable. Missing measurements use `null`. Capabilities use `available`, `unavailable`, or `unknown`. Memory and disk measurements are in bytes, and free disk refers to the filesystem containing the current directory. RAM and logical CPU counts describe the environment as reported by the OS and may exceed container limits.

GPU inventory is best effort. PyTorch availability comes from PyTorch itself, and readiness requires a small tensor operation and a successful scalar readback. CUDA and ROCm test the first visible device. Apple unified memory and unreported dedicated VRAM appear as unknown. Tool statuses reflect executable discovery and, for the repository's AI Hub helper, file presence. Docker daemon access, WSL distributions, AI Hub credentials, and remote hosts are outside the probe. Container detection relies on visible runtime markers.

`local_cpu` covers CLI diagnostics and CPU preprocessing. The `torch_*` modes describe the tested device operations. The WSL and container modes show whether the current process is already running in those environments or their readiness is unverified. External hardware commands have individual timeouts of at most eight seconds, and the isolated PyTorch process has a 20-second timeout.

## Run tests

```text
python -m pip install -e ".[test]"
python -m pytest
```

The CLI tests use a temporary directory and run without dataset files, credentials, or a GPU.
