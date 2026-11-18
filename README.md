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

`doctor`, `setup`, `run`, `status`, `resume`, `data`, `train`, `evaluate`, `compare`, `report`, and `clean` are placeholders. Invoking one prints a message to stderr and exits with status `1`. Workflow options and subcommands are planned in [PLANS.md](PLANS.md). Existing dataset tools can still be invoked directly from `scripts/` with their own dependencies.

## Run tests

```text
python -m pip install -e ".[test]"
python -m pytest
```

The CLI tests use a temporary directory and run without dataset files, credentials, or a GPU.
