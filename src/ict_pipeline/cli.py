"""Command-line interface for the ICT pipeline."""

import json
import os
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .environment.doctor import collect_environment
from .environment.model import EnvironmentReport

app = typer.Typer(
    name="ict",
    help="ICT manufacturing defect detection pipeline.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"ict {version('ict-pipeline')}")
        raise typer.Exit()


@app.callback()
def main(
    show_version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_show_version,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Register options shared by the top-level command."""


def _not_implemented(command: str) -> NoReturn:
    Console(stderr=True).print(f"ict {command} is not yet implemented.", highlight=False)
    raise typer.Exit(code=1)


def _display_value(value: str | int | bool | None) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "unknown" if value is None else str(value)


def _display_bytes(value: int | None) -> str:
    return "unknown" if value is None else f"{value / 1024**3:,.2f} GiB"


def _print_environment(report: EnvironmentReport) -> None:
    host, python, resources = report.host, report.python, report.resources
    graphics, torch = report.graphics, report.torch
    torch_info = torch.version or (torch.detail or "version unknown").replace("_", " ")
    table = Table(title="ICT environment", show_header=False)
    table.add_column(style="cyan")
    table.add_column()
    rows = [
        ("OS", host.os),
        ("OS/kernel release", _display_value(host.release)),
        ("Architecture", _display_value(host.architecture)),
        ("In WSL", _display_value(host.wsl)),
        ("In container", _display_value(host.container)),
        ("Python", f"{python.implementation} {python.version}"),
        ("Python venv", _display_value(python.venv)),
        ("CPU", _display_value(resources.cpu_model)),
        ("Logical CPUs", _display_value(resources.logical_cpus)),
        ("RAM", _display_bytes(resources.ram_bytes)),
        ("Free disk (working directory)", _display_bytes(resources.working_directory_free_bytes)),
        ("GPU inventory", graphics.inventory),
    ]
    for gpu in graphics.devices:
        rows.append((
            "GPU",
            f"{_display_value(gpu.vendor)} / {_display_value(gpu.model)} / {_display_bytes(gpu.vram_bytes)} VRAM",
        ))
    rows.extend([
        ("nvidia-smi probe", graphics.nvidia_smi),
        ("NVIDIA driver", _display_value(graphics.nvidia_driver)),
        ("PyTorch", f"{torch.status} ({torch_info})"),
        ("CUDA build", _display_value(torch.cuda_build)),
        ("ROCm build", _display_value(torch.rocm_build)),
    ])
    for name, backend in (("CPU", torch.cpu), ("CUDA", torch.cuda), ("ROCm", torch.rocm), ("MPS", torch.mps)):
        rows.append((f"PyTorch {name}", f"Availability  {backend.availability}\nReadiness     {backend.readiness}"))
    for label, value in rows:
        table.add_row(label, Text(value))
    console = Console()
    console.print(table)
    console.print("Tool discovery")
    for tool, status in report.tools.items():
        console.print(f"  {tool}  {status}")
    console.print("Execution modes")
    for mode, status in report.execution_modes.items():
        console.print(f"  {mode}  {status}")
    if report.recommendations:
        console.print("Recommendations")
        for recommendation in report.recommendations:
            console.print(f"  - {recommendation}")


@app.command(help="Check the local runtime and PyTorch device readiness.")
def doctor(
    json_output: Annotated[bool, typer.Option("--json", help="Print the versioned JSON report.")] = False,
) -> None:
    report = collect_environment(Path.cwd())
    if json_output:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _print_environment(report)


@app.command(help="Configure the pipeline with a wizard or validate and save headless settings.")
def setup(
    config: Annotated[Path | None, typer.Option(help="Read settings from this YAML file.")] = None,
    non_interactive: Annotated[bool, typer.Option(help="Validate settings without prompting.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Validate and preview settings without saving.")] = False,
    replace: Annotated[bool, typer.Option(help="Allow headless replacement of configs/local.yaml.")] = False,
    execution_mode: Annotated[str | None, typer.Option(help="local, wsl, docker, or remote.")] = None,
    dataset_source: Annotated[str | None, typer.Option(help="aihub, existing, or skip.")] = None,
    data_path: Annotated[str | None, typer.Option(help="Dataset directory.")] = None,
    output_path: Annotated[str | None, typer.Option(help="Output directory.")] = None,
    model: Annotated[str | None, typer.Option(help="yolo, faster_rcnn, or rt_detr.")] = None,
    device: Annotated[str | None, typer.Option(help="auto, cpu, cuda, rocm, or mps.")] = None,
    training_profile: Annotated[str | None, typer.Option(help="smoke, standard, or full.")] = None,
    tracking: Annotated[str | None, typer.Option(help="off, local, or mlflow.")] = None,
) -> None:
    from .config.settings import SetupError
    from .wizard import run_setup

    overrides = {
        "execution_mode": execution_mode,
        "dataset_source": dataset_source,
        "data_path": data_path,
        "output_path": output_path,
        "model": model,
        "device": device,
        "training_profile": training_profile,
        "tracking": tracking,
    }
    try:
        run_setup(
            Path.cwd(), config, os.environ,
            {name: value for name, value in overrides.items() if value is not None},
            non_interactive=non_interactive, dry_run=dry_run, replace=replace,
        )
    except SetupError as exc:
        typer.echo(f"Setup failed. {exc}", err=True)
        raise typer.Exit(code=2) from None
    except (typer.Abort, KeyboardInterrupt, EOFError):
        typer.echo("Setup cancelled.", err=True)
        raise typer.Exit(code=130) from None


@app.command(help="Run pipeline stages (planned).")
def run() -> None:
    _not_implemented("run")


@app.command(help="Show pipeline run status (planned).")
def status() -> None:
    _not_implemented("status")


@app.command(help="Resume an interrupted run (planned).")
def resume() -> None:
    _not_implemented("resume")


@app.command(help="Manage pipeline datasets (planned).")
def data() -> None:
    _not_implemented("data")


@app.command(help="Train detection models (planned).")
def train() -> None:
    _not_implemented("train")


@app.command(help="Evaluate trained models (planned).")
def evaluate() -> None:
    _not_implemented("evaluate")


@app.command(help="Compare model evaluation results (planned).")
def compare() -> None:
    _not_implemented("compare")


@app.command(help="Generate pipeline reports (planned).")
def report() -> None:
    _not_implemented("report")


@app.command(help="Clean generated artifacts (planned).")
def clean() -> None:
    _not_implemented("clean")
