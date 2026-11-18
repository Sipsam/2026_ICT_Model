"""Command-line interface for the ICT pipeline."""

from importlib.metadata import version
from typing import Annotated, NoReturn

import typer
from rich.console import Console

app = typer.Typer(
    name="ict",
    help="ICT manufacturing defect detection pipeline.",
    no_args_is_help=True,
    add_completion=False,
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


@app.command(help="Check the local runtime (planned).")
def doctor() -> None:
    _not_implemented("doctor")


@app.command(help="Configure the pipeline (planned).")
def setup() -> None:
    _not_implemented("setup")


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
