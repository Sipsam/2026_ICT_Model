"""CLI presentation for the local run catalog and report exports."""

import json
import os
from pathlib import Path
from typing import Annotated

import typer

from ..config.settings import SetupError
from ..evaluation.contracts import EvaluationError
from .catalog import Run, list_runs
from .export import ReportFormat, report_document, write_report

runs_app = typer.Typer(help="Inspect saved runs.", no_args_is_help=True)
TrainingRoot = Annotated[Path, typer.Option("--training-root", help="Root passed to ict train --runs-dir.")]
EvaluationRoot = Annotated[Path, typer.Option("--evaluation-root", help="Root passed to ict evaluate --output-dir.")]
TRAINING_ROOT = Path("runs/training")
EVALUATION_ROOT = Path("outputs/evaluation")


def _load(training_root: Path, evaluation_root: Path) -> list[Run]:
    try:
        return list_runs(Path.cwd(), training_root, evaluation_root)
    except SetupError as exc:
        typer.echo(str(exc), err=True)
    except (OSError, ValueError):
        typer.echo("Cannot read run storage. Check directories and permissions.", err=True)
    raise typer.Exit(code=1)


def _find(runs: list[Run], run_id: str | None) -> Run:
    for run in runs:
        if run_id is None or run["run_id"] == run_id:
            return run
    typer.echo("No saved run found. Use ict runs list to check the available IDs.", err=True)
    raise typer.Exit(code=1)


def _print_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, allow_nan=False))


@runs_app.command("list", help="List saved pipeline, training and evaluation runs.")
def runs_list(
    training_root: TrainingRoot = TRAINING_ROOT, evaluation_root: EvaluationRoot = EVALUATION_ROOT,
) -> None:
    runs = _load(training_root, evaluation_root)
    if not runs:
        typer.echo("No saved runs.")
    for run in runs:
        updated = run.get("updated_at") or run.get("finished_at") or run.get("started_at") or "unknown"
        typer.echo(f"{run['run_id']}   {run['kind']}   {run['status']}   {updated}")


@runs_app.command("show", help="Print a run's metadata, results and log paths as JSON.")
def runs_show(
    run_id: str, training_root: TrainingRoot = TRAINING_ROOT, evaluation_root: EvaluationRoot = EVALUATION_ROOT,
) -> None:
    _print_json(_find(_load(training_root, evaluation_root), run_id))


def status(
    run_id: Annotated[str | None, typer.Option(help="Run ID from ict runs list. Defaults to the latest run.")] = None,
    training_root: TrainingRoot = TRAINING_ROOT, evaluation_root: EvaluationRoot = EVALUATION_ROOT,
) -> None:
    """Show the status of the latest or selected run."""
    run = _find(_load(training_root, evaluation_root), run_id)
    updated = run.get("updated_at") or run.get("finished_at") or run.get("started_at") or "unknown"
    typer.echo(f"Run {run['run_id']}   {run['status']}   updated {updated}")
    if run.get("failure_reason"):
        typer.echo(run["failure_reason"])
    if run.get("selection"):
        typer.echo("Selection   " + ", ".join(run["selection"]))
    if run.get("pending_force"):
        typer.echo("Pending forced stages   " + ", ".join(run["pending_force"]))
    stages = run.get("stages", {})
    for name in dict.fromkeys((*run.get("selection", []), *stages)):
        record = stages.get(name, {})
        elapsed = f"   {record['elapsed_seconds']:.1f}s" if record.get("elapsed_seconds") is not None else ""
        detail = f"   {record['error']}" if record.get("error") else ""
        typer.echo(f"{name}   {record.get('status', 'pending')}{elapsed}{detail}")
    for model in run.get("models", []):
        typer.echo(f"{model['model']}   {model['status']}")


def compare(
    reports: Annotated[list[Path] | None, typer.Argument(help="Optional completed evaluation reports.")] = None,
    training_root: TrainingRoot = TRAINING_ROOT, evaluation_root: EvaluationRoot = EVALUATION_ROOT,
) -> None:
    """Compare saved evaluations grouped by matching split, code and protocol."""
    try:
        if reports:
            from ..evaluation.runner import compare_reports

            _print_json(compare_reports(Path.cwd(), tuple(reports)))
        else:
            _print_json(report_document(_load(training_root, evaluation_root)))
    except (EvaluationError, SetupError) as exc:
        typer.echo(f"Comparison failed. {exc}", err=True)
        raise typer.Exit(code=1) from None
    except (OSError, ValueError, RuntimeError):
        typer.echo("Comparison failed. Check report contents and file permissions.", err=True)
        raise typer.Exit(code=1) from None


def report(
    format: Annotated[ReportFormat, typer.Option("--format", help="Output format.")] = ReportFormat.HTML,
    output_dir: Annotated[Path, typer.Option(help="Parent directory for new reports.")] = Path("outputs/reports"),
    training_root: TrainingRoot = TRAINING_ROOT, evaluation_root: EvaluationRoot = EVALUATION_ROOT,
    mlflow: Annotated[bool, typer.Option("--mlflow", help="Also send a JSON snapshot to the configured MLflow service.")] = False,
    experiment_id: Annotated[str, typer.Option(help="MLflow experiment ID for --mlflow.")] = "0",
) -> None:
    """Write a local comparison report and print the saved path."""
    from ..config.storage import SetupFiles

    workspace = Path.cwd()
    try:
        document = report_document(_load(training_root, evaluation_root))
        credentials = SetupFiles.read(workspace, None).credentials(os.environ)
        credentials.protect_path(str((workspace / output_dir).resolve()), "Report output directory")
        path = write_report(workspace, output_dir, format, document)
        typer.echo(str(path))
        if mlflow:
            from .mlflow import export_to_mlflow

            export_to_mlflow(workspace, experiment_id, document)
    except SetupError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from None
    except (OSError, ValueError, RuntimeError):
        typer.echo("Cannot generate the report. Check output permissions and choose a directory outside source data.", err=True)
        raise typer.Exit(code=1) from None
