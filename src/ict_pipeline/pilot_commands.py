"""CLI entry points for detector recommendations and bounded pilots."""

import json
from dataclasses import asdict
from pathlib import Path
from time import monotonic
from typing import Annotated, NoReturn

import typer

from .adaptive import Device, Overrides, Precision, recommend
from .config.settings import SetupError
from .disk import StorageError, volume_at
from .environment.doctor import collect_environment
from .evaluation.contracts import EvaluationError
from .models.contracts import ModelName, Progress, TrainingError
from .pilot import pilot_scope, plan_pilot, prepare_pilot, run_pilot
from .pipeline.model import PipelineError

pilot_app = typer.Typer(no_args_is_help=True, help="Prepare and run bounded pretrained detector pilots.")
_SAFE_ERRORS = (TrainingError, SetupError, StorageError, PipelineError, EvaluationError)


def _fail(exc: Exception) -> NoReturn:
    message = str(exc) if isinstance(exc, _SAFE_ERRORS) else "Check the input artifacts, permissions, and free storage."
    typer.echo(message, err=True)
    raise typer.Exit(1) from None


def _progress() -> Progress:
    stages: dict[str, tuple[float, float]] = {}

    def report(stage: str, done: int, total: int) -> None:
        if not total:
            return
        now = monotonic()
        start, last = stages.get(stage, (now, float("-inf")))
        if done in (0, total) or now - last >= 1:
            elapsed = now - start
            eta = f"{elapsed * (total - done) / done:.1f}s" if done else "unknown"
            typer.echo(f"{stage}  {done}/{total}  elapsed {elapsed:.1f}s  ETA {eta}", err=True)
            stages[stage] = (start, now)

    return report


def _models(model: ModelName | None, all_models: bool) -> tuple[ModelName, ...]:
    if (model is not None) == all_models:
        raise typer.BadParameter("Select exactly one of --model or --all.")
    return (model,) if model is not None else tuple(ModelName)


def recommend_command(
    model: Annotated[ModelName | None, typer.Option()] = None,
    all_models: Annotated[bool, typer.Option("--all")] = False,
    output: Annotated[Path | None, typer.Option(help="Destination path or directory for the disk probe.")] = None,
    device: Annotated[Device | None, typer.Option()] = None,
    precision: Annotated[Precision | None, typer.Option()] = None,
    image_size: Annotated[int | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option()] = None,
    workers: Annotated[int | None, typer.Option()] = None,
    gradient_accumulation: Annotated[int | None, typer.Option()] = None,
) -> None:
    """Report device and model checks and conservative starting settings as JSON."""
    names = _models(model, all_models)
    try:
        volume = volume_at(output or Path.cwd())
        environment = collect_environment(volume.path)
        overrides = Overrides(device, precision, image_size, batch_size, workers, gradient_accumulation)
        models = [recommend(name, environment, overrides, free_disk_bytes=volume.free_bytes) for name in names]
        typer.echo(json.dumps({"schema_version": 1, "hardware": environment.to_dict(),
                               "models": [asdict(item) for item in models]}, indent=2))
        if not all(item.checks_passed for item in models):
            raise typer.Exit(1)
    except _SAFE_ERRORS as exc:
        _fail(exc)


@pilot_app.command("prepare")
def prepare(
    bundle: Annotated[Path, typer.Option(help="Published group-aware split bundle with a detection seal.")],
    out: Annotated[Path, typer.Option(help="Fresh output directory outside production runs.")],
    train_images: Annotated[int, typer.Option(min=1, max=256)] = 64,
    val_images: Annotated[int, typer.Option(min=1, max=256)] = 32,
    min_per_stratum: Annotated[int, typer.Option(min=1, help="Minimum per defect class and background, per split.")] = 1,
    seed: Annotated[int, typer.Option(min=0, max=2**32 - 1)] = 42,
) -> None:
    """Materialize and verify a deterministic Train and Validation cohort."""
    try:
        result = prepare_pilot(Path.cwd(), bundle, out, train_images, val_images, min_per_stratum, seed, _progress())
    except (*_SAFE_ERRORS, OSError, ValueError, TypeError, KeyError) as exc:
        _fail(exc)
    typer.echo(json.dumps(result, indent=2))


@pilot_app.command("run")
def run(
    cohort: Annotated[Path, typer.Option(help="Prepared pilot cohort directory.")],
    config_dir: Annotated[Path, typer.Option(help="Model profiles pointing to staged pretrained files.")],
    model: Annotated[ModelName | None, typer.Option()] = None,
    all_models: Annotated[bool, typer.Option("--all")] = False,
    epochs: Annotated[int, typer.Option(min=1, max=3)] = 1,
    device: Annotated[Device | None, typer.Option()] = None,
    precision: Annotated[Precision | None, typer.Option()] = None,
    image_size: Annotated[int | None, typer.Option()] = None,
    batch_size: Annotated[int | None, typer.Option()] = None,
    workers: Annotated[int | None, typer.Option()] = None,
    gradient_accumulation: Annotated[int | None, typer.Option()] = None,
    resume_from: Annotated[Path | None, typer.Option(help="Unfinished pilot model directory from this cohort.")] = None,
    approve_train: Annotated[bool, typer.Option("--approve-train")] = False,
    dry_run: Annotated[bool, typer.Option(help="Preview without training. This is the default.")] = False,
) -> None:
    """Preview by default. Train and evaluate on Validation after explicit approval."""
    names = _models(model, all_models)
    if dry_run and approve_train:
        raise typer.BadParameter("Choose either --dry-run or --approve-train.")
    try:
        plan = plan_pilot(
            Path.cwd(), cohort, names, config_dir, epochs,
            Overrides(device, precision, image_size, batch_size, workers, gradient_accumulation), resume_from,
        )
        typer.echo(json.dumps(pilot_scope(plan), indent=2))
        if approve_train:
            typer.echo(str(run_pilot(plan, approved=True, progress=_progress())))
    except (*_SAFE_ERRORS, OSError, ValueError, TypeError, KeyError) as exc:
        _fail(exc)
    except KeyboardInterrupt:
        typer.echo("Pilot interrupted. See pilot.json and the unfinished model's run.json.", err=True)
        raise typer.Exit(130) from None
