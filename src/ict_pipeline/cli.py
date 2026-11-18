"""Command-line interface for the ICT pipeline."""

import json
import os
from importlib.metadata import version
from pathlib import Path
from time import monotonic
from typing import Annotated, Literal, NoReturn
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table
from rich.text import Text

from .config.settings import SetupError, merge_settings, validate_settings
from .data.commands import data_app, dataset_paths, provider_for, show_download_plan
from .data.ingestion import Acquisition, has_dataset
from .data.model import DataError, load_selection
from .data.stages import DataStages
from .environment.doctor import collect_environment
from .environment.model import EnvironmentReport
from .evaluation.contracts import EvaluationConfig, EvaluationError, EvaluationSplit
from .models.contracts import ModelName, TrainingError
from .pipeline import PipelineError, StageRunner
from .pipeline.model import valid_id
from .pipeline.runner import StageEvent
from .pipeline.stages import pipeline_stages
from .reporting.commands import compare, report, runs_app, status

app = typer.Typer(
    name="ict",
    help="ICT manufacturing defect detection pipeline.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


app.add_typer(data_app, name="data")
app.add_typer(runs_app, name="runs")
app.command()(status)
app.command()(compare)
app.command()(report)


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


def _pipeline_progress(event: StageEvent) -> None:
    units = ""
    if event.completed is not None:
        total = event.total if event.total is not None else "?"
        units = f"   {event.completed}/{total} units"
    eta = f"{event.eta_seconds:.1f}s" if event.eta_seconds is not None else "unknown"
    typer.echo(
        f"[{event.stages_completed}/{event.stages_total}] {event.stage} {event.status}{units}"
        f"   elapsed {event.elapsed_seconds:.1f}s   ETA {eta}"
    )


def _pipeline(config: Path | None, run_id: str | None, approve_download: bool = False) -> StageRunner:
    from .config.storage import SetupFiles

    root = Path.cwd()
    files = SetupFiles.read(root, config)
    credentials = files.credentials(os.environ)
    credentials.protect_path(run_id, "run ID")
    values = merge_settings(files.config_text, os.environ, {})
    values.setdefault("dataset_source", "skip")
    settings = validate_settings(values, credentials).model_dump(exclude_none=True)
    for key in ("data_path", "output_path"):
        if key in settings:
            try:
                settings[key] = (root / settings[key]).resolve().as_posix()
                credentials.protect_path(settings[key], f"resolved {key}")
            except (OSError, RuntimeError, ValueError):
                raise SetupError(f"Cannot resolve {key}. Check the path and permissions.") from None
    data = None
    if settings["dataset_source"] != "skip":
        selection = load_selection()
        raw_root, dataset = dataset_paths(root, selection, credentials, None, Path(settings["data_path"]))
        settings["raw_path"] = raw_root.as_posix()
        existing = dataset if settings["dataset_source"] == "existing" or has_dataset(dataset, selection) else None
        acquisition = Acquisition(selection, raw_root)
        data = DataStages(
            acquisition, provider_for(credentials, os.environ), existing, approve_download,
            lambda missing: show_download_plan(acquisition, missing),
        )
    if data is not None:
        stages = pipeline_stages(data)
    elif "data_path" in settings:
        stages = pipeline_stages(dataset_root=Path(settings["data_path"]))
    else:
        stages = pipeline_stages()
    return StageRunner(stages, root, settings, on_event=_pipeline_progress, credentials=credentials)


def _pipeline_failure(exc: Exception) -> NoReturn:
    if isinstance(exc, (OSError, ValueError)):
        typer.echo("Cannot access run or configuration files. Check permissions and available storage.", err=True)
    else:
        typer.echo(str(exc), err=True)
    raise typer.Exit(code=1) from None


RunIdOption = Annotated[str | None, typer.Option(help="Identify a saved run under runs/.")]
ConfigOption = Annotated[Path | None, typer.Option(help="Read settings from this YAML file.")]
ApprovalOption = Annotated[bool, typer.Option("--approve-download", help="Authorize missing AI Hub transfers.")]


@app.command(help="Plan or execute selected pipeline stages.")
def run(
    dry_run: Annotated[bool, typer.Option(help="Print the plan without writing run files.")] = False,
    start: Annotated[str | None, typer.Option("--from", help="First stage in the range.")] = None,
    until: Annotated[str | None, typer.Option(help="Last stage in the range, inclusive.")] = None,
    only: Annotated[list[str] | None, typer.Option(help="Select a stage. Repeat to select multiple stages.")] = None,
    force: Annotated[bool, typer.Option(help="Recompute the selected stages in new artifact directories.")] = False,
    run_id: RunIdOption = None,
    config: ConfigOption = None,
    approve_download: ApprovalOption = False,
) -> None:
    try:
        runner = _pipeline(config, run_id, approve_download)
        selection = runner.graph.select(start, until, only or ())
        if dry_run:
            for step in runner.plan(selection, run_id=run_id, force=force):
                typer.echo(f"{step.stage}   {step.action}")
            return
        run_id = uuid4().hex if run_id is None else run_id
        if valid_id(run_id):
            typer.echo(f"Run ID   {run_id}")
        runner.run(run_id, selection, force=force)
        typer.echo(f"Run {run_id} completed.")
    except (PipelineError, SetupError, DataError, OSError, ValueError) as exc:
        _pipeline_failure(exc)
    except KeyboardInterrupt:
        typer.echo("Run cancelled. Use ict status and ict resume to continue.", err=True)
        raise typer.Exit(code=130) from None


@app.command(help="Resume the last selection for a saved run and verify its cached results.")
def resume(
    run_id: RunIdOption = None, config: ConfigOption = None, approve_download: ApprovalOption = False,
) -> None:
    try:
        state = _pipeline(config, run_id, approve_download).resume(run_id)
        typer.echo(f"Run {state.run_id} completed.")
    except (PipelineError, SetupError, DataError, OSError, ValueError) as exc:
        _pipeline_failure(exc)
    except KeyboardInterrupt:
        typer.echo("Run cancelled. Use ict status and ict resume to continue.", err=True)
        raise typer.Exit(code=130) from None


@app.command(help="Train a detector or all three on generated detection data.")
def train(
    model: Annotated[ModelName | None, typer.Option(help="Select one detector.")] = None,
    all_models: Annotated[bool, typer.Option("--all", help="Train all detectors sequentially.")] = False,
    dataset: Annotated[Path, typer.Option(help="Directory containing the generated yolo/ and coco/ outputs.")] = Path(
        "data/processed/detection"
    ),
    runs_dir: Annotated[Path, typer.Option(help="Parent directory for new training runs.")] = Path("runs/training"),
    config_dir: Annotated[Path | None, typer.Option(help="Read the selected model YAML files from this directory.")] = None,
    resume_from: Annotated[Path | None, typer.Option(help="Continue a trusted earlier ICT model run into a new directory.")] = None,
    approve_train: Annotated[bool, typer.Option("--approve-train", help="Authorize training and checkpoint writes.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Preview the selected configs and output paths.")] = False,
) -> None:
    from .models.registry import REGISTRY
    from .models.training import execute_training, plan_training

    if (model is not None) == all_models:
        raise typer.BadParameter("Select exactly one of --model or --all.")
    names = tuple(REGISTRY) if model is None else (model,)
    timings: dict[str, tuple[float, float]] = {}

    def progress(stage: str, done: int, total: int) -> None:
        now = monotonic()
        started, last = timings.get(stage, (now, float("-inf")))
        if done == total or now - last >= 1:
            elapsed = now - started
            eta = f"{elapsed * (total - done) / done:.1f}s" if done else "unknown"
            typer.echo(f"{stage}   {done}/{total}   elapsed {elapsed:.1f}s   ETA {eta}")
            timings[stage] = (started, now)

    try:
        plan = plan_training(Path.cwd(), names, dataset, runs_dir, config_dir, resume_from)
        for name, config in plan.models.items():
            typer.echo(f"{name}   input {plan.dataset / REGISTRY[name].data_format}   output {plan.destination / name}")
            typer.echo(config.model_dump_json())
        if dry_run:
            return
        if not approve_train:
            raise TrainingError("Training requires explicit approval. Review the scope above, then pass --approve-train.")
        for name, checkpoints in execute_training(plan, progress):
            typer.echo(f"{name} completed   best {checkpoints.best}   last {checkpoints.last}")
    except (TrainingError, SetupError, PipelineError) as exc:
        typer.echo(f"Training failed. {exc}", err=True)
        raise typer.Exit(code=1) from None
    except (OSError, ValueError, RuntimeError):
        typer.echo("Training could not access local files. Check paths, permissions, and available storage.", err=True)
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        typer.echo("Training interrupted. Use --resume-from with the model run directory and its original config.", err=True)
        raise typer.Exit(code=130) from None


@app.command(help="Benchmark completed detector runs on one labeled COCO split.")
def evaluate(
    runs: Annotated[list[Path], typer.Option("--run", help="Trusted completed model run directory. Repeat for each model.")],
    dataset: Annotated[Path, typer.Option(help="Directory containing coco/ with images and annotations.")] = Path(
        "data/processed/detection"
    ),
    split: Annotated[EvaluationSplit, typer.Option(help="Validation (val) or independent Test (test).")] = EvaluationSplit.VALIDATION,
    held_out: Annotated[bool, typer.Option(help="Attest that Test was independently held out and never used for tuning.")] = False,
    reference_dataset: Annotated[Path | None, typer.Option(help="Original Train/Validation COCO dataset for Test overlap checks.")] = None,
    checkpoint: Annotated[Literal["best", "last"], typer.Option(help="Training checkpoint to evaluate.")] = "best",
    output_dir: Annotated[Path, typer.Option(help="Parent directory for new evaluation attempts.")] = Path("outputs/evaluation"),
    resume_from: Annotated[Path | None, typer.Option(help="Resume a previous evaluation with the original arguments.")] = None,
    image_size: Annotated[int, typer.Option(help="Model input resize bound, a multiple of 32.")] = 640,
    score_floor: Annotated[float, typer.Option(help="Minimum confidence retained for AP.")] = 0.001,
    confidence: Annotated[float, typer.Option(help="Confidence cutoff for precision, recall, F1 and confusion.")] = 0.25,
    match_iou: Annotated[float, typer.Option(help="Match IoU for fixed-confidence metrics.")] = 0.5,
    nms_iou: Annotated[float, typer.Option(help="Class-aware NMS IoU for YOLO and Faster R-CNN.")] = 0.7,
    warmup: Annotated[int, typer.Option(help="Number of untimed single-image warmup calls.")] = 3,
    device: Annotated[Literal["cpu", "cuda", "rocm", "mps"], typer.Option()] = "cpu",
    precision: Annotated[Literal["fp32", "fp16"], typer.Option()] = "fp32",
) -> None:
    from pydantic import ValidationError

    from .evaluation.runner import evaluate_runs

    timings: dict[str, tuple[float, float]] = {}

    def progress(stage: str, done: int, total: int) -> None:
        if total == 0:
            if stage.startswith("evaluation directory "):
                typer.echo(stage, err=True)
            return
        now = monotonic()
        started, last = timings.get(stage, (now, float("-inf")))
        if done in (0, total) or now - last >= 1:
            elapsed = now - started
            eta = f"{elapsed * (total - done) / done:.1f}s" if done else "unknown"
            typer.echo(f"{stage}   {done}/{total}   elapsed {elapsed:.1f}s   ETA {eta}", err=True)
            timings[stage] = started, now

    try:
        config = EvaluationConfig(
            image_size=image_size, score_floor=score_floor, confidence=confidence, match_iou=match_iou,
            nms_iou=nms_iou, warmup=warmup, device=device, precision=precision,
        )
        for report_path in evaluate_runs(
            Path.cwd(), tuple(runs), dataset, split, config, output_dir, checkpoint, held_out,
            reference_dataset, resume_from, progress,
        ):
            typer.echo(str(report_path))
    except ValidationError as exc:
        fields = ", ".join(str(error["loc"][0]) for error in exc.errors(include_input=False))
        typer.echo(f"Invalid evaluation options. Check {fields}.", err=True)
        raise typer.Exit(code=2) from None
    except (EvaluationError, SetupError, TrainingError, PipelineError) as exc:
        typer.echo(f"Evaluation failed. {exc}", err=True)
        raise typer.Exit(code=1) from None
    except (OSError, ValueError, RuntimeError):
        typer.echo("Evaluation could not access local files. Check inputs, permissions, and storage.", err=True)
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        typer.echo("Evaluation interrupted. Resume with the evaluation directory and original arguments.", err=True)
        raise typer.Exit(code=130) from None


@app.command(help="Clean generated artifacts (planned).")
def clean() -> None:
    _not_implemented("clean")
