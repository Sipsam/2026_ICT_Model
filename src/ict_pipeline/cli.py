"""Command-line interface for the ICT pipeline."""

import json
import os
from dataclasses import asdict, dataclass
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
from .data.integrity_commands import integrity_app
from .data.model import DataError, load_selection
from .data.stages import DataStages
from .environment.doctor import collect_environment
from .environment.model import EnvironmentReport
from .evaluation.contracts import EvaluationConfig, EvaluationError, EvaluationSplit
from .models.contracts import ModelName, TrainingError
from .pipeline import PipelineError, StageRunner
from .pipeline.detection import DETECTION_STAGES, DetectionStages
from .pipeline.model import valid_id
from .pipeline.runner import StageEvent
from .pipeline.stages import CLASSIFICATION_STAGES, pipeline_stages
from .pipeline.store import RunState, RunStore
from .pilot_commands import pilot_app, recommend_command
from .reporting.commands import compare, report, runs_app, status
from .runtime.commands import runtime_app

app = typer.Typer(
    name="ict",
    help="ICT manufacturing defect detection pipeline.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


app.add_typer(data_app, name="data")
app.add_typer(integrity_app, name="integrity")
app.add_typer(runs_app, name="runs")
app.add_typer(runtime_app, name="runtime")
app.add_typer(pilot_app, name="pilot")
app.command("recommend")(recommend_command)
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
        ("Available RAM", _display_bytes(resources.available_ram_bytes)),
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
        value = f"Availability  {backend.availability}\nTensor test   {backend.readiness}"
        if backend.device_name is not None:
            value += f"\nDevice 0   {backend.device_name}"
        if backend.total_device_bytes is not None:
            value += f"\nDevice 0 VRAM   {_display_bytes(backend.free_device_bytes)} free / {_display_bytes(backend.total_device_bytes)} total"
        if backend.shared_memory_limit_bytes is not None:
            value += f"\nShared memory recommendation   {_display_bytes(backend.shared_memory_limit_bytes)}"
        if backend.precisions_tested:
            value += "\nTested precisions   " + ", ".join(backend.precisions_tested)
        rows.append((f"PyTorch {name}", value))
    for label, value in rows:
        table.add_row(label, Text(value))
    console = Console()
    console.print(table)
    console.print("Tool discovery")
    for tool, availability in report.tools.items():
        console.print(f"  {tool}  {availability}")
    console.print("Execution modes")
    for mode, availability in report.execution_modes.items():
        console.print(f"  {mode}  {availability}")
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


EvaluationDevice = Literal["cpu", "cuda", "rocm", "mps"]


@dataclass(frozen=True)
class DetectorOptions:
    model: ModelName | None
    all_models: bool
    config_dir: Path | None
    evaluation_image_size: int
    evaluation_device: EvaluationDevice


def _pipeline(
    config: Path | None, run_id: str | None, approve_download: bool = False, approve_train: bool = False,
    *, detector_options: DetectorOptions | None = None, approve_detection: bool = False,
    resume: bool = False, split_bundle: Path | None = None,
) -> tuple[StageRunner, DetectionStages | None, RunState | None]:
    from .config.storage import SetupFiles

    root = Path.cwd()
    files = SetupFiles.read(root, config)
    credentials = files.credentials(os.environ)
    credentials.protect_path(run_id, "run ID")
    values = merge_settings(files.config_text, os.environ, {})
    values.setdefault("dataset_source", "skip")
    if values["dataset_source"] == "skip" and values.get("data_path") is None:
        values["data_path"] = str(load_selection().extracted_root)
    resolved = validate_settings(values, credentials)
    settings = resolved.model_dump(exclude_none=True)
    if settings["execution_mode"] in {"docker", "remote"}:
        raise SetupError("Use ict runtime with --executor docker or --executor ssh for this execution mode.")
    if settings["execution_mode"] == "wsl":
        from .environment.probes import probe_host

        if not probe_host().wsl:
            raise SetupError("Open an existing WSL2 distribution and invoke ict inside it, using Linux paths.")
    for key in ("data_path", "output_path"):
        if key in settings:
            try:
                settings[key] = (root / settings[key]).resolve().as_posix()
                credentials.protect_path(settings[key], f"resolved {key}")
            except (OSError, RuntimeError, ValueError):
                raise SetupError(f"Cannot resolve {key}. Check the path and permissions.") from None
    data = None
    if settings["dataset_source"] != "skip" and split_bundle is None:
        selection = load_selection()
        raw_root, dataset = dataset_paths(root, selection, credentials, None, Path(settings["data_path"]))
        settings["raw_path"] = raw_root.as_posix()
        existing = dataset if settings["dataset_source"] == "existing" or has_dataset(dataset, selection) else None
        acquisition = Acquisition(selection, raw_root)
        data = DataStages(
            acquisition, provider_for(credentials, os.environ), existing, approve_download,
            lambda missing: show_download_plan(acquisition, missing),
        )
    def select_bundle(bundle: Path) -> Path:
        credentials.protect_path(str(bundle), "split bundle path")
        bundle = (root / bundle).resolve()
        credentials.protect_path(str(bundle), "resolved split bundle path")
        settings["detection_split_bundle"] = bundle.as_posix()
        return bundle

    if split_bundle is not None:
        split_bundle = select_bundle(split_bundle)

    def build(detectors: DetectionStages | None) -> StageRunner:
        stages = pipeline_stages(
            data=data, dataset_root=Path(settings["data_path"]) if data is None else None,
            classification=resolved.classification, approve_train=approve_train, detectors=detectors,
            split_bundle=split_bundle,
        )
        return StageRunner(stages, root, settings, on_event=_pipeline_progress, credentials=credentials)

    saved = None
    if resume:
        runner = build(None)
        saved = runner.status(run_id)
        if saved_bundle := saved.configuration.get("detection_split_bundle"):
            split_bundle = select_bundle(Path(saved_bundle))
        if not set(saved.selection).intersection(DETECTION_STAGES) or "detection_training" not in saved.configuration:
            return build(None) if split_bundle else runner, None, saved
        training = saved.configuration["detection_training"]
        names = tuple(ModelName(name) for name in training["models"])
        directory = Path(training["config_dir"]) if training["config_dir"] is not None else None
        benchmark = EvaluationConfig.model_validate(saved.configuration["detection_evaluation"])
    elif detector_options is not None:
        names = tuple(ModelName) if detector_options.all_models else (detector_options.model or ModelName(resolved.model),)
        directory = detector_options.config_dir
        try:
            benchmark = EvaluationConfig(
                image_size=detector_options.evaluation_image_size, device=detector_options.evaluation_device,
            )
        except ValueError:
            raise SetupError("Evaluation image size must be a multiple of 32 and at least 64.") from None
    else:
        return build(None), None, None
    if directory is not None:
        credentials.protect_path(str(directory), "detector config directory")
        directory = (root / directory).resolve()
        credentials.protect_path(str(directory), "resolved detector config directory")
    settings["detection_training"] = {
        "models": [name.value for name in names], "config_dir": directory.as_posix() if directory else None,
    }
    settings["detection_evaluation"] = benchmark.model_dump(mode="json")
    detectors = DetectionStages.configure(
        root, credentials, names, directory, benchmark, approve_detection, typer.echo,
    )
    return build(detectors), detectors, saved


def _pipeline_failure(exc: Exception) -> NoReturn:
    if isinstance(exc, (OSError, ValueError)):
        typer.echo("Cannot access run or configuration files. Check permissions and available storage.", err=True)
    else:
        typer.echo(str(exc), err=True)
    raise typer.Exit(code=1) from None


RunIdOption = Annotated[str | None, typer.Option(help="Identify a saved run under runs/.")]
ConfigOption = Annotated[Path | None, typer.Option(help="Read settings from this YAML file.")]
ApprovalOption = Annotated[bool, typer.Option("--approve-download", help="Authorize missing AI Hub transfers.")]
ClassificationApproval = Annotated[
    bool, typer.Option("--approve-train", help="Authorize CPU image-level classification training.")
]
DetectionApproval = Annotated[
    bool, typer.Option("--approve-detection-train", help="Authorize the selected detector training for this invocation.")
]


def _print_completed(runner: StageRunner, state: RunState) -> None:
    typer.echo(f"Run {state.run_id} completed.")
    store = RunStore(runner.runs_dir, state.run_id)
    if "convert" in state.selection:
        detection = store.artifact_dir("convert", state.stages["convert"].attempt) / "detection"
        typer.echo(f"Detection dataset   {detection}")
    for stage, label, path in (
        ("train", "Detector checkpoints", "checkpoints"),
        ("validate_models", "Validation predictions", "predictions"),
        ("evaluate", "Detection metrics", "metrics.json"),
        ("report", "Detection report", "report.html"),
    ):
        if stage in state.selection:
            record = state.stages[stage]
            typer.echo(f"{label}   {store.artifact_dir(stage, record.attempt) / path}")
    evaluated = CLASSIFICATION_STAGES[-1]
    if evaluated in state.selection:
        comparison = store.artifact_dir(evaluated, state.stages[evaluated].attempt) / "comparison.json"
        typer.echo(f"Classification comparison   {comparison}")
    elif (prepared := CLASSIFICATION_STAGES[0]) in state.selection:
        alignment = store.artifact_dir(prepared, state.stages[prepared].attempt) / "alignment.json"
        typer.echo(f"Classification alignment   {alignment}")


@app.command(help="Run detection preprocessing or the complete detector workflow, or select specific stages.")
def run(
    dry_run: Annotated[bool, typer.Option(help="Print the plan without writing run files.")] = False,
    start: Annotated[str | None, typer.Option("--from", help="First stage in the range.")] = None,
    until: Annotated[str | None, typer.Option(help="Last stage in the range, inclusive.")] = None,
    only: Annotated[list[str] | None, typer.Option(help="Select a stage. Repeat to select multiple stages.")] = None,
    force: Annotated[bool, typer.Option(help="Recompute the selected stages in new artifact directories.")] = False,
    run_id: RunIdOption = None,
    config: ConfigOption = None,
    approve_download: ApprovalOption = False,
    approve_train: ClassificationApproval = False,
    approve_detection_train: DetectionApproval = False,
    model: Annotated[ModelName | None, typer.Option(help="Select one detector and run through report by default.")] = None,
    all_models: Annotated[bool, typer.Option("--all", help="Select all three detectors sequentially.")] = False,
    config_dir: Annotated[Path | None, typer.Option(help="Read detector YAML profiles from this directory.")] = None,
    evaluation_image_size: Annotated[int, typer.Option(help="Shared Validation inference image size.")] = 640,
    evaluation_device: Annotated[EvaluationDevice, typer.Option(help="Shared Validation inference device.")] = "cpu",
    split_bundle: Annotated[Path | None, typer.Option(help="Use a verified split bundle published with --export-detection.")] = None,
) -> None:
    if model is not None and all_models:
        raise typer.BadParameter("Select --model or --all.")
    if (model is not None or all_models) and until is None and not only:
        until = "report"
    options = None
    if start in DETECTION_STAGES or until in DETECTION_STAGES or set(only or ()).intersection(DETECTION_STAGES):
        options = DetectorOptions(model, all_models, config_dir, evaluation_image_size, evaluation_device)
    try:
        runner, detectors, _ = _pipeline(
            config, run_id, approve_download, approve_train,
            detector_options=options, approve_detection=approve_detection_train,
            split_bundle=split_bundle,
        )
        if (
            not only and start is None and CLASSIFICATION_STAGES[0] in runner.graph.stages
            and (until is None or until in CLASSIFICATION_STAGES)
        ):
            prepare, _, evaluate = CLASSIFICATION_STAGES
            acquisition = ["download", "extract"] if runner.graph.stages[prepare].dependencies else []
            only = [*acquisition, *CLASSIFICATION_STAGES[:CLASSIFICATION_STAGES.index(until or evaluate) + 1]]
            until = None
        elif not only and "validate_detection" in runner.graph.stages:
            order = runner.graph.order
            preparation = order[:order.index("validate_detection") + 1]
            if start is None and split_bundle is not None:
                start = "validate_detection"
            elif start is None and runner.settings["dataset_source"] == "skip" and until not in {"download", "extract"}:
                start = "audit"
            if until is None and (start is None or start in preparation):
                until = "validate_detection"
            elif until is None and start in DETECTION_STAGES:
                until = "report"
        selection = runner.graph.select(start, until, only or ())
        if (dry_run or approve_train) and CLASSIFICATION_STAGES[1] in selection:
            typer.echo(f"Classification training   CPU   artifacts under {runner.runs_dir}")
        if dry_run:
            for step in runner.plan(selection, run_id=run_id, force=force):
                typer.echo(f"{step.stage}   {step.action}")
            if detectors and "train" in selection:
                detectors.describe_training(
                    "the verified detection conversion", runner.runs_dir / (run_id or "<id>") / "artifacts/train",
                )
            return
        run_id = uuid4().hex if run_id is None else run_id
        if valid_id(run_id):
            typer.echo(f"Run ID   {run_id}")
        _print_completed(runner, runner.run(run_id, selection, force=force))
    except (PipelineError, SetupError, DataError, TrainingError, OSError, ValueError) as exc:
        _pipeline_failure(exc)
    except KeyboardInterrupt:
        typer.echo("Run cancelled. Use ict status and ict resume to continue.", err=True)
        raise typer.Exit(code=130) from None


@app.command(help="Resume the last selection for a saved run and verify its cached results.")
def resume(
    run_id: RunIdOption = None, config: ConfigOption = None, approve_download: ApprovalOption = False,
    approve_train: ClassificationApproval = False,
    approve_detection_train: DetectionApproval = False,
) -> None:
    try:
        runner, _, saved = _pipeline(
            config, run_id, approve_download, approve_train, resume=True, approve_detection=approve_detection_train,
        )
        assert saved is not None
        _print_completed(runner, runner.resume(saved.run_id, expected_updated_at=saved.updated_at))
    except (PipelineError, SetupError, DataError, TrainingError, OSError, ValueError) as exc:
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
    from .models.weights import training_scope
    from .training_storage import storage_estimates

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
            typer.echo(plan.credentials.redact_text(json.dumps(training_scope(name, config, plan.weight_sources.get(name)))))
        for estimate in storage_estimates(plan):
            typer.echo(plan.credentials.redact_text(json.dumps({"storage": asdict(estimate)})))
        if dry_run:
            return
        if not approve_train:
            raise TrainingError("Training requires explicit approval. Review the scope above, then pass --approve-train.")
        for name, checkpoints in execute_training(plan, progress):
            typer.echo(f"{name} completed   best {checkpoints.best}   last {checkpoints.last}")
            typer.echo(f"Model run   {plan.destination / name}")
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


@app.command(help="Preview or remove an explicitly selected managed pytest session.")
def clean(
    root: Annotated[Path | None, typer.Option(help="Parent used by pytest --ict-artifact-root.")] = None,
    older_than_hours: Annotated[float, typer.Option(min=0, help="Minimum age of an eligible session.")] = 24,
    dry_run: Annotated[bool, typer.Option(help="Preview managed sessions. This is the default.")] = False,
    delete: Annotated[str | None, typer.Option(help="Exact session ID from a preview.")] = None,
    confirm: Annotated[bool, typer.Option(help="Authorize deletion of the specified session ID.")] = False,
) -> None:
    from math import isfinite

    from .test_artifacts import artifact_root, delete_session, preview_cleanup

    if not isfinite(older_than_hours):
        raise typer.BadParameter("Provide a finite minimum age.")
    if dry_run and delete or confirm and not delete:
        raise typer.BadParameter("Use preview or specify --delete ID --confirm.")
    try:
        managed = artifact_root(root)
        candidates = list(preview_cleanup(managed, age_hours=older_than_hours))
        for item in candidates:
            size = f"{item.logical_bytes} logical bytes" if item.logical_bytes is not None else "size unavailable"
            typer.echo(f"{item.path} | {item.owner} | {item.status} | {size} | {item.detail}")
        if not candidates:
            typer.echo(f"No registered sessions under {managed}.")
        if delete:
            if not confirm:
                raise PipelineError("Review the preview and pass --delete ID --confirm to authorize that session.")
            delete_session(managed, delete, age_hours=older_than_hours)
            typer.echo(f"Removed session {delete}.")
    except (OSError, PipelineError, ValueError) as exc:
        detail = str(exc) if isinstance(exc, PipelineError) else "Check access to the managed test artifact root."
        typer.echo(f"Cleanup failed. {detail}", err=True)
        raise typer.Exit(code=1) from None
