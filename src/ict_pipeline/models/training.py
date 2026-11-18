"""Plan and execute new per-model runs with reproducible metadata."""

import json
import math
import os
import platform
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass, replace
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from shutil import copyfile
from time import monotonic
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import ValidationError

from ..config.settings import Credentials, SetupError, is_filesystem_path
from ..config.storage import SetupFiles
from ..data.preprocessing.common import require_separate_output
from ..data.preprocessing.console import protect_source_data
from ..data.preprocessing.policy import label_policy
from ..data.preprocessing.revision import read_dataset_seal
from ..pipeline.model import PipelineError
from ..pipeline.store import RunStore, digest_json, fingerprint_path, timestamp
from ..reporting.attempt import track_attempt
from .contracts import Checkpoints, ModelAdapter, ModelName, Progress, TrainConfig, TrainingError, TrainingRequest
from .dataset import DetectionData, inspect_data, verify_shared_labels
from .registry import REGISTRY, failure_message, resolve_device


@dataclass(frozen=True)
class TrainingPlan:
    models: dict[ModelName, TrainConfig]
    dataset: Path
    destination: Path
    resume_from: Path | None
    credentials: Credentials
    workspace: Path
    dataset_revision: Mapping[str, object] | None = None
    profile_fingerprint: str | None = None
    verified_data: Mapping[str, DetectionData] | None = None


def read_training_profile(
    name: ModelName, directory: Path | None, device: str | None = None,
) -> tuple[TrainConfig, str]:
    try:
        source = files(__package__) if directory is None else directory
        contents = source.joinpath(f"{name}.yaml").read_bytes()
        values = yaml.safe_load(contents.decode("utf-8-sig"))
        if device is not None and isinstance(values, dict):
            values["device"] = device
        return TrainConfig.model_validate(values), sha256(contents).hexdigest()
    except ValidationError as exc:
        fields = [
            str(error["loc"][0]) if error["loc"] and error["loc"][0] in TrainConfig.model_fields
            else "unsupported or missing mapping"
            for error in exc.errors(include_input=False, include_context=False, include_url=False)
        ]
        raise TrainingError(f"Invalid {name} YAML fields. " + ", ".join(fields)) from None
    except (OSError, ValueError, yaml.YAMLError, RecursionError):
        raise TrainingError(f"Cannot read {name}.yaml. Check the config directory and YAML syntax.") from None


def plan_training(
    workspace: Path, names: tuple[ModelName, ...], dataset: Path, runs: Path,
    config_dir: Path | None, resume_from: Path | None, *,
    device: Literal["cpu", "cuda", "rocm", "mps"] | None = None,
) -> TrainingPlan:
    credentials = SetupFiles.read(workspace, None).credentials(os.environ)

    def local(path: Path, label: str) -> Path:
        if not is_filesystem_path(str(path)):
            raise TrainingError(f"Use a local filesystem path for {label}.")
        credentials.protect_path(str(path), label)
        resolved = (workspace / path).resolve()
        credentials.protect_path(str(resolved), label)
        return resolved

    dataset = local(dataset, "detection dataset")
    destination = local(runs, "runs directory") / uuid4().hex
    directory = local(config_dir, "config directory") if config_dir else None
    resume_from = local(resume_from, "resume run") if resume_from else None
    protect_source_data(destination, workspace)
    require_separate_output(dataset, destination)
    if resume_from is not None:
        if len(names) != 1 or not REGISTRY[names[0]].supports_resume:
            raise TrainingError("Resume requires one model with checkpoint resumption support.")
        if resume_from.name != names[0] or not (resume_from / "run.json").is_file():
            raise TrainingError("Select the model directory of an earlier ICT training run.")
        require_separate_output(resume_from, destination)
    models = {name: read_training_profile(name, directory, device)[0] for name in names}
    for name in names:
        source = dataset / REGISTRY[name].data_format
        if not source.is_dir() or not source.resolve().is_relative_to(dataset):
            raise TrainingError("Generated detection data is missing. Pass --dataset with the detection conversion directory.")
    return TrainingPlan(models, dataset, destination, resume_from, credentials, workspace)


def checkpoint_paths(name: ModelName, output: Path) -> Checkpoints:
    directory = output / "weights" if name == "yolo" else output
    return Checkpoints(directory / "best.pt", directory / "last.pt")


def _records(output: Path, checkpoints: Checkpoints) -> dict[str, dict[str, str]]:
    records = {}
    for key, path in (("best", checkpoints.best), ("last", checkpoints.last)):
        if path.is_file():
            records[key] = {"path": path.relative_to(output).as_posix(), "fingerprint": fingerprint_path(path)}
    return records


def _resume_checkpoint(
    name: ModelName, source: Path, output: Path, expected: str, *, verified: bool,
) -> tuple[Checkpoints, dict[str, object]]:
    # The original run remains intact. Hold its lock while checking/copying checkpoints.
    with RunStore(source.parent, source.name).lock():
        try:
            metadata = json.loads((source / "run.json").read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != 1 or metadata.get("model") != name
                or metadata.get("status") not in {"running", "failed", "interrupted"}
            ):
                raise TrainingError("Resume requires an unfinished run from the same adapter.")
            if metadata.get("run_fingerprint") != expected:
                raise TrainingError("Resume requires the original dataset, config, device, code and library versions.")
            paths = checkpoint_paths(name, source)
            if not paths.last.is_file():
                raise TrainingError("The previous run has no last checkpoint. Start a fresh run.")
            records = _records(source, paths)
            if (verified or metadata.get("checkpoints")) and metadata.get("checkpoints") != records:
                raise TrainingError("A saved checkpoint has changed since the previous run ended.")
            folder = output / "resume"
            folder.mkdir()
            for path in (paths.last, paths.best):
                if path.is_file():
                    copyfile(path, folder / path.name)
            return Checkpoints(folder / "best.pt", folder / "last.pt"), metadata
        except (OSError, ValueError, TypeError, AttributeError):
            raise TrainingError("Cannot read the previous run or checkpoints. Check the run directory and storage.") from None


def execute_training(plan: TrainingPlan, progress: Progress) -> Iterator[tuple[ModelName, Checkpoints]]:
    plan.destination.mkdir(parents=True, exist_ok=False)
    store = RunStore(plan.destination.parent, plan.destination.name, plan.credentials)
    configuration = {
        "models": {name.value: config.model_dump(mode="json") for name, config in plan.models.items()},
        "dataset": str(plan.dataset), "resume_from": str(plan.resume_from) if plan.resume_from else None,
    }
    packages = tuple(package for name in plan.models for package in REGISTRY[name].libraries)
    with store.lock(), ExitStack() as attempt:
        attempt.enter_context(track_attempt(
            store, plan.workspace, "training", configuration, packages, (TrainingError, SetupError, PipelineError),
        ))
        store.log({"time": timestamp(), "stage": "training preflight"})
        for index, result in enumerate(_execute_training(plan, progress), 1):
            if index == len(plan.models):
                attempt.close()
            yield result


def _execute_training(plan: TrainingPlan, progress: Progress) -> Iterator[tuple[ModelName, Checkpoints]]:
    datasets = dict(plan.verified_data or {})
    try:
        seal = read_dataset_seal(plan.dataset)
    except ValueError as exc:
        raise TrainingError(str(exc)) from None
    if seal is not None:
        revision = {**(plan.dataset_revision or {}), "selected_split": seal}
        plan = replace(plan, dataset_revision=revision)
    if seal is not None or all((plan.dataset / fmt).is_dir() for fmt in ("yolo", "coco")):
        for data_format in ("yolo", "coco"):
            if data_format not in datasets:
                datasets[data_format] = inspect_data(plan.dataset, data_format, plan.credentials, progress)
        if plan.verified_data is None and seal is None:
            verify_shared_labels(datasets["yolo"], datasets["coco"])
    runtimes = {}
    for name, config in plan.models.items():
        backend = REGISTRY[name]
        library_versions = backend.versions()
        try:
            torch = import_module("torch")
            device = resolve_device(torch, config, backend)
            adapter = backend.load()
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(f"{name}. {failure_message(exc, backend)}") from None
        runtimes[name] = (device, library_versions, adapter)

    for name in plan.models:
        data_format = REGISTRY[name].data_format
        if data_format not in datasets:
            datasets[data_format] = inspect_data(plan.dataset, data_format, plan.credentials, progress)
    fingerprints = [dataset.split_fingerprints for dataset in datasets.values()]
    if any(value != fingerprints[0] for value in fingerprints):
        raise TrainingError("YOLO and COCO Train/Validation images differ. Rerun conversion and detection validation.")

    code = {source.name: fingerprint_path(source) for source in Path(__file__).parent.glob("*.py")}
    for name, config in plan.models.items():
        device, library_versions, adapter = runtimes[name]
        yield name, _train_model(
            name, config, plan, datasets[REGISTRY[name].data_format], device, library_versions, adapter, code, progress,
        )


def _train_model(
    name: ModelName, config: TrainConfig, plan: TrainingPlan, data: DetectionData, device: str,
    library_versions: dict[str, str], adapter: ModelAdapter, code: dict[str, str], progress: Progress,
) -> Checkpoints:
    backend = REGISTRY[name]
    output = plan.destination / name
    metadata: dict[str, object] = {
        "schema_version": 1, "model": name, "architecture": adapter.architecture,
        "configuration": config.model_dump(mode="json"), "device": device,
        "dataset_format": backend.data_format, "split_fingerprints": data.split_fingerprints,
        "annotation_fingerprint": data.annotation_fingerprint, "label_policy": label_policy(),
        "split_annotation_fingerprints": data.split_annotation_fingerprints,
        "libraries": library_versions, "ict_pipeline": version("ict-pipeline"), "code_fingerprints": code,
        "python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
        "checkpoint_format": backend.checkpoint_format, "library_license": backend.license,
        "best_criterion": backend.best_criterion, "initialization": "resume" if plan.resume_from else "random",
    }
    if plan.dataset_revision is not None:
        metadata.update(dataset_revision=plan.dataset_revision, profile_fingerprint=plan.profile_fingerprint)
    # Initialization origin and workspace paths are descriptive, so moving identical input data
    # or resuming an already resumed attempt does not change the compatibility fingerprint.
    metadata["run_fingerprint"] = digest_json({k: v for k, v in metadata.items() if k != "initialization"})
    metadata.update({
        "status": "running", "started_at": timestamp(), "dataset_path": str(plan.dataset),
        "resume_from": str(plan.resume_from) if plan.resume_from else None, "checkpoints": {},
    })
    output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    checkpoints = checkpoint_paths(name, output)
    store = RunStore(output.parent, name, plan.credentials)
    with store.lock():
        last_logged = 0.0

        def report(stage: str, done: int, total: int) -> None:
            nonlocal last_logged
            now = monotonic()
            if done in (0, total) or now - last_logged >= 1:
                store.log({
                    "time": timestamp(), "stage": stage, "completed": done, "total": total,
                    "elapsed_seconds": round(now - started, 3),
                })
                last_logged = now
            progress(f"{name} {plan.credentials.redact_text(stage)}", done, total)

        def record_metrics(values: Mapping[str, float]) -> None:
            metrics = {key: value for key, value in values.items() if math.isfinite(value)}
            metadata["metrics"] = plan.credentials.redact(metrics)
            if plan.dataset_revision is not None and "epoch" in metrics:
                metadata["checkpoints"] = _records(output, checkpoints)
                metadata["checkpoint_epoch"] = metrics["epoch"]
            store.write_json("run.json", metadata)

        store.write_json("run.json", metadata)
        try:
            resume = None
            saved: dict[str, object] = {}
            if plan.resume_from:
                resume, saved = _resume_checkpoint(
                    name, plan.resume_from, output, str(metadata["run_fingerprint"]),
                    verified=plan.dataset_revision is not None,
                )
            report(f"starting on {device}", 0, config.epochs)
            if resume and plan.dataset_revision is not None and saved.get("checkpoint_epoch") == config.epochs:
                checkpoints.last.parent.mkdir(parents=True, exist_ok=True)
                copyfile(resume.best, checkpoints.best)
                copyfile(resume.last, checkpoints.last)
                metadata.update(metrics=saved["metrics"], checkpoint_epoch=saved["checkpoint_epoch"])
                result = checkpoints
            else:
                result = adapter.train(TrainingRequest(config, data, output, device, report, resume, record_metrics))
            if result != checkpoints or len(_records(output, result)) != 2:
                raise TrainingError("Training did not produce both best and last checkpoints.")
            if plan.dataset_revision is not None:
                expected = plan.dataset_revision
                try:
                    if ("detection" in expected and fingerprint_path(plan.dataset) != expected["detection"]
                            or "selected_split" in expected and read_dataset_seal(plan.dataset) != expected["selected_split"]):
                        raise ValueError
                except (ValueError, OSError):
                    raise TrainingError("The verified detection dataset changed during training. Restore stable inputs.") from None
            metadata["status"] = "completed"
        except KeyboardInterrupt:
            metadata["status"] = "interrupted"
            raise
        except Exception as exc:
            metadata["status"] = "failed"
            message = str(exc) if isinstance(exc, TrainingError) else failure_message(exc, backend)
            message = plan.credentials.redact_text(message)
            metadata["error"] = message
            raise TrainingError(f"{name}. {message}") from None
        finally:
            metadata["finished_at"] = timestamp()
            metadata["elapsed_seconds"] = round(monotonic() - started, 3)
            if plan.dataset_revision is None or metadata["status"] == "completed":
                metadata["checkpoints"] = _records(output, checkpoints)
            store.write_json("run.json", metadata)
    return result
