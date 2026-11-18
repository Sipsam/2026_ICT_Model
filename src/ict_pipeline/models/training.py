"""Plan and execute new per-model runs with reproducible metadata."""

import json
import math
import os
import platform
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from importlib import import_module
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
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
from ..environment.monitor import logical_file_bytes, observe_training
from ..pipeline.model import PipelineError
from ..pipeline.store import RunStore, digest_json, fingerprint_path, timestamp
from ..reporting.attempt import track_attempt
from ..training_storage import (
    CHECKPOINT_ALLOWANCE, cache_root, check_training_storage, guard_training_write, storage_estimates,
)
from .contracts import Checkpoints, ModelAdapter, ModelName, Progress, TrainConfig, TrainingError, TrainingRequest
from .dataset import DetectionData, inspect_data, verify_shared_labels
from .registry import REGISTRY, failure_message, resolve_device
from .weights import VerifiedWeights, checkpoint_source, verify_weights, weight_libraries, weight_packages


# Pretrained support changes the code fingerprint even for scratch. Recognize
# only the exact predecessor adapter sources for legacy scratch resumption.
_LEGACY_SCRATCH_CHANGES = ({
    "coco.py": "8c4073b7f68327ff05ac20d2016162f6569826ee1f9abc2ea3030e3317a1194b",
    "contracts.py": "0bad004705a8a4c4a3b9656fb8eae3554483c06a3c78d6fe07455c0acaa795b1",
    "training.py": "430971a746dd27e7c7b050d0be9117ab259ef4c1d29d9e34bf91b17690fd06d8",
    "yolo.py": "519c327610bb02215bb92b002dcfba34618faea33ef31222868bb8f3b6260017",
}, {
    "coco.py": "34c02598c9f901889bca450e39cf062de2093da0cacc023f9db732c3b69f8ac1",
    "contracts.py": "132e0576ccdc562ba3ef87c0c289370a9d14c184c50cb96a007faaa12c4802e8",
    "training.py": "140d7901dc6483b4fc3043279d81096db155ee812126077977cf0043c012dd8b",
    "yolo.py": "99a8d4c91ae247ca918d4d013c843af02c1dc43f44d84e8442e3467c9265df94",
}, {
    "coco.py": "de8b4a410c93a89469ee79c3e7c0dd026f9e738cdd2ac3327e0a654eff8e5d30",
    "contracts.py": "18cc503b9c0b8ae0401ab5b6e6a30972fd2e63c0992b80c0ea6cd96eee02bb4c",
    "training.py": "f16471a377cb640dee43ebff8c27e182a14366be42bd4f5edc8661b025ee5c7f",
    "yolo.py": "91971c5ff40ad52fc46e4e98b161413518de028c36af99a5db8058197f97a781",
})

# Exact predecessors for storage-only changes, including Windows checkout bytes.
STORAGE_PREDECESSORS = ({
    "coco.py": "bfba3786944e3b1afb2d61b2d78bfd4bbc9a650a7c60286675b48ba3a7ad6059",
    "contracts.py": "8cddd503fc88c92d7c2168446db6fd2e09cac278efcf0c838cf8f855a7b9c7a4",
    "training.py": "dc8b9c6678f41b3c97a18d4cc2c64baa5700ad4b20ab8999e918ff14e55d8173",
    "yolo.py": "5a49c5005b6cb24a1c3e0f74d56f29b5f9b76099a4857f40ee27e4146b21eb09",
}, {
    "coco.py": "5f79b10022329e47b9a21fa7d490f8a8691a852160ca9bbfc4682d344130d9d9",
    "contracts.py": "2e1adc99a96bbe958ba9f203fb7d629f52d0de154906e54d47e1a64e1de69232",
    "training.py": "434177bd7d09fa03f65a83058ef65e468a6daa3e15b605ad6970d52dff27e934",
    "yolo.py": "7705417465f9bb5faadc17b06799967505901eaf12938fc05b2675fff78e6559",
}, {
    "coco.py": "3cce1555978745da9e619e8ff994bd5569a36f46934cb8e6ab27470f2715f134",
    "contracts.py": "af490150a21f1579ce9641ef499e5d3c040e12ec8d7bf45edbea177488fbd61b",
    "training.py": "66456f5c256f832cee32faadcde9530d12d1f07be37a1df8c81a0c2f9014b023",
    "yolo.py": "ef4f03d3c6aa8ffa8be604f66b5a78df6c0598fafac754a347eae6b7824dd732",
})


# Previous checkpoints are compatible only with the original worker and accumulation defaults.
ADAPTIVE_PREDECESSORS = ({
    "contracts.py": "3c6588f5de32b14833fbe88676e75882d6a6470af57597a3426a233518ddd226",
    "training.py": "42186b1de4503213c0c481439af4915622f333b9c701bdca019070dd06cee078",
    "coco.py": "6a43e47fb1452b7a9863e417c53910840ad9d5d15330a948027d94cca56e07e0",
    "yolo.py": "0fea6e9c2a92ab18dd3178b0caa005b47b5835149c2ad9b9c228f4a06ffb03ef",
}, {
    "contracts.py": "d2e52436b2b8af7ad3b914562d88a3705c3927264b56e19ae43f235abd2c08d1",
    "training.py": "42186b1de4503213c0c481439af4915622f333b9c701bdca019070dd06cee078",
    "coco.py": "779a32bfd80ac5f9a8151d825bd0bb44bb2e7d870fb324c88bb5f23999443b03",
    "yolo.py": "458310a5aa01eae8c05784236937dc3d1e1dd608974b4fb3e9878dfbaa3b92d2",
}, {
    "contracts.py": "0fcce5ac483dc89e1056f98c8b6a4f7c3f068b3cf643c37febf71328c26d8d48",
    "training.py": "82bec46704fac2ddc3750a6b9b6092c7bc510eb0d374b7f01010538d7e9974dd",
    "coco.py": "48ef96f951b7c8f2bcc54e82293fb9ce177628f7f6ba65da9ccfc18b8735e179",
    "yolo.py": "7f3baab68205a62021891f08ed0b1a1fb665e62ad935a69352dbc178e133a576",
},)

ADAPTIVE_FIELDS = {"workers", "gradient_accumulation"}


def compatible_predecessors(config: TrainConfig) -> tuple[dict[str, str], ...]:
    if config.workers == 0 and config.gradient_accumulation == 1:
        return (*STORAGE_PREDECESSORS, *ADAPTIVE_PREDECESSORS)
    return ()


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
    weight_sources: Mapping[ModelName, Path] = field(default_factory=dict)
    publish_checkpoints: bool = False
    monitor_resources: bool = False


def _validate_adapter_options(name: ModelName, config: TrainConfig, device: str) -> None:
    if name == ModelName.YOLO:
        if config.gradient_accumulation != 1:
            raise TrainingError("YOLO uses its native accumulation schedule. Remove gradient_accumulation.")
        if config.workers and device in ("cpu", "mps"):
            raise TrainingError("YOLO forces zero DataLoader workers on CPU and MPS.")


def read_training_profile(
    name: ModelName, directory: Path | None, device: str | None = None,
) -> tuple[TrainConfig, str]:
    try:
        source = files(__package__) if directory is None else directory
        contents = source.joinpath(f"{name}.yaml").read_bytes()
        values = yaml.safe_load(contents.decode("utf-8-sig"))
        if device is not None and isinstance(values, dict):
            values["device"] = device
        config = TrainConfig.model_validate(values)
        _validate_adapter_options(name, config, config.device)
        return config, sha256(contents).hexdigest()
    except ValidationError as exc:
        fields = [
            ".".join(map(str, error["loc"])) if error["loc"] and error["loc"][0] in TrainConfig.model_fields
            else str(error["msg"]) if not error["loc"] else "unsupported model field"
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
    weights = {
        name: path for name, config in models.items()
        if (path := checkpoint_source(name, config, directory, credentials)) is not None
    }
    for name in names:
        source = dataset / REGISTRY[name].data_format
        if not source.is_dir() or not source.resolve().is_relative_to(dataset):
            raise TrainingError("Generated detection data is missing. Pass --dataset with the detection conversion directory.")
    return TrainingPlan(models, dataset, destination, resume_from, credentials, workspace, weight_sources=weights)


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
    name: ModelName, source: Path, folder: Path, expected: set[str], *, verified: bool,
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
            if metadata.get("run_fingerprint") not in expected:
                raise TrainingError("Resume requires the original dataset, initialization, config, device, code and library versions.")
            paths = checkpoint_paths(name, source)
            if not paths.last.is_file():
                raise TrainingError("The previous run has no last checkpoint. Start a fresh run.")
            records = _records(source, paths)
            if (verified or metadata.get("checkpoints")) and metadata.get("checkpoints") != records:
                raise TrainingError("A saved checkpoint has changed since the previous run ended.")
            for path in (paths.last, paths.best):
                if path.is_file():
                    guard_training_write(folder, path.stat().st_size)
                    copyfile(path, folder / path.name)
            return Checkpoints(folder / "best.pt", folder / "last.pt"), metadata
        except (OSError, ValueError, TypeError, AttributeError):
            raise TrainingError("Cannot read the previous run or checkpoints. Check the run directory and storage.") from None


def execute_training(plan: TrainingPlan, progress: Progress) -> Iterator[tuple[ModelName, Checkpoints]]:
    estimates = storage_estimates(plan)
    check_training_storage(estimates)
    plan.destination.mkdir(parents=True, exist_ok=False)
    store = RunStore(plan.destination.parent, plan.destination.name, plan.credentials)
    configuration = {
        "models": {name.value: config.model_dump(mode="json") for name, config in plan.models.items()},
        "dataset": str(plan.dataset), "resume_from": str(plan.resume_from) if plan.resume_from else None,
    }
    packages = tuple(
        package for name, config in plan.models.items() for package in REGISTRY[name].libraries + weight_packages(config)
    )
    with store.lock(), ExitStack() as attempt:
        attempt.enter_context(track_attempt(
            store, plan.workspace, "training", configuration, packages, (TrainingError, SetupError, PipelineError),
        ))
        store.log({"time": timestamp(), "stage": "training preflight", "storage": [asdict(item) for item in estimates]})
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
        if plan.dataset_revision is not None and plan.dataset_revision.get("selected_split", seal) != seal:
            raise TrainingError("The verified split selection changed before training.")
        revision = {**(plan.dataset_revision or {}), "selected_split": seal}
        plan = replace(plan, dataset_revision=revision)
    elif plan.dataset_revision is not None and "selected_split" in plan.dataset_revision:
        raise TrainingError("The verified split seal is missing.")
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
        weights = verify_weights(name, config, plan.weight_sources.get(name), library_versions)
        try:
            torch = import_module("torch")
            device = resolve_device(torch, config, backend)
            _validate_adapter_options(name, config, device)
            adapter = backend.load()
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(f"{name}. {failure_message(exc, backend)}") from None
        if weights is not None and weights.spec.architecture != adapter.architecture:
            raise TrainingError("The installed adapter architecture differs from the verified checkpoint layout.")
        runtimes[name] = (device, {**library_versions, **weight_libraries(config)}, adapter, weights)

    for name in plan.models:
        data_format = REGISTRY[name].data_format
        if data_format not in datasets:
            datasets[data_format] = inspect_data(plan.dataset, data_format, plan.credentials, progress)
    fingerprints = [dataset.split_fingerprints for dataset in datasets.values()]
    if any(value != fingerprints[0] for value in fingerprints):
        raise TrainingError("YOLO and COCO Train/Validation images differ. Rerun conversion and detection validation.")

    code = {source.name: fingerprint_path(source) for source in Path(__file__).parent.glob("*.py")}
    for name, config in plan.models.items():
        device, library_versions, adapter, weights = runtimes[name]
        yield name, _train_model(
            name, config, plan, datasets[REGISTRY[name].data_format], device, library_versions, adapter, code, progress,
            weights,
        )


def _train_model(
    name: ModelName, config: TrainConfig, plan: TrainingPlan, data: DetectionData, device: str,
    library_versions: dict[str, str], adapter: ModelAdapter, code: dict[str, str], progress: Progress,
    weights: VerifiedWeights | None,
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
        "best_criterion": backend.best_criterion,
    }
    if plan.dataset_revision is not None:
        metadata.update(dataset_revision=plan.dataset_revision, profile_fingerprint=plan.profile_fingerprint)
    resume_fingerprints = set()
    legacy_options = config.workers == 0 and config.gradient_accumulation == 1
    if weights is None and legacy_options:
        old_config = config.model_dump(exclude={
            "initialization", "checkpoint", "freeze_backbone", "fine_tuning", *ADAPTIVE_FIELDS,
        })
        old_code = {key: value for key, value in code.items() if key != "weights.py"}
        for changes in _LEGACY_SCRATCH_CHANGES:
            resume_fingerprints.add(digest_json({
                **metadata, "configuration": old_config, "code_fingerprints": {**old_code, **changes},
            }))
    if weights is not None:
        metadata["pretrained_weights"] = {
            **weights.spec.model_dump(mode="json"), "source_classes": weights.source_classes,
            "file_sha256": weights.spec.sha256,
        }
    metadata["initialization_mode"] = config.initialization
    run_fingerprint = digest_json(metadata)
    if plan.resume_from and legacy_options:
        old_metadata = {**metadata, "configuration": config.model_dump(exclude=ADAPTIVE_FIELDS)}
        resume_fingerprints.update(
            digest_json({**old_metadata, "code_fingerprints": {**code, **changes}})
            for changes in compatible_predecessors(config)
        )
    resume_fingerprints.add(run_fingerprint)
    metadata.update({
        "run_fingerprint": run_fingerprint,
        "initialization": "resume" if plan.resume_from else ("pretrained" if weights else "random"),
        "status": "running", "started_at": timestamp(), "dataset_path": str(plan.dataset),
        "resume_from": str(plan.resume_from) if plan.resume_from else None, "checkpoints": {},
    })
    output.mkdir(parents=True, exist_ok=False)
    started = monotonic()
    checkpoints = checkpoint_paths(name, output)
    verified_resume = plan.dataset_revision is not None or weights is not None
    store = RunStore(output.parent, name, plan.credentials)
    scratch = cache_root(plan) if name == ModelName.YOLO else None
    resources: dict[str, object] = {}
    overhead: dict[str, float] = {}
    epoch_metrics: list[object] = []
    if plan.monitor_resources:
        metadata.update(resources=resources, overhead=overhead, epoch_metrics=epoch_metrics)
    with store.lock():
        last_logged = 0.0
        last_storage_check = float("-inf")

        def report(stage: str, done: int, total: int) -> None:
            nonlocal last_logged, last_storage_check
            now = monotonic()
            if now - last_storage_check >= 1:
                guard_training_write(output)
                if scratch is not None:
                    guard_training_write(scratch)
                last_storage_check = now
            if done in (0, total) or now - last_logged >= 1:
                store.log({
                    "time": timestamp(), "stage": stage, "completed": done, "total": total,
                    "elapsed_seconds": round(now - started, 3),
                })
                last_logged = now
            progress(f"{name} {plan.credentials.redact_text(stage)}", done, total)

        def check_checkpoint_space() -> None:
            measured = checkpoints.last.stat().st_size if checkpoints.last.is_file() else 0
            copies = 2 if name == ModelName.YOLO else 1
            guard_training_write(output, copies * max(measured, CHECKPOINT_ALLOWANCE[name]))

        def record_metrics(values: Mapping[str, float]) -> None:
            metrics = {key: value for key, value in values.items() if math.isfinite(value)}
            metadata["metrics"] = plan.credentials.redact(metrics)
            if plan.monitor_resources and "epoch" in metrics:
                epoch_metrics.append(metadata["metrics"])
            if verified_resume and "epoch" in metrics:
                metadata["checkpoints"] = _records(output, checkpoints)
                metadata["checkpoint_epoch"] = metrics["epoch"]
            store.write_json("run.json", metadata)

        def record_transfer(result: Mapping[str, object]) -> None:
            metadata["weight_transfer"] = plan.credentials.redact(dict(result))
            store.write_json("run.json", metadata)

        def record_overhead(key: str, seconds_or_bytes: float) -> None:
            overhead[key] = overhead.get(key, 0) + seconds_or_bytes

        store.write_json("run.json", metadata)
        try:
            with ExitStack() as staged:
                resume = None
                saved: dict[str, object] = {}
                if plan.resume_from:
                    folder = Path(staged.enter_context(TemporaryDirectory(prefix="resume-", dir=output)))
                    resume, saved = _resume_checkpoint(
                        name, plan.resume_from, folder, resume_fingerprints, verified=verified_resume,
                    )
                    if weights is not None and isinstance(saved.get("weight_transfer"), dict):
                        record_transfer(saved["weight_transfer"])
                    if plan.monitor_resources and isinstance(saved.get("epoch_metrics"), list):
                        epoch_metrics.extend(saved["epoch_metrics"])
                report(f"starting on {device}", 0, config.epochs)
                if resume and verified_resume and saved.get("checkpoint_epoch") == config.epochs:
                    guard_training_write(output, resume.best.stat().st_size + resume.last.stat().st_size)
                    checkpoints.last.parent.mkdir(parents=True, exist_ok=True)
                    copyfile(resume.best, checkpoints.best)
                    copyfile(resume.last, checkpoints.last)
                    metadata.update(metrics=saved["metrics"], checkpoint_epoch=saved["checkpoint_epoch"])
                    result = checkpoints
                else:
                    if plan.monitor_resources:
                        staged.enter_context(observe_training(device, resources))
                    result = adapter.train(TrainingRequest(
                        config, data, output, device, report, resume, record_metrics, weights, record_transfer,
                        scratch, check_checkpoint_space, record_overhead if plan.monitor_resources else None,
                    ))
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
            if plan.monitor_resources:
                try:
                    resources["checkpoint_logical_bytes"] = logical_file_bytes(checkpoints.last.parent.glob("*.pt"))
                except OSError:
                    resources["checkpoint_logical_bytes"] = None
            metadata["finished_at"] = timestamp()
            metadata["elapsed_seconds"] = round(monotonic() - started, 3)
            if not verified_resume or metadata["status"] == "completed":
                metadata["checkpoints"] = _records(output, checkpoints)
            store.write_json("run.json", metadata)
    return result
