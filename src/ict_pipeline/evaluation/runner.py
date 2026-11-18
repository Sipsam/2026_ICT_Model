"""Validate inputs, benchmark a shared split, and publish resumable reports."""

import hashlib
import json
import os
import platform
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from tempfile import NamedTemporaryFile
from time import monotonic, perf_counter
from typing import Any, Literal
from uuid import uuid4

from ..config.settings import Credentials, SetupError, is_filesystem_path
from ..config.storage import SetupFiles
from ..data.preprocessing.common import require_separate_output
from ..data.preprocessing.console import protect_source_data
from ..data.preprocessing.policy import label_policy
from ..models.contracts import ModelName, Progress, TrainConfig, TrainingError
from ..models.dataset import Split, read_coco_split
from ..models.registry import REGISTRY
from ..models.training import checkpoint_paths
from ..pipeline.model import PipelineError
from ..pipeline.store import RunStore, digest_json, fingerprint_path, timestamp
from ..reporting.attempt import track_attempt
from .adapters import PredictorFactory, load_predictor
from .contracts import MAX_DETECTIONS, EvaluationConfig, EvaluationError, EvaluationSplit, Prediction, Predictor
from .metrics import IOU_THRESHOLDS, detection_metrics

CheckpointKind = Literal["best", "last"]


@dataclass(frozen=True)
class Checkpoint:
    name: ModelName
    path: Path
    fingerprint: str
    config: TrainConfig
    split_fingerprints: dict[str, str]
    bytes: int


def _local(path: Path, workspace: Path, credentials: Credentials) -> Path:
    if not is_filesystem_path(str(path)):
        raise EvaluationError("Evaluation requires local filesystem paths.")
    credentials.protect_path(str(path), "evaluation path")
    resolved = (workspace / path).resolve()
    credentials.protect_path(str(resolved), "resolved evaluation path")
    return resolved


def _checkpoint(run: Path, kind: CheckpointKind, credentials: Credentials) -> Checkpoint:
    try:
        metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
        name = ModelName(metadata["model"])
        if (
            metadata["schema_version"] != 1 or metadata["status"] != "completed"
            or metadata["checkpoint_format"] != REGISTRY[name].checkpoint_format
            or digest_json(metadata["label_policy"]) != digest_json(label_policy())
        ):
            raise EvaluationError("Use a completed ICT training run with the canonical label policy.")
        path = getattr(checkpoint_paths(name, run), kind)
        record = metadata["checkpoints"][kind]
        if (
            record["path"] != path.relative_to(run).as_posix() or not path.is_file()
            or not path.resolve().is_relative_to(run)
        ):
            raise EvaluationError(f"The {kind} checkpoint is missing or has an invalid path.")
        credentials.protect_path(str(path.resolve()), "checkpoint path")
        fingerprint = fingerprint_path(path)
        if record["fingerprint"] != fingerprint:
            raise EvaluationError(f"The {kind} checkpoint fingerprint differs from the training record.")
        return Checkpoint(
            name, path, fingerprint, TrainConfig.model_validate(metadata["configuration"]),
            metadata["split_fingerprints"], path.stat().st_size,
        )
    except FileNotFoundError:
        raise EvaluationError("Training run metadata or checkpoint is missing. Pass a completed model directory.") from None
    except KeyError:
        raise EvaluationError(f"The training record is missing checkpoint or reproducibility metadata for {kind}.") from None
    except (ValueError, TypeError, OSError):
        raise EvaluationError("Cannot read the training run, configuration, or checkpoint.") from None


def _stat(path: Path) -> tuple[int, int, int, int, int]:
    info = path.stat()
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _split(
    root: Path, split: str, credentials: Credentials, progress: Progress, *, labels_required: bool,
) -> tuple[Split, dict[str, str], str, dict[Path, tuple[int, int, int, int, int]]]:
    directory = root / "coco"
    annotation = directory / "annotations" / f"instances_{split}.json"
    credentials.protect_path(str(annotation.resolve()), "COCO annotation path")
    if not annotation.resolve().is_relative_to(root):
        raise EvaluationError("COCO annotations must resolve inside the dataset.")
    before = _stat(annotation)
    labels = read_coco_split(directory, split)
    if not labels.samples or (labels_required and not any(sample.objects for sample in labels.samples)):
        raise EvaluationError("The selected COCO split requires images and ground-truth objects.")
    if len({sample.image.stem for sample in labels.samples}) != len(labels.samples):
        raise EvaluationError("The selected COCO split requires unique sample IDs.")
    snapshots = {annotation: before}
    hashes = {}
    stage = f"fingerprint {split} images"
    total = len(labels.images)
    progress(stage, 0, total)
    for index, image in enumerate(labels.images, 1):
        resolved = image.resolve()
        credentials.protect_path(str(image), "dataset image path")
        credentials.protect_path(str(resolved), "resolved dataset image path")
        if (
            not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}
            or not resolved.is_relative_to(root)
        ):
            raise EvaluationError("A COCO image is missing, unsupported, or outside the dataset.")
        snapshots[image] = _stat(image)
        with image.open("rb") as stream:
            hashes[image.name] = hashlib.file_digest(stream, "sha256").hexdigest()
        progress(stage, index, total)
    annotation_hash = fingerprint_path(annotation)
    _unchanged(snapshots)
    return labels, hashes, annotation_hash, snapshots


def _unchanged(snapshots: dict[Path, tuple[int, int, int, int, int]]) -> None:
    if any(_stat(path) != previous for path, previous in snapshots.items()):
        raise EvaluationError("Dataset inputs changed during evaluation. Rerun with stable inputs.")


def _benchmark(
    predictor: Predictor, labels: Split, config: EvaluationConfig, progress: Progress,
) -> tuple[dict[str, Prediction], dict[str, object]]:
    samples = sorted(labels.samples, key=lambda sample: sample.image.stem)
    progress("warmup", 0, config.warmup)
    for index in range(config.warmup):
        predictor.predict(samples[index % len(samples)])
        predictor.synchronize()
        progress("warmup", index + 1, config.warmup)
    predictions = {}
    durations = []
    progress("inference", 0, len(samples))
    for index, sample in enumerate(samples, 1):
        predictor.synchronize()
        started = perf_counter()
        prediction = predictor.predict(sample)
        predictor.synchronize()
        durations.append(perf_counter() - started)
        predictions[sample.image.stem] = prediction
        progress("inference", index, len(samples))

    durations.sort()
    return predictions, {
        "batch_size": 1, "timed_images": len(durations), "total_seconds": sum(durations),
        "fps": len(durations) / sum(durations),
        "latency_ms": {
            "mean": 1000 * mean(durations), "median": 1000 * median(durations),
            "p95": 1000 * durations[(95 * len(durations) + 99) // 100 - 1],
        },
        "hardware": predictor.hardware,
        "observed_input_sizes_wh": sorted({pred.preprocessing.input_size for pred in predictions.values()}),
    }


def _publish(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise EvaluationError("Evaluation output already exists. Choose a fresh output or resume.")
    with NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        partial = Path(stream.name)
        try:
            stream.write(json.dumps(payload, indent=2, allow_nan=False).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            partial.rename(path)
        finally:
            partial.unlink(missing_ok=True)


def read_report(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        result = document["result"]
        if (
            result["schema_version"] != 1 or result["status"] != "completed"
            or digest_json(result) != document["fingerprint"]
        ):
            raise ValueError
        return result
    except (OSError, ValueError, KeyError, TypeError):
        raise EvaluationError("An evaluation report is missing, incomplete, corrupt, or unsupported.") from None


def evaluate_runs(
    workspace: Path, runs: tuple[Path, ...], dataset: Path, split: EvaluationSplit, config: EvaluationConfig,
    output_root: Path, kind: CheckpointKind, held_out: bool, reference_dataset: Path | None,
    resume_from: Path | None, progress: Progress, factory: PredictorFactory = load_predictor,
) -> Iterator[Path]:
    if not runs or config.confidence < config.score_floor:
        raise EvaluationError("Supply model runs and use a confidence threshold at or above the score floor.")
    if split == EvaluationSplit.TEST and not held_out:
        raise EvaluationError("Test requires --held-out to attest independent selection and no tuning on Test.")
    if split == EvaluationSplit.VALIDATION and (held_out or reference_dataset is not None):
        raise EvaluationError("Held-out options apply only to Test.")

    credentials = SetupFiles.read(workspace, None).credentials(os.environ)
    dataset = _local(dataset, workspace, credentials)
    reference = _local(reference_dataset, workspace, credentials) if reference_dataset is not None else dataset
    sources = tuple(_local(run, workspace, credentials) for run in runs)
    output = (
        _local(resume_from, workspace, credentials) if resume_from is not None
        else _local(output_root, workspace, credentials) / uuid4().hex
    )
    protect_source_data(output, workspace)
    for source in (dataset, reference, *sources):
        require_separate_output(source, output)
    store = RunStore(output.parent, output.name, credentials)
    configuration = {
        **config.model_dump(mode="json"), "split": split.value, "dataset": str(dataset),
        "checkpoint": kind, "training_runs": [str(source) for source in sources],
        "held_out": held_out, "reference_dataset": str(reference) if reference_dataset is not None else None,
    }
    packages = ("torch", "torchvision", "ultralytics", "transformers", "numpy", "scipy")
    safe_errors = (EvaluationError, TrainingError, PipelineError, SetupError)
    if resume_from is not None and not (output / "evaluation.json").is_file():
        raise EvaluationError("Select a previous evaluation directory for --resume-from.")
    with store.lock(), ExitStack() as attempt:
        if resume_from is None:
            attempt.enter_context(track_attempt(store, workspace, "evaluation", configuration, packages, safe_errors))
            store.log({"time": timestamp(), "stage": "evaluation preflight"})

        checkpoints = [_checkpoint(source, kind, credentials) for source in sources]

        try:
            labels, hashes, annotation_hash, snapshots = _split(dataset, split, credentials, progress, labels_required=True)
            reference_hashes = {}
            if split == EvaluationSplit.TEST:
                for name in ("train", "val"):
                    _, other, _, reference_snapshots = _split(reference, name, credentials, progress, labels_required=False)
                    reference_hashes[name] = digest_json(other)
                    if hashes.keys() & other.keys() or set(hashes.values()) & set(other.values()):
                        raise EvaluationError("Test shares image filenames or bytes with Train or Validation.")
                    snapshots.update(reference_snapshots)
                if any(reference_hashes != checkpoint.split_fingerprints for checkpoint in checkpoints):
                    raise EvaluationError("Reference splits must match the recorded Train and Validation of every model.")
        except (OSError, ValueError, TypeError, KeyError):
            raise EvaluationError("Cannot read or validate the labeled COCO split. Check annotations and image paths.") from None

        dataset_record = {
            "split": split.value, "role": "held_out_test" if held_out else "validation_for_tuning",
            "held_out_attested": held_out, "image_count": len(labels.images), "image_fingerprint": digest_json(hashes),
            "annotation_fingerprint": annotation_hash, "reference_split_fingerprints": reference_hashes,
        }
        package = Path(__file__).parents[1]
        code = {path.relative_to(package).as_posix(): fingerprint_path(path)
                for part in ("evaluation", "models") for path in sorted((package / part).glob("*.py"))}
        protocol = {
            "ap": "COCO bbox all-area (non-crowd)", "ap_iou_thresholds": IOU_THRESHOLDS,
            "recall_points": 101, "max_detections_per_image_per_class": MAX_DETECTIONS,
            "configuration": config.model_dump(mode="json"), "label_policy": label_policy(),
        }
        records = [
            {"model": checkpoint.name.value, "checkpoint": {
                "path": str(checkpoint.path), "kind": kind, "fingerprint": checkpoint.fingerprint,
                "size_bytes": checkpoint.bytes,
            }, "training_configuration": checkpoint.config.model_dump(mode="json"),
             "training_split_fingerprints": checkpoint.split_fingerprints}
            for checkpoint in checkpoints
        ]
        request = {"dataset": dataset_record, "protocol": protocol, "code": code, "models": records}
        identity = digest_json(request)
        marker = output / "evaluation.json"
        if resume_from is None:
            _publish(marker, {"request_fingerprint": identity, **request})
        else:
            original = json.loads(marker.read_text(encoding="utf-8"))
            if (not isinstance(original, dict) or original.pop("request_fingerprint", None) != identity
                    or digest_json(original) != identity):
                raise EvaluationError("Resume requires the same checkpoints, split, configuration, and code.")
            attempt.enter_context(track_attempt(store, workspace, "evaluation", configuration, packages, safe_errors))
        progress(f"evaluation directory {output}", 0, 0)

        for index, (checkpoint, record) in enumerate(zip(checkpoints, records, strict=True), 1):
            target = output / f"{index:02d}-{checkpoint.name}.json"
            if target.exists():
                saved = read_report(target)
                if saved.get("request_fingerprint") != identity or saved.get("model_index") != index:
                    raise EvaluationError("A completed report does not match this evaluation request.")
                if index == len(checkpoints):
                    attempt.close()
                yield target
                continue

            started = monotonic()
            last_log = float("-inf")
            active_stage = "load checkpoint"

            def report(stage: str, done: int, total: int) -> None:
                nonlocal last_log, active_stage
                active_stage = stage
                now = monotonic()
                if done in (0, total) or now - last_log >= 1:
                    store.log({
                        "time": timestamp(), "stage": stage, "model_index": index, "model": checkpoint.name.value,
                        "completed": done, "total": total, "elapsed_seconds": round(now - started, 3),
                    })
                    last_log = now
                progress(f"{index} {checkpoint.name} {stage}", done, total)

            report("load checkpoint", 0, 1)
            try:
                predictor = factory(checkpoint.name, checkpoint.path, checkpoint.config, config, output)
                report("load checkpoint", 1, 1)
                predictions, speed = _benchmark(predictor, labels, config, report)
                metrics = detection_metrics(labels.samples, predictions, config, report)
                report("verify inputs", 0, 1)
                _unchanged(snapshots)
                if fingerprint_path(checkpoint.path) != checkpoint.fingerprint:
                    raise EvaluationError("The checkpoint changed during inference. Retry with a stable checkpoint.")
                report("verify inputs", 1, 1)
            except KeyboardInterrupt:
                store.log({"time": timestamp(), "stage": active_stage, "model_index": index, "status": "interrupted"})
                raise
            except EvaluationError as exc:
                store.log({"time": timestamp(), "stage": active_stage, "model_index": index, "status": "failed"})
                raise EvaluationError(credentials.redact_text(f"{checkpoint.name} {active_stage}. {exc}")) from None
            except Exception:
                store.log({"time": timestamp(), "stage": active_stage, "model_index": index, "status": "failed"})
                raise EvaluationError(
                    f"{checkpoint.name} {active_stage} failed. Check image integrity, device operators and checkpoint compatibility."
                ) from None
            result = {
                **record, "schema_version": 1, "status": "completed", "finished_at": timestamp(),
                "model_index": index, "request_fingerprint": identity, "dataset": dataset_record,
                "protocol": protocol, "metrics": credentials.redact(metrics), "speed": credentials.redact(speed),
                "postprocessing": credentials.redact(predictor.postprocessing), "libraries": credentials.redact(predictor.libraries),
                "python": platform.python_version(), "code_fingerprints": code,
            }
            del predictor, predictions
            credentials.protect_path(json.dumps(result, ensure_ascii=False), "evaluation report")
            _publish(target, {"fingerprint": digest_json(result), "result": result})
            store.log({"time": timestamp(), "stage": "publish report", "model_index": index, "status": "completed"})
            if index == len(checkpoints):
                attempt.close()
            yield target


def compare_reports(workspace: Path, paths: tuple[Path, ...]) -> dict[str, object]:
    if len(paths) < 2:
        raise EvaluationError("Supply at least two completed evaluation reports.")
    credentials = SetupFiles.read(workspace, None).credentials(os.environ)
    reports = [read_report(_local(path, workspace, credentials)) for path in paths]
    comparison = compare_results(reports)
    credentials.protect_path(json.dumps(comparison, ensure_ascii=False), "comparison report")
    return comparison


def compare_results(reports: list[dict[str, Any]]) -> dict[str, object]:
    anchor = reports[0]
    try:
        if any(
            entry["dataset"] != anchor["dataset"] or entry["protocol"] != anchor["protocol"]
            or entry["code_fingerprints"] != anchor["code_fingerprints"]
            for entry in reports[1:]
        ):
            raise EvaluationError("Compare reports from the same labeled split, code and evaluation protocol.")
        hardware = anchor["speed"]["hardware"]
        speed_comparable = bool(hardware.get("cpu") and (
            hardware.get("device") == "cpu" or hardware.get("accelerator")
        )) and all(
            entry["speed"]["hardware"] == hardware and entry["python"] == anchor["python"]
            and entry["libraries"].get("torch") == anchor["libraries"].get("torch")
            and entry["libraries"].get("torchvision") == anchor["libraries"].get("torchvision")
            for entry in reports[1:]
        )
        return {"dataset": anchor["dataset"], "matching_speed_environment": speed_comparable, "models": reports}
    except (KeyError, AttributeError, TypeError):
        raise EvaluationError("An evaluation report is missing comparison fields.") from None
