"""Detection stage adapters over the standalone model and reporting APIs."""

import json
import math
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import wraps
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from shutil import copyfile
from typing import Any
from uuid import uuid4

from pydantic import TypeAdapter

from ..config.settings import Credentials, SetupError
from ..data.preprocessing.policy import label_policy
from ..data.stages import limited_progress
from ..evaluation import runner as evaluation
from ..evaluation.contracts import CLASSES, EvaluationConfig, EvaluationError, EvaluationSplit, Prediction
from ..evaluation.metrics import detection_metrics
from ..models.contracts import ModelName, Progress, TrainConfig, TrainingError
from ..models.dataset import Split, inspect_data
from ..models.registry import REGISTRY
from ..models.training import TrainingPlan, checkpoint_paths, execute_training, read_training_profile
from ..reporting.export import ReportFormat, report_document, write_report_file
from .model import StageContext, StageFailure, StageResult
from .store import RunStore, digest_json, fingerprint_path, timestamp

DETECTION_STAGES = ("train", "validate_models", "evaluate", "report")
_PREDICTIONS = TypeAdapter(dict[str, Prediction])


def _adapter(
    execute: Callable[["DetectionStages", StageContext], StageResult],
) -> Callable[["DetectionStages", StageContext], StageResult]:
    @wraps(execute)
    def run(stages: "DetectionStages", context: StageContext) -> StageResult:
        try:
            return execute(stages, context)
        except (TrainingError, EvaluationError, SetupError) as exc:
            raise StageFailure(stages.credentials.redact_text(str(exc))) from None

    return run


def _object(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise StageFailure("Expected a JSON object in a detection artifact.")
    return result


def _package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name in names:
        try:
            values[name] = version(name)
        except PackageNotFoundError:
            values[name] = None
    return values


@dataclass(frozen=True)
class DetectionStages:
    workspace: Path
    credentials: Credentials
    profiles: dict[ModelName, tuple[TrainConfig, str]]
    config: EvaluationConfig
    approved: bool = False
    show_scope: Callable[[str], None] | None = None

    @classmethod
    def configure(
        cls, workspace: Path, credentials: Credentials, names: tuple[ModelName, ...], config_dir: Path | None,
        config: EvaluationConfig, approved: bool, show_scope: Callable[[str], None],
    ) -> "DetectionStages":
        profiles = {name: read_training_profile(name, config_dir) for name in names}
        return cls(workspace, credentials, profiles, config, approved, show_scope)

    def describe_training(self, dataset: str, output: Path) -> None:
        if self.show_scope:
            for name, (config, fingerprint) in self.profiles.items():
                self.show_scope(self.credentials.redact_text(
                    f"Detector {name}   input {dataset}   output {output}   YAML SHA-256 {fingerprint}\n"
                    f"{config.model_dump_json()}"
                ))

    def model_revision(self, name: ModelName) -> str:
        config, fingerprint = self.profiles[name]
        return digest_json({
            "config": config.model_dump(mode="json"), "profile": fingerprint,
            "runtime": [platform.python_version(), platform.system(), platform.machine()],
            "packages": _package_versions(REGISTRY[name].libraries),
            "code": {
                path.name: fingerprint_path(path)
                for path in sorted((Path(__file__).parents[1] / "models").glob("*.py"))
            },
        })

    @_adapter
    def train(self, context: StageContext) -> StageResult:
        dataset = context.inputs["detection"]
        receipt = _object(context.inputs["detection_validation"])
        policy = label_policy()
        if (
            receipt.get("schema_version") != 1 or receipt.get("passed") is not True
            or receipt.get("inputs") != {
                key: context.input_fingerprints[key] for key in ("manifest", "manifest_summary", "detection")
            }
            or digest_json(receipt.get("label_policy")) != digest_json(policy)
        ):
            raise StageFailure("Training requires a passing conversion validation receipt for these inputs.")
        if not self.profiles or context.recovery_dir is None:
            raise StageFailure("Configure detector model profiles and run training through the stage runner.")

        progress = self._progress(context)
        data = {fmt: inspect_data(dataset, fmt, self.credentials, progress) for fmt in ("yolo", "coco")}
        if data["yolo"].split_fingerprints != data["coco"].split_fingerprints:
            raise StageFailure("YOLO and COCO split membership or image bytes differ. Rerun conversion.")
        revision = {
            "schema_version": 1, **receipt["inputs"],
            "detection_validation": context.input_fingerprints["detection_validation"],
            "label_policy": policy,
            "splits": data["coco"].split_fingerprints,
            "annotations": {fmt: item.split_annotation_fingerprints for fmt, item in data.items()},
        }
        if "selected_split" in receipt:
            revision["selected_split"] = receipt["selected_split"]
        target = context.output_dir / "checkpoints"
        target.mkdir()
        self.describe_training(str(dataset), context.artifact_dir / "checkpoints")
        for index, (name, (config, profile_hash)) in enumerate(self.profiles.items(), 1):
            identity = digest_json([revision, name, self.model_revision(name)])
            root = RunStore(context.recovery_dir, identity, self.credentials).directory
            model, resume = self._recover(root, name, config, profile_hash, revision)
            if model is None:
                # A saved attempt or a previous approval never authorizes new work.
                if not self.approved:
                    raise StageFailure(
                        f"Training {name} requires explicit approval. Review the detector scope and pass "
                        "--approve-detection-train for this invocation."
                    )
                plan = TrainingPlan(
                    {name: config}, dataset, root / uuid4().hex, resume, self.credentials, self.workspace,
                    revision, profile_hash, data,
                )
                list(execute_training(plan, progress))
                model = plan.destination / name
            self._publish_model(model, target / name, name)
            context.progress(index, len(self.profiles))
        return StageResult.SUCCESS

    def _recover(
        self, root: Path, name: ModelName, config: TrainConfig, profile_hash: str, revision: dict[str, Any],
    ) -> tuple[Path | None, Path | None]:
        resume = None
        if root.is_dir():
            attempts = sorted(root.iterdir(), key=lambda path: path.stat().st_mtime_ns, reverse=True)
            for attempt in attempts:
                if not attempt.is_dir():
                    continue
                RunStore(root, attempt.name)
                RunStore(attempt, name)
                model = attempt / name
                try:
                    fingerprint_path(model / "run.json")
                    record = _object(model / "run.json")
                    if (
                        record.get("schema_version") != 1 or record.get("model") != name
                        or record.get("profile_fingerprint") != profile_hash
                        or record.get("configuration") != config.model_dump(mode="json")
                        or digest_json(record.get("dataset_revision")) != digest_json(revision)
                    ):
                        continue
                    if record.get("status") == "completed":
                        for kind in ("best", "last"):
                            evaluation.inspect_checkpoint(model, kind, self.credentials)
                        return model, None
                    epoch = record.get("checkpoint_epoch")
                    if (
                        resume is None and record.get("status") in {"running", "interrupted", "failed"}
                        and type(epoch) is int and 1 <= epoch <= config.epochs
                    ):
                        checkpoints = checkpoint_paths(name, model)
                        if all(
                            record["checkpoints"][kind] == {
                                "path": path.relative_to(model).as_posix(), "fingerprint": fingerprint_path(path),
                            }
                            for kind, path in (("best", checkpoints.best), ("last", checkpoints.last))
                        ):
                            resume = model
                except (EvaluationError, OSError, KeyError, TypeError, ValueError):
                    continue
        return None, resume

    def _publish_model(self, source: Path, output: Path, name: ModelName) -> None:
        origin, targets = checkpoint_paths(name, source), checkpoint_paths(name, output)
        targets.last.parent.mkdir(parents=True)
        copyfile(origin.best, targets.best)
        copyfile(origin.last, targets.last)
        copyfile(source / "run.json", output / "run.json")
        for kind in ("best", "last"):
            evaluation.inspect_checkpoint(output, kind, self.credentials)

    @_adapter
    def validate_models(self, context: StageContext) -> StageResult:
        labels, data, checkpoints = self._validation_data(context)
        output = context.output_dir / "predictions"
        output.mkdir()
        scratch = context.output_dir / "inference"
        scratch.mkdir()
        for index, (name, checkpoint) in enumerate(checkpoints.items(), 1):
            model = context.inputs["checkpoints"] / name
            predictions, inference = evaluation.benchmark_checkpoint(
                checkpoint, labels, self.config, self._progress(context), scratch,
            )
            _validate_predictions(labels, predictions)
            result = {
                "schema_version": 1, "status": "completed", "model_index": index,
                **evaluation.checkpoint_record(checkpoint, "best"),
                "run_metadata_fingerprint": fingerprint_path(model / "run.json"),
                "dataset": data, "protocol": evaluation.evaluation_protocol(self.config),
                "code_fingerprints": evaluation.evaluation_code(),
                **self.credentials.redact(inference),
                "predictions": {key: asdict(prediction) for key, prediction in predictions.items()},
            }
            self.credentials.protect_path(json.dumps(result, ensure_ascii=False), "prediction artifact")
            evaluation.write_result(output / f"{name}.json", result)
        return StageResult.SUCCESS

    @_adapter
    def evaluate(self, context: StageContext) -> StageResult:
        labels, dataset, checkpoints = self._validation_data(context)
        protocol = evaluation.evaluation_protocol(self.config)
        code = evaluation.evaluation_code()
        output = context.output_dir / "model_reports"
        output.mkdir()
        models = []
        identity = digest_json([dict(context.input_fingerprints), protocol, code])
        for index, (name, checkpoint) in enumerate(checkpoints.items(), 1):
            path = context.inputs["predictions"] / f"{name}.json"
            result = evaluation.read_result(path, "A Validation prediction artifact")
            model = context.inputs["checkpoints"] / name
            expected = {
                **evaluation.checkpoint_record(checkpoint, "best"), "model_index": index,
                "dataset": dataset, "protocol": protocol, "code_fingerprints": code,
                "run_metadata_fingerprint": fingerprint_path(model / "run.json"),
            }
            if any(digest_json(result.get(key)) != digest_json(value) for key, value in expected.items()):
                raise StageFailure("Validation predictions do not match the dataset, checkpoint or evaluation protocol.")
            predictions = _PREDICTIONS.validate_json(json.dumps(result.pop("predictions")), strict=True)
            _validate_predictions(labels, predictions)
            result.update(
                finished_at=timestamp(), request_fingerprint=identity,
                prediction_fingerprint=fingerprint_path(path),
                metrics=detection_metrics(labels.samples, predictions, self.config, self._progress(context)),
            )
            evaluation.write_result(output / f"{index:02d}-{name}.json", result)
            models.append(result)
        evaluation.write_result(context.output_dir / "metrics.json", {
            "schema_version": 1, "status": "completed", "models": models,
        })
        return StageResult.SUCCESS

    @_adapter
    def report(self, context: StageContext) -> StageResult:
        models = evaluation.read_result(context.inputs["metrics"], "The detection metrics index")["models"]
        document = report_document([{
            "run_id": context.artifact_dir.parents[2].name, "kind": "evaluation",
            "status": "completed", "models": models,
            "metrics_fingerprint": context.input_fingerprints["metrics"],
        }])
        for fmt in (ReportFormat.HTML, ReportFormat.JSON):
            write_report_file(context.output_dir / f"report.{fmt.value}", fmt, document)
        return StageResult.SUCCESS

    def _validation_data(
        self, context: StageContext,
    ) -> tuple[Split, dict[str, Any], dict[ModelName, evaluation.Checkpoint]]:
        if not self.profiles:
            raise StageFailure("Select detector profiles before running Validation.")
        labels, hashes, annotation, _ = evaluation.inspect_split(
            context.inputs["detection"], "val", self.credentials, self._progress(context), labels_required=True,
        )
        image_fingerprint = digest_json(hashes)
        revision = None
        checkpoints = {
            name: evaluation.inspect_checkpoint(context.inputs["checkpoints"] / name, "best", self.credentials)
            for name in self.profiles
        }
        for name, checkpoint in checkpoints.items():
            recorded = checkpoint.dataset_revision
            if (
                checkpoint.name != name or recorded is None
                or (revision is not None and recorded != revision)
                or recorded["detection"] != context.input_fingerprints["detection"]
                or recorded["annotations"]["coco"]["val"] != digest_json({"instances_val.json": annotation})
                or checkpoint.split_fingerprints["val"] != image_fingerprint
            ):
                raise StageFailure("Validation images and labels must match the verified training dataset revision.")
            revision = recorded
        dataset = evaluation.dataset_record(
            EvaluationSplit.VALIDATION, len(labels.images), image_fingerprint, annotation, {}, revision,
        )
        return labels, dataset, checkpoints

    @staticmethod
    def _progress(context: StageContext) -> Progress:
        update = limited_progress(context.progress)
        return lambda _stage, done, total: update(done, total)


def _validate_predictions(labels: Split, predictions: dict[str, Prediction]) -> None:
    if predictions.keys() != {sample.image.stem for sample in labels.samples}:
        raise EvaluationError("Predictions must cover every Validation sample exactly once.")
    for sample in labels.samples:
        pred = predictions[sample.image.stem]
        if (
            pred.sample_id != sample.image.stem
            or pred.preprocessing.original_size != (sample.width, sample.height)
            or min(pred.preprocessing.input_size) < 1
            or len(pred.boxes) != len(pred.class_ids) or len(pred.boxes) != len(pred.confidence_scores)
        ):
            raise EvaluationError("Prediction rows or preprocessing metadata do not match their Validation sample.")
        for box, category, score in zip(pred.boxes, pred.class_ids, pred.confidence_scores, strict=True):
            x1, y1, x2, y2 = box
            if (
                type(category) is not int or category not in CLASSES or not 0 <= score <= 1
                or not all(math.isfinite(value) for value in box)
                or not (0 <= x1 < x2 <= sample.width and 0 <= y1 < y2 <= sample.height)
            ):
                raise EvaluationError("Predictions contain an invalid box, class or confidence score.")
