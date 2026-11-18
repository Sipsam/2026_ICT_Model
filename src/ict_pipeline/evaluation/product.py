"""Evaluate frozen Validation predictions and compare like-for-like experiment reports."""

import json
import math
import os
import platform
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from itertools import groupby
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TypeVar

from pydantic import TypeAdapter

from ..config.settings import Credentials
from ..data.preprocessing.console import protect_source_data
from ..inspection.contracts import InspectionResult, Modality, ProductInspection, ProductModelIdentity
from ..models.contracts import Progress
from ..models.dataset import Object, Sample
from ..models.registry import REGISTRY
from ..multimodal.model_config import FusionConfig
from ..pipeline.store import digest_json, fingerprint_path
from .contracts import CLASSES, EvaluationError, Prediction, validate_predictions
from .metrics import detection_metrics
from .product_protocol import (
    ClassificationMetric, DetectionProtocol, DetectorExperimentModel, ExperimentReport, FrozenValidation,
    ProductProtocol, Protocol, ValidationIdentity,
)
from .runner import evaluation_protocol, inspect_checkpoint, inspect_split, read_result

_Artifact = TypeVar("_Artifact", ExperimentReport, FrozenValidation)


def _metric(name: str, rows: Iterable[tuple[bool | None, float]], threshold: float) -> ClassificationMetric:
    tn = fp = fn = tp = missing = 0
    ranked = []
    for label, score in rows:
        if label is None:
            missing += 1
            continue
        ranked.append((score, label))
        if score >= threshold:
            tp += int(label)
            fp += int(not label)
        else:
            fn += int(label)
            tn += int(not label)
    positives, negatives = tp + fn, tn + fp
    auc = None
    if positives and negatives:
        lower_negatives = favorable_pairs = 0
        for _, tie_rows in groupby(sorted(ranked), key=lambda row: row[0]):
            tie = [label for _, label in tie_rows]
            tied_positives = sum(tie)
            tied_negatives = len(tie) - tied_positives
            favorable_pairs += tied_positives * (2 * lower_negatives + tied_negatives)
            lower_negatives += tied_negatives
        auc = favorable_pairs / (2 * positives * negatives)
    return ClassificationMetric(
        name=name, missing_labels=missing, confusion_matrix=((tn, fp), (fn, tp)),
        precision=tp / (tp + fp) if tp + fp else None,
        recall=tp / positives if positives else None,
        f1=2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
        auroc=auc,
    )


def _environment() -> dict[str, object]:
    package = Path(__file__).resolve().parents[1]
    code = sorted(
        path.relative_to(package).as_posix()
        for directory in ("evaluation", "multimodal", "inspection") for path in (package / directory).glob("*.py")
    )
    runtime = {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()}
    for name in ("ict-pipeline", "pydantic", "Pillow", "torch", "torchvision", "ultralytics", "transformers"):
        try:
            runtime[name] = version(name)
        except PackageNotFoundError:
            continue
    return {"code_fingerprint": digest_json({name: fingerprint_path(package / name) for name in code}),
            "runtime": runtime}


def pending_experiment(
    experiment_id: str, protocol: Protocol, *, branch: Modality | None = None,
    model: ProductModelIdentity | None = None, fusion: FusionConfig | None = None,
    detector: DetectorExperimentModel | None = None,
) -> ExperimentReport:
    return ExperimentReport(
        experiment_id=experiment_id, protocol=protocol, status="pending", branch=branch,
        product_model=model, fusion=fusion, detector_model=detector, **_environment(),
    )


def classification_metrics(
    cohort: FrozenValidation, protocol: ProductProtocol, scores: Mapping[str, tuple[float, ...]],
) -> tuple[ClassificationMetric, ...]:
    """Score canonical probability vectors against a frozen Validation identity."""
    if protocol.cohort != cohort.identity or scores.keys() != set(protocol.cohort.sample_ids):
        raise ValueError("Scores must cover exactly the frozen Validation cohort.")
    names = ("any_target_defect",) if protocol.target_policy == "smt-target-defect-presence-v1" else tuple(
        CLASSES.values()
    )
    if any(len(row) != len(names) or any(not math.isfinite(score) or not 0 <= score <= 1 for score in row)
           for row in scores.values()):
        raise ValueError("Use finite probabilities in canonical target order for every Validation sample.")
    rows = [(sample.labels(protocol.target_policy), scores[sample.sample_id]) for sample in cohort.samples]
    return tuple(_metric(name, ((truth[i], score[i]) for truth, score in rows), protocol.threshold)
                 for i, name in enumerate(names))


def evaluate_products(
    experiment_id: str, cohort: FrozenValidation, protocol: ProductProtocol, branch: Modality,
    records: Iterable[InspectionResult], *, fusion: FusionConfig | None = None,
) -> ExperimentReport:
    """Require exact branch coverage and input identity before calculating any metrics."""
    if protocol.alignment_policy != "verified_receipt_at_capture-v1":
        raise ValueError("Inspection records require the capture-time Validation protocol.")
    if protocol.cohort != cohort.identity:
        raise ValueError("The evaluation protocol differs from the frozen Validation cohort.")
    predictions: dict[str, tuple[InspectionResult, ProductInspection]] = {}
    for record in records:
        prediction = record.multimodal
        if (record.outcome != "review_required" or prediction is None or prediction.branch != branch
                or prediction.input_mode != branch or prediction.sample_id in predictions
                or prediction.target_policy != protocol.target_policy or prediction.threshold != protocol.threshold):
            raise ValueError("Supply exactly one successful prediction for the requested branch and label policy per sample.")
        predictions[prediction.sample_id] = record, prediction
    if predictions.keys() != set(protocol.cohort.sample_ids):
        raise ValueError("Predictions must match the frozen Validation sample IDs exactly.")

    ordered = []
    for sample in cohort.samples:
        record, prediction = predictions[sample.sample_id]
        if (record.image is not None and (record.image.sha256 != sample.image_sha256
                                          or (record.image.width, record.image.height) != sample.image_size)
                or (branch != "image" and prediction.sensor_fingerprint != sample.sensor_fingerprint)
                or (prediction.capture_timestamp is not None and prediction.capture_timestamp != sample.capture_timestamp)):
            raise ValueError("Prediction input evidence differs from the frozen Validation sample.")
        ordered.append(prediction)
    first = ordered[0]
    if any(row.model != first.model or row.weight_status != first.weight_status for row in ordered):
        raise ValueError("Use one checkpoint and weight status per experiment.")
    status = {"simulated": "simulated", "untrained": "untrained", "trusted_trained": "measured"}[first.weight_status]
    metrics = None
    if status == "measured":
        truths = [sample.labels(protocol.target_policy) for sample in cohort.samples]
        if any(label is not None for truth in truths for label in truth):
            metrics = classification_metrics(cohort, protocol, {
                row.sample_id: tuple(score.score for score in row.scores) for row in ordered
            })
        else:
            status = "unscorable"
    return ExperimentReport(
        experiment_id=experiment_id, protocol=protocol, status=status, branch=branch,
        product_model=first.model, fusion=fusion, classification_metrics=metrics,
        prediction_fingerprint=digest_json([row.model_dump(mode="json") for row in ordered]), **_environment(),
    )


def evaluate_detector(
    experiment_id: str, cohort: FrozenValidation, protocol: DetectionProtocol, *,
    model: DetectorExperimentModel, input_cohort: ValidationIdentity,
    predictions: Mapping[str, Prediction], progress: Progress | None = None,
) -> ExperimentReport:
    """Use raw detector predictions from the trusted adapter on the same frozen image cohort."""
    if protocol.cohort != cohort.identity or input_cohort != protocol.cohort:
        raise ValueError("Detector prediction provenance differs from the frozen Validation cohort.")
    samples = []
    for sample in cohort.samples:
        if sample.targets is None or sample.missing_classes:
            raise ValueError("Detection mAP requires complete original box labels for every cohort image.")
        width, height = sample.image_size
        samples.append(Sample(Path(sample.sample_id + ".jpg"), width, height, tuple(
            Object(target.category, target.box) for target in sample.targets
        )))
    labels = tuple(samples)
    validate_predictions(labels, predictions)
    return ExperimentReport(
        experiment_id=experiment_id, protocol=protocol, status="measured", detector_model=model,
        prediction_fingerprint=digest_json({key: asdict(value) for key, value in predictions.items()}),
        detection_metrics=detection_metrics(labels, predictions, protocol.evaluation, progress), **_environment(),
    )


def evaluate_detector_artifact(
    experiment_id: str, cohort: FrozenValidation, protocol: DetectionProtocol, *,
    artifact_path: Path, detection_root: Path, run: Path, trust_run: bool,
    credentials: Credentials, progress: Progress,
) -> ExperimentReport:
    """Bind a native validate_models artifact and its trusted run to the common cohort."""
    if trust_run is not True:
        raise EvaluationError("Use a trusted completed detector run and its Validation predictions.")
    saved = read_result(artifact_path, "A detector Validation prediction artifact")
    try:
        dataset = saved["dataset"]
        checkpoint = saved["checkpoint"]
        kind = checkpoint["kind"]
        if dataset["split"] != "val" or dataset["held_out_attested"] is not False or kind not in ("best", "last"):
            raise ValueError
        if digest_json(saved["protocol"]) != digest_json(evaluation_protocol(protocol.evaluation)):
            raise ValueError
        run = run.resolve()
        verified_run = inspect_checkpoint(run, kind, credentials)
        if (saved["model"] != verified_run.name.value or checkpoint["fingerprint"] != verified_run.fingerprint
                or saved["run_metadata_fingerprint"] != fingerprint_path(run / "run.json")
                or saved["training_split_fingerprints"] != verified_run.split_fingerprints):
            raise ValueError
        labels, hashes, annotation, _ = inspect_split(
            detection_root.resolve(), "val", credentials, progress, labels_required=True,
        )
        image_fingerprint = digest_json(hashes)
        if (dataset["image_count"] != len(labels.samples) or dataset["image_fingerprint"] != image_fingerprint
                or verified_run.split_fingerprints.get("val") != image_fingerprint
                or dataset["annotation_fingerprint"] != annotation):
            raise ValueError
        predictions = TypeAdapter(dict[str, Prediction]).validate_json(json.dumps(saved["predictions"]), strict=True)
        validate_predictions(labels.samples, predictions)
        native = {sample.image.stem: sample for sample in labels.samples}
        for sample in cohort.samples:
            matched = native[sample.sample_id]
            if (sample.image_sha256 != hashes[matched.image.name] or sample.image_size != (matched.width, matched.height)
                    or sample.targets is None or sample.missing_classes
                    or sorted((target.category, target.box) for target in sample.targets)
                    != sorted((obj.category, obj.box) for obj in matched.objects)):
                raise ValueError
        model = DetectorExperimentModel(
            model=verified_run.name.value, checkpoint_fingerprint=verified_run.fingerprint,
            checkpoint_format=REGISTRY[verified_run.name].checkpoint_format,
            run_metadata_fingerprint=saved["run_metadata_fingerprint"], training_seed=verified_run.config.seed,
            adapter_version=digest_json([saved["code_fingerprints"], saved["libraries"], saved["postprocessing"]]),
            weight_status="trusted_trained", libraries=credentials.redact(saved["libraries"]),
        )
    except (KeyError, TypeError, ValueError, OSError):
        raise EvaluationError("Detector artifact, checkpoint or original Validation evidence is incompatible.") from None
    return evaluate_detector(
        experiment_id, cohort, protocol, model=model, input_cohort=cohort.identity,
        predictions={sample.sample_id: predictions[sample.sample_id] for sample in cohort.samples}, progress=progress,
    )


def compare_experiments(reports: Iterable[ExperimentReport]) -> tuple[ExperimentReport, ...]:
    """Check protocol equality and preserve statuses alongside any measured metrics."""
    ordered = tuple(sorted(reports, key=lambda report: report.experiment_id))
    if len(ordered) < 2 or len({report.experiment_id for report in ordered}) != len(ordered):
        raise ValueError("Compare at least two distinct experiment IDs.")
    first = ordered[0]
    if any(report.protocol != first.protocol or report.code_fingerprint != first.code_fingerprint for report in ordered):
        raise ValueError("Comparison requires matching task, cohort, label policy, preprocessing and scoring protocol.")
    return ordered


def write_artifact(path: Path, artifact: ExperimentReport | FrozenValidation, *, credentials: Credentials | None = None) -> None:
    """Publish a new fingerprinted report or cohort without overwriting another writer's output."""
    path = path.absolute()
    protect_source_data(path.resolve(), Path.cwd())
    if path.exists() or path.is_symlink():
        raise FileExistsError("Choose a fresh experiment artifact path.")
    saved = artifact.model_dump(mode="json")
    if credentials is not None and credentials.redact(saved) != saved:
        raise ValueError("Experiment artifact contains unsanitized credential material.")
    payload = {"artifact": saved, "fingerprint": digest_json(saved)}
    with NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.link(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def read_artifact(path: Path, kind: type[_Artifact]) -> _Artifact:
    try:
        saved = json.loads(path.read_bytes())
        artifact = kind.model_validate_json(json.dumps(saved["artifact"]))
        if (saved.keys() != {"fingerprint", "artifact"} or "schema_version" not in artifact.model_fields_set
                or saved["fingerprint"] != digest_json(saved["artifact"])
                or saved["fingerprint"] != digest_json(artifact.model_dump(mode="json"))):
            raise ValueError
        return artifact
    except (OSError, KeyError, TypeError, ValueError):
        raise ValueError("Experiment artifact is missing, corrupt or incompatible.") from None
