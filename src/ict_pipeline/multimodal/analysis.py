"""Read-only cohort leakage, subgroup scoring, and sensor intervention analysis."""

import csv
import hashlib
import json
import os
import random
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from ..config.settings import Credentials
from ..data.detector_finalization import RECEIPT, verify_finalized_detector
from ..data.group_retraining import _assignments, _check_prior_test, read_verification_receipt
from ..data.model import Progress
from ..data.preprocessing.common import require_separate_output, write_json
from ..data.preprocessing.console import fresh_output, protect_source_data
from ..data.preprocessing.leakage import METHODS, MODALITIES, file_sha
from ..data.preprocessing.policy import SPLIT_MAP
from ..data.preprocessing.revision import read_publication
from ..evaluation.contracts import CLASSES
from ..evaluation.product import _environment, _metric, classification_metrics, read_artifact
from ..evaluation.product_protocol import ExperimentReport, FrozenValidation, ProductProtocol
from ..models.contracts import ModelName, TrainConfig
from ..models.registry import REGISTRY, resolve_device
from ..models.weights import LAYOUTS
from ..pipeline.store import _reject_link, digest_json, fingerprint_path
from .audit import Cohort
from .contracts import Record
from .dataset import MultimodalDataset, collate
from .model_config import image_preprocessing_fingerprint
from .preprocessing import PreprocessingState, training_fingerprint, transform_sensor
from .training_plan import ProductTrainingConfig, _in_training_scope, _load_index, _views

AnalysisMode = Literal["all", "leakage", "groups", "ablation"]
ModelRole = Literal["image", "concat"]
Scores = dict[str, tuple[float, ...]]
ProgressFactory = Callable[[str], Progress]


@dataclass(frozen=True)
class AnalysisRun:
    role: ModelRole
    model_path: Path
    state: PreprocessingState
    profile: ProductTrainingConfig
    train: tuple[Record, ...]
    validation: MultimodalDataset
    protocol: ProductProtocol
    provenance: dict[str, Any]
    architecture_fingerprint: str


@dataclass(frozen=True)
class GroupAnalysisInputs:
    verification: Path
    bundle: Path
    detector: Path
    prior_test_manifests: tuple[Path, ...] = ()


def _verify_group_run(
    saved: dict[str, Any], profile: ProductTrainingConfig, cohort: Cohort, frozen: FrozenValidation,
    inputs: GroupAnalysisInputs, progress: Progress,
) -> tuple[Cohort, dict[str, Any]]:
    receipt, fingerprint = read_verification_receipt(inputs.verification)
    if fingerprint != saved["profile"]["group_verification"]:
        raise ValueError("Group Verification Receipt fingerprint differs from run.json.")
    if receipt["revision_id"] != profile.revision_id or receipt["revision_id"] != cohort.revision_id:
        raise ValueError("Group Split Revision differs from the verification receipt.")
    if (
        receipt["cohort_fingerprint"] != cohort.fingerprint
        or receipt["split_fingerprint"] != cohort.split_fingerprint
        or receipt["manifest_fingerprint"] != cohort.manifest_fingerprint
        or saved.get("split_fingerprint") != cohort.split_fingerprint
        or saved.get("dataset_fingerprint") != cohort.input_fingerprint
        or receipt["preprocessing_fingerprint"] != saved["preprocessing_fingerprint"]
        or receipt["training_fingerprint"] != saved["train_state_fingerprint"]
    ):
        raise ValueError("Group Cohort or Train preprocessing differs from the verification receipt or run.json.")
    if (receipt["frozen_validation_fingerprint"] != frozen.identity.cohort_fingerprint
            or receipt["frozen_validation_count"] != len(frozen.identity.sample_ids)):
        raise ValueError("FrozenValidation differs from the group verification receipt.")
    revision, seal, rows = read_publication(inputs.bundle, profile.revision_id)
    _assignments(revision, rows)
    if (seal != receipt["publication_fingerprint"] or seal != saved.get("seal_fingerprint")
            or revision["selected_manifest_fingerprint"] != receipt["manifest_fingerprint"]
            or receipt["test"]["origin"] != revision["test_origin"]
            or receipt["test"]["payloads_opened"] is not False):
        raise ValueError("Group publication or Test custody differs from the verification receipt.")
    prior = _check_prior_test(rows, inputs.prior_test_manifests)
    if sorted(prior) != sorted(receipt["prior_test_manifest_fingerprints"]):
        raise ValueError("Supply the exact prior Test reservation manifests recorded in the group verification receipt.")

    initialization = profile.detector_initialization
    layout = profile.architecture.image.detector
    backend = REGISTRY[ModelName.FASTER_RCNN]
    provenance = {
        "format": backend.checkpoint_format, "architecture": LAYOUTS["faster_rcnn"].architecture,
        "sha256": initialization.sha256,
    }
    _reject_link(inputs.detector.parent)
    _reject_link(inputs.detector)
    detector_run = inputs.detector.parent / "run.json"
    finalization = inputs.detector.parent / RECEIPT
    if finalization.exists() or finalization.is_symlink():
        if detector_run.exists() or detector_run.is_symlink():
            raise ValueError("Use a separate finalized Detector directory.")
        provenance["finalization_fingerprint"] = verify_finalized_detector(
            inputs.detector, initialization.sha256, layout, receipt, fingerprint, inputs.bundle,
        )
    else:
        _reject_link(detector_run)
        detector = json.loads(detector_run.read_bytes())
        if (
            detector.get("status") != "completed" or detector.get("model") != "faster_rcnn"
            or detector.get("architecture") != provenance["architecture"]
            or detector.get("dataset_revision") != {
                "group_split_revision": profile.revision_id, "group_verification": fingerprint,
                "detector_scope": receipt["detector_scope"],
            }
            or detector.get("initialization_mode") != layout.layout
            or (detector.get("configuration") or {}).get("image_size") != layout.image_size
            or (detector.get("checkpoints") or {}).get("best") != {
                "path": inputs.detector.name, "fingerprint": fingerprint_path(inputs.detector),
            }
            or file_sha(inputs.detector) != initialization.sha256
        ):
            raise ValueError("Detector initialization differs from the completed verified group Detector.")
    if provenance != saved.get("detector_initialization"):
        raise ValueError("Detector initialization provenance differs from run.json.")
    scoped, live_seal, _ = _load_index(profile, frozen, progress)
    if live_seal != seal:
        raise ValueError("Group publication changed during analysis verification.")
    return scoped, {
        "role_split": "verified_group_revision", "group_verification": fingerprint,
        "group_split_revision": profile.revision_id, "publication_fingerprint": seal,
        "prior_test_manifest_fingerprints": prior, "detector_initialization": provenance,
    }


def _inspect_run(
    role: ModelRole, run: Path, root: Path, cohort_path: Path, frozen_path: Path,
    cohort: Cohort, frozen: FrozenValidation,
    *, group: GroupAnalysisInputs | None = None, progress: Progress = lambda *_: None,
) -> AnalysisRun:
    raw = (run / "run.json").read_bytes()
    saved = json.loads(raw)
    if saved["schema_version"] != 1 or saved["task"] != "product_classification" or saved["status"] != "completed":
        raise ValueError(f"{role}: expected a completed product training run.")
    has_group = saved["profile"].get("group_verification") is not None
    if has_group != (group is not None):
        raise ValueError(f"{role}: group analysis requires matching run provenance and --group-verification, "
                         "--group-bundle and --detector-initialization.")
    state_path = run / "sensor-state.json"
    profile = ProductTrainingConfig.model_validate_json(json.dumps({
        **saved["profile"], "source": str(root), "bundle": str(group.bundle if group else root),
        "cohort": str(cohort_path), "group_verification": str(group.verification) if group else None,
        "preprocessing": str(state_path),
        "validation": {**saved["profile"]["validation"], "cohort": str(frozen_path)},
        "detector_initialization": {**saved["profile"]["detector_initialization"],
                                    "path": str(group.detector if group else root)},
    }))
    if (
        profile.branch != ("image" if role == "image" else "paired")
        or profile.architecture.image.kind != "faster_rcnn_fpn"
        or (role == "concat" and (profile.architecture.fusion.kind != "concat"
                                  or profile.architecture.sensor.kind != "mlp"))
        or profile.architecture.target_policy != "smt-view-class-presence-v1"
        or profile.architecture.fusion.threshold != 0.5
        or profile.checkpoint_policy != "best_validation" or profile.selection_metric != "macro_auroc"
        or profile.cohort_fingerprint != cohort.fingerprint
        or profile.validation is None or profile.validation.fingerprint != frozen.identity.cohort_fingerprint
        or saved["validation_identity"] != frozen.identity.model_dump(mode="json")
    ):
        raise ValueError(f"{role}: run profile, cohort or FrozenValidation differs from the analysis protocol.")
    state = PreprocessingState.load(state_path, expected_training_fingerprint=saved["train_state_fingerprint"])
    if (state.fingerprint != saved["preprocessing_fingerprint"]
            or state.fingerprint != profile.architecture.preprocessing_fingerprint
            or state.training_fingerprint != training_fingerprint(cohort)):
        raise ValueError(f"{role}: frozen preprocessing fingerprint mismatch.")
    expected_ids = frozen.identity.sample_ids
    selected = set(expected_ids)
    scoped = cohort.model_copy(update={"records": tuple(
        row for row in cohort.records if _in_training_scope(row.split, row.sample_id, selected)
    )})
    split_provenance = {"role_split": "provider_split"}
    if group is not None:
        scoped, split_provenance = _verify_group_run(saved, profile, cohort, frozen, group, progress)
    train, validation, counts = _views(profile, scoped, state, frozen)
    recorded = saved["samples"]
    if not isinstance(recorded, dict):
        raise ValueError(f"{role}: invalid run.json samples metadata.")
    if counts != recorded:
        mismatches = [split for split in ("train", "validation") if counts.get(split) != recorded.get(split)]
        details = ", ".join(mismatches) or "split keys"
        raise ValueError(f"{role}: reconstructed sample counts or sample fingerprints differ from run.json samples "
                         f"for {details}.")
    if validation is None:
        raise ValueError(f"{role}: no Validation view was reconstructed for FrozenValidation.")
    actual_ids = tuple(sorted(row.sample_id for row in validation.records))
    if actual_ids != expected_ids:
        raise ValueError(f"{role}: reconstructed Validation sample IDs differ from FrozenValidation "
                         f"(expected {len(expected_ids)}, got {len(actual_ids)}).")
    if group is None and any(row.split != row.provider_split for row in (*train.records, *validation.records)):
        raise ValueError(f"{role}: reconstructed sample split differs from the original provider split.")

    selected = saved["selected_checkpoint"]
    epoch = selected["epoch"]
    if type(epoch) is not int or not 1 <= epoch <= saved["completed_epochs"]:
        raise ValueError(f"{role}: selected checkpoint epoch is invalid.")
    folder = f"epoch-{epoch:06d}"
    model_path, report_path = run / folder / "model.pt", run / folder / "validation.json"
    recorded_report = saved["validation_report"]
    if (selected["path"] != f"{folder}/model.pt" or recorded_report["path"] != f"{folder}/validation.json"
            or any(not path.resolve().is_relative_to(run.resolve()) for path in (model_path, report_path))
            or file_sha(model_path) != selected["sha256"] or file_sha(report_path) != recorded_report["sha256"]):
        raise ValueError(f"{role}: selected model or Validation report fingerprint mismatch.")
    report = read_artifact(report_path, ExperimentReport)
    protocol = report.protocol
    if (
        not isinstance(protocol, ProductProtocol) or protocol.cohort != frozen.identity
        or protocol.threshold != profile.architecture.fusion.threshold
        or protocol.target_policy != profile.architecture.target_policy
        or protocol.preprocessing_fingerprint != state.fingerprint
        or report.status != "measured" or report.branch != profile.branch or report.product_model is None
        or report.product_model.checkpoint_sha256 != selected["sha256"]
        or report.product_model.architecture_fingerprint != saved["architecture_fingerprint"]
        or report.fusion != profile.architecture.fusion or report.macro_metric("auroc") is None
    ):
        raise ValueError(f"{role}: selected Validation report differs from the run protocol.")
    return AnalysisRun(role, model_path, state, profile, train.records, validation, protocol, {
        "epoch": epoch, "checkpoint": selected["path"], "checkpoint_sha256": selected["sha256"],
        "validation_sha256": recorded_report["sha256"],
        "run_metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "architecture_fingerprint": saved["architecture_fingerprint"],
        "preprocessing_fingerprint": state.fingerprint, "training_seed": profile.architecture.seed,
        "training_samples": counts["train"], "validation_samples": counts["validation"],
        "selected_macro_auroc": report.macro_metric("auroc"), "selected_macro_f1": report.macro_metric("f1"),
        "protocol": protocol.model_dump(mode="json"),
        **split_provenance,
    }, saved["architecture_fingerprint"])


def _load_leakage_report(raw: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (ValueError, RecursionError):
        raise ValueError("leakage.json must contain a valid JSON document.") from None
    if not isinstance(payload, dict):
        raise ValueError("leakage.json must contain a JSON object.")
    fingerprint = payload.get("fingerprint")
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)):
        raise ValueError("leakage.json fingerprint must be a lowercase SHA-256 digest.")
    report = payload.get("report")
    if not isinstance(report, dict):
        raise ValueError("leakage.json report must be an object.")
    try:
        expected = digest_json(report)
    except RecursionError:
        raise ValueError("leakage.json report is nested too deeply.") from None
    if fingerprint != expected:
        raise ValueError("leakage.json fingerprint does not match report contents.")
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("leakage.json report.schema_version must be 1.")
    if report.get("dataset") != "aihub-smt-71900":
        raise ValueError("leakage.json report.dataset must be aihub-smt-71900.")
    for key in ("samples", "groups", "cross_split_groups"):
        if not isinstance(report.get(key), list):
            raise ValueError(f"leakage.json report.{key} must be an array.")
    if "configuration" in report and not isinstance(report["configuration"], dict):
        raise ValueError("leakage.json report.configuration must be an object.")

    samples = report["samples"]
    for index, sample in enumerate(samples):
        field = f"report.samples[{index}]"
        if not isinstance(sample, dict):
            raise ValueError(f"leakage.json {field} must be an object.")
        split = sample.get("original_split")
        if not isinstance(split, str) or split not in SPLIT_MAP:
            raise ValueError(f"leakage.json {field}.original_split must be a supported provider split.")
        if not isinstance(sample.get("sample_id"), str) or not sample["sample_id"]:
            raise ValueError(f"leakage.json {field}.sample_id must be a nonempty string.")
        for key in ("paths", "hashes", "groups"):
            if not isinstance(sample.get(key), dict):
                raise ValueError(f"leakage.json {field}.{key} must be an object.")
        for key in MODALITIES:
            if not isinstance(sample["paths"].get(key), str):
                raise ValueError(f"leakage.json {field}.paths.{key} must be a string.")
            if key in sample["hashes"] and not isinstance(sample["hashes"][key], str):
                raise ValueError(f"leakage.json {field}.hashes.{key} must be a string.")

    ids: set[str] = set()
    cross_split: set[str] = set()
    for index, group in enumerate(report["groups"]):
        field = f"report.groups[{index}]"
        if not isinstance(group, dict):
            raise ValueError(f"leakage.json {field} must be an object.")
        group_id = group.get("group_id")
        if not isinstance(group_id, str) or not group_id or group_id in ids:
            raise ValueError(f"leakage.json {field}.group_id must be a unique nonempty string.")
        ids.add(group_id)
        method = group.get("method")
        if not isinstance(method, str) or method not in METHODS:
            raise ValueError(f"leakage.json {field}.method must be a supported audit method.")
        members = group.get("members")
        if (not isinstance(members, list)
                or any(type(i) is not int or not 0 <= i < len(samples) for i in members)
                or len(set(members)) != len(members)):
            raise ValueError(f"leakage.json {field}.members must contain unique valid sample indices.")
        if len({samples[i]["original_split"] for i in members}) > 1:
            cross_split.add(group_id)
    listed = report["cross_split_groups"]
    if (any(not isinstance(item, str) or item not in ids for item in listed)
            or len(set(listed)) != len(listed) or set(listed) != cross_split):
        raise ValueError("leakage.json report.cross_split_groups must list each cross-split group ID exactly once.")
    return report


def project_leakage(
    audit: dict[str, Any], train: tuple[Record, ...], validation: tuple[Record, ...],
) -> dict[str, Any]:
    """Project provider audit groups onto exactly one model's verified data views."""
    if audit["schema_version"] != 1 or audit["dataset"] != "aihub-smt-71900":
        raise ValueError("Expected an SMT 71900 leakage.json audit.")
    samples = audit["samples"]
    indexed: dict[tuple[str, str, str], int] = {}
    for index, sample in enumerate(samples):
        key = sample["original_split"], sample["sample_id"], sample["paths"]["annotation_path"]
        if key in indexed:
            raise ValueError("Repeated source identity in leakage audit.")
        indexed[key] = index
    membership: dict[int, Record] = {}
    for record in (*train, *validation):
        index = indexed.get((record.provider_split, record.sample_id, record.annotation_path))
        if index is None:
            raise ValueError(f"Leakage audit is missing cohort sample {record.sample_id}.")
        sample = samples[index]
        for field, path, sha in (
            ("image_path", record.image_path, record.image_sha256),
            ("annotation_path", record.annotation_path, record.annotation_sha256),
            ("sensor_path", record.sensor_path, record.sensor_sha256),
        ):
            if sample["paths"][field] != path or (sha is not None and sample["hashes"].get(field) != sha):
                raise ValueError(f"Leakage audit source differs for {record.sample_id} ({field}).")
        membership[index] = record

    sensor_workbooks: dict[str, list[int]] = defaultdict(list)
    for index in membership:
        if sha := samples[index]["hashes"].get("sensor_path"):
            sensor_workbooks[sha].append(index)
    workbook_groups = [
        {"method": "sensor_workbook_sha256", "members": members,
         "group_id": digest_json(["sensor_workbook_sha256", sha])}
        for sha, members in sorted(sensor_workbooks.items()) if len(members) > 1
    ]
    groups = []
    methods: dict[str, list[dict[str, Any]]] = {name: [] for name in (*METHODS, "sensor_workbook_sha256")}
    for group in (*audit["groups"], *workbook_groups):
        if group["method"] not in methods:
            raise ValueError("Unknown leakage audit method.")
        members = group["members"]
        if len(set(members)) != len(members) or any(type(i) is not int or not 0 <= i < len(samples) for i in members):
            raise ValueError("Invalid sample indices in leakage audit group.")
        projected: dict[str, list[str]] = defaultdict(list)
        for index in members:
            if index in membership:
                row = membership[index]
                projected[row.split].append(row.sample_id)
        if projected["train"] and projected["validation"]:
            entry = {
                "group_id": group["group_id"], "method": group["method"],
                "train_ids": sorted(projected["train"]), "validation_ids": sorted(projected["validation"]),
            }
            groups.append(entry)
            methods[group["method"]].append(entry)
    totals = {
        method: {
            "group_count": len(entries),
            **{f"{split}_ids": sorted({sid for entry in entries for sid in entry[f"{split}_ids"]})
               for split in ("train", "validation")},
        }
        for method, entries in methods.items()
    }
    return {
        "train_ids": sorted(row.sample_id for row in train),
        "validation_ids": sorted(row.sample_id for row in validation),
        "groups": groups, "methods": totals,
        "metadata_missing": {
            field: {split: sum(record.split == split and field not in samples[index]["groups"]
                               for index, record in membership.items()) for split in ("train", "validation")}
            for field in ("manufacturer", "capture_timestamp", "capture_group", "production_session", "sensor_exact")
        },
        "any_method_ids": {split: sorted({sid for entry in groups for sid in entry[f"{split}_ids"]})
                           for split in ("train", "validation")},
    }


def sensor_donors(
    sample_ids: tuple[str, ...], manufacturers: dict[str, str], seed: int, *, within_manufacturer: bool,
) -> dict[str, str]:
    """Permute each eligible pool with no fixed points; retain singleton/unknown donors."""
    pools: dict[str, list[str]] = defaultdict(list)
    donors = dict(zip(sample_ids, sample_ids, strict=True))
    for sid in sorted(sample_ids):
        key = manufacturers[sid] if within_manufacturer else "all"
        if key:
            pools[key].append(sid)
    rng = random.Random(seed)
    for key in sorted(pools):
        pool = pools[key]
        rng.shuffle(pool)
        donors.update(zip(pool, pool[1:] + pool[:1], strict=True))
    return donors


def score_run(
    run: AnalysisRun, device: str, batch_size: int, donors: dict[str, dict[str, str]],
    progress: ProgressFactory, checkpoint_dir: Path | None = None,
) -> tuple[dict[str, Scores], dict[str, str]]:
    import torch

    from .checkpoint import load_checkpoint
    from .training_torch import _branch_logits

    model = load_checkpoint(run.model_path, run.state, expected_config=run.profile.architecture)
    if (model.architecture_fingerprint != run.architecture_fingerprint
            or image_preprocessing_fingerprint(model.config.image, model.image_encoder.input_contract)
            != run.protocol.image_preprocessing_fingerprint
            or model.branch_status(run.profile.branch) != "trained"):
        raise ValueError(f"{run.role}: checkpoint architecture or trained branch differs from the run receipt.")
    resolved = resolve_device(torch, TrainConfig(device=device), REGISTRY[ModelName.FASTER_RCNN])
    model.to(resolved).eval()
    skipped = {}
    conditions = {"original": run.profile.branch, **{key: "paired" for key in donors}}
    if donors:
        if model.branch_status("image") == "trained":
            conditions["sensor_removed"] = "image"
        else:
            skipped["sensor_removed"] = "Missing sensor routes to the checkpoint's untrained image branch."
    windows = {
        row.sample_id: transform_sensor(row.sequence, row.capture_timestamp, run.state)
        for row in run.validation.records
    } if donors else {}
    scores: dict[str, Scores] = {condition: {} for condition in conditions}
    if checkpoint_dir is not None:
        accelerator = torch.cuda.get_device_name(resolved) if resolved.startswith("cuda") else resolved
        fingerprint = digest_json([run.provenance, _environment(), resolved, accelerator, batch_size, donors, conditions])
        checkpoint_dir = checkpoint_dir / fingerprint
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report = progress(f"Score {run.role} checkpoint ({len(conditions)} conditions)")
    total = len(run.validation)
    report(0, total)
    with torch.inference_mode():
        for start in range(0, total, batch_size):
            end = min(total, start + batch_size)
            samples = [run.validation[index] for index in range(start, end)]
            ids = tuple(sample.sample_id for sample in samples)
            batch_path = checkpoint_dir / f"{start:08d}.json" if checkpoint_dir is not None else None
            if batch_path is not None and batch_path.exists():
                cached = _read_batch(batch_path, ids, set(conditions))
                for condition, predictions in cached.items():
                    scores[condition].update(predictions)
                report(end, total)
                continue
            batch_scores = {}
            for condition, branch in conditions.items():
                variant = samples
                if condition == "sensor_removed":
                    variant = [replace(sample, sensor=None) for sample in samples]
                elif condition in donors:
                    variant = [replace(sample, sensor=windows[donors[condition][sample.sample_id]]) for sample in samples]
                output = model(collate(variant))
                if output.sample_ids != ids or output.split != "validation":
                    raise ValueError("Scoring must preserve batch sample IDs and the Validation split.")
                batch_scores[condition] = dict(zip(output.sample_ids, map(tuple, _branch_logits(
                    output, branch, len(samples),
                ).sigmoid().tolist()), strict=True))
            if batch_path is not None:
                _save_batch(batch_path, batch_scores)
            for condition, predictions in batch_scores.items():
                scores[condition].update(predictions)
            report(end, total)
    return scores, skipped


def _read_batch(path: Path, ids: tuple[str, ...], conditions: set[str]) -> dict[str, Scores]:
    payload = json.loads(path.read_bytes())
    scores = payload["scores"]
    if (payload["fingerprint"] != digest_json(scores) or scores.keys() != conditions
            or any(set(rows) != set(ids) for rows in scores.values())
            or any(len(vector) != len(CLASSES) or any(
                type(score) not in (int, float) or not 0 <= score <= 1 for score in vector
            ) for rows in scores.values() for vector in rows.values())):
        raise ValueError("Analysis checkpoint batch is corrupt. Use a fresh --checkpoint-dir.")
    return {condition: {sid: tuple(row) for sid, row in rows.items()} for condition, rows in scores.items()}


def _save_batch(path: Path, scores: dict[str, Scores]) -> None:
    payload = {"scores": scores, "fingerprint": digest_json(scores)}
    with NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json.dumps(payload, allow_nan=False).encode())
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.link(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def group_metrics(
    frozen: FrozenValidation, protocol: ProductProtocol, scores: Scores, ids: list[str],
) -> dict[str, Any]:
    subset = set(ids)
    if subset:
        selection = frozen.model_copy(update={"samples": tuple(row for row in frozen.samples if row.sample_id in subset)})
        metrics = classification_metrics(selection, protocol.model_copy(update={"cohort": selection.identity}),
                                         {sid: scores[sid] for sid in subset})
    else:
        metrics = tuple(_metric(name, (), protocol.threshold) for name in CLASSES.values())
    per_class = []
    for metric in metrics:
        (tn, fp), (fn, tp) = metric.confusion_matrix
        per_class.append({**metric.model_dump(mode="json"), "positive": tp + fn, "negative": tn + fp})
    macro = {}
    for name in ("auroc", "f1", "precision", "recall"):
        values = [row[name] for row in per_class if row[name] is not None]
        macro[f"macro_{name}"] = sum(values) / len(values) if values else None
        macro[f"defined_{name}_classes"] = len(values)
    return {"sample_count": len(ids), "sample_ids": ids, **macro, "classes": per_class}

LIMITATIONS = (
    "Similarity membership is the projection of original audit connected components. "
    "A connection may be indirect or pass through excluded records. Direct cohort pair counts are unavailable.",
    "Leakage methods overlap. Use the union counts for unique samples. Shared metadata is a risk indicator.",
    "Unknown metadata is retained as its own stratum. The audit has no verified process-condition categories.",
    "Sensor exchange can break the image/sensor relationship and move inputs outside the training distribution. "
    "Differences measure sensitivity to the intervention on this Validation cohort.",
    "These Validation samples were used for checkpoint selection; independent generalization remains unmeasured.",
)


def _strata(
    projected: dict[str, Any], manufacturers: dict[str, str], dates: dict[str, str],
) -> list[tuple[str, str, list[str]]]:
    ids = projected["validation_ids"]
    similar = set(projected["methods"]["image_similarity"]["validation_ids"])
    result = [
        ("overall", "all", ids),
        ("image_similarity", "cross_group", [sid for sid in ids if sid in similar]),
        ("image_similarity", "outside_cross_group", [sid for sid in ids if sid not in similar]),
    ]
    for kind, metadata in (("manufacturer", manufacturers), ("capture_date", dates)):
        grouped: dict[str, list[str]] = defaultdict(list)
        for sid in ids:
            grouped[metadata[sid] or "unknown"].append(sid)
        result.extend((kind, key, members) for key, members in sorted(grouped.items()))
    return result


def _table(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    write_json(path.with_suffix(".json"), {"results": rows})
    columns = (
        "model", "kind", "group", "status", "reason", "sample_count", "macro_auroc", "macro_f1",
        "macro_precision", "macro_recall", "delta_macro_auroc", "delta_macro_f1", "changed_donors",
        "name", "positive", "negative", "missing_labels", "auroc", "f1", "precision", "recall", "confusion_matrix",
    )
    _table(path.with_suffix(".csv"), columns, [
        {**row, **metric} for row in rows for metric in row.get("classes", []) or [{}]
    ])


def analyze(
    root: Path, audit_path: Path, cohort_path: Path, frozen_path: Path, runs: dict[ModelRole, Path],
    out: Path, *, mode: AnalysisMode, seed: int, batch_size: int, device: str,
    credentials: Credentials, progress: ProgressFactory, checkpoint_dir: Path | None = None,
    group: GroupAnalysisInputs | None = None,
) -> None:
    """Verify run identities and publish a complete analysis in a fresh directory."""
    if batch_size < 1 or seed < 0 or mode not in ("all", "leakage", "groups", "ablation"):
        raise ValueError("Use a positive batch size, a nonnegative seed and a supported analysis mode.")
    required = {"concat"} if mode == "ablation" else {"image", "concat"}
    if not required <= runs.keys():
        raise ValueError("Supply both --image-run and --concat-run, or only --concat-run for ablation.")
    if mode == "ablation":
        runs = {"concat": runs["concat"]}
    inputs = (root, audit_path, cohort_path, frozen_path, *runs.values())
    if group is not None:
        inputs += (group.verification, group.bundle, group.detector.parent, *group.prior_test_manifests)
    for path in (*inputs, out):
        credentials.protect_path(str(path.resolve()), "Multimodal analysis path")
    for path in inputs:
        require_separate_output(path, out)
    protect_source_data(out, Path.cwd())
    if out.exists() or out.is_symlink():
        raise ValueError("Choose a fresh analysis output directory.")
    if checkpoint_dir is not None:
        credentials.protect_path(str(checkpoint_dir.resolve()), "Multimodal analysis checkpoint path")
        for path in (*inputs, out):
            require_separate_output(path, checkpoint_dir)
        require_separate_output(checkpoint_dir, out)
        protect_source_data(checkpoint_dir, Path.cwd())

    prepare = progress("Verify cohort, audit and selected checkpoints")
    prepare(0, len(runs) + 1)
    cohort = Cohort.load(cohort_path)
    frozen = read_artifact(frozen_path, FrozenValidation)
    credentials.protect_path(json.dumps(frozen.identity.sample_ids), "Validation sample IDs")
    audit_bytes = audit_path.read_bytes()
    audit = _load_leakage_report(audit_bytes)
    prepare(1, len(runs) + 1)
    inspected = {}
    leakage = {}
    for done, (role, path) in enumerate(runs.items(), 2):
        verify = progress(f"Verify {role} run and project leakage")
        run = _inspect_run(role, path, root, cohort_path, frozen_path, cohort, frozen, group=group, progress=verify)
        inspected[role] = run
        leakage[role] = project_leakage(audit, run.train, run.validation.records)
        prepare(done, len(runs) + 1)

    grouped, ablations = [], []
    for role, run in inspected.items():
        if mode == "leakage":
            break
        progress(f"Prepare {role} checkpoint scoring")
        manufacturers = {row.sample_id: row.source_group for row in run.validation.records}
        dates = {row.sample_id: (row.capture_timestamp or "")[:10] for row in run.validation.records}
        ids = frozen.identity.sample_ids
        donors = {}
        if role == "concat" and mode in ("all", "ablation"):
            donors = {
                "within_manufacturer_shuffle": sensor_donors(ids, manufacturers, seed, within_manufacturer=True),
                "global_shuffle": sensor_donors(ids, manufacturers, seed, within_manufacturer=False),
            }
        scores, skipped = score_run(run, device, batch_size, donors, progress, checkpoint_dir)
        progress(f"Compute {role} analysis metrics")
        if mode in ("all", "groups"):
            grouped.extend(
                {"model": role, "kind": kind, "group": key, "status": "measured" if members else "empty",
                 **group_metrics(frozen, run.protocol, scores["original"], members)}
                for kind, key, members in _strata(leakage[role], manufacturers, dates)
            )
        if donors:
            baseline = group_metrics(frozen, run.protocol, scores["original"], list(ids))
            for condition, predictions in scores.items():
                result = group_metrics(frozen, run.protocol, predictions, list(ids))
                mapping = donors.get(condition)
                ablations.append({
                    "model": role, "kind": "sensor_intervention", "group": condition, "status": "measured", **result,
                    "donor_by_sample": mapping,
                    "changed_donors": sum(sid != donor for sid, donor in mapping.items()) if mapping else None,
                    "prediction_fingerprint": digest_json(predictions),
                    **{f"delta_{key}": result[key] - baseline[key]
                       if result[key] is not None and baseline[key] is not None else None
                       for key in ("macro_auroc", "macro_f1")},
                })
            ablations.extend({"model": role, "kind": "sensor_intervention", "group": condition,
                              "status": "skipped", "reason": reason, "sample_count": len(ids), "sample_ids": list(ids)}
                             for condition, reason in skipped.items())

    progress("Assemble multimodal analysis reports")
    leakage_rows = []
    for role, result in leakage.items():
        for method, stats in result["methods"].items():
            leakage_rows.append({
                "model": role, "method": method, "groups": stats["group_count"],
                "train": len(stats["train_ids"]), "validation": len(stats["validation_ids"]),
            })
        leakage_rows.append({
            "model": role, "method": "any_method", "groups": len(result["groups"]),
            **{split: len(result["any_method_ids"][split]) for split in ("train", "validation")},
        })
    config = {
        "schema_version": 1, "mode": mode, "seed": seed, "batch_size": batch_size, "device": device,
        "audit_sha256": hashlib.sha256(audit_bytes).hexdigest(), "audit_configuration": audit.get("configuration"),
        "audit_fingerprint": digest_json(audit), "audit_source_identity_split": "provider_split",
        "cohort_fingerprint": cohort.fingerprint, "validation_identity": frozen.identity.model_dump(mode="json"),
        "models": {role: run.provenance for role, run in inspected.items()}, **_environment(),
        "group_policy": {"image_similarity": "project_provider_components-v1", "manufacturer": "cohort.source_group",
                         "capture_date": "date in cohort.capture_timestamp", "process_conditions": "unavailable"},
        "sensor_shuffle_policy": "sorted seed-shuffled cycles; unknown manufacturer and singletons retain own sensor",
        "averaging_policy": "macro mean over defined class metrics; JSON null and CSV blank for undefined values",
        "limitations": list(LIMITATIONS),
    }
    summary = {
        "leakage_counts": leakage_rows,
        "group_metrics": [{key: value for key, value in row.items() if key not in ("classes", "sample_ids")}
                          for row in grouped if row["kind"] in ("overall", "image_similarity")],
        "ablation_metrics": [{key: value for key, value in row.items()
                              if key not in ("classes", "sample_ids", "donor_by_sample")} for row in ablations],
        "limitations": list(LIMITATIONS),
    }
    credentials.protect_path(json.dumps([config, leakage, grouped, ablations, summary], ensure_ascii=False),
                             "Multimodal analysis report")
    publish = progress("Publish multimodal analysis reports")
    publish(0, 1)
    with fresh_output(out, root) as working:
        write_json(working / "analysis.json", config)
        write_json(working / "summary.json", summary)
        write_json(working / "leakage.json", {"models": leakage, "counts": leakage_rows})
        _table(working / "leakage.csv", ("model", "method", "groups", "train", "validation"), leakage_rows)
        if mode in ("all", "groups"):
            _write_metrics(working / "groups", grouped)
            _write_metrics(working / "manufacturers", [row for row in grouped if row["kind"] == "manufacturer"])
        if mode in ("all", "ablation"):
            _write_metrics(working / "ablation", ablations)
    publish(1, 1)
