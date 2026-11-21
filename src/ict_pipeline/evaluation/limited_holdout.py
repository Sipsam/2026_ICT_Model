"""Limited Internal Holdout evaluation without changing the strict Test Release Gate."""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Literal

from ..data.model import Progress
from ..data.preprocessing.common import raw_file_sha256 as _sha
from ..data.preprocessing.leakage import AnalysisConfig, analyze_leakage
from ..data.preprocessing.policy import NAMES
from ..data.preprocessing.revision import read_publication
from ..data.preprocessing.split import REQUIRED_GROUPS
from ..data.test_release_gate import preflight_test_release
from ..evaluation.product import _metric
from ..multimodal.analysis import ProgressFactory, _inspect_run, match_leakage_records, score_run
from ..multimodal.audit import Cohort, audit_cohort
from ..multimodal.dataset import MultimodalDataset
from ..multimodal.preprocessing import PreprocessingState
from ..pipeline.model import PipelineError
from ..pipeline.store import _reject_link, digest_json


LIMITATION_FIELDS: dict[str, Any] = {
    "evaluation_origin": "internal_same_source",
    "evaluation_protocol": "limited_internal_holdout",
    "historical_reservation_status": "unverified",
    "historical_payload_seal_verified": False,
    "external_independence_verified": False,
    "production_session_independence_verified": False,
}
FIXED_TEST_COUNT = 1472


def _read_json(path: Path) -> dict:
    _reject_link(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object.")
    return value


def revalidate_limited_holdout(
    *,
    root: Path,
    bundle: Path,
    revision_id: str,
    preflight: Path,
    group_verification: Path,
    cohort: Path,
    frozen_validation: Path,
    preprocessing: Path,
    image_run: Path,
    concat_run: Path,
    detector_initialization: Path | None = None,
    prior_manifests: tuple[Path, ...] = (),
) -> tuple[dict, dict]:
    """Revalidate receipt and fixed checkpoints before any Test payload is opened."""
    preflight_payload = _read_json(preflight)
    if preflight_payload.get("schema_version") != 1 or preflight_payload.get("stage") != "preflight":
        raise ValueError("A valid Test preflight receipt is required.")
    if digest_json({key: value for key, value in preflight_payload.items() if key != "fingerprint"}) != preflight_payload.get("fingerprint"):
        raise ValueError("Preflight receipt fingerprint mismatch.")
    if preflight_payload.get("passed") is not True:
        raise ValueError("Limited Holdout evaluation requires a passing existing Preflight.")
    if preflight_payload.get("final_evaluation_allowed") is not False:
        raise ValueError("Existing strict Test Release Gate final-evaluation state is invalid.")

    live = preflight_test_release(
        root=root, bundle=bundle, revision_id=revision_id,
        group_verification=group_verification, cohort=cohort,
        frozen_validation=frozen_validation, preprocessing=preprocessing,
        image_run=image_run, concat_run=concat_run,
        prior_manifests=prior_manifests, detector_initialization=detector_initialization,
    )
    pinned = (
        "test_sample_ids", "test_sample_count", "group_split_revision", "publication_fingerprint",
        "manifest_fingerprint", "cohort_fingerprint", "frozen_validation_fingerprint",
        "preprocessing_fingerprint", "group_verification_fingerprint",
        "prior_test_manifest_fingerprints", "evaluation_plan",
    )
    if any(live.get(field) != preflight_payload.get(field) for field in pinned):
        raise ValueError("Preflight receipt disagrees with live split, checkpoint or receipt revalidation.")
    image_epoch = live["evaluation_plan"]["image_only_checkpoint"]["epoch"]
    concat_epoch = live["evaluation_plan"]["sensor_concat_checkpoint"]["epoch"]
    if type(image_epoch) is not int or image_epoch != 27:
        raise ValueError("Limited Holdout requires the fixed Image-only Epoch 27 checkpoint.")
    if type(concat_epoch) is not int or concat_epoch != 15:
        raise ValueError("Limited Holdout requires the fixed Sensor Concat Epoch 15 checkpoint.")
    if live["evaluation_plan"]["threshold"] != 0.5:
        raise ValueError("Limited Holdout requires threshold 0.5.")
    if tuple(live["evaluation_plan"]["class_order"]) != tuple(NAMES):
        raise ValueError("Limited Holdout class order differs from the canonical five classes.")
    if len(live["test_sample_ids"]) != FIXED_TEST_COUNT:
        raise ValueError("Limited Holdout requires exactly 1472 fixed Test samples.")
    if live["test_sample_ids"] != preflight_payload["test_sample_ids"]:
        raise ValueError("Live fixed Test sample set differs from the Preflight receipt.")
    if live["evaluation_plan"]["test_sample_fingerprint"] != digest_json(live["test_sample_ids"]):
        raise ValueError("Fixed Test sample fingerprint mismatch.")
    return preflight_payload, live


def snapshot_test_payload(
    root: Path, bundle: Path, revision_id: str, test_ids: Iterable[str],
    *, progress: Progress | None = None,
) -> dict[str, Any]:
    """Create an evaluation-time snapshot; never label this as a historical seal."""
    root = root.resolve()
    _reject_link(root)
    revision, _, rows = read_publication(bundle, revision_id)
    test_rows = [row for row in rows if row.get("split") == "test"]
    ids = sorted(str(row["sample_id"]) for row in test_rows if row.get("sample_id"))
    if ids != sorted(set(test_ids)) or len(ids) != FIXED_TEST_COUNT or len(set(ids)) != FIXED_TEST_COUNT:
        raise ValueError("Test sample set differs from the fixed Preflight or manifest.")

    coco = bundle / "detection" / "coco" / "annotations" / "instances_test.json"
    _reject_link(coco)
    if not coco.is_file():
        raise ValueError("Missing Test COCO annotation file.")
    document = json.loads(coco.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("images"), list) or not isinstance(document.get("annotations"), list):
        raise ValueError("Malformed Test COCO annotation file.")

    sample_files: list[dict[str, str]] = []
    if progress:
        progress(0, len(test_rows))
    for done, row in enumerate(test_rows, 1):
        sample_id = str(row["sample_id"])
        record: dict[str, str] = {"sample_id": sample_id}
        for kind, key in (("image", "image_path"), ("annotation", "annotation_path"), ("sensor", "sensor_path")):
            relative = str(row.get(key, ""))
            if not relative:
                raise ValueError(f"{sample_id}: missing {kind} path.")
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"{sample_id}: unsafe {kind} payload path.")
            resolved = (root / path).resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                raise ValueError(f"{sample_id}: missing or escaping {kind} payload.")
            _reject_link(resolved)
            record[f"{kind}_sha256"] = _sha(resolved)
        sample_files.append(record)
        if progress:
            progress(done, len(test_rows))

    snapshot = {
        "schema_version": 1,
        "kind": "limited_internal_holdout_payload_snapshot",
        "snapshot_status": "evaluation_time_only",
        "not_historical_original_hash_index": True,
        "historical_payload_seal_verified": False,
        "revision_id": revision["revision_id"],
        "test_sample_count": len(ids),
        "test_sample_ids": ids,
        "test_sample_fingerprint": digest_json(ids),
        "payload_file_fingerprint": digest_json(sample_files),
        "coco_test_sha256": _sha(coco),
        "samples": sample_files,
    }
    snapshot["fingerprint"] = digest_json(snapshot)
    return snapshot


def verify_test_payload_integrity(
    *, root: Path, bundle: Path, revision_id: str, snapshot: dict,
    split_cohort: Cohort, progress: ProgressFactory | None = None,
) -> dict[str, Any]:
    """Block on cross-split hashes, sample mismatch, similarity or malformed evidence."""
    test_ids = set(snapshot["test_sample_ids"])
    if split_cohort.revision_id != revision_id:
        raise ValueError("Audited cohort differs from the pinned Holdout revision.")
    test_records = [record for record in split_cohort.records if record.split == "test"]
    record_ids = {record.sample_id for record in test_records}
    if record_ids != test_ids or len(test_records) != len(test_ids):
        raise ValueError("Audited Test cohort differs from the fixed 1472-sample manifest.")
    if any(not record.detection_candidate or not record.image_available or record.targets is None
           for record in test_records):
        raise ValueError("Test records require candidate status, an image and the label contract.")

    duplicate: list[dict[str, Any]] = []
    hash_keys = (
        ("image_sha256", "image"), ("annotation_sha256", "annotation"),
        ("sensor_sha256", "sensor"),
    )
    for field, kind in hash_keys:
        groups: dict[str, list[tuple[str, bool]]] = {}
        for record in split_cohort.records:
            value = getattr(record, field)
            if value:
                groups.setdefault(str(value), []).append((record.split, record.detection_candidate))
        for value, assignments in groups.items():
            includes_test = any(split == "test" for split, _ in assignments)
            splits = {split for split, candidate in assignments if includes_test or candidate}
            if len(splits) > 1:
                duplicate.append({"kind": kind, "sha256": value, "splits": sorted(splits)})
    if duplicate:
        raise ValueError("Cross-split file hash duplication detected: " + json.dumps(duplicate[:10]))
    if any(record.quarantined for record in test_records):
        raise ValueError("Test records are quarantined; the Holdout split must not be changed.")

    audit = analyze_leakage(
        root, bundle / "manifest.csv", root, AnalysisConfig(image_hamming=4, image_mae=8.0,
                                                            sensor_scale=1.0, verify_xlsx=True),
        progress("Train/Validation/Test leakage: inspect source evidence") if progress else None,
        similarity_progress=progress, provider_column="original_split",
    )
    membership = match_leakage_records(audit, split_cohort.records)
    cross = []
    block_methods = {*REQUIRED_GROUPS, "image_similarity", "sensor_similarity"}
    for group in audit["groups"]:
        if group["method"] not in block_methods:
            continue
        members = [membership[index] for index in group["members"]]
        if all(record.split != "test" for record in members):
            members = [record for record in members if record.detection_candidate]
        splits = {record.split for record in members}
        if len(splits) > 1:
            cross.append({
                "group_id": group["group_id"], "method": group["method"],
                "splits": sorted(splits),
                "test_ids": sorted(record.sample_id for record in members if record.split == "test"),
                "development_ids": sorted(record.sample_id for record in members
                                           if record.split in ("train", "validation")),
            })
    if cross:
        raise ValueError("Cross-split similarity or leakage evidence detected: " + json.dumps(cross[:20]))

    coco = json.loads((bundle / "detection" / "coco" / "annotations" / "instances_test.json").read_text())
    coco_ids = sorted(str(image["file_name"]).rsplit(".", 1)[0] for image in coco["images"])
    if coco_ids != sorted(test_ids):
        raise ValueError("COCO Test sample set differs from the fixed manifest.")
    coco_categories: dict[str, set[int]] = {sample_id: set() for sample_id in coco_ids}
    coco_boxes: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {
        sample_id: [] for sample_id in coco_ids
    }
    coco_annotation_count = 0
    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        image = next((row for row in coco["images"] if row["id"] == image_id), None)
        if image is None:
            raise ValueError("COCO Test annotation references a missing image.")
        sample_id = str(image["file_name"]).rsplit(".", 1)[0]
        if sample_id not in coco_categories:
            continue
        coco_annotation_count += 1
        category_id = annotation["category_id"]
        if type(category_id) is not int or category_id not in range(1, len(NAMES) + 1):
            raise ValueError("COCO Test annotation has an invalid canonical class ID.")
        coco_categories[sample_id].add(category_id)
        box = annotation.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("COCO Test annotation has a malformed bbox.")
        try:
            coco_boxes[sample_id].append((category_id, tuple(float(value) for value in box)))
        except (TypeError, ValueError):
            raise ValueError("COCO Test annotation bbox values must be numeric.") from None
    for record in test_records:
        original_categories = {target.category for target in record.targets or ()}
        if coco_categories[record.sample_id] != original_categories:
            raise ValueError(f"COCO/original Test label mismatch for {record.sample_id}.")
        original_boxes = [(target.category, target.box) for target in record.targets or ()]
        if sorted(coco_boxes[record.sample_id]) != sorted(original_boxes):
            raise ValueError(f"COCO/original Test box mismatch for {record.sample_id}.")
        if not record.image_available or record.image_size is None or record.sequence is None:
            raise ValueError(f"Test modality or sensor alignment missing for {record.sample_id}.")
        if record.alignment != "matched" or record.sensor_sha256 is None:
            raise ValueError(f"Test sensor is not explicitly aligned for {record.sample_id}.")
    if len(coco_categories) != FIXED_TEST_COUNT:
        raise ValueError("COCO Test image count differs from the fixed Holdout cohort.")
    return {
        "file_hash_duplicates": duplicate,
        "cross_split_similarity": cross,
        "coco_test_sample_count": len(coco_ids),
        "coco_annotation_count": coco_annotation_count,
    }


def _subset_metrics(frozen_ids: list[str], scores: dict[str, tuple[float, ...]],
                    records: dict[str, dict[str, Any]], subset_ids: Iterable[str]) -> dict[str, Any]:
    ids = [sample_id for sample_id in frozen_ids if sample_id in set(subset_ids)]
    metrics = tuple(_metric(name, (
        (bool(records[sample_id]["labels"][index]), scores[sample_id][index])
        for sample_id in ids
    ), 0.5) for index, name in enumerate(NAMES))
    per_class = []
    positive_supported_f1 = []
    all_tp = all_fp = all_fn = 0
    for metric in metrics:
        (tn, fp), (fn, tp) = metric.confusion_matrix
        all_tp += tp
        all_fp += fp
        all_fn += fn
        positive = tp + fn
        negative = tn + fp
        fpr = fp / negative if negative else None
        per_class.append({
            "name": metric.name, "auroc": metric.auroc, "precision": metric.precision,
            "recall": metric.recall, "f1": metric.f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "positive": positive, "negative": negative, "fpr": fpr,
        })
        if positive and metric.f1 is not None:
            positive_supported_f1.append(metric.f1)
    defined_f1 = [row["f1"] for row in per_class if row["f1"] is not None]
    defined_auroc = [row["auroc"] for row in per_class if row["auroc"] is not None]
    return {
        "sample_count": len(ids),
        "macro_auroc": sum(defined_auroc) / len(defined_auroc) if defined_auroc else None,
        "macro_f1": sum(defined_f1) / len(defined_f1) if defined_f1 else None,
        "positive_supported_macro_f1": (
            sum(positive_supported_f1) / len(positive_supported_f1) if positive_supported_f1 else None
        ),
        "micro_f1": (
            2 * all_tp / (2 * all_tp + all_fp + all_fn)
            if (2 * all_tp + all_fp + all_fn) else None
        ),
        "classes": per_class,
    }


def evaluate_limited_holdout(
    *,
    root: Path, bundle: Path, revision_id: str, preflight: Path,
    group_verification: Path, cohort: Path, frozen_validation: Path, preprocessing: Path,
    image_run: Path, concat_run: Path, out: Path, approve_test_access: bool,
    detector_initialization: Path | None = None, prior_manifests: tuple[Path, ...] = (),
    resume: bool = False,
    device: Literal["cpu", "cuda", "rocm"] = "cpu",
    batch_size: int = 32,
    seed: int = 42,
    checkpoint_dir: Path | None = None,
    progress: ProgressFactory | None = None,
) -> dict[str, Any]:
    """Approve, verify, score both fixed FPN checkpoints, and export JSON/CSV summaries."""
    root = root.resolve()
    bundle = bundle.resolve()
    if not approve_test_access:
        raise ValueError("Limited Holdout requires explicit --approve-test-access before opening Test payloads.")

    def stage(name: str) -> Progress:
        if progress is not None:
            return progress(name)
        return lambda completed, total: None

    preflight_report = stage("Revalidate Preflight and fixed checkpoints")
    preflight_report(0, 1)
    preflight_payload, live = revalidate_limited_holdout(
        root=root, bundle=bundle, revision_id=revision_id, preflight=preflight,
        group_verification=group_verification, cohort=cohort,
        frozen_validation=frozen_validation, preprocessing=preprocessing,
        image_run=image_run, concat_run=concat_run,
        detector_initialization=detector_initialization, prior_manifests=prior_manifests,
    )
    preflight_report(1, 1)
    snapshot = snapshot_test_payload(
        root, bundle, revision_id, live["test_sample_ids"], progress=stage("Snapshot Test payload"),
    )
    if batch_size < 1:
        raise ValueError("Use a positive Holdout inference batch size.")
    if seed < 0:
        raise ValueError("Use a nonnegative Holdout seed.")
    if device not in ("cpu", "cuda", "rocm"):
        raise ValueError("Use cpu, cuda, or rocm for Holdout evaluation.")
    if checkpoint_dir is not None:
        checkpoint_dir = checkpoint_dir.expanduser().resolve()
    plan_fingerprint = digest_json({
        "preflight_fingerprint": preflight_payload["fingerprint"],
        "live_evaluation_plan": live["evaluation_plan"],
        "test_snapshot_fingerprint": snapshot["fingerprint"],
        "device": device,
        "batch_size": batch_size,
        "seed": seed,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else "output-default",
    })

    if resume:
        state_path = out / "resume-state.json"
        if not out.is_dir() or not state_path.is_file():
            raise ValueError("Interrupted Holdout output or resume state is missing. Use a fresh --out without --resume.")
        try:
            previous = _read_json(state_path)
        except (ValueError, OSError, PipelineError):
            raise ValueError("Holdout resume state is unreadable. Use a fresh --out without --resume.") from None
        if (previous.get("schema_version") != 1 or previous.get("status") != "running"
                or previous.get("plan_fingerprint") != plan_fingerprint):
            raise ValueError("Holdout resume state, plan or snapshot is incompatible. Use a fresh --out without --resume.")
        if any((out / name).exists() or (out / name).is_symlink() for name in (
            "limited-holdout.json", "payload-snapshot.json", "limited-holdout.csv",
        )):
            raise ValueError("Holdout reports already exist. Preserve them and use a fresh --out without --resume.")
    else:
        if out.exists():
            raise ValueError("Output already exists. Choose a fresh directory.")
        out.mkdir(parents=True)

        with (out / "resume-state.json").open("x", encoding="utf-8") as stream:
            json.dump({
                "schema_version": 1, "status": "running", "plan_fingerprint": plan_fingerprint,
                "preflight_fingerprint": preflight_payload["fingerprint"],
                "test_snapshot_fingerprint": snapshot["fingerprint"],
                "device": device, "batch_size": batch_size, "seed": seed,
            }, stream, indent=2)

    training_cohort = Cohort.load(cohort)
    split_cohort = audit_cohort(
        root, bundle / "manifest.csv", revision_path=bundle / "split.json",
        progress=stage("Check Test payload and published cohort integrity"), isolate_source_capture=False,
    )
    _, _, test_rows = read_publication(bundle, revision_id)
    leakage_report = stage("Check Train/Validation/Test leakage and Test COCO labels")
    leakage_report(0, 1)
    integrity = verify_test_payload_integrity(
        root=root, bundle=bundle, revision_id=revision_id, snapshot=snapshot,
        split_cohort=split_cohort, progress=progress,
    )
    leakage_report(1, 1)

    from ..evaluation.product_protocol import FrozenValidation
    from ..evaluation.product import read_artifact
    from ..multimodal.analysis import GroupAnalysisInputs

    frozen_validation_data = read_artifact(frozen_validation, FrozenValidation)
    state = PreprocessingState.load(preprocessing)
    group_inputs = GroupAnalysisInputs(
        verification=group_verification,
        bundle=bundle,
        detector=detector_initialization if detector_initialization is not None else root,
        prior_test_manifests=prior_manifests,
    )
    image_inspected = _inspect_run(
        "image", image_run, root, cohort, frozen_validation, training_cohort,
        frozen_validation_data, group=group_inputs, progress=stage("Verify Image-only run inputs"),
    )
    concat_inspected = _inspect_run(
        "concat", concat_run, root, cohort, frozen_validation, training_cohort,
        frozen_validation_data, group=group_inputs, progress=stage("Verify Sensor Concat run inputs"),
    )

    if image_inspected.provenance["epoch"] != 27 or concat_inspected.provenance["epoch"] != 15:
        raise ValueError("Checkpoint Epoch 27/15 contract changed before inference.")

    row_by_id = {row["sample_id"]: row for row in test_rows if row["split"] == "test"}
    test_records: dict[str, dict[str, Any]] = {}
    for record in split_cohort.records:
        if record.split != "test":
            continue
        test_records[record.sample_id] = {
            "labels": tuple(
                category in {target.category for target in record.targets or ()}
                for category in range(1, len(NAMES) + 1)
            ),
            "manufacturer": row_by_id[record.sample_id].get("manufacturer", "unknown"),
            "capture_date": (
                record.capture_timestamp[:10] if record.capture_timestamp else
                str(row_by_id[record.sample_id].get("capture_date", "unknown"))
            ),
        }

    ids = tuple(sorted(test_records))
    # Test uses live verified records. Sensor preprocessing stays bound to the pinned Train view.
    scoring_cohort = training_cohort.model_copy(update={"records": (
        *(record for record in training_cohort.records if record.split != "test"),
        *(record for record in split_cohort.records if record.split == "test"),
    )})
    scores: dict[str, dict[str, tuple[float, ...]]] = {"image_only": {}, "sensor_concat": {}}
    batch_checkpoint_root = checkpoint_dir if checkpoint_dir is not None else out / "batch-checkpoints"
    batch_checkpoint_root.mkdir(parents=True, exist_ok=True)

    for role, run in (("image_only", image_inspected), ("sensor_concat", concat_inspected)):
        mode = "image" if role == "image_only" else "paired"
        dataset = MultimodalDataset(
            scoring_cohort, root, "test", mode=mode, preprocessing=None if mode == "image" else state,
            missing="error", sample_ids=frozenset(ids),
        )
        test_run = replace(run, validation=dataset)
        model_name = "Image-only" if role == "image_only" else "Sensor Concat"
        inference = stage(f"{model_name} inference (including restored batches)")
        condition_scores, skipped = score_run(
            test_run, device, batch_size, {}, lambda _: inference, batch_checkpoint_root, seed,
        )
        if skipped:
            raise ValueError(f"{role}: unexpected skipped Holdout branch: {skipped}")
        scores[role] = condition_scores["original"]
        if set(scores[role]) != set(ids):
            raise ValueError(f"{role}: model did not score the complete fixed Test cohort.")

    metrics_report = stage("Calculate metrics and save reports")
    metrics_report(0, 1)
    manufacturers: dict[str, set[str]] = {}
    dates: dict[str, set[str]] = {}
    for sample_id, record in test_records.items():
        manufacturers.setdefault(record["manufacturer"], set()).add(sample_id)
        dates.setdefault(record["capture_date"], set()).add(sample_id)

    model_results = {}
    for role in ("image_only", "sensor_concat"):
        overall = _subset_metrics(ids, scores[role], test_records, ids)
        model_results[role] = {
            "overall": overall,
            "manufacturer": {
                name: _subset_metrics(ids, scores[role], test_records, ids_for_group)
                for name, ids_for_group in sorted(manufacturers.items())
            },
            "capture_date": {
                name: _subset_metrics(ids, scores[role], test_records, ids_for_group)
                for name, ids_for_group in sorted(dates.items())
            },
        }

    image_overall = model_results["image_only"]["overall"]
    concat_overall = model_results["sensor_concat"]["overall"]
    comparison = {
        "cohort": "same_fixed_1472_test_samples",
        "macro_auroc_delta": (
            (concat_overall["macro_auroc"] - image_overall["macro_auroc"])
            if concat_overall["macro_auroc"] is not None and image_overall["macro_auroc"] is not None else None
        ),
        "macro_f1_delta": (
            (concat_overall["macro_f1"] - image_overall["macro_f1"])
            if concat_overall["macro_f1"] is not None and image_overall["macro_f1"] is not None else None
        ),
    }
    final_payload_snapshot = snapshot_test_payload(root, bundle, revision_id, live["test_sample_ids"])
    if final_payload_snapshot["fingerprint"] != snapshot["fingerprint"]:
        raise ValueError("Test payload or evaluation snapshot changed during Limited Holdout scoring.")
    final_preflight, final_live = revalidate_limited_holdout(
        root=root, bundle=bundle, revision_id=revision_id, preflight=preflight,
        group_verification=group_verification, cohort=cohort,
        frozen_validation=frozen_validation, preprocessing=preprocessing,
        image_run=image_run, concat_run=concat_run,
        detector_initialization=detector_initialization, prior_manifests=prior_manifests,
    )
    if final_preflight["fingerprint"] != preflight_payload["fingerprint"] or final_live != live:
        raise ValueError("Preflight, checkpoint or receipt changed before Holdout completion.")

    result = {
        "schema_version": 1,
        "stage": "limited_internal_holdout_evaluation",
        "status": "completed",
        **LIMITATION_FIELDS,
        "preflight": {
            "path": str(preflight),
            "fingerprint": preflight_payload["fingerprint"],
            "revalidated_live": True,
        },
        "group_split_revision": live["group_split_revision"],
        "fixed_test": {
            "sample_count": FIXED_TEST_COUNT,
            "sample_ids": ids,
            "sample_fingerprint": digest_json(ids),
        },
        "evaluation_snapshot": snapshot,
        "integrity": integrity,
        "checkpoint_contract": {
            "image_only_epoch": 27,
            "sensor_concat_epoch": 15,
            "threshold": 0.5,
            "class_order": list(NAMES),
            "detector_initialization": live["evaluation_plan"]["detector_initialization"],
            "sensor_preprocessing": "frozen_train_only",
            "device": device,
            "batch_size": batch_size,
            "seed": seed,
            "batch_checkpoint_dir": str(batch_checkpoint_root),
            "resume_scope": "completed inference batches only under matching checkpoint, dataset, device, batch-size and seed fingerprint",
        },
        "models": model_results,
        "comparison": comparison,
    }
    result["fingerprint"] = digest_json(result)
    with (out / "limited-holdout.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    snapshot_path = out / "payload-snapshot.json"
    with snapshot_path.open("x", encoding="utf-8") as stream:
        json.dump(snapshot, stream, indent=2, allow_nan=False)

    csv_path = out / "limited-holdout.csv"
    with csv_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "scope", "scope_value", "metric", "value"])
        for role, result_for_role in model_results.items():
            for scope_name, scope_values in (
                ("overall", {"overall": result_for_role["overall"]}),
                ("manufacturer", result_for_role["manufacturer"]),
                ("capture_date", result_for_role["capture_date"]),
            ):
                for scope_value, metrics in scope_values.items():
                    for metric in ("macro_auroc", "macro_f1", "micro_f1", "positive_supported_macro_f1"):
                        writer.writerow([role, scope_name, scope_value, metric, metrics[metric]])
                    for class_row in metrics["classes"]:
                        for metric in ("auroc", "precision", "recall", "f1", "tp", "fp", "fn", "tn", "fpr",
                                       "positive", "negative"):
                            writer.writerow([role, scope_name, f'{scope_value}:{class_row["name"]}', metric,
                                             class_row[metric]])

    completed_state = {
        "schema_version": 1,
        "status": "completed",
        "plan_fingerprint": plan_fingerprint,
        "preflight_fingerprint": preflight_payload["fingerprint"],
        "test_snapshot_fingerprint": snapshot["fingerprint"],
        "result_fingerprint": result["fingerprint"],
        "device": device, "batch_size": batch_size, "seed": seed,
        "batch_checkpoint_dir": str(batch_checkpoint_root),
    }
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=out, suffix=".partial", delete=False) as stream:
        json.dump(completed_state, stream, indent=2)
    Path(stream.name).replace(out / "resume-state.json")
    metrics_report(1, 1)
    return result
