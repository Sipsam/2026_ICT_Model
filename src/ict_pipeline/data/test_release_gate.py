"""Test Release Gate receipts for the separate preflight and seal phases."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..config.settings import Credentials
from ..data.preprocessing.common import read_csv
from ..data.preprocessing.policy import NAMES, label_policy
from ..data.preprocessing.revision import read_publication
from ..evaluation.product_protocol import FrozenValidation
from ..multimodal.checkpoint import FORMAT as MULTIMODAL_CHECKPOINT_FORMAT
from ..multimodal.preprocessing import PreprocessingState
from ..pipeline.store import _reject_link, digest_json, fingerprint_path


CANONICAL_CLASSES = tuple(NAMES)
SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict:
    _reject_link(path)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Invalid JSON receipt or metadata.")
    return value


def read_public_index(path: Path) -> dict[str, dict[str, str]]:
    """Read sanitized public hash metadata without opening image, label or sensor payloads."""
    _reject_link(path)
    if path.suffix.casefold() == ".csv":
        rows = read_csv(path)
        return {
            str(row.get("sample_id") or row.get("stem") or row.get("image_path")): dict(row)
            for row in rows
            if row.get("sample_id") or row.get("stem") or row.get("image_path")
        }
    value = json.loads(path.read_bytes())
    rows = value.get("samples") if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError("Public Test index must contain a samples array or be a JSON array.")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Public Test index samples must be objects.")
        sample_id = str(row.get("sample_id") or row.get("stem") or row.get("image_path") or "")
        if not sample_id:
            raise ValueError("Every public Test index row requires sample_id.")
        result[sample_id] = {key: str(value) for key, value in row.items()}
    return result


def _test_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("split") == "test"]


def _metadata_overlap(test: list[dict[str, str]], development: list[dict[str, str]]) -> dict[str, object]:
    fields = ("manufacturer", "capture_date", "capture_time", "source_capture_time")
    result: dict[str, object] = {}
    for field in fields:
        test_values = {str(row.get(field, "")) for row in test if row.get(field)}
        development_values = {str(row.get(field, "")) for row in development if row.get(field)}
        shared = sorted(test_values.intersection(development_values))
        result[field] = {
            "test_distinct_values": len(test_values),
            "shared_values": shared,
            "shared_value_count": len(shared),
            "blocker": False,
        }
    return result


def _prior_reservation_conflicts(
    development: list[dict[str, str]], manifests: tuple[Path, ...],
) -> tuple[list[str], list[str]]:
    fingerprints: list[str] = []
    conflicts: list[str] = []
    keys = ("sample_id", "image_path", "annotation_path", "sensor_path")
    development_keys = {
        (key, str(row.get(key, "")).replace("\\", "/").casefold())
        for row in development
        for key in keys
        if row.get(key)
    }
    for path in manifests:
        _reject_link(path)
        previous = read_csv(path)
        fingerprints.append(digest_json(previous))
        reserved = {
            (key, str(row.get(key, "")).replace("\\", "/").casefold())
            for row in previous if row.get("split") == "test"
            for key in keys
            if row.get(key)
        }
        if not reserved:
            conflicts.append(f"{path}: prior reservation manifest contains no Test rows")
        elif development_keys.intersection(reserved):
            conflicts.append(f"{path}: reserved Test custody conflicts with Train/Validation")
    return fingerprints, conflicts


def _run_receipt(run: Path, role: str) -> dict:
    _reject_link(run)
    metadata_path = run / "run.json"
    saved = _read_json(metadata_path)
    if saved.get("schema_version") != 1 or saved.get("task") != "product_classification":
        raise ValueError(f"{role}: expected a completed product classification run.")
    if saved.get("status") != "completed":
        raise ValueError(f"{role}: the run is not completed.")
    selected = saved.get("selected_checkpoint")
    if not isinstance(selected, dict) or not isinstance(selected.get("path"), str):
        raise ValueError(f"{role}: run.json has no selected checkpoint.")
    checkpoint = (run / selected["path"]).resolve()
    _reject_link(checkpoint)
    if fingerprint_path(checkpoint) != selected.get("sha256"):
        raise ValueError(f"{role}: selected checkpoint fingerprint mismatch.")
    profile = saved.get("profile")
    if not isinstance(profile, dict):
        raise ValueError(f"{role}: run.json has no resolved profile.")
    architecture = profile.get("architecture")
    if not isinstance(architecture, dict):
        raise ValueError(f"{role}: run.json has no architecture configuration.")
    fusion = architecture.get("fusion")
    if not isinstance(fusion, dict):
        raise ValueError(f"{role}: run.json has no fusion configuration.")
    return {
        "role": role,
        "run": str(run),
        "run_metadata_fingerprint": fingerprint_path(metadata_path),
        "selected_checkpoint": {
            "path": selected["path"],
            "sha256": selected["sha256"],
            "epoch": selected.get("epoch"),
        },
        "branch": profile.get("branch"),
        "architecture": architecture,
        "fusion": fusion,
        "detector_initialization": saved.get("detector_initialization"),
        "preprocessing_fingerprint": saved.get("preprocessing_fingerprint"),
        "train_state_fingerprint": saved.get("train_state_fingerprint"),
        "validation_identity": saved.get("validation_identity"),
        "checkpoint_format": MULTIMODAL_CHECKPOINT_FORMAT,
    }


def preflight_test_release(
    *, root: Path, bundle: Path, revision_id: str, group_verification: Path,
    cohort: Path, frozen_validation: Path, preprocessing: Path,
    image_run: Path, concat_run: Path, prior_manifests: tuple[Path, ...] = (),
) -> dict:
    """Verify all metadata and training receipts without opening Test payloads."""
    root = root.resolve()
    bundle = bundle.resolve()
    revision, publication_fingerprint, rows = read_publication(bundle, revision_id)
    if revision["test_origin"] != "internal_same_source":
        raise ValueError("Test Release Gate requires an internal_same_source holdout.")

    verification_payload = _read_json(group_verification)
    verification = verification_payload.get("verification")
    if (
        not isinstance(verification, dict)
        or digest_json(verification) != verification_payload.get("fingerprint")
        or verification.get("schema_version") != SCHEMA_VERSION
        or verification.get("passed") is not True
        or verification.get("blockers") != []
        or verification.get("revision_id") != revision_id
        or verification.get("publication_fingerprint") != publication_fingerprint
    ):
        raise ValueError("A passing group verification receipt is required.")
    test = verification.get("test")
    if not isinstance(test, dict) or test.get("payloads_opened") is not False:
        raise ValueError("The group verification receipt must keep Test payloads unopened.")
    if test.get("payload_seal_reverified") is not False:
        raise ValueError("The Test payload seal must not already be reverified.")

    prior_fingerprints, conflicts = _prior_reservation_conflicts(
        [row for row in rows if row.get("split") != "test"], prior_manifests,
    )
    historical_status = "provided_verified" if prior_manifests else "unverified"
    if conflicts:
        historical_status = "conflict"
        raise ValueError("Historical Test reservation conflict: " + "; ".join(conflicts))
    if prior_manifests:
        prior_set = set(prior_fingerprints)
        receipt_set = set(verification.get("prior_test_manifest_fingerprints", []))
        if prior_set != receipt_set:
            raise ValueError("Prior Test manifests differ from the group verification receipt.")

    test_rows = _test_rows(rows)
    development = [row for row in rows if row.get("split") in ("train", "val", "validation")]
    test_ids = sorted({
        str(row["sample_id"]) for row in test_rows if row.get("sample_id")
    })
    development_ids = {
        str(row["sample_id"]).casefold()
        for row in development if row.get("sample_id")
    }
    id_conflicts = sorted(
        sample_id for sample_id in test_ids
        if sample_id.casefold() in development_ids
    )
    path_keys = ("image_path", "annotation_path", "sensor_path")
    path_conflicts = sorted({
        str(row[key]).replace("\\", "/").casefold()
        for row in test_rows for key in path_keys
        if row.get(key)
    }.intersection({
        str(row[key]).replace("\\", "/").casefold()
        for row in development for key in path_keys
        if row.get(key)
    }))

    cohort_payload = _read_json(cohort)
    if cohort_payload.get("fingerprint") != verification.get("cohort_fingerprint"):
        raise ValueError("Pinned cohort fingerprint differs from group verification.")
    frozen = FrozenValidation.model_validate_json(frozen_validation.read_bytes())
    if frozen.identity.cohort_fingerprint != verification.get("frozen_validation_fingerprint"):
        raise ValueError("FrozenValidation fingerprint differs from group verification.")
    state = PreprocessingState.load(preprocessing)
    if state.fingerprint != verification.get("preprocessing_fingerprint"):
        raise ValueError("Sensor preprocessing fingerprint differs from group verification.")

    image = _run_receipt(image_run, "image")
    concat = _run_receipt(concat_run, "concat")
    if image["branch"] != "image" or concat["branch"] != "paired":
        raise ValueError("Evaluation plan requires the Image-only and Sensor Concat branches.")
    if image["fusion"].get("kind") != "late" or concat["fusion"].get("kind") != "concat":
        raise ValueError("Evaluation plan requires Image late fusion and Sensor Concat fusion.")
    if image["fusion"].get("threshold") != 0.5 or concat["fusion"].get("threshold") != 0.5:
        raise ValueError("Both ablation checkpoints must use the frozen 0.5 threshold.")
    if image["preprocessing_fingerprint"] != concat["preprocessing_fingerprint"]:
        raise ValueError("Image-only and Sensor Concat preprocessing fingerprints differ.")

    detector = verification.get("detector_initialization")
    for run in (image, concat):
        saved = run["detector_initialization"]
        if saved != verification.get("detector_initialization"):
            raise ValueError("Run detector initialization differs from the passing group receipt.")

    blockers = []
    if id_conflicts:
        blockers.append("Test sample IDs overlap Train/Validation.")
    if path_conflicts:
        blockers.append("Test file paths overlap Train/Validation.")
    if verification.get("blockers"):
        blockers.extend(verification["blockers"])
    if conflicts:
        blockers.extend(conflicts)

    plan = {
        "schema_version": SCHEMA_VERSION,
        "stage": "preflight",
        "passed": not blockers,
        "blockers": blockers,
        "test_origin": "internal_same_source",
        "test_sample_count": len(test_ids),
        "test_sample_ids": test_ids,
        "historical_reservation_status": historical_status,
        "prior_test_manifest_fingerprints": prior_fingerprints,
        "prior_test_manifest_provided": bool(prior_manifests),
        "group_split_revision": revision_id,
        "publication_fingerprint": publication_fingerprint,
        "manifest_fingerprint": revision["selected_manifest_fingerprint"],
        "cohort_fingerprint": cohort_payload["fingerprint"],
        "frozen_validation_fingerprint": frozen.identity.cohort_fingerprint,
        "preprocessing_fingerprint": state.fingerprint,
        "group_verification_fingerprint": fingerprint_path(group_verification),
        "group_verification_digest": digest_json(verification),
        "class_order": list(CANONICAL_CLASSES),
        "label_policy": label_policy(),
        "test_payload_coverage": "publication and sanitized index metadata only",
        "payloads_opened": False,
        "payload_seal_reverified": False,
        "metadata_overlap": _metadata_overlap(test_rows, development),
        "duplicate_evidence": {
            "sample_id_conflicts": id_conflicts,
            "file_path_conflicts": path_conflicts,
            "file_hash_conflicts": [],
            "image_similarity_groups": [],
            "sensor_similarity_groups": [],
        },
        "evaluation_plan": {
            "test_split": "test",
            "test_sample_count": len(test_ids),
            "test_sample_fingerprint": digest_json(test_ids),
            "same_test_sample_set": True,
            "class_order": list(CANONICAL_CLASSES),
            "image_only_checkpoint": image["selected_checkpoint"],
            "sensor_concat_checkpoint": concat["selected_checkpoint"],
            "threshold": 0.5,
            "threshold_source": "frozen_validation",
            "preprocessing_fingerprint": state.fingerprint,
            "image_preprocessing": "rgb/faster_rcnn_fpn",
            "sensor_preprocessing": "frozen_train_only",
            "evaluation_protocol": "same_labeled_test_coco_no_tuning",
            "metrics": "heldout classification and ablation protocol from existing analysis",
            "model_checkpoint_selection": "image_only=selected_validation_checkpoint;sensor_concat=selected_validation_checkpoint",
            "detector_initialization": detector,
        },
        "model_receipts": {"image_only": image, "sensor_concat": concat},
        "leakage_claim": "Only available metadata and published evidence were checked; no payload-level duplicate claim is made.",
        "final_evaluation_allowed": not blockers,
        "seal_verification_allowed": not blockers,
    }
    plan["fingerprint"] = digest_json(plan)
    return plan


def verify_test_payload_seal(
    *, root: Path, bundle: Path, preflight_receipt: dict,
    preflight_path: Path, public_index: Path,
) -> dict:
    """Open Test payloads once explicitly approved and verify sanitized public seals."""
    if preflight_receipt.get("schema_version") != SCHEMA_VERSION or preflight_receipt.get("stage") != "preflight":
        raise ValueError("A valid Test preflight receipt is required.")
    if preflight_receipt.get("passed") is not True or preflight_receipt.get("seal_verification_allowed") is not True:
        raise ValueError("Preflight passed=false; Seal Verification is not allowed.")
    if digest_json({k: v for k, v in preflight_receipt.items() if k != "fingerprint"}) != preflight_receipt.get("fingerprint"):
        raise ValueError("Preflight receipt fingerprint mismatch.")
    if _read_json(preflight_path).get("fingerprint") != preflight_receipt.get("fingerprint"):
        raise ValueError("Preflight receipt changed after it was saved.")

    public = read_public_index(public_index)
    revision = json.loads((bundle / "split.json").read_bytes())
    manifest = read_csv(bundle / "manifest.csv")
    test_rows = _test_rows(manifest)
    test_ids = sorted(str(row["sample_id"]) for row in test_rows if row.get("sample_id"))
    if test_ids != preflight_receipt["test_sample_ids"]:
        raise ValueError("Test sample set differs from the preflight receipt.")

    verified = 0
    mismatches: list[str] = []
    for row in test_rows:
        sample_id = str(row["sample_id"])
        metadata = public.get(sample_id) or public.get(str(row.get("image_path", "")))
        if metadata is None:
            mismatches.append(f"{sample_id}: absent from public index")
            continue
        for kind, key in (
            ("image", "image_path"), ("annotation", "annotation_path"), ("sensor", "sensor_path"),
        ):
            relative = str(row.get(key, ""))
            if not relative:
                continue
            path = root / relative
            _reject_link(path)
            if not path.is_file():
                mismatches.append(f"{sample_id}: missing {kind} payload")
                continue
            actual = fingerprint_path(path)
            for field in (f"{kind}_sha256", f"{kind}_hash", f"{kind}_fingerprint", "sha256"):
                expected = metadata.get(field)
                if expected:
                    if actual != expected:
                        mismatches.append(f"{sample_id}: {kind} payload hash mismatch")
                    break
            else:
                manifest_hash = row.get(f"{kind}_sha256")
                if manifest_hash and actual != manifest_hash:
                    mismatches.append(f"{sample_id}: {kind} manifest hash mismatch")
            if kind == "image":
                try:
                    from PIL import Image
                    with Image.open(path) as image:
                        image.verify()
                except Exception:
                    mismatches.append(f"{sample_id}: unreadable Test image")
            elif kind == "annotation":
                try:
                    document = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(document, dict):
                        raise ValueError
                    if not document.get("images") and not document.get("annotations"):
                        raise ValueError
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    mismatches.append(f"{sample_id}: unreadable Test annotation")
            elif kind == "sensor":
                try:
                    if path.suffix.casefold() in (".xlsx", ".xlsm"):
                        from openpyxl import load_workbook
                        with load_workbook(path, read_only=True, data_only=True) as workbook:
                            if not workbook.sheetnames:
                                raise ValueError
                    elif path.stat().st_size <= 0:
                        raise ValueError
                except Exception:
                    mismatches.append(f"{sample_id}: unreadable Test sensor payload")
        verified += 1

    coco = bundle / "detection" / "coco" / "annotations" / "instances_test.json"
    _reject_link(coco)
    labels = json.loads(coco.read_bytes())
    coco_ids = sorted(
        str(image["file_name"]).rsplit(".", 1)[0]
        for image in labels.get("images", [])
    )
    if coco_ids != test_ids:
        mismatches.append("COCO Test sample set differs from the published manifest")

    if mismatches:
        raise ValueError("Test payload seal verification failed: " + "; ".join(mismatches))

    receipt = {
        "schema_version": SCHEMA_VERSION,
        "stage": "seal_verification",
        "passed": True,
        "preflight_passed": True,
        "preflight_fingerprint": preflight_receipt["fingerprint"],
        "historical_reservation_status": preflight_receipt["historical_reservation_status"],
        "test_origin": "internal_same_source",
        "test_sample_count": len(test_ids),
        "test_sample_set_fingerprint": digest_json(test_ids),
        "public_index_fingerprint": fingerprint_path(public_index),
        "test_payload_coverage": "images labels and sensor files verified",
        "payloads_opened": True,
        "payload_seal_reverified": True,
        "model_inference_performed": False,
        "performance_computed": False,
        "evaluation_plan": preflight_receipt["evaluation_plan"],
        "publisher_revision": revision["revision_id"],
        "verified_payload_count": verified,
        "mismatches": [],
        "final_evaluation_allowed": True,
        "blocking_reason": [],
    }
    receipt["fingerprint"] = digest_json(receipt)
    return receipt
