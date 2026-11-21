"""Fail-closed Test Release Gate receipts for preflight and payload seal verification."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from ..data.preprocessing.common import read_csv
from ..data.preprocessing.policy import NAMES, label_policy
from ..data.preprocessing.revision import read_publication
from ..evaluation.product import read_artifact
from ..evaluation.product_protocol import FrozenValidation
from ..multimodal.analysis import _inspect_run
from ..multimodal.checkpoint import FORMAT as MULTIMODAL_CHECKPOINT_FORMAT
from ..multimodal.audit import Cohort
from ..multimodal.analysis import GroupAnalysisInputs
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


def _raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def read_public_index(path: Path) -> dict[str, dict[str, str]]:
    """Read the official sanitized public hash index without opening Test payloads."""
    _reject_link(path)
    if path.suffix.casefold() != ".json":
        raise ValueError("The trusted public Test index must be JSON, not a CSV or spreadsheet.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("Public Test index must use schema_version 1.")
    rows = value.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Public Test index must contain a nonempty samples array.")
    identity = {
        "schema_version": 1,
        "revision_id": value.get("revision_id"),
        "publication_fingerprint": value.get("publication_fingerprint"),
        "manifest_fingerprint": value.get("manifest_fingerprint"),
        "samples": rows,
    }
    if value.get("fingerprint") != digest_json(identity):
        raise ValueError("Public Test index fingerprint mismatch.")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Public Test index samples must be objects.")
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError("Every public Test index row requires sample_id.")
        if sample_id in result:
            raise ValueError("Public Test index contains duplicate sample IDs.")
        for field in ("image_sha256", "annotation_sha256", "sensor_sha256"):
            if not _is_sha256(row.get(field)):
                raise ValueError(f"Public Test index sample {sample_id} lacks a valid {field}.")
        result[sample_id] = {key: str(item) for key, item in row.items()}
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
        if not path.is_file():
            conflicts.append(f"{path}: prior Test reservation manifest is missing")
            continue
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


def _run_receipt(
    run: Path, role: str, root: Path, bundle: Path, cohort_path: Path,
    frozen_path: Path, frozen: FrozenValidation, cohort: Cohort,
    group: GroupAnalysisInputs | None,
) -> dict:
    """Verify a run through the production multimodal analysis serializer contract."""
    _reject_link(run)
    inspected = _inspect_run(
        "image" if role == "image" else "concat",
        run, root, cohort_path, frozen_path, cohort, frozen,
        group=group,
    )
    saved = _read_json(run / "run.json")
    selected = saved["selected_checkpoint"]
    profile = saved["profile"]
    architecture = profile["architecture"]
    fusion = architecture["fusion"]
    return {
        "role": role,
        "run": str(run),
        "run_metadata_fingerprint": fingerprint_path(run / "run.json"),
        "selected_checkpoint": {
            "path": selected["path"],
            "sha256": selected["sha256"],
            "epoch": selected["epoch"],
        },
        "validation_report": saved["validation_report"],
        "branch": profile["branch"],
        "architecture": architecture,
        "fusion": fusion,
        "preprocessing_fingerprint": saved["preprocessing_fingerprint"],
        "train_state_fingerprint": saved["train_state_fingerprint"],
        "validation_identity": saved["validation_identity"],
        "architecture_fingerprint": saved["architecture_fingerprint"],
        "samples": saved["samples"],
        "protocol": inspected.protocol.model_dump(mode="json"),
        "detector_initialization": inspected.provenance["detector_initialization"]
        if "detector_initialization" in inspected.provenance else None,
        "checkpoint_format": MULTIMODAL_CHECKPOINT_FORMAT,
    }


def preflight_test_release(
    *, root: Path, bundle: Path, revision_id: str, group_verification: Path,
    cohort: Path, frozen_validation: Path, preprocessing: Path,
    image_run: Path, concat_run: Path, prior_manifests: tuple[Path, ...] = (),
    detector_initialization: Path | None = None, public_index: Path | None = None,
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
    receipt_prior = verification.get("prior_test_manifest_fingerprints", [])
    if not isinstance(receipt_prior, list):
        receipt_prior = []
    if prior_manifests:
        historical_status = "provided_verified" if not conflicts else "conflict"
    elif receipt_prior:
        conflicts.append("prior Test reservation manifests are missing; the group receipt records reservations")
        historical_status = "missing"
    else:
        historical_status = "unverified"
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

    public_index_status = "unavailable_no_trusted_public_hash_index"
    public_index_fingerprint = None
    if public_index is not None:
        public_index = public_index.resolve()
        public_payload = _read_json(public_index)
        public = read_public_index(public_index)
        if set(public) != set(test_ids):
            raise ValueError("Public Test index sample set differs from the published manifest.")
        if (public_payload.get("revision_id") != revision_id
                or public_payload.get("publication_fingerprint") != publication_fingerprint
                or public_payload.get("manifest_fingerprint") != revision["selected_manifest_fingerprint"]):
            raise ValueError("Public Test index disagrees with the pinned split publication.")
        public_index_status = "pinned_verified_metadata_only"
        public_index_fingerprint = fingerprint_path(public_index)

    cohort_payload = _read_json(cohort)
    cohort_model = Cohort.load(cohort)
    if (cohort_payload.get("fingerprint") != verification.get("cohort_fingerprint")
            or cohort_model.fingerprint != verification.get("cohort_fingerprint")
            or cohort_model.split_fingerprint != verification.get("split_fingerprint")
            or cohort_model.manifest_fingerprint != verification.get("manifest_fingerprint")):
        raise ValueError("Pinned cohort fingerprint differs from group verification.")
    frozen = read_artifact(frozen_validation, FrozenValidation)
    if frozen.identity.cohort_fingerprint != verification.get("frozen_validation_fingerprint"):
        raise ValueError("FrozenValidation fingerprint differs from group verification.")
    state = PreprocessingState.load(preprocessing)
    if state.fingerprint != verification.get("preprocessing_fingerprint"):
        raise ValueError("Sensor preprocessing fingerprint differs from group verification.")

    group_inputs = None
    if verification.get("detector_scope") is not None:
        if detector_initialization is None:
            raise ValueError("Group runs require the --detector-initialization artifact or finalized checkpoint.")
        group_inputs = GroupAnalysisInputs(
            verification=group_verification,
            bundle=bundle,
            detector=detector_initialization.resolve(),
            prior_test_manifests=prior_manifests,
        )
    image = _run_receipt(
        image_run, "image", root, bundle, cohort, frozen_validation, frozen, cohort_model, group_inputs,
    )
    concat = _run_receipt(
        concat_run, "concat", root, bundle, cohort, frozen_validation, frozen, cohort_model, group_inputs,
    )
    if image["branch"] != "image" or concat["branch"] != "paired":
        raise ValueError("Evaluation plan requires the Image-only and Sensor Concat branches.")
    if image["fusion"].get("kind") != "late" or concat["fusion"].get("kind") != "concat":
        raise ValueError("Evaluation plan requires Image late fusion and Sensor Concat fusion.")
    if image["fusion"].get("threshold") != 0.5 or concat["fusion"].get("threshold") != 0.5:
        raise ValueError("Both ablation checkpoints must use the frozen 0.5 threshold.")
    if image["preprocessing_fingerprint"] != concat["preprocessing_fingerprint"]:
        raise ValueError("Image-only and Sensor Concat preprocessing fingerprints differ.")

    detector = image["detector_initialization"]
    if detector is None or concat["detector_initialization"] != detector:
        raise ValueError("Both runs require matching verified Detector initialization provenance.")

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
        "public_test_hash_index": {
            "status": public_index_status,
            "fingerprint": public_index_fingerprint,
        },
        "payloads_opened": False,
        "payload_seal_reverified": False,
        "metadata_overlap": _metadata_overlap(test_rows, development),
        "duplicate_evidence": {
            "sample_id_conflicts": id_conflicts,
            "file_path_conflicts": path_conflicts,
            "file_hash_conflicts": [
                {"method": method, "cross_split_group_count": count}
                for method, count in sorted(
                    verification.get("cross_split_groups_by_evidence", {}).items()
                )
                if method in ("image_sha256", "sensor_workbook_sha256", "annotation_sha256")
                and isinstance(count, int) and count > 0
            ],
            "image_similarity_groups": (
                [{"status": "verified", "cross_split_near_pair_count":
                  verification.get("train_validation_near_pairs_by_method", {}).get("image_similarity", 0)}]
                if "image_similarity" in verification.get("train_validation_near_pairs_by_method", {})
                else [{"status": "unverified", "reason": "No verification receipt similarity result"}]
            ),
            "sensor_similarity_groups": (
                [{"status": "verified", "cross_split_near_pair_count":
                  verification.get("train_validation_near_pairs_by_method", {}).get("sensor_similarity", 0)}]
                if "sensor_similarity" in verification.get("train_validation_near_pairs_by_method", {})
                else [{"status": "unverified", "reason": "No verification receipt similarity result"}]
            ),
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
        "final_evaluation_allowed": False,
        "seal_verification_allowed": not blockers and public_index_fingerprint is not None,
    }
    plan["fingerprint"] = digest_json(plan)
    return plan


def verify_test_payload_seal(
    *, root: Path, bundle: Path, preflight_receipt: dict,
    preflight_path: Path, public_index: Path,
    revision_id: str | None = None, group_verification: Path | None = None,
    cohort: Path | None = None, frozen_validation: Path | None = None,
    preprocessing: Path | None = None, image_run: Path | None = None,
    concat_run: Path | None = None, detector_initialization: Path | None = None,
    prior_manifests: tuple[Path, ...] = (),
) -> dict:
    """Reverify every metadata and checkpoint receipt, then open and seal the Test payloads."""
    if preflight_receipt.get("schema_version") != SCHEMA_VERSION or preflight_receipt.get("stage") != "preflight":
        raise ValueError("A valid Test preflight receipt is required.")
    if preflight_receipt.get("passed") is not True or preflight_receipt.get("seal_verification_allowed") is not True:
        raise ValueError("Preflight passed=false; Seal Verification is not allowed.")
    if preflight_receipt.get("final_evaluation_allowed") is not False:
        raise ValueError("Preflight must keep final_evaluation_allowed=false.")
    if digest_json({k: v for k, v in preflight_receipt.items() if k != "fingerprint"}) != preflight_receipt.get("fingerprint"):
        raise ValueError("Preflight receipt fingerprint mismatch.")
    if _read_json(preflight_path).get("fingerprint") != preflight_receipt.get("fingerprint"):
        raise ValueError("Preflight receipt changed after it was saved.")
    if not all((revision_id, group_verification, cohort, frozen_validation, preprocessing, image_run, concat_run)):
        raise ValueError("Seal Verification requires all Preflight revalidation inputs.")

    live = preflight_test_release(
        root=root, bundle=bundle, revision_id=revision_id,
        group_verification=group_verification, cohort=cohort,
        frozen_validation=frozen_validation, preprocessing=preprocessing,
        image_run=image_run, concat_run=concat_run,
        prior_manifests=prior_manifests,
        detector_initialization=detector_initialization,
        public_index=public_index,
    )
    if live.get("passed") is not True or live.get("seal_verification_allowed") is not True:
        raise ValueError("Live Preflight revalidation failed; split, checkpoint or receipt changed.")
    pinned_fields = (
        "test_sample_ids", "group_split_revision", "publication_fingerprint",
        "manifest_fingerprint", "cohort_fingerprint", "frozen_validation_fingerprint",
        "preprocessing_fingerprint", "group_verification_fingerprint",
        "prior_test_manifest_fingerprints", "evaluation_plan",
    )
    if any(live.get(field) != preflight_receipt.get(field) for field in pinned_fields):
        raise ValueError("Preflight and Seal receipts disagree after live revalidation.")

    public = read_public_index(public_index)
    public_payload = _read_json(public_index)
    revision, publication_fingerprint, manifest = read_publication(bundle, revision_id)
    test_rows = _test_rows(manifest)
    test_ids = sorted(str(row["sample_id"]) for row in test_rows if row.get("sample_id"))
    if test_ids != preflight_receipt["test_sample_ids"]:
        raise ValueError("Test sample set differs from the preflight receipt.")
    if len(test_ids) != len(set(test_ids)):
        raise ValueError("Duplicate Test sample IDs in the published manifest.")
    if set(public) != set(test_ids):
        raise ValueError("Public Test index sample set differs from the manifest.")
    if (public_payload.get("revision_id") != revision_id
            or public_payload.get("publication_fingerprint") != publication_fingerprint
            or public_payload.get("manifest_fingerprint") != digest_json(manifest)):
        raise ValueError("Public Test index disagrees with the live split publication.")
    if preflight_receipt["public_test_hash_index"]["fingerprint"] != fingerprint_path(public_index):
        raise ValueError("Public Test hash index changed after Preflight.")

    verified = 0
    mismatches: list[str] = []
    for row in test_rows:
        sample_id = str(row["sample_id"])
        metadata = public[sample_id]
        for kind, key in (
            ("image", "image_path"), ("annotation", "annotation_path"), ("sensor", "sensor_path"),
        ):
            relative = str(row.get(key, ""))
            if not relative:
                mismatches.append(f"{sample_id}: missing {kind} manifest path")
                continue
            if Path(relative).is_absolute() or ".." in Path(relative).parts:
                mismatches.append(f"{sample_id}: unsafe {kind} payload path")
                continue
            path = (root / relative).resolve()
            if not path.is_relative_to(root.resolve()):
                mismatches.append(f"{sample_id}: {kind} payload escapes the source root")
                continue
            _reject_link(path)
            if not path.is_file():
                mismatches.append(f"{sample_id}: missing {kind} payload")
                continue
            expected = metadata.get(f"{kind}_sha256")
            if not _is_sha256(expected):
                mismatches.append(f"{sample_id}: missing expected {kind} SHA-256")
                continue
            actual = _raw_sha256(path)
            if actual != expected:
                mismatches.append(f"{sample_id}: {kind} payload hash mismatch")
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
                    if not isinstance(document, dict) or not isinstance(document.get("images"), list) or not isinstance(document.get("annotations"), list):
                        raise ValueError
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    mismatches.append(f"{sample_id}: unreadable Test annotation")
            elif kind == "sensor":
                try:
                    if path.suffix.casefold() in (".xlsx", ".xlsm"):
                        from openpyxl import load_workbook
                        workbook = load_workbook(path, read_only=True, data_only=True)
                        try:
                            if not workbook.sheetnames:
                                raise ValueError
                        finally:
                            workbook.close()
                    elif path.stat().st_size <= 0:
                        raise ValueError
                except Exception:
                    mismatches.append(f"{sample_id}: unreadable Test sensor payload")
        verified += 1

    coco = bundle / "detection" / "coco" / "annotations" / "instances_test.json"
    _reject_link(coco)
    labels = json.loads(coco.read_bytes())
    coco_images = labels.get("images") if isinstance(labels, dict) else None
    coco_annotations = labels.get("annotations") if isinstance(labels, dict) else None
    if not isinstance(coco_images, list) or not isinstance(coco_annotations, list):
        mismatches.append("COCO Test annotations have invalid JSON structure")
        coco_ids = []
    else:
        coco_ids = sorted(
            str(image["file_name"]).rsplit(".", 1)[0]
            for image in coco_images
            if isinstance(image, dict) and image.get("file_name")
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
        "public_index_status": "trusted_pin_present",
        "public_index_metadata": {
            "revision_id": public_payload["revision_id"],
            "publication_fingerprint": public_payload["publication_fingerprint"],
            "manifest_fingerprint": public_payload["manifest_fingerprint"],
        },
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
