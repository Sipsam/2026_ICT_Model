"""Bind the SMT preprocessing functions to the resumable pipeline runner."""

import json

from ...pipeline.model import StageContext, StageFailure, StageResult
from ...pipeline.store import digest_json
from ..stages import limited_progress
from .audit import audit_dataset
from .common import read_csv, write_json
from .convert import convert_detection
from .manifest import build_manifest
from .policy import label_policy
from .revision import read_dataset_seal
from .validate import validate_detection


def audit(context: StageContext) -> StageResult:
    try:
        audit_dataset(
            context.inputs["dataset"], context.output_dir, deep=True, hash_images=True,
            progress=limited_progress(context.progress),
            source_fingerprint=context.input_fingerprints.get("dataset"),
        )
    except ImportError:
        raise StageFailure("Deep audit requires Pillow and openpyxl. Install the preprocessing extra.") from None
    return StageResult.SUCCESS


def manifest(context: StageContext) -> StageResult:
    audit_files = {"samples": "samples.csv", "issues": "issues.csv", "audit_summary": "summary.json"}
    build_manifest(
        context.inputs["dataset"], context.inputs["samples"].parent, context.output_dir,
        progress=limited_progress(context.progress),
        source_fingerprint=context.input_fingerprints.get("dataset"),
        audit_fingerprints={
            filename: context.input_fingerprints[name] for name, filename in audit_files.items()
        } if context.input_fingerprints else None,
    )
    return StageResult.SUCCESS


def convert(context: StageContext) -> StageResult:
    convert_detection(
        context.inputs["dataset"], context.inputs["manifest"], context.output_dir / "detection",
        published_out=context.artifact_dir / "detection", progress=limited_progress(context.progress),
    )
    return StageResult.SUCCESS


def validate(context: StageContext) -> StageResult:
    _validate(context)
    return StageResult.SUCCESS


def validate_revision(context: StageContext) -> StageResult:
    try:
        seal = read_dataset_seal(context.inputs["detection"])
        revision = json.loads(context.inputs["manifest_summary"].read_text(encoding="utf-8"))
        revision_id = revision.pop("revision_id")
        provenance = {key: value for key, value in revision.items() if key != "assignments"}
        provenance["revision_id"] = revision_id
        if (
            seal is None or digest_json(revision) != revision_id
            or seal["split"] != provenance
            or digest_json(read_csv(context.inputs["manifest"])) != revision["selected_manifest_fingerprint"]
        ):
            raise ValueError
    except (OSError, ValueError, KeyError, TypeError):
        raise StageFailure("Use a verified split bundle published with --export-detection.") from None
    _validate(context, selected_split=seal)
    return StageResult.SUCCESS


def _validate(context: StageContext, *, selected_split: dict | None = None) -> None:
    report = validate_detection(
        context.inputs["manifest"], context.inputs["detection"], context.output_dir,
        progress=limited_progress(context.progress),
        include_test=selected_split is not None and selected_split["split"]["test_origin"] == "internal_same_source",
    )
    if not report["passed"]:
        raise StageFailure(
            f"Detection validation found {report['issue_count']} issues. Inspect the reports in the failed attempt."
        )
    receipt = {
        "schema_version": 1, "passed": True, "inputs": dict(context.input_fingerprints),
        "label_policy": label_policy(),
    }
    if selected_split is not None:
        receipt["selected_split"] = selected_split
    write_json(context.output_dir / "receipt.json", receipt)
