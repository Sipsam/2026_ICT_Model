"""Bind the SMT preprocessing functions to the resumable pipeline runner."""

from ...pipeline.model import StageContext, StageFailure, StageResult
from ..stages import limited_progress
from .audit import audit_dataset
from .convert import convert_detection
from .manifest import build_manifest
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
    report = validate_detection(
        context.inputs["manifest"], context.inputs["detection"], context.output_dir,
        progress=limited_progress(context.progress),
    )
    if not report["passed"]:
        raise StageFailure(
            f"Detection validation found {report['issue_count']} issues. Inspect the reports in the failed attempt."
        )
    return StageResult.SUCCESS
