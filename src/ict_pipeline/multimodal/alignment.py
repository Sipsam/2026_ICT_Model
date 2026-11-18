"""Verify the documented AI Hub record relationship without searching by time."""

import hashlib
import json
from datetime import datetime
from io import BytesIO
from pathlib import Path

from pydantic import ValidationError

from ..classification.sensors import SampleIssue, parse_timestamp, read_xlsx
from ..data.preprocessing.common import CsvRow
from ..data.preprocessing.convert import check_annotation
from ..data.preprocessing.leakage import Observation, is_candidate, metadata_groups
from ..data.preprocessing.policy import CLASS_MAP, SENSOR_FIELDS
from ..pipeline.store import digest_json
from .contracts import Record, SensorSequence, Target, sensor_step

PAIRING_RULE = "aihub-smt-provider-record-v1"
VALUE_ATOL = 0.001
EXTENSIONS = {
    "annotation_path": (".json",),
    "image_path": (".jpg", ".jpeg"),
    "sensor_path": (".xlsx",),
}


def _check_path(row: CsvRow, key: str, annotation: Path) -> None:
    path = Path(row[key])
    if not row[key] or path.stem != row["sample_id"] or path.suffix.lower() not in EXTENSIONS[key]:
        raise SampleIssue(f"missing_or_conflicting_{key}")
    modality = {"annotation_path": "annotations", "image_path": "images", "sensor_path": "sensors"}[key]
    folder = annotation.parent.with_name(annotation.parent.name.replace("_annotations_", f"_{modality}_", 1))
    if path.parent != folder:
        raise SampleIssue(f"cross_group_{key}")


def _metadata(document: dict, stem: str) -> dict:
    images = document.get("images")
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise SampleIssue("ambiguous_image_metadata")
    image = images[0]
    name = image.get("file_name")
    if not isinstance(name, str) or Path(name).name != name or any(c in name for c in "/\\:") or (
        Path(name).stem != stem or Path(name).suffix.lower() not in EXTENSIONS["image_path"]
    ):
        raise SampleIssue("image_metadata_filename_mismatch")
    if any(type(image.get(key)) is not int or image[key] <= 0 for key in ("width", "height")):
        raise SampleIssue("invalid_image_dimensions")
    return image


def _sensor_group(document: dict) -> dict:
    groups = document.get("sensor_data")
    if groups is None or groups == []:
        raise SampleIssue("missing_json_sensor_group")
    if (not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict)
            or document.get("sensor_sequence") is not None):
        raise SampleIssue("ambiguous_json_sensor_group")
    return groups[0]


def _capture(image: dict, token: str) -> tuple[str, datetime]:
    try:
        filename = datetime.strptime(token, "%Y%m%d-%H%M%S")
        if image.get("timestamp") is None:
            return filename.isoformat(), filename
        text = image["timestamp"]
        actual = parse_timestamp(text)
        if datetime.fromisoformat(text.strip()).replace(tzinfo=None) != filename:
            raise ValueError
        return text, actual
    except (ValueError, TypeError, OverflowError):
        raise SampleIssue("invalid_or_conflicting_capture_time") from None


def _verified_sequence(group: dict, raw: bytes, capture: datetime) -> SensorSequence:
    source = group.get("sensor_sequence")
    if not isinstance(source, list) or not source or any(not isinstance(item, dict) for item in source):
        raise SampleIssue("missing_or_invalid_json_sequence")
    try:
        embedded = SensorSequence(steps=tuple(
            sensor_step(item.get("timestamp"), tuple(item.get(name) for name in SENSOR_FIELDS)) for item in source
        ))
        workbook = SensorSequence(steps=read_xlsx(BytesIO(raw), sensor_step))
    except ValidationError:
        raise SampleIssue("non_increasing_or_incompatible_sensor_timestamps") from None
    if len(embedded.steps) != len(workbook.steps):
        raise SampleIssue("sensor_length_mismatch")
    if embedded.steps[0].instant != capture:
        raise SampleIssue("sensor_capture_timestamp_mismatch")
    for left, right in zip(embedded.steps, workbook.steps, strict=True):
        if left.instant != right.instant:
            raise SampleIssue("sensor_timestamp_mismatch")
        for a, b in zip(left.values, right.values, strict=True):
            if (a is None) != (b is None):
                raise SampleIssue("sensor_missingness_mismatch")
            if a is not None and b is not None and abs(a - b) > VALUE_ATOL:
                raise SampleIssue("sensor_value_mismatch")
    return embedded


def observe_record(
    row: CsvRow, sources: dict[str, bytes], hashes: dict[str, str], ambiguous: frozenset[str],
) -> Record:
    """Create a record from a single provider relationship and its captured input bytes."""
    from PIL import Image

    stem = row["sample_id"]
    tokens = stem.split("_")
    annotation = Path(row["annotation_path"])
    issues = []
    sample = Observation(row, groups={"sample_id": stem.casefold()})
    if "image_path" in hashes:
        sample.groups["image_sha256"] = hashes["image_path"]
    if "sensor_path" in hashes:
        sample.groups["sensor_workbook_sha256"] = hashes["sensor_path"]
    if row.get("group_id"):
        sample.groups["revision_group"] = row["group_id"]
    fields = {
        "sample_id": stem, "split": row["split"], "provider_split": row["original_split"],
        "decision": row["decision"], "detection_candidate": is_candidate(row),
        "source_class": row.get("source_class", ""), "source_group": row.get("manufacturer", ""),
        **{key: row[key] for key in EXTENSIONS},
        **{key.replace("_path", "_sha256"): value for key, value in hashes.items()},
    }
    try:
        if "annotation_path" in ambiguous:
            raise SampleIssue("ambiguous_annotation_files")
        _check_path(row, "annotation_path", annotation)
        if not annotation.parent.name.startswith(f"{row['original_split']}_annotations_soldering_"):
            raise SampleIssue("cross_group_annotation_path")
        if len(tokens) != 6 or (row.get("manufacturer") and tokens[3] != row["manufacturer"]):
            raise SampleIssue("invalid_or_conflicting_source_group")
        fields["source_group"] = tokens[3]
        if "annotation_path" not in sources:
            raise SampleIssue("missing_annotation")
        document = json.loads(sources["annotation_path"].decode("utf-8-sig"))
        if not isinstance(document, dict):
            raise SampleIssue("invalid_annotation")
        image = _metadata(document, stem)
    except (SampleIssue, UnicodeError, json.JSONDecodeError) as exc:
        issue = str(exc) if isinstance(exc, SampleIssue) else "invalid_annotation"
        return Record(**fields, groups=tuple(sorted(sample.groups.items())), issues=(issue,),
                      alignment="ambiguous" if issue.startswith("ambiguous") else "unmatched")

    fields["image_size"] = (image["width"], image["height"])
    fields["image_id"] = image.get("id") if type(image.get("id")) is int else None
    try:
        _, _, boxes = check_annotation(document, stem)
        fields["targets"] = tuple(Target(category=CLASS_MAP[cid] + 1, box=tuple(box)) for cid, box in boxes)
    except (KeyError, AttributeError, TypeError, ValueError, OverflowError):
        issues.append("invalid_canonical_detection_targets")

    image_name = image["file_name"]
    try:
        if "image_path" in ambiguous:
            raise SampleIssue("ambiguous_image_files")
        _check_path(row, "image_path", annotation)
        if Path(row["image_path"]).name.casefold() != image_name.casefold():
            raise SampleIssue("image_metadata_filename_mismatch")
        if "image_path" not in sources:
            raise SampleIssue("missing_image")
        with Image.open(BytesIO(sources["image_path"])) as opened:
            if opened.format != "JPEG" or opened.size != fields["image_size"]:
                raise SampleIssue("image_format_or_size_mismatch")
            opened.verify()
        fields["image_available"] = True
    except SampleIssue as exc:
        issues.append(str(exc))
    except (OSError, ValueError, Image.DecompressionBombError):
        issues.append("invalid_image")

    alignment = "unmatched"
    group = {}
    try:
        if "sensor_path" in ambiguous:
            raise SampleIssue("ambiguous_sensor_files")
        group = _sensor_group(document)
        fields["sensor_image_id"] = group.get("image_id") if type(group.get("image_id")) is int else None
        fields["capture_timestamp"], capture = _capture(image, tokens[4])
        _check_path(row, "sensor_path", annotation)
        if ("image_file_name" in group and group["image_file_name"] != image_name) or (
            "file_name" in group and group["file_name"] != Path(row["sensor_path"]).name
        ):
            raise SampleIssue("ambiguous_sensor_filename_reference")
        if "sensor_path" not in sources:
            raise SampleIssue("missing_sensor_workbook")
        sequence = _verified_sequence(group, sources["sensor_path"], capture)
        sample.groups["sensor_exact"] = digest_json([
            [step.instant.isoformat(), step.values] for step in sequence.steps
        ])
        fields.update(sequence=sequence, alignment_method="provider_record_json_xlsx")
        alignment = "matched"
        if not sequence.available:
            issues.append("all_measurements_missing")
    except SampleIssue as exc:
        issues.append(str(exc))
        if str(exc).startswith("ambiguous"):
            alignment = "ambiguous"

    metadata_groups(sample, document, image, group)
    if sample.issues:
        issues.extend(sample.issues)
        if alignment == "matched":
            fields.update(sequence=None, alignment_method=None)
            alignment = "ambiguous"
    if alignment == "matched" and any(issue in issues for issue in (
        "ambiguous_image_files", "cross_group_image_path", "image_metadata_filename_mismatch",
        "missing_or_conflicting_image_path",
    )) and row["image_path"]:
        fields.update(sequence=None, alignment_method=None)
        alignment = "ambiguous" if "ambiguous_image_files" in issues else "unmatched"
    return Record(**fields, groups=tuple(sorted(sample.groups.items())), issues=tuple(issues),
                  alignment=alignment, quarantined=any(
                      issue.startswith(("conflicting_", "unusable_")) for issue in sample.issues
                  ))


def read_sources(root: Path, row: CsvRow) -> tuple[dict[str, bytes], dict[str, str]]:
    sources = {}
    for key in EXTENSIONS:
        if row[key]:
            try:
                sources[key] = (root / row[key]).read_bytes()
            except FileNotFoundError:
                continue
    return sources, {key: hashlib.sha256(raw).hexdigest() for key, raw in sources.items()}
