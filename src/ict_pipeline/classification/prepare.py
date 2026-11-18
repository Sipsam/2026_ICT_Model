"""Prepare classification samples and audit alignment and split provenance."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PureWindowsPath

from ..data.model import Progress
from ..data.preprocessing.audit import analyze, sha256
from ..data.preprocessing.common import Row, require_separate_output, safe_component, write_csv, write_json
from ..data.preprocessing.policy import POLICY_VERSION, SENSOR_FIELDS, SPLIT_MAP, TARGET_IDS
from ..pipeline.store import digest_json
from .contracts import IMAGE_SIZE, LABEL_POLICY, SCHEMA_VERSION, ClassificationConfig, Sample, Split
from .sensors import SampleIssue, aligned_sequence, json_readings, parse_timestamp, xlsx_readings

_CLASSIFICATION_CHECKS = {
    # Classification accepts absent modalities, optional IDs, and XLSX-only sequences.
    "missing_image", "missing_sensor", "sensor_sequence_missing",
    "json_sensor_status_conflict", "filename_status_sensor_conflict",
    "mixed_normal_defect_boxes",
}
_GROUP_KINDS = ("sample_id", "image_bytes", "sensor_sequence", "session", "source_capture", "source_day", "source")


@dataclass(frozen=True)
class AuditedSample:
    sample: Sample
    groups: dict[str, str]


def _paired_file(folder: Path, stem: str, extensions: tuple[str, ...]) -> Path | None:
    matches = [folder / f"{stem}{ext}" for ext in extensions if (folder / f"{stem}{ext}").is_file()]
    # Case-insensitive filesystems can resolve multiple spellings of one entry.
    existing = sorted({path.resolve() for path in matches})
    if len(existing) > 1:
        raise SampleIssue("ambiguous_modality_files")
    return existing[0] if existing else None


def _pixels(path: Path, metadata: dict[str, object]) -> tuple[int, ...]:
    from PIL import Image

    try:
        with Image.open(path) as image:
            if image.format != "JPEG" or image.size != (metadata["width"], metadata["height"]):
                raise SampleIssue("image_format_or_size_mismatch")
            return tuple(image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR).tobytes())
    except (OSError, Image.DecompressionBombError):
        raise SampleIssue("invalid_image") from None


def _sensor_group(data: dict[str, object]) -> dict[str, object]:
    groups = data.get("sensor_data")
    if groups is None or groups == []:
        return {"sensor_sequence": data.get("sensor_sequence")}
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        raise SampleIssue("ambiguous_json_sensor_group")
    if data.get("sensor_sequence") is not None:
        raise SampleIssue("ambiguous_json_sensor_group")
    return groups[0]


def _capture_time(token: str, metadata: dict[str, object]) -> datetime | None:
    try:
        filename_time = datetime.strptime(token, "%Y%m%d-%H%M%S")
    except ValueError:
        filename_time = None
    explicit = metadata.get("timestamp")
    if explicit is None:
        return filename_time
    parsed = parse_timestamp(explicit)
    wall_time = datetime.fromisoformat(str(explicit).strip()).replace(tzinfo=None)
    if filename_time is not None and filename_time != wall_time:
        raise SampleIssue("image_capture_timestamp_mismatch")
    return parsed


def _groups(
    stem: str, sample: Sample, image_hash: str | None, *metadata: dict[str, object],
) -> dict[str, str]:
    _, _, _, source, capture, *_ = stem.split("_")
    groups = {"sample_id": stem, "source": source}
    if len(capture) >= 8 and capture[:8].isdigit():
        groups["source_day"] = digest_json((source, capture[:8]))
    if sample.capture_time is not None:
        wall_capture = capture if "-" in capture else sample.capture_time.isoformat()
        groups["source_capture"] = digest_json((source, wall_capture))
    sessions = set()
    for item in metadata:
        session = item.get("production_session_id")
        if session is not None:
            if type(session) not in (str, int) or not str(session).strip():
                raise SampleIssue("invalid_production_session_id")
            sessions.add(str(session).strip())
    if len(sessions) > 1:
        raise SampleIssue("production_session_conflict")
    if sessions:
        groups["session"] = digest_json((source, next(iter(sessions))))
    if image_hash is not None:
        groups["image_bytes"] = image_hash
    if sample.sequence:
        groups["sensor_sequence"] = digest_json([row.model_dump(mode="json") for row in sample.sequence])
    return groups


def _prepare_sample(path: Path, root: Path, split: Split, config: ClassificationConfig) -> AuditedSample:
    stem = path.stem
    if not safe_component(stem) or not safe_component(path.parent.name):
        raise SampleIssue("invalid_filename")
    legacy_issues: list[Row] = []
    audited = analyze(path, root, False, False, config.sensor_value_atol, legacy_issues, Counter(), {}, defaultdict(set))
    problems = {str(issue["issue"]) for issue in legacy_issues} - _CLASSIFICATION_CHECKS
    if stem.startswith("SD_NOR_") and audited["boxes"] == 0:
        problems.discard("expected_category_missing")
    if problems:
        raise SampleIssue(";".join(sorted(problems)))

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if len(data["images"]) != 1:
        raise SampleIssue("ambiguous_image_metadata")
    metadata = data["images"][0]
    label = int(any(box["category_id"] in TARGET_IDS for box in data.get("annotations", [])))
    if label != int(stem.startswith("SD_DEF_")):
        raise SampleIssue("image_label_conflict")
    capture = _capture_time(stem.split("_")[4], metadata)
    group = _sensor_group(data)
    image_name = metadata["file_name"]
    if not isinstance(image_name, str) or not safe_component(image_name):
        raise SampleIssue("image_metadata_filename_mismatch")
    if PureWindowsPath(image_name).suffix.lower() not in {".jpg", ".jpeg"}:
        raise SampleIssue("image_metadata_filename_mismatch")
    if group.get("image_file_name", image_name) != image_name:
        raise SampleIssue("sensor_image_filename_mismatch")

    folder = path.parent.name
    image_path = _paired_file(path.parent.with_name(folder.replace("_annotations_", "_images_", 1)),
                              stem, (".JPG", ".jpg", ".JPEG", ".jpeg"))
    sensor_path = _paired_file(path.parent.with_name(folder.replace("_annotations_", "_sensors_", 1)),
                               stem, (".xlsx", ".XLSX"))
    if image_path is not None and image_path.name.lower() != image_name.lower():
        raise SampleIssue("image_metadata_filename_mismatch")
    sensor_name = group.get("file_name")
    if sensor_name is not None and (
        not isinstance(sensor_name, str) or not safe_component(sensor_name)
        or PureWindowsPath(sensor_name).stem != stem or PureWindowsPath(sensor_name).suffix.lower() != ".xlsx"
        or (sensor_path is not None and sensor_name.lower() != sensor_path.name.lower())
    ):
        raise SampleIssue("sensor_filename_mismatch")
    if any(not item.resolve().is_relative_to(root) for item in (path, image_path, sensor_path) if item is not None):
        raise SampleIssue("modality_path_outside_dataset")

    embedded = json_readings(group.get("sensor_sequence"), label)
    workbook = xlsx_readings(sensor_path, label) if sensor_path is not None else ()
    sequence, sensor_source = aligned_sequence(embedded, workbook, capture, config.sensor_value_atol)
    pixels = _pixels(image_path, metadata) if image_path is not None else None
    if pixels is None and not sequence:
        raise SampleIssue("no_modalities")
    image_id, sensor_id = metadata.get("id"), group.get("image_id")
    offset = sensor_id - image_id if type(image_id) is int and type(sensor_id) is int else None
    sample = Sample(
        sample_id=stem, split=split, label=label, annotation_path=path.relative_to(root).as_posix(),
        image_path=image_path.relative_to(root).as_posix() if image_path else None,
        sensor_path=sensor_path.relative_to(root).as_posix() if sensor_path else None,
        sensor_source=sensor_source, sensor_image_id_offset=offset, capture_time=capture,
        sequence=sequence, pixels=pixels,
    )
    return AuditedSample(sample, _groups(stem, sample, sha256(image_path) if image_path else None, data, metadata, group))


def prepare_samples(
    root: Path, output: Path, config: ClassificationConfig, source_fingerprint: str,
    progress: Progress,
) -> None:
    root = root.resolve()
    require_separate_output(root, output)
    paths = [
        (split, path) for split in SPLIT_MAP
        for path in sorted(root.glob(f"{split}_annotations_*/*.json"))
    ]
    if not paths:
        raise ValueError("No Train/Validation SMT annotation files found.")

    issues: list[Row] = []
    audited = []
    progress(0, len(paths))
    for done, (split, path) in enumerate(paths, 1):
        try:
            audited.append(_prepare_sample(path, root, split, config))
        except SampleIssue as exc:
            issues.append({"split": split, "sample_id": path.stem if safe_component(path.stem) else "",
                           "reason": str(exc)})
        progress(done, len(paths))

    grouped: defaultdict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for entry in audited:
        for name, group in entry.groups.items():
            grouped[name, group][entry.sample.split] += 1
    overlaps = {key for key, counts in grouped.items() if len(counts) > 1}
    enforced = set(_GROUP_KINDS[:5])
    if config.group_policy != "session":
        enforced.add(config.group_policy)
    retained: list[Sample] = []
    for entry in audited:
        sample = entry.sample
        conflicts = [name for name, group in entry.groups.items() if name in enforced and (name, group) in overlaps]
        if conflicts:
            issues.append({"split": sample.split, "sample_id": sample.sample_id,
                           "reason": ";".join(f"cross_split_{name}" for name in sorted(conflicts))})
        else:
            retained.append(sample)

    coverage = Counter(
        f"{sample.split}/" + ("paired" if sample.sequence and sample.pixels else "sensor_only" if sample.sequence else "image_only")
        for sample in retained
    )
    with (output / "samples.jsonl").open("x", encoding="utf-8") as stream:
        for sample in retained:
            stream.write(sample.model_dump_json() + "\n")
    write_csv(output / "alignment_issues.csv", issues, ["split", "sample_id", "reason"])
    write_json(output / "alignment.json", {
        "schema_version": SCHEMA_VERSION, "label_policy": LABEL_POLICY,
        "detection_policy_reference": POLICY_VERSION, "source_fingerprint": source_fingerprint,
        "split_fingerprint": digest_json(sorted((s.split, s.annotation_path, s.label) for s in retained)),
        "sensor_fields": SENSOR_FIELDS, "sensor_value_atol": config.sensor_value_atol,
        "image_size": IMAGE_SIZE, "group_policy": config.group_policy,
        "coverage": dict(coverage), "annotation_count": len(paths), "retained": len(retained),
        "excluded_reasons": dict(Counter(str(item["reason"]) for item in issues)),
        "sensor_sources": dict(Counter(sample.sensor_source or "absent" for sample in retained)),
        "sensor_image_id_offsets": dict(Counter(str(sample.sensor_image_id_offset) for sample in retained)),
        "group_audit": {
            kind: {
                "cross_split_groups": sum(name == kind for name, _ in overlaps),
                "cross_split_rows": {
                    split: sum(grouped[key][split] for key in overlaps if key[0] == kind)
                    for split in SPLIT_MAP
                },
                "rows_with_group": sum(kind in entry.groups for entry in audited),
                "exclude_overlaps": kind in enforced,
            } for kind in _GROUP_KINDS
        },
    })
