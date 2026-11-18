"""Read-only evidence collection for the source split of AI Hub SMT 71900."""

import hashlib
import json
import math
from bisect import bisect_right
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

from ...classification.sensors import SampleIssue, aligned_sequence, parse_timestamp, reading, xlsx_readings
from ...pipeline.store import digest_json, fingerprint_path
from ..model import Progress
from .common import CsvRow, read_csv, safe_component
from .leakage_cache import ObservationCache, open_observation_cache
from .policy import CLASS_MAP, SENSOR_FIELDS, SPLIT_MAP, label_policy

MODALITIES = ("image_path", "annotation_path", "sensor_path")
SENSOR_TOLERANCES = (0.05, 0.1, 0.001, 0.001, 0.1)
METHODS = {
    "image_sha256": ("confirmed", "high", "Identical encoded image bytes. Keep together and review labels."),
    "sample_id": ("potential", "high", "Same filename stem. Inspect identity and annotations."),
    "image_similarity": ("potential", "medium", "Review image pairs for recaptures, edits, or lookalikes."),
    "production_session": ("confirmed", "high", "Same explicit source-scoped session. Keep together."),
    "capture_group": ("potential", "high", "Same explicit source-scoped capture group. Keep together."),
    "source_capture": ("potential", "medium", "Same source and capture time. Check production records."),
    "source_day": ("potential", "low", "Same source and capture day. Check acquisition boundaries."),
    "capture_timestamp": ("potential", "low", "Same capture time. Check clocks, sources, and records."),
    "manufacturer": ("potential", "info", "Shared source token alone does not establish leakage."),
    "source_identifier": ("potential", "info", "Shared source identifier alone does not establish leakage."),
    "sensor_exact": ("potential", "high", "Identical timestamped measurements. Check sensor and production provenance."),
    "sensor_similarity": ("potential", "medium", "Review closely matching sensor windows and their provenance."),
}


@dataclass(frozen=True)
class AnalysisConfig:
    image_hamming: int = 4
    image_mae: float = 8.0
    sensor_scale: float = 1.0
    verify_xlsx: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.image_hamming <= 8 or not math.isfinite(self.image_mae) or not 0 <= self.image_mae <= 255:
            raise ValueError("Use image Hamming distance 0–8 and finite thumbnail MAE 0–255.")
        if not math.isfinite(self.sensor_scale) or not 0 < self.sensor_scale <= 10:
            raise ValueError("Use a finite sensor tolerance scale greater than 0 and at most 10.")


@dataclass
class Observation:
    row: CsvRow
    hashes: dict[str, str] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    classes: list[int] = field(default_factory=list)
    image: tuple[int, bytes] | None = None
    sensor: tuple[tuple[float, ...], tuple[float, ...]] | None = None

    def checkpoint(self) -> dict[str, Any]:
        return {
            "hashes": self.hashes, "groups": self.groups, "issues": self.issues, "classes": self.classes,
            "image": [self.image[0], self.image[1].hex()] if self.image is not None else None,
            "sensor": self.sensor,
        }

    @classmethod
    def restore(cls, row: CsvRow, saved: dict[str, Any]) -> "Observation":
        pixels = saved["image"]
        sensor = saved["sensor"]
        return cls(row, saved["hashes"], saved["groups"], saved["issues"], saved["classes"],
                   (pixels[0], bytes.fromhex(pixels[1])) if pixels is not None else None,
                   (tuple(sensor[0]), tuple(sensor[1])) if sensor is not None else None)


class Components:
    """Track transitive evidence across the entire cohort."""

    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.size = [1] * count

    def root(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def join(self, left: int, right: int) -> None:
        left, right = self.root(left), self.root(right)
        if left != right:
            if self.size[left] < self.size[right]:
                left, right = right, left
            self.parent[right] = left
            self.size[left] += self.size[right]

    def groups(self, members: Iterable[int]) -> list[list[int]]:
        groups: dict[int, list[int]] = defaultdict(list)
        for member in members:
            groups[self.root(member)].append(member)
        return list(groups.values())


def is_candidate(row: CsvRow) -> bool:
    return row.get("decision") == row.get("use_for_detection") == "candidate"


def source_rows(manifest: Path, root: Path, path_base: Path) -> list[CsvRow]:
    """Normalize either manifest dialect to paths relative to the extracted root."""
    return normalize_source_rows(read_csv(manifest), root, path_base)


def normalize_source_rows(
    rows: list[CsvRow], root: Path, path_base: Path, *, provider_column: str = "split",
) -> list[CsvRow]:
    if not rows:
        raise ValueError("The source manifest is empty.")
    for row in rows:
        if row.get(provider_column) not in SPLIT_MAP or not safe_component(row.get("sample_id", "")):
            raise ValueError("The source manifest contains an invalid original split or sample ID.")
        for key in MODALITIES:
            if row.get(key):
                path = (path_base / row[key]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Manifest paths must resolve inside --root. Check --path-base.")
                row[key] = path.relative_to(root).as_posix()
            else:
                row[key] = ""
    rows.sort(key=lambda row: (row[provider_column], row["sample_id"], row["annotation_path"]))
    keys = {(row[provider_column], row["sample_id"], row["annotation_path"]) for row in rows}
    if len(keys) != len(rows):
        raise ValueError("The manifest contains repeated source records.")
    return rows


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_hashes(root: Path, row: CsvRow) -> dict[str, str]:
    return {key: file_sha(root / row[key]) for key in MODALITIES if row[key] and (root / row[key]).is_file()}


def inspect_sample(
    root: Path, row: CsvRow, config: AnalysisConfig, cache: ObservationCache | None, index: int,
) -> Observation:
    if cache is None:
        return observe(root, row, config)
    saved = cache.load(index)
    if saved is not None:
        try:
            current = source_hashes(root, row)
        except OSError:
            current = None
        if saved["hashes"] == current:
            return Observation.restore(row, saved)
    item = observe(root, row, config)
    cache.save(index, item.checkpoint())
    return item


def image_signature(raw: bytes) -> tuple[int, bytes]:
    from PIL import Image

    with Image.open(BytesIO(raw)) as image:
        reduced = image.convert("RGB")
        thumb = reduced.resize((16, 16), Image.Resampling.LANCZOS).tobytes()
        pixels = reduced.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    signature = 0
    for y in range(8):
        for x in range(8):
            offset = 9 * y + x
            signature = (signature << 1) | (pixels[offset] > pixels[offset + 1])
    return signature, thumb


def metadata_groups(item: Observation, document: dict, image: dict, sensor: dict) -> datetime | None:
    tokens = item.row["sample_id"].split("_")
    source = item.row.get("manufacturer") or (tokens[3] if len(tokens) >= 6 else "")
    if source:
        item.groups["manufacturer"] = source
    for field_name in ("source_id", "manufacturer_id"):
        raw = [record[field_name] for record in (document, image, sensor) if field_name in record]
        values = {str(value).strip() for value in raw if type(value) in (int, str) and str(value).strip()}
        if len(values) == 1 and all(type(value) in (int, str) and str(value).strip() for value in raw):
            source = digest_json([field_name, values.pop()])
            item.groups["source_identifier"] = source
            break
        if raw:
            item.issues.append(f"conflicting_or_invalid_{field_name}")
    capture = None
    if len(tokens) >= 6:
        try:
            capture = datetime.strptime(tokens[4], "%Y%m%d-%H%M%S")
        except ValueError:
            pass
    if image.get("timestamp") is not None:
        try:
            explicit = parse_timestamp(image["timestamp"])
            if capture is not None and capture != datetime.fromisoformat(image["timestamp"]).replace(tzinfo=None):
                raise ValueError
            capture = explicit
        except (ValueError, TypeError):
            item.issues.append("conflicting_or_invalid_capture_time")
            capture = None
    if capture is not None:
        timestamp = capture.isoformat()
        item.groups["capture_timestamp"] = timestamp
        if source:
            item.groups["source_capture"] = digest_json([source, timestamp])
            item.groups["source_day"] = digest_json([source, capture.date().isoformat()])
    for field_name, group in (("production_session_id", "production_session"), ("capture_group_id", "capture_group")):
        raw = [record[field_name] for record in (document, image, sensor) if field_name in record]
        if not raw:
            continue
        if not source or any(type(value) not in (str, int) or not str(value).strip() for value in raw):
            item.issues.append(f"unusable_{field_name}")
        elif len(values := {str(value).strip() for value in raw}) != 1:
            item.issues.append(f"conflicting_{field_name}")
        else:
            item.groups[group] = digest_json([source, values.pop()])
    return capture


def observe(root: Path, row: CsvRow, config: AnalysisConfig) -> Observation:
    from PIL import Image

    item = Observation(row, groups={"sample_id": row["sample_id"].casefold()})
    annotation = {}
    for key in MODALITIES:
        if not row[key]:
            item.issues.append(f"missing_{key}")
            continue
        try:
            path = root / row[key]
            if key == "sensor_path":
                item.hashes[key] = file_sha(path)
                continue
            raw = path.read_bytes()
            item.hashes[key] = hashlib.sha256(raw).hexdigest()
            if key == "image_path":
                item.groups["image_sha256"] = item.hashes[key]
                item.image = image_signature(raw)
            else:
                data = json.loads(raw.decode("utf-8-sig"))
                if not isinstance(data, dict):
                    raise ValueError
                annotation = data
        except (OSError, ValueError, Image.DecompressionBombError):
            item.issues.append(f"unreadable_{key}")

    images = annotation.get("images")
    image = images[0] if isinstance(images, list) and len(images) == 1 and isinstance(images[0], dict) else {}
    sensors = annotation.get("sensor_data")
    sensor = sensors[0] if isinstance(sensors, list) and len(sensors) == 1 and isinstance(sensors[0], dict) else {}
    capture = metadata_groups(item, annotation, image, sensor)
    boxes = annotation.get("annotations", [])
    if isinstance(boxes, list):
        item.classes = sorted({
            CLASS_MAP[ann["category_id"]] for ann in boxes
            if isinstance(ann, dict) and type(ann.get("category_id")) is int and ann["category_id"] in CLASS_MAP
        })
    try:
        raw_sequence = sensor.get("sensor_sequence", annotation.get("sensor_sequence", []))
        if not isinstance(raw_sequence, list) or any(not isinstance(record, dict) for record in raw_sequence):
            raise SampleIssue("invalid_json_sensor_sequence")
        embedded = tuple(
            reading(record.get("timestamp"), tuple(record.get(key) for key in SENSOR_FIELDS))
            for record in raw_sequence
        )
        workbook = ()
        if config.verify_xlsx and "sensor_path" in item.hashes:
            workbook = xlsx_readings(root / row["sensor_path"], None)
        sequence, _ = aligned_sequence(embedded, workbook, capture, 0.001)
        if sequence:
            item.groups["sensor_exact"] = digest_json([
                [record.timestamp.isoformat(), record.values] for record in sequence
            ])
        if len(sequence) >= 2:
            item.sensor = (
                tuple((record.timestamp - sequence[0].timestamp).total_seconds() for record in sequence),
                tuple(value for record in sequence for value in record.values),
            )
        else:
            item.issues.append("insufficient_sensor_readings")
    except SampleIssue as exc:
        item.issues.append(str(exc))
    return item


def image_pairs(
    items: list[Observation], config: AnalysisConfig, progress: Progress | None = None,
) -> Iterator[tuple[int, int, dict]]:
    # With d + 1 disjoint blocks, any hashes within distance d share at least one block.
    block_count = config.image_hamming + 1
    blocks = [(64 * index // block_count, 64 * (index + 1) // block_count) for index in range(block_count)]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    if progress:
        progress(0, len(items))
    for index, item in enumerate(items):
        if progress:
            progress(index + 1, len(items))
        if item.image is None:
            continue
        hashes, thumb = item.image
        possible: set[int] = set()
        for start, stop in blocks:
            bucket = buckets[start, (hashes >> start) & ((1 << (stop - start)) - 1)]
            possible.update(bucket)
            bucket.append(index)
        for other in sorted(possible):
            other_hash, other_thumb = items[other].image
            distance = (hashes ^ other_hash).bit_count()
            if distance > config.image_hamming or item.hashes["image_path"] == items[other].hashes["image_path"]:
                continue
            mae = sum(abs(a - b) for a, b in zip(thumb, other_thumb)) / len(thumb)
            if mae <= config.image_mae:
                yield other, index, {"hamming_distance": distance, "thumbnail_mae": round(mae, 6)}


def sensor_pairs(
    items: list[Observation], config: AnalysisConfig, progress: Progress | None = None,
) -> Iterator[tuple[int, int, dict]]:
    tolerances = tuple(value * config.sensor_scale for value in SENSOR_TOLERANCES)
    by_cadence: dict[tuple[float, ...], list[tuple[float, int, tuple[float, ...]]]] = defaultdict(list)
    for index, item in enumerate(items):
        if item.sensor is not None:
            cadence, values = item.sensor
            by_cadence[cadence].append((values[0], index, values))
    total = sum(map(len, by_cadence.values()))
    done = 0
    if progress:
        progress(done, total)
    for entries in by_cadence.values():
        entries.sort()
        ordered = [entry[0] for entry in entries]
        for position, (first, index, values) in enumerate(entries):
            end = bisect_right(ordered, first + tolerances[0], lo=position + 1)
            for other_position in range(position + 1, end):
                _, other, measurements = entries[other_position]
                if items[index].groups["sensor_exact"] == items[other].groups["sensor_exact"]:
                    continue
                differences = []
                for axis, (a, b) in enumerate(zip(values, measurements)):
                    ratio = abs(a - b) / tolerances[axis % len(tolerances)]
                    if ratio > 1:
                        break
                    differences.append(ratio)
                else:
                    yield index, other, {"max_tolerance_fraction": round(max(differences), 6),
                                         "reading_count": len(values) // len(tolerances)}
            done += 1
            if progress:
                progress(done, total)


def finding(method: str, members: list[int], items: list[Observation], evidence: dict) -> dict:
    status, severity, handling = METHODS[method]
    return {
        "method": method, "status": status, "severity": severity, "members": members,
        "group_id": digest_json([method, members]), "evidence": evidence, "recommended_handling": handling,
        "by_original_split": dict(Counter(items[i].row["split"] for i in members)),
        "candidates_by_original_split": dict(Counter(
            items[i].row["split"] for i in members if is_candidate(items[i].row)
        )),
    }


def pair_groups(method: str, pairs: Iterable[tuple[int, int, dict]], items: list[Observation]) -> list[dict]:
    components = Components(len(items))
    involved: set[int] = set()
    evidence_by_root: dict[int, dict] = {}
    for left, right, evidence in pairs:
        previous = {components.root(left), components.root(right)}
        records = [evidence_by_root.pop(root) for root in sorted(previous) if root in evidence_by_root]
        components.join(left, right)
        involved.update((left, right))
        crosses = items[left].row["split"] != items[right].row["split"]
        candidate_crosses = crosses and all(is_candidate(items[i].row) for i in (left, right))
        counts = sum(record["pair_count"] for record in records)
        cross_counts = sum(record["cross_split_pair_count"] for record in records)
        example = next((record["example"] for record in records if record["cross_split_pair_count"]), None)
        if example is None:
            example = {"members": [left, right], **evidence} if crosses or not records else records[0]["example"]
        evidence_by_root[components.root(left)] = {
            "pair_count": counts + 1, "cross_split_pair_count": cross_counts + crosses, "example": example,
            "direct_candidate_cross_split_pairs": (
                sum(record["direct_candidate_cross_split_pairs"] for record in records) + candidate_crosses
            ),
        }
    return [
        finding(method, group, items, evidence_by_root[components.root(group[0])])
        for group in components.groups(sorted(involved))
    ]


def analyze_leakage(
    root: Path, manifest: Path, path_base: Path, config: AnalysisConfig, progress: Progress | None = None,
    *, similarity_progress: Callable[[str], Progress] | None = None, checkpoint: Path | None = None,
) -> dict:
    from importlib.metadata import version

    root = root.resolve()
    rows = source_rows(manifest, root, path_base)
    configuration = {**asdict(config), "pillow": version("Pillow"), "sensor_atol": SENSOR_TOLERANCES,
                     "openpyxl": version("openpyxl") if config.verify_xlsx else None}
    manifest_fingerprint = digest_json(rows)
    identity = ""
    if checkpoint is not None:
        from ...classification import contracts, sensors

        identity = digest_json([manifest_fingerprint, configuration, label_policy(),
                                [fingerprint_path(Path(path)) for path in (__file__, contracts.__file__, sensors.__file__)]])
    with open_observation_cache(checkpoint, root, identity) as cache:
        return _analyze(root, rows, config, configuration, manifest_fingerprint, progress, similarity_progress, cache)


def _analyze(
    root: Path, rows: list[CsvRow], config: AnalysisConfig, configuration: dict[str, Any],
    manifest_fingerprint: str, progress: Progress | None,
    similarity_progress: Callable[[str], Progress] | None, cache: ObservationCache | None,
) -> dict:
    items = []
    if cache:
        cache.stage("inspect_sources")
    if progress:
        progress(0, len(rows))
    for done, row in enumerate(rows, 1):
        items.append(inspect_sample(root, row, config, cache, done - 1))
        if progress:
            progress(done, len(rows))

    indexed: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        for method, value in item.groups.items():
            indexed[method, value].append(index)
    groups = [
        finding(method, members, items, {"key": key})
        for (method, key), members in sorted(indexed.items()) if len(members) > 1
    ]
    image_progress = similarity_progress("Compare image signatures") if similarity_progress else None
    sensor_progress = similarity_progress("Compare sensor sequences") if similarity_progress else None
    if cache:
        cache.stage("compare_images")
    groups.extend(pair_groups("image_similarity", image_pairs(items, config, image_progress), items))
    if cache:
        cache.stage("compare_sensors")
    groups.extend(pair_groups("sensor_similarity", sensor_pairs(items, config, sensor_progress), items))
    records = [
        {"original_split": row["split"], "sample_id": row["sample_id"], "candidate": is_candidate(row),
         "decision": row["decision"], "paths": {key: row[key] for key in MODALITIES}, "hashes": item.hashes,
         "classes": item.classes, "groups": item.groups, "issues": item.issues}
        for row, item in zip(rows, items, strict=True)
    ]
    if cache:
        cache.stage("complete")
    return {
        "schema_version": 1, "dataset": "aihub-smt-71900", "label_policy": label_policy(),
        "configuration": configuration, "manifest_fingerprint": manifest_fingerprint,
        "dataset_fingerprint": digest_json([[record["paths"], record["hashes"]] for record in records]),
        "original_split_counts": dict(Counter(row["split"] for row in rows)),
        "candidate_split_counts": dict(Counter(row["split"] for row in rows if is_candidate(row))),
        "unknown_provenance": {
            "status": "unknown",
            "production_session": sum("production_session" not in record["groups"] for record in records),
            "capture_timestamp": sum("capture_timestamp" not in record["groups"] for record in records),
            "manufacturer": sum("manufacturer" not in record["groups"] for record in records),
            "explicit_source_identifier": sum("source_identifier" not in record["groups"] for record in records),
            "issues": dict(Counter(issue for item in items for issue in item.issues)),
            "recommended_handling": "Obtain production logs and validate provenance before claiming session independence.",
        },
        "samples": records, "groups": groups,
        "cross_split_groups": [group["group_id"] for group in groups if len(group["by_original_split"]) > 1],
    }
