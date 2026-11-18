"""Create a deterministic cohort index and summarize alignment and sensor quality."""

import json
from collections import Counter, defaultdict
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, pstdev

from ..classification import sensors
from ..data.model import Progress
from ..data.preprocessing import convert, leakage
from ..data.preprocessing.audit import XLSX_FIELDS
from ..data.preprocessing.common import CsvRow, read_csv
from ..data.preprocessing.leakage import Components, normalize_source_rows
from ..data.preprocessing.leakage_cache import open_observation_cache
from ..data.preprocessing.policy import NAMES, label_policy
from ..inspection.contracts import Contract, SchemaVersion
from ..pipeline.store import Fingerprint, digest_json
from . import alignment, contracts
from .alignment import EXTENSIONS, PAIRING_RULE, VALUE_ATOL, observe_record, read_sources
from .contracts import CHANNELS, UNITS, Record

ISOLATION_GROUPS = frozenset({
    "sample_id", "image_sha256", "sensor_workbook_sha256", "sensor_exact",
    "production_session", "capture_group", "source_capture", "revision_group",
})


class Cohort(Contract):
    schema_version: SchemaVersion = 1
    manifest_fingerprint: Fingerprint
    input_fingerprint: Fingerprint
    split_fingerprint: Fingerprint
    implementation_fingerprint: Fingerprint
    revision_id: Fingerprint | None
    records: tuple[Record, ...]

    @property
    def fingerprint(self) -> str:
        return digest_json(self.model_dump(mode="json"))

    @classmethod
    def load(cls, path: Path) -> "Cohort":
        saved = json.loads(path.read_text(encoding="utf-8"))
        if digest_json(saved["cohort"]) != saved["fingerprint"]:
            raise ValueError("Multimodal cohort fingerprint mismatch. Rebuild the index.")
        return cls.model_validate_json(json.dumps(saved["cohort"]))


def _read_manifest(
    root: Path, manifest: Path, path_base: Path, revision_path: Path | None,
) -> tuple[list[CsvRow], dict | None]:
    rows = read_csv(manifest)
    revision = None
    if revision_path is not None:
        revision = json.loads(revision_path.read_text(encoding="utf-8"))
        revision_id = revision["revision_id"]
        if (revision.get("schema_version") != 1
                or digest_json({k: v for k, v in revision.items() if k != "revision_id"}) != revision_id
                or digest_json(rows) != revision["selected_manifest_fingerprint"]
                or digest_json(revision["label_policy"]) != digest_json(label_policy())):
            raise ValueError("The split revision, manifest, or label policy differs from the published assignment.")
    elif any(row.get("original_split") or row.get("group_id") for row in rows):
        raise ValueError("Supply the published split.json with a revised manifest.")
    for row in rows:
        if row.get("decision") not in ("candidate", "hold_review", "exclude"):
            raise ValueError("The manifest contains an unknown decision.")
        if row.get("split") not in ("train", "validation", "test") or (
            revision is None and row["split"] == "test"
        ):
            raise ValueError("The manifest contains an invalid split assignment.")
        if revision is None:
            row["original_split"] = row["split"]
    return normalize_source_rows(rows, root, path_base, provider_column="original_split"), revision


def _folder_counts(folder: Path, extensions: tuple[str, ...]) -> Counter[str]:
    try:
        return Counter(entry.stem.casefold() for entry in folder.iterdir()
                       if entry.suffix.lower() in extensions and entry.is_file())
    except FileNotFoundError:
        return Counter()


def _ambiguous_files(
    root: Path, row: CsvRow, cache: dict[tuple[Path, str], Counter[str]],
) -> frozenset[str]:
    ambiguous = set()
    for key, extensions in EXTENSIONS.items():
        if not row[key]:
            continue
        folder = (root / row[key]).parent
        if (folder, key) not in cache:
            cache[folder, key] = _folder_counts(folder, extensions)
        if cache[folder, key][row["sample_id"].casefold()] > 1:
            ambiguous.add(key)
    return frozenset(ambiguous)


def _isolate(records: list[Record]) -> tuple[Record, ...]:
    groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    components = Components(len(records))
    flags: defaultdict[int, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        for kind, group in record.groups:
            if kind in ISOLATION_GROUPS:
                groups[kind, group].append(index)
    for (kind, _), members in groups.items():
        assigned = {records[index].split for index in members if records[index].detection_candidate}
        for index in members[1:]:
            components.join(members[0], index)
        if len(assigned) > 1:
            for index in members:
                flags[index].add(f"cross_split_{kind}")
        if kind == "sample_id" and len(members) > 1:
            for index in members:
                flags[index].add("ambiguous_duplicate_sample_id")
    active = [i for i, record in enumerate(records) if record.detection_candidate]
    for members in components.groups(active):
        if len({records[index].split for index in members}) > 1:
            for index in members:
                flags[index].add("cross_split_connected_group")
    return tuple(
        record.model_copy(update={
            "quarantined": True, "issues": (*record.issues, *sorted(flags[index])),
            **({"alignment": "ambiguous", "sequence": None, "alignment_method": None}
               if "ambiguous_duplicate_sample_id" in flags[index] else {}),
        }) if index in flags else record
        for index, record in enumerate(records)
    )


def audit_cohort(
    root: Path, manifest: Path, *, path_base: Path | None = None, revision_path: Path | None = None,
    progress: Progress | None = None, checkpoint: Path | None = None,
) -> Cohort:
    root = root.resolve()
    rows, revision = _read_manifest(root, manifest, (path_base or root).resolve(), revision_path)
    manifest_fingerprint = digest_json(rows)
    revision_id = revision["revision_id"] if revision else None
    implementation = digest_json([
        [(module.__name__, leakage.file_sha(Path(module.__file__)))
         for module in (alignment, contracts, sensors, leakage, convert)],
        leakage.file_sha(Path(__file__)), version("Pillow"), version("openpyxl"),
    ])
    identity = digest_json([manifest_fingerprint, revision_id, implementation, PAIRING_RULE, VALUE_ATOL])
    records = []
    inputs = []
    folders: dict[tuple[Path, str], Counter[str]] = {}
    with open_observation_cache(checkpoint, root, identity) as cache:
        if cache:
            cache.stage("scan")
        if progress:
            progress(0, len(rows))
        for index, row in enumerate(rows):
            sources, hashes = read_sources(root, row)
            ambiguity = _ambiguous_files(root, row, folders)
            inputs.append([{key: row[key] for key in EXTENSIONS}, hashes])
            saved = cache.load(index) if cache else None
            if saved is not None and saved["hashes"] == hashes and saved["ambiguity"] == sorted(ambiguity):
                record = Record.model_validate_json(json.dumps(saved["record"]))
            else:
                record = observe_record(row, sources, hashes, ambiguity)
                if cache:
                    cache.save(index, {"hashes": hashes, "ambiguity": sorted(ambiguity),
                                       "record": record.model_dump(mode="json")})
            records.append(record)
            if progress:
                progress(index + 1, len(rows))
        input_fingerprint = digest_json(inputs)
        if revision is not None:
            expected_source = revision.get("selected_source_fingerprint", revision["dataset_fingerprint"])
            if input_fingerprint != expected_source:
                raise ValueError("Source files changed since the split revision was published.")
        records = _isolate(records)
        if cache:
            cache.stage("complete")
    return Cohort(
        manifest_fingerprint=manifest_fingerprint, input_fingerprint=input_fingerprint,
        split_fingerprint=digest_json([
            [record.sample_id, record.annotation_path, record.provider_split, record.split,
             record.detection_candidate, record.groups, record.quarantined] for record in records
        ]),
        implementation_fingerprint=implementation, revision_id=revision_id, records=records,
    )


def _availability(record: Record) -> str:
    if record.quarantined:
        return "quarantined"
    if record.image_available and record.sensor_available:
        return "paired"
    if record.image_available:
        return "image_only"
    if record.sensor_available:
        return "sensor_only"
    return "no_modalities"


def _coverage(records: list[Record]) -> dict[str, object]:
    return {
        "samples": len(records),
        "alignment": dict(Counter(record.alignment for record in records)),
        "verified_pairs": sum(record.image_available and record.sensor_available for record in records),
        "timestamp_inferred_pairs": 0,
        "availability": dict(Counter(map(_availability, records))),
    }


def _quality(records: list[Record]) -> dict[str, object]:
    sequences = [record.sequence for record in records if record.sequence is not None]
    length_counts = Counter(str(len(seq.steps)) for seq in sequences)
    rows = [step.values for seq in sequences for step in seq.steps]
    columns = {}
    for index, name in enumerate(CHANNELS):
        observed = [row[index] for row in rows if row[index] is not None]
        columns[name] = {
            "unit": UNITS[index], "observed": len(observed), "missing": len(rows) - len(observed),
            "recorded_zeros": observed.count(0),
            "min": min(observed, default=None), "max": max(observed, default=None),
            "mean": fmean(observed) if observed else None, "std": pstdev(observed) if observed else None,
        }
    return {"sequence_lengths": dict(length_counts), "steps": len(rows), "channels": columns}


def audit_report(cohort: Cohort, *, preview: bool = False) -> dict[str, object]:
    records = list(cohort.records)
    candidates = [record for record in records if record.detection_candidate]
    by_split = {}
    for split in sorted({record.split for record in records}):
        selected = [record for record in records if record.split == split]
        eligible = [record for record in candidates if record.split == split]
        classes: defaultdict[str, list[Record]] = defaultdict(list)
        sources: defaultdict[str, list[Record]] = defaultdict(list)
        for record in eligible:
            names = ("unknown",) if record.targets is None else (
                tuple(NAMES[category - 1] for category in sorted({target.category for target in record.targets}))
                or ("background",)
            )
            for name in names:
                classes[name].append(record)
            sources[record.source_group or "unknown"].append(record)
        by_split[split] = {
            **_coverage(selected), "detection_candidates": _coverage(eligible),
            "candidate_classes": {name: _coverage(samples) for name, samples in sorted(classes.items())},
            "candidate_source_groups": {name: _coverage(samples) for name, samples in sorted(sources.items())},
            "sensor_quality": _quality(selected),
        }
    report: dict[str, object] = {
        "schema_version": 1, "label_policy": label_policy(),
        "pairing": {"rule": PAIRING_RULE, "sensor_value_atol": VALUE_ATOL, "timestamp_only": "disabled",
                    "required_copies": ["annotation JSON", "sibling XLSX"], "isolated_groups": sorted(ISOLATION_GROUPS)},
        "sensor_schema": {"channels": CHANNELS, "units": UNITS, "workbook_columns": ("timestamp", *XLSX_FIELDS)},
        **cohort.model_dump(exclude={"records"}), "cohort_fingerprint": cohort.fingerprint,
        **_coverage(records), "detection_candidates": _coverage(candidates), "by_split": by_split,
        "decisions": dict(Counter(record.decision for record in records)),
        "issues": dict(Counter(issue for record in records for issue in record.issues)),
        "candidate_issues": dict(Counter(issue for record in candidates for issue in record.issues)),
        "unknown_session": sum("production_session" not in dict(record.groups) for record in records),
        "sensor_id_offsets": dict(Counter(
            str(record.sensor_image_id - record.image_id) for record in records
            if record.image_id is not None and record.sensor_image_id is not None
        )),
        "unsupported": ["unverified sensor relationships", "timestamp-only matching", "non-SMT measurement channels"],
    }
    if preview:
        example = next((record for record in candidates if record.split == "train"
                        and not record.quarantined and record.sensor_available), None)
        report["sensor_preview"] = {
            "sample_id": example.sample_id, "steps": [step.model_dump() for step in example.sequence.steps[:8]],
        } if example and example.sequence else None
    return report
