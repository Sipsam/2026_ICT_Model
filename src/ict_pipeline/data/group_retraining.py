"""Verify an existing group revision while keeping Test payloads sealed."""

import json
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from ..classification import contracts, sensors
from ..config.settings import Credentials
from ..evaluation.product_protocol import FrozenValidation
from ..models.dataset import inspect_data
from ..multimodal.audit import ISOLATION_GROUPS, Cohort
from ..multimodal.preprocessing import PreprocessingState, fit_preprocessor, window_available
from ..pipeline.store import _reject_link, digest_json, fingerprint_path
from .model import Progress
from .preprocessing import leakage
from .preprocessing.common import read_csv
from .preprocessing.convert import check_source
from .preprocessing.leakage_cache import ObservationCache, open_observation_cache
from .preprocessing.policy import CLASS_MAP, NAMES
from .preprocessing.revision import read_publication
from .preprocessing.split import SPLITS, SplitConfig, grouping_methods


@dataclass(frozen=True)
class GroupInputs:
    source: Path
    bundle: Path
    revision_id: str
    cohort: Path
    cohort_fingerprint: str


def detector_scope(dataset: Path) -> dict[str, str]:
    """Fingerprint only the COCO inputs consumed by Faster R-CNN."""
    for folder in (dataset, dataset / "coco", dataset / "coco/images", dataset / "coco/annotations"):
        _reject_link(folder)
    return {
        entry: fingerprint_path(dataset / "coco" / entry)
        for entry in ("images/train", "images/val", "annotations/instances_train.json",
                      "annotations/instances_val.json")
    }


def read_verification_receipt(path: Path) -> tuple[dict, str]:
    receipt = json.loads(path.read_bytes())
    result = receipt["verification"]
    if digest_json(result) != receipt["fingerprint"] or result["schema_version"] != 1 or result["passed"] is not True:
        raise ValueError("A passing group retraining verification receipt is required.")
    return result, receipt["fingerprint"]


def read_group_verification(dataset: Path, path: Path) -> dict:
    result, fingerprint = read_verification_receipt(path)
    revision, seal, _ = read_publication(dataset.parent, result["revision_id"])
    if (dataset.name != "detection" or seal != result["publication_fingerprint"]
            or detector_scope(dataset) != result["detector_scope"]):
        raise ValueError("Verified group Detector inputs changed. Rerun group verification.")
    return {"group_verification": fingerprint, "group_split_revision": revision["revision_id"],
            "detector_scope": result["detector_scope"]}


def _assignments(revision: dict, rows: list[dict[str, str]]) -> set[str]:
    config = SplitConfig(**revision["configuration"])
    if config.strategy != "group-aware" or revision["test_origin"] != "internal_same_source":
        raise ValueError("Select a published group-aware revision with an internal Test.")
    required = set(grouping_methods(config))
    expected = {i: (row["split"], row["group_id"]) for i, row in enumerate(rows) if leakage.is_candidate(row)}
    assignments = {item["sample_index"]: (item["split"], item["group_id"]) for item in revision["assignments"]}
    counts = Counter(split for split, _ in expected.values())
    if (assignments != expected or len(assignments) != len(revision["assignments"])
            or counts != revision["counts"] or set(counts) != set(SPLITS)
            or any(not group for _, group in expected.values())
            or required != set(revision["grouping_methods"])):
        raise ValueError("Published assignments, counts, or grouping rules disagree with the manifest.")
    return required


def _check_prior_test(rows: list[dict[str, str]], manifests: tuple[Path, ...]) -> list[str]:
    """Check known older Test reservations before opening development sources."""
    keys = ("sample_id", *leakage.MODALITIES)
    development = {
        (key, row[key].replace("\\", "/").casefold())
        for row in rows if row["split"] != "test" and leakage.is_candidate(row)
        for key in keys if row[key]
    }
    fingerprints = []
    for path in manifests:
        previous = read_csv(path)
        reserved = {
            (key, row[key].replace("\\", "/").casefold())
            for row in previous if row["split"] == "test"
            for key in keys if row[key]
        }
        if not reserved:
            raise ValueError("A prior Test reservation manifest has no Test rows.")
        if development.intersection(reserved):
            raise ValueError("Prior Test custody conflict: reserved samples appear in group Train/Validation.")
        fingerprints.append(digest_json(previous))
    return fingerprints


def _evidence(
    cohort: Cohort, inputs: GroupInputs, rows: list[dict[str, str]], required: set[str],
    progress: Callable[[str], Progress], config: leakage.AnalysisConfig, cache: ObservationCache | None,
) -> tuple[dict, list[str]]:
    records = cohort.records
    items = []
    evidence: dict[tuple[str, str], list[int]] = defaultdict(list)
    distributions: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    gaps: Counter = Counter()
    scan = progress("Inspect Train/Validation image and sensor evidence")
    scan(0, len(records))
    if cache:
        cache.stage("inspect_development_sources")
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        groups = dict(record.groups)
        if record.image_sha256:
            groups["image_sha256"] = record.image_sha256
        if record.sensor_sha256:
            groups["sensor_workbook_sha256"] = record.sensor_sha256
        groups["sample_id"] = record.sample_id.casefold()
        item = leakage.Observation(row)
        if record.split != "test" and record.detection_candidate:
            item = leakage.inspect_sample(inputs.source, row, config, cache, index)
            expected = {key: digest for key, digest in (
                ("image_path", record.image_sha256), ("sensor_path", record.sensor_sha256),
                ("annotation_path", record.annotation_sha256),
            ) if digest}
            if item.hashes != expected:
                raise ValueError("Train/Validation source bytes changed during verification.")
            for kind, key in item.groups.items():
                if groups.get(kind) != key:
                    evidence[kind, key].append(index)
            gaps.update(item.issues)
        # The manifest manufacturer and filename capture tokens are permitted metadata.
        metadata = leakage.Observation(row)
        capture = leakage.metadata_groups(metadata, {}, {}, {})
        if record.detection_candidate:
            distributions[record.split]["manufacturer"][metadata.groups.get("manufacturer", "unknown")] += 1
            distributions[record.split]["capture_date"][capture.date().isoformat() if capture else "unknown"] += 1
        for kind, key in metadata.groups.items():
            groups.setdefault(kind, key)
        for kind, key in groups.items():
            evidence[kind, key].append(index)
        items.append(item)
        scan(index + 1, len(records))

    candidates = [i for i, record in enumerate(records) if record.detection_candidate]
    components = leakage.Components(len(records))
    metadata_overlap: Counter = Counter()
    near_overlap: Counter = Counter()
    within_split_near_cross_group: Counter = Counter()
    blockers = []
    strict = required | ISOLATION_GROUPS
    policy_components = leakage.Components(len(records))
    for (method, _), indices in evidence.items():
        active = [records[i] for i in indices if records[i].detection_candidate]
        if len({record.split for record in active}) > 1:
            metadata_overlap[method] += 1
        if method in strict:
            for i in indices[1:]:
                components.join(indices[0], i)
                if method in required:
                    policy_components.join(indices[0], i)
    if cache:
        cache.stage("compare_development_signatures")
    for method, pairs in (
        ("image_similarity", leakage.image_pairs(items, config, progress("Compare Train/Validation images"))),
        ("sensor_similarity", leakage.sensor_pairs(items, config, progress("Compare Train/Validation sensors"))),
    ):
        for left, right, _ in pairs:
            a, b = records[left], records[right]
            if a.split != b.split:
                near_overlap[method] += 1
            elif rows[left]["group_id"] != rows[right]["group_id"]:
                within_split_near_cross_group[method] += 1
            if method in required:
                components.join(left, right)
                policy_components.join(left, right)
    for members in components.groups(candidates):
        if len({records[i].split for i in members}) > 1:
            blockers.append("An isolation component contains candidates in multiple splits.")
            break
    inconsistent_groups = 0
    for members in policy_components.groups(candidates):
        if len({rows[i]["group_id"] for i in members}) > 1:
            inconsistent_groups += 1
    if inconsistent_groups:
        blockers.append("A required isolation component crosses published groups.")
    if any(record.quarantined for record in records if record.detection_candidate):
        blockers.append("The index contains quarantined detection candidates.")
    if any(issue.startswith("unreadable_") for issue in gaps):
        blockers.append("A Train/Validation modality could not be inspected.")
    unknown_sessions = Counter(records[i].split for i in candidates if not dict(records[i].groups).get("production_session"))
    return {
        "cross_split_groups_by_evidence": dict(metadata_overlap),
        "train_validation_near_pairs_by_method": dict(near_overlap),
        "within_split_near_pairs_crossing_groups": dict(within_split_near_cross_group),
        "components_crossing_published_groups": inconsistent_groups,
        "manufacturer_and_capture_date": distributions,
        "unknown_production_session": unknown_sessions,
        "train_validation_evidence_gaps": gaps,
        "similarity_thresholds": {"image_hamming": config.image_hamming, "image_mae": config.image_mae,
                                  "sensor_scale": config.sensor_scale},
    }, blockers


def verify_group_retraining(
    inputs: GroupInputs, frozen: FrozenValidation, state: PreprocessingState, credentials: Credentials,
    progress: Callable[[str], Progress], *, prior_manifests: tuple[Path, ...] = (), checkpoint: Path | None = None,
) -> dict:
    from ..multimodal.training_plan import _load_index

    revision, seal, rows = read_publication(inputs.bundle, inputs.revision_id)
    required = _assignments(revision, rows)
    prior_fingerprints = _check_prior_test(rows, prior_manifests)
    scoped, _, _ = _load_index(inputs, frozen, progress("Verify Train/Validation source bytes"), all_validation=True)
    if frozen != FrozenValidation.from_audit(scoped, frozen.identity.sample_ids):
        raise ValueError("Frozen Validation differs from the group cohort.")
    if state != fit_preprocessor(scoped, state.config):
        raise ValueError("Saved sensor statistics differ from a fresh fit on group Train.")
    cohort = Cohort.load(inputs.cohort)
    if cohort.fingerprint != inputs.cohort_fingerprint:
        raise ValueError("The pinned cohort changed during verification.")
    rows.sort(key=lambda row: (row["original_split"], row["sample_id"], row["annotation_path"]))
    records = tuple(sorted(cohort.records, key=lambda r: (r.provider_split, r.sample_id, r.annotation_path)))
    config = leakage.AnalysisConfig(verify_xlsx=True)
    identity = ""
    if checkpoint is not None:
        identity = digest_json([
            "group-development-observations-v1", inputs.revision_id, inputs.cohort_fingerprint,
            asdict(config), version("Pillow"), version("openpyxl"),
            [fingerprint_path(Path(module.__file__)) for module in (leakage, contracts, sensors)],
        ])
    with open_observation_cache(checkpoint, inputs.source, identity) as cache:
        evidence, blockers = _evidence(
            cohort.model_copy(update={"records": records}), inputs, rows, required, progress, config, cache,
        )
        if cache:
            cache.stage("development_evidence_complete")
    frozen_ids = set(frozen.identity.sample_ids)
    if any(
        record.targets is None or not record.image_available or not window_available(record, state.config)
        for record in scoped.records
        if record.detection_candidate and (record.split == "train" or record.sample_id in frozen_ids)
    ):
        blockers.append("A group Train or FrozenValidation sample lacks an image, targets, or a usable sensor window.")
    historical = revision.get("remaining_cross_split_groups")
    if historical is None and required.intersection(("image_similarity", "sensor_similarity")):
        blockers.append("The publication lacks the historical cross-split similarity summary.")
    elif historical and any(historical.get(method, 0) for method in required):
        blockers.append("The published analysis reports residual overlap under a required grouping method.")

    dataset = inputs.bundle / "detection"
    fingerprint = progress("Fingerprint scoped COCO inputs")
    fingerprint(0, 2)
    scope = detector_scope(dataset)
    fingerprint(1, 2)
    report = progress("Inspect scoped COCO inputs")
    data = inspect_data(dataset, "coco", credentials, lambda _, done, total: report(done, total))
    class_samples = {}
    for role, split in (("train", "train"), ("validation", "val")):
        expected = {record.sample_id: record for record in scoped.records
                    if record.detection_candidate and record.split == role}
        samples = data.splits[split].samples
        if len(expected) != len(samples) or set(expected) != {sample.image.stem for sample in samples}:
            raise ValueError(f"COCO {role} membership differs from the group manifest.")
        verify = progress(f"Verify COCO {role} images and labels")
        verify(0, len(samples))
        support = Counter(dict.fromkeys((*NAMES, "background"), 0))
        for done, sample in enumerate(samples, 1):
            record = expected[sample.image.stem]
            width, height, boxes = check_source(inputs.source / record.annotation_path, record.sample_id)
            if (sample.image.name != f"{record.sample_id}{Path(record.image_path).suffix}"
                    or leakage.file_sha(sample.image) != record.image_sha256
                    or (sample.width, sample.height) != (width, height)
                    or Counter((obj.category, obj.box) for obj in sample.objects)
                    != Counter((CLASS_MAP[source_id] + 1, tuple(box)) for source_id, box in boxes)):
                raise ValueError(f"COCO {role} images or labels differ from verified source.")
            support.update({NAMES[obj.category - 1] for obj in sample.objects} or {"background"})
            verify(done, len(samples))
        class_samples[role] = dict(support)
        if support != revision["class_samples"][role]:
            blockers.append(f"Live {role} class sample counts differ from the published summary.")
    if detector_scope(dataset) != scope:
        raise ValueError("Scoped COCO inputs changed during verification.")
    fingerprint(2, 2)
    return {
        "schema_version": 1, "passed": not blockers, "blockers": blockers, "revision_id": inputs.revision_id,
        "publication_fingerprint": seal, "manifest_fingerprint": revision["selected_manifest_fingerprint"],
        "cohort_fingerprint": inputs.cohort_fingerprint, "split_fingerprint": cohort.split_fingerprint,
        "frozen_validation_fingerprint": frozen.identity.cohort_fingerprint,
        "frozen_validation_count": len(frozen.identity.sample_ids),
        "preprocessing_fingerprint": state.fingerprint, "training_fingerprint": state.training_fingerprint,
        "counts": revision["counts"], "class_samples_at_publication": revision["class_samples"],
        "live_train_validation_class_samples": class_samples,
        "grouping_methods": sorted(required), "detector_scope": scope,
        "historical_cross_split_groups": historical,
        "prior_test_manifest_fingerprints": prior_fingerprints,
        "test": {"origin": revision["test_origin"], "coverage": "publication and sanitized index metadata only",
                 "payloads_opened": False, "payload_seal_reverified": False},
        **evidence,
    }
