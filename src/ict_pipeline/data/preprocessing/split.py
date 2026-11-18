"""Draft and publish immutable dataset split revisions."""

import math
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from ...pipeline.store import digest_json
from ..model import Progress
from .common import write_csv, write_json
from .console import fresh_output
from .convert import convert_detection
from .leakage import Components, file_sha, is_candidate, source_hashes, source_rows
from .policy import NAMES, label_policy
from .revision import SEAL, seal_dataset
from .validate import validate_detection

Strategy = Literal["aihub", "group-aware"]
Grouping = Literal["capture", "source-day", "manufacturer"]
SPLITS = ("train", "validation", "test")
REQUIRED_GROUPS = ("sample_id", "image_sha256", "sensor_exact", "production_session", "capture_group")
LABELS = (*NAMES, "background")


@dataclass(frozen=True)
class SplitConfig:
    strategy: Strategy = "group-aware"
    grouping: Grouping = "capture"
    val_fraction: float = 0.15
    test_fraction: float = 0.10
    seed: int = 42
    group_near_images: bool = False
    group_near_sensors: bool = False

    def __post_init__(self) -> None:
        if (
            self.strategy not in ("aihub", "group-aware")
            or self.grouping not in ("capture", "source-day", "manufacturer")
            or not math.isfinite(self.val_fraction) or not math.isfinite(self.test_fraction)
            or self.val_fraction <= 0 or self.test_fraction < 0 or self.val_fraction + self.test_fraction >= 1
        ):
            raise ValueError("Use supported split options, positive Validation, non-negative Test, and a positive Train fraction.")


def _allocate(
    sizes: list[int], profiles: list[Counter[int]], fractions: list[float], seed: int,
) -> list[int]:
    """Try seeded greedy assignments, ranking coverage ahead of the distribution objective."""
    total = sum(sizes)
    counts: Counter[int] = sum(profiles, Counter())
    support = Counter(label for profile in profiles for label in profile)
    targets = [fraction * total for fraction in fractions]
    active = len(fractions)
    rng = random.Random(seed)
    best = None
    chosen: list[int] = []

    def distribution_cost(delta: float, target: float) -> float:
        return delta * delta / max(1.0, target)

    for _ in range(8):
        order = list(range(len(sizes)))
        rng.shuffle(order)
        order.sort(key=lambda i: (min(support[label] for label in profiles[i]), -sizes[i]))
        assigned_sizes = [0] * active
        assigned_labels: list[Counter[int]] = [Counter() for _ in fractions]
        assigned = [-1] * len(sizes)
        for position, group in enumerate(order):
            size, profile = sizes[group], profiles[group]

            def rank(split: int) -> tuple[float, float]:
                gain = sum(1 / support[label] for label in profile if not assigned_labels[split][label])
                before = assigned_sizes[split] - targets[split]
                delta = distribution_cost(before + size, targets[split]) - distribution_cost(before, targets[split])
                for label, amount in profile.items():
                    target = counts[label] * fractions[split]
                    before = assigned_labels[split][label] - target
                    delta += distribution_cost(before + amount, target) - distribution_cost(before, target)
                return -gain, delta

            empty_splits = [split for split in range(active) if assigned_sizes[split] == 0]
            eligible = empty_splits if len(order) - position <= len(empty_splits) else range(active)
            split = min(eligible, key=rank)
            assigned_sizes[split] += size
            assigned_labels[split].update(profile)
            assigned[group] = split
        missing = sum(not counter[label] for counter in assigned_labels for label in counts if label < len(NAMES))
        empty = assigned_sizes.count(0)
        imbalance = sum(
            distribution_cost(size - target, target) for size, target in zip(assigned_sizes, targets, strict=True)
        ) + sum(
            distribution_cost(counter[label] - counts[label] * fractions[split], counts[label] * fractions[split])
            for split, counter in enumerate(assigned_labels) for label in counts
        )
        score = (empty, missing, imbalance)
        if best is None or score < best:
            best, chosen = score, assigned
    return chosen


def plan_split(report: dict, config: SplitConfig) -> dict:
    if report["schema_version"] != 1 or digest_json(report["label_policy"]) != digest_json(label_policy()):
        raise ValueError("Use an analysis report with the current schema and label policy.")
    samples = report["samples"]
    candidates = [index for index, row in enumerate(samples) if row["candidate"]]
    if not candidates:
        raise ValueError("The source manifest has no detection candidates.")
    methods = [*REQUIRED_GROUPS, "source_capture"]
    if config.grouping in ("source-day", "manufacturer"):
        methods.append("source_day")
    if config.grouping == "manufacturer":
        methods.extend(("manufacturer", "source_identifier"))
    if config.group_near_images:
        methods.append("image_similarity")
    if config.group_near_sensors:
        methods.append("sensor_similarity")
    components = Components(len(samples))
    for group in report["groups"]:
        if group["method"] in methods:
            first, *others = group["members"]
            for other in others:
                components.join(first, other)
    groups = components.groups(candidates)
    profiles = [
        Counter(label for index in group for label in (samples[index]["classes"] or [len(NAMES)]))
        for group in groups
    ]
    fractions = [1 - config.val_fraction - config.test_fraction, config.val_fraction]
    if config.test_fraction:
        fractions.append(config.test_fraction)
    selected = SPLITS[:len(fractions)] if config.strategy == "group-aware" else SPLITS[:2]
    allocation = (
        _allocate([len(group) for group in groups], profiles, fractions, config.seed)
        if config.strategy == "group-aware" else []
    )
    assignment = {}
    distributions: dict[str, Counter[int]] = {split: Counter() for split in selected}
    group_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    for group_index, group in enumerate(groups):
        group_id = digest_json(group)
        seen_splits = set()
        for index in group:
            split = selected[allocation[group_index]] if allocation else samples[index]["original_split"]
            assignment[index] = {"split": split, "group_id": group_id}
            distributions[split].update(samples[index]["classes"] or [len(NAMES)])
            split_counts[split] += 1
            seen_splits.add(split)
        group_counts.update(seen_splits)
    remaining_cross_split_groups: Counter[str] = Counter()
    for group in report["groups"]:
        if len({assignment[index]["split"] for index in group["members"] if index in assignment}) > 1:
            remaining_cross_split_groups[group["method"]] += 1
    supports = Counter(label for counts in profiles for label in counts)
    missing = {split: [name for label, name in enumerate(NAMES) if not counts[label]]
               for split, counts in distributions.items()}
    return {
        "schema_version": 1, "configuration": asdict(config), "grouping_methods": methods,
        "allocation_algorithm": "seeded-group-stratification-v1" if allocation else "aihub-provided",
        "label_policy": label_policy(), "analysis_fingerprint": digest_json(report),
        "source_manifest_fingerprint": report["manifest_fingerprint"],
        "dataset_fingerprint": report["dataset_fingerprint"],
        "test_origin": "internal_same_source" if "test" in selected else None,
        "requested_fractions": dict(zip(selected, fractions)) if allocation else {},
        "counts": {split: split_counts[split] for split in selected},
        "groups_per_split": {split: group_counts[split] for split in selected},
        "candidate_components": len(groups), "largest_candidate_component": max(map(len, groups)),
        "class_samples": {split: {name: counts[label] for label, name in enumerate(LABELS)}
                          for split, counts in distributions.items()},
        "class_group_support": {name: supports[label] for label, name in enumerate(LABELS)},
        "missing_classes": missing,
        "insufficient_class_group_support": [name for label, name in enumerate(NAMES)
                                             if supports[label] < len(selected)],
        "remaining_cross_split_groups": dict(remaining_cross_split_groups),
        "assignments": [{"sample_index": index, **assignment[index]} for index in candidates],
    }


def publish_split(
    root: Path, manifest: Path, path_base: Path, report: dict, plan: dict, output: Path, *,
    export_detection: bool, allow_missing_classes: bool,
    progress: Callable[[str], Progress],
) -> None:
    root = root.resolve()
    if output.exists():
        raise ValueError("The split destination already exists. Choose a fresh directory.")
    if not all(plan["counts"].values()):
        raise ValueError("At least one requested split is empty. Review group sizes and change the split options.")
    if (plan["configuration"]["strategy"] == "group-aware"
            and any(plan["missing_classes"].values()) and not allow_missing_classes):
        raise ValueError("At least one class is missing from a split. Review the preview and group constraints.")
    rows = source_rows(manifest, root, path_base)
    if digest_json(rows) != plan["source_manifest_fingerprint"] or digest_json(report) != plan["analysis_fingerprint"]:
        raise ValueError("Manifest or leakage evidence changed. Rerun analysis and planning.")
    check = progress("Verify source bytes")
    check(0, len(rows))
    for done, (row, sample) in enumerate(zip(rows, report["samples"], strict=True), 1):
        observed = source_hashes(root, row)
        if observed != sample["hashes"]:
            raise ValueError("Source files changed since leakage analysis. Rerun analysis.")
        if is_candidate(row) and (not {"image_path", "annotation_path"} <= observed.keys()
                                  or any(issue.startswith("unreadable_image") for issue in sample["issues"])):
            raise ValueError("A detection candidate has unavailable image or annotation data.")
        check(done, len(rows))
    assignment = {item["sample_index"]: item for item in plan["assignments"]}
    revised = [
        {**row, "original_split": row["split"], "split": assignment.get(index, {}).get("split", row["split"]),
         "group_id": assignment.get(index, {}).get("group_id", "")}
        for index, row in enumerate(rows)
    ]
    keys = [(row["split"], row["sample_id"].casefold()) for row in revised if is_candidate(row)]
    if len(keys) != len(set(keys)):
        raise ValueError("Multiple candidates have the same output stem. Review those source rows before publishing.")
    revision = {**plan, "missing_classes_accepted": allow_missing_classes,
                "selected_manifest_fingerprint": digest_json(revised)}
    revision = {**revision, "revision_id": digest_json(revision)}

    with fresh_output(output, root) as working:
        csv = working / "manifest.csv"
        write_csv(csv, revised, list(revised[0]))
        write_json(working / "split.json", revision)
        if export_detection:
            dataset = working / "detection"
            include_test = "test" in plan["counts"]
            convert_detection(root, csv, dataset, include_test=include_test, portable_yaml=True,
                              progress=progress("Export split"))
            validation = validate_detection(csv, dataset, working / "validation", include_test=include_test,
                                            progress=progress("Validate labels"))
            if not validation["passed"]:
                raise ValueError("Split conversion failed YOLO/COCO validation.")
            for index in assignment:
                row = revised[index]
                split = "val" if row["split"] == "validation" else row["split"]
                image = root / row["image_path"]
                expected = report["samples"][index]["hashes"]["image_path"]
                if any(file_sha(dataset / fmt / "images" / split / image.name) != expected
                       for fmt in ("yolo", "coco")):
                    raise ValueError("An exported image differs from the analyzed source.")
                if file_sha(root / row["annotation_path"]) != report["samples"][index]["hashes"]["annotation_path"]:
                    raise ValueError("A source annotation changed during split conversion.")
            write_json(dataset / SEAL, seal_dataset(dataset, revision))
