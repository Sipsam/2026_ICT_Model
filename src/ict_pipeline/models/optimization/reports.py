"""Summarize a converted cohort and render annotated Train previews."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...data.preprocessing.common import read_csv
from ...data.preprocessing.policy import NAMES, label_policy
from ...pipeline.store import digest_json, fingerprint_path
from ..contracts import TrainingError
from ..dataset import DetectionData, Sample, Split
from .augmentation import augment, read_rgb
from .config import Augmentation
from .sampling import frequencies

if TYPE_CHECKING:
    from PIL.Image import Image


def imbalance_report(
    dataset: Path, data: DetectionData, seal: dict[str, Any] | None,
    manifest: Path | None, test: Split | None,
) -> dict[str, object]:
    records: dict[tuple[str, str], tuple[str, str]] = {}
    verified = False
    manifest_fingerprint = None
    if manifest is not None:
        rows = read_csv(manifest)
        if seal is not None:
            if digest_json(rows) != seal["split"]["selected_manifest_fingerprint"]:
                raise TrainingError("The manifest differs from the selected split's published manifest.")
            verified = True
        manifest_fingerprint = digest_json(rows)
        candidate = [row for row in rows if row.get("decision") == row.get("use_for_detection") == "candidate"]
        split_names = {"train": "train", "validation": "val", "test": "test"}
        group_splits: dict[str, set[str]] = defaultdict(set)
        for row in candidate:
            if row["split"] not in split_names:
                raise TrainingError("A candidate has an unsupported split.")
            split = split_names[row["split"]]
            key = split, row["sample_id"]
            if key in records:
                raise TrainingError("The candidate manifest repeats a sample within a split.")
            manufacturer, group = row.get("manufacturer", ""), row.get("group_id", "")
            records[key] = manufacturer or "(unknown)", group or "(unknown)"
            if group:
                group_splits[group].add(split)
        if verified and seal["split"].get("configuration", {}).get("strategy") == "group-aware" and (
            any(group == "(unknown)" for _, group in records.values())
            or any(len(splits) > 1 for splits in group_splits.values())
        ):
            raise TrainingError("The selected group-aware manifest has missing or cross-split group IDs.")

    def summarize(split: str, samples: tuple[Sample, ...]) -> dict[str, object]:
        manufacturers: dict[str, list[Sample]] = defaultdict(list)
        groups: dict[str, list[Sample]] = defaultdict(list)
        expected = {(split, sample.image.stem) for sample in samples}
        if manifest is not None and expected != {key for key in records if key[0] == split}:
            raise TrainingError("The candidate manifest does not match the converted split membership.")
        for sample in samples:
            manufacturer, group = records.get((split, sample.image.stem), ("(unknown)", "(unknown)"))
            manufacturers[manufacturer].append(sample)
            groups[group].append(sample)
        return {
            **frequencies(samples),
            "by_manufacturer": {key: frequencies(tuple(items)) for key, items in sorted(manufacturers.items())},
            "by_group": {key: frequencies(tuple(items)) for key, items in sorted(groups.items())},
        }

    splits = data.splits | ({"test": test} if test is not None else {})
    return {
        "schema_version": 1, "label_policy": label_policy(),
        "split_revision": digest_json(seal) if seal else None,
        "manifest_fingerprint": manifest_fingerprint,
        "manifest_verified_by_seal": verified,
        "train_validation_image_fingerprints": data.split_fingerprints,
        "annotation_fingerprints": data.split_annotation_fingerprints | (
            {"test": digest_json({path.name: fingerprint_path(path) for path in test.annotations})}
            if test is not None else {}
        ),
        "test_available": (dataset / "coco" / "annotations" / "instances_test.json").is_file(),
        "splits": {name: summarize(name, split.samples) for name, split in splits.items()},
    }


def _draw_panel(image: Image, sample: Sample, heading: str) -> Image:
    from PIL import Image, ImageDraw

    header = 22
    panel = Image.new("RGB", (max(image.width, 160), image.height + header), (24, 24, 24))
    panel.paste(image, (0, header))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 4), heading, fill="white")
    for index, obj in enumerate(sample.objects):
        x, y, w, h = obj.box
        color = "#7ef0ff"
        draw.rectangle((x, y + header, min(image.width - 1, x + w), min(panel.height - 1, y + h + header)),
                       outline=color, width=1)
        draw.text((x, max(header, y + header - 11)), f"{index} {NAMES[obj.category - 1]}", fill=color,
                  stroke_width=1, stroke_fill="black")
    return panel


def preview_sample(
    sample: Sample, policy: Augmentation, seed: int, output: Path,
) -> dict[str, object]:
    from PIL import Image

    original = read_rgb(sample)
    image, transformed, changes = augment(original, sample, policy, seed)
    left = _draw_panel(original, sample, "Original")
    right = _draw_panel(image, transformed, "Geometry rejected" if changes.geometry_rejected else "Transformed")
    joined = Image.new("RGB", (left.width + right.width + 4, max(left.height, right.height)), (24, 24, 24))
    joined.paste(left, (0, 0))
    joined.paste(right, (left.width + 4, 0))
    with output.open("xb") as stream:
        joined.save(stream, format="PNG")
    return {
        "sample_id": sample.image.stem,
        "source": sample.image.name,
        "preview": output.name,
        "proposed_annotation_changes": asdict(changes),
        "dropped_annotations": [],
        "invalid_annotations": [],
        "applied_clipping": changes.clipped if not changes.geometry_rejected else [],
        "original_boxes": [asdict(obj) for obj in sample.objects],
        "transformed_boxes": [asdict(obj) for obj in transformed.objects],
    }
