"""Read the published detection formats and fingerprint their split membership."""

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..config.settings import Credentials
from ..data.preprocessing.common import safe_component
from ..data.preprocessing.policy import NAMES
from ..pipeline.store import digest_json, fingerprint_path
from .contracts import DataFormat, Progress, TrainingError

Box = tuple[float, float, float, float]
SPLITS = ("train", "val")


@dataclass(frozen=True)
class Object:
    category: int  # COCO one-based ID
    box: Box       # absolute xywh


@dataclass(frozen=True)
class Sample:
    image: Path
    width: int
    height: int
    objects: tuple[Object, ...]


@dataclass(frozen=True)
class Split:
    images: tuple[Path, ...]
    annotations: tuple[Path, ...]
    samples: tuple[Sample, ...] = ()


@dataclass(frozen=True)
class DetectionData:
    splits: dict[str, Split]
    split_fingerprints: dict[str, str]
    annotation_fingerprint: str
    split_annotation_fingerprints: dict[str, str] = field(default_factory=dict)


def _read_yolo(root: Path) -> dict[str, Split]:
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8-sig"))
    if (
        not isinstance(data, dict)
        or data.get("names") != dict(enumerate(NAMES))
        or any(data.get(split) != f"images/{split}" for split in SPLITS)
        or set(data) - {"path", "train", "val", "names", "nc"}
        or data.get("nc", len(NAMES)) != len(NAMES)
    ):
        raise TrainingError("YOLO data.yaml does not match the generated split and class mapping. Rerun conversion.")
    splits = {}
    for split in SPLITS:
        images = tuple(sorted((root / "images" / split).iterdir()))
        labels = tuple(root / "labels" / split / f"{image.stem}.txt" for image in images)
        if len(labels) != len(set(labels)):
            raise TrainingError("YOLO image filenames have ambiguous label stems.")
        has_objects = False
        for label in labels:
            for line in label.read_text(encoding="utf-8").splitlines():
                category, *box = line.split()
                cx, cy, width, height = map(float, box)
                if (
                    int(category) not in range(len(NAMES))
                    or not all(0 <= value <= 1 for value in (cx, cy, width, height))
                    or width <= 0 or height <= 0
                ):
                    raise ValueError
                has_objects = True
        if split == "train" and not has_objects:
            raise TrainingError("YOLO training requires at least one labeled object in Train.")
        splits[split] = Split(images, labels)
    return splits


def read_coco_split(root: Path, split: str) -> Split:
    path = root / "annotations" / f"instances_{split}.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    categories = data["categories"]
    if len(categories) != len(NAMES) or {row["id"]: row["name"] for row in categories} != dict(enumerate(NAMES, 1)):
        raise TrainingError("COCO categories do not match the canonical five-class mapping. Rerun conversion.")
    images = {row["id"]: row for row in data["images"]}
    if len(images) != len(data["images"]):
        raise ValueError

    objects: dict[int, list[Object]] = defaultdict(list)
    for index, row in enumerate(data["annotations"]):
        try:
            category = row["category_id"]
            image = images[row["image_id"]]
            box = row["bbox"]
            if not isinstance(box, list) or any(type(value) not in (int, float) for value in box):
                raise ValueError
            x, y, w, h = map(float, box)
            if (
                type(category) is not int or category not in range(1, len(NAMES) + 1) or row.get("iscrowd", 0) != 0
                or not all(math.isfinite(value) for value in (x, y, w, h))
                or not (0 <= x < min(image["width"], x + w) and 0 <= y < min(image["height"], y + h)
                        and x + w <= image["width"] + 1e-6 and y + h <= image["height"] + 1e-6)
            ):
                raise ValueError
        except (KeyError, ValueError, TypeError, OverflowError):
            raise TrainingError(f"Invalid COCO annotation at index {index} in {split}. Rerun conversion validation.") from None
        objects[row["image_id"]].append(Object(category, (x, y, w, h)))

    samples = []
    for image_id, row in images.items():
        name, width, height = row["file_name"], row["width"], row["height"]
        if (
            type(image_id) is not int or not isinstance(name, str) or not safe_component(name)
            or type(width) is not int or type(height) is not int or min(width, height) <= 0
        ):
            raise ValueError
        samples.append(Sample(root / "images" / split / name, width, height, tuple(objects[image_id])))
    if len({sample.image for sample in samples}) != len(samples):
        raise ValueError
    return Split(tuple(sample.image for sample in samples), (path,), tuple(samples))


def inspect_data(
    root: Path, data_format: DataFormat, credentials: Credentials, progress: Progress | None = None,
) -> DetectionData:
    try:
        directory = root / data_format
        splits = _read_yolo(directory) if data_format == "yolo" else {
            split: read_coco_split(directory, split) for split in SPLITS
        }
        if data_format == "coco" and splits["train"].samples and not any(
            sample.objects for sample in splits["train"].samples
        ):
            raise TrainingError("COCO training requires at least one labeled object in Train.")
        stage = f"fingerprint {data_format} images"
        total = sum(len(split.images) for split in splits.values())
        done = 0
        if progress:
            progress(stage, done, total)
        membership: dict[str, dict[str, str]] = {}
        annotations = {}
        for name, split in splits.items():
            if not split.images:
                raise TrainingError("Train and Validation must both contain images. Check the conversion output.")
            membership[name] = {}
            for path in split.images:
                credentials.protect_path(str(path), "dataset image path")
                if (
                    path.suffix.lower() not in (".jpg", ".jpeg", ".png") or not path.is_file()
                    or not path.resolve().is_relative_to(directory.resolve())
                ):
                    raise ValueError
                with path.open("rb") as stream:
                    membership[name][path.name] = hashlib.file_digest(stream, "sha256").hexdigest()
                done += 1
                if progress:
                    progress(stage, done, total)
            for path in split.annotations:
                if not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError
            annotations[name] = {path.name: fingerprint_path(path) for path in split.annotations}
        if (
            set(membership["train"]) & set(membership["val"])
            or set(membership["train"].values()) & set(membership["val"].values())
        ):
            raise TrainingError("Train and Validation contain duplicate filenames or image bytes. Review the dataset audit.")
        return DetectionData(
            splits, {name: digest_json(hashes) for name, hashes in membership.items()}, digest_json(annotations),
            {name: digest_json(hashes) for name, hashes in annotations.items()},
        )
    except (OSError, ValueError, TypeError, KeyError, yaml.YAMLError, RecursionError):
        raise TrainingError(f"Invalid generated {data_format.upper()} dataset. Check the files and rerun detection validation.") from None


def verify_shared_labels(yolo: DetectionData, coco: DetectionData, *, tolerance: float = 1e-6) -> None:
    """Check converted labels in normalized coordinates, including empty backgrounds."""
    if yolo.split_fingerprints != coco.split_fingerprints:
        raise TrainingError("YOLO and COCO Train/Validation images differ. Rerun conversion and detection validation.")
    for name, split in coco.splits.items():
        label_paths = dict(zip((image.name for image in yolo.splits[name].images), yolo.splits[name].annotations, strict=True))
        for sample in split.samples:
            remaining = []
            for obj in sample.objects:
                x, y, w, h = obj.box
                remaining.append((obj.category - 1, (x + w / 2) / sample.width, (y + h / 2) / sample.height,
                                  w / sample.width, h / sample.height))
            for line in label_paths[sample.image.name].read_text(encoding="utf-8").splitlines():
                numbers = tuple(map(float, line.split()))
                match = next((i for i, box in enumerate(remaining) if numbers[0] == box[0] and all(
                    abs(a - b) <= tolerance for a, b in zip(numbers[1:], box[1:], strict=True)
                )), None)
                if match is None:
                    raise TrainingError("YOLO and COCO annotations differ. Rerun conversion and detection validation.")
                remaining.pop(match)
            if remaining:
                raise TrainingError("YOLO and COCO annotations differ. Rerun conversion and detection validation.")
