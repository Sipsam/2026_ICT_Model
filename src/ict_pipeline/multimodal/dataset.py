"""Indexed modality views and deterministic CPU collation for future fusion."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple

from ..models.dataset import Object, Sample
from ..models.optimization.augmentation import augment
from ..models.optimization.config import Augmentation
from .audit import Cohort
from .contracts import CHANNELS, Record, Split
from .preprocessing import (
    FloatRow, MaskRow, PreprocessingState, SensorWindow, eligible, training_fingerprint,
    transform_sensor, window_available,
)

if TYPE_CHECKING:
    from PIL.Image import Image
    from torch import Tensor

Mode = Literal["image", "sensor", "paired"]
MissingModality = Literal["drop", "preserve", "error"]

PAD_VALUES: FloatRow = (0.0, 0.0, 0.0, 0.0, 0.0)
PAD_OBSERVED: MaskRow = (False, False, False, False, False)


@dataclass(frozen=True)
class DatasetSample:
    sample_id: str
    split: Split
    provider_split: str
    source_group: str
    image: Image | None
    sensor: SensorWindow | None
    original_size: tuple[int, int]
    original_targets: tuple[Object, ...]
    image_targets: tuple[Object, ...] | None
    preprocessing_fingerprint: str | None

    @property
    def modalities(self) -> tuple[bool, bool]:
        return self.image is not None, self.sensor is not None and self.sensor.available


class SensorTensors(NamedTuple):
    values: Tensor
    observed: Tensor
    timestep_valid: Tensor
    lengths: Tensor
    modalities: Tensor


@dataclass(frozen=True)
class Batch:
    samples: tuple[DatasetSample, ...]
    sensor_values: tuple[tuple[FloatRow, ...], ...]
    sensor_observed: tuple[tuple[MaskRow, ...], ...]
    timestep_valid: tuple[tuple[bool, ...], ...]
    lengths: tuple[int, ...]
    modalities: tuple[tuple[bool, bool], ...]  # image, sensor

    def sensor_tensors(self) -> SensorTensors:
        """Materialize CPU tensors when an existing torch installation is available."""
        import torch

        size, length = len(self.samples), max(self.lengths)
        shape = (size, length, len(CHANNELS))
        return SensorTensors(
            torch.tensor(self.sensor_values, dtype=torch.float64, device="cpu").reshape(shape),
            torch.tensor(self.sensor_observed, dtype=torch.bool, device="cpu").reshape(shape),
            torch.tensor(self.timestep_valid, dtype=torch.bool, device="cpu").reshape(size, length),
            torch.tensor(self.lengths, dtype=torch.int64, device="cpu"),
            torch.tensor(self.modalities, dtype=torch.bool, device="cpu"),
        )


class MultimodalDataset:
    """Read eligible records in deterministic cohort order, with lazy image decoding."""

    def __init__(
        self, cohort: Cohort, root: Path, split: Split, *, mode: Mode,
        preprocessing: PreprocessingState | None = None, missing: MissingModality = "drop",
        augmentation: Augmentation | None = None, seed: int = 42,
    ) -> None:
        if split not in ("train", "validation", "test") or mode not in ("image", "sensor", "paired"):
            raise ValueError("Use a supported split and modality view.")
        if missing not in ("drop", "preserve", "error"):
            raise ValueError("Use a supported missing-modality policy.")
        if mode != "image":
            if preprocessing is None:
                raise ValueError("Sensor views require frozen Train preprocessing statistics.")
            if preprocessing.training_fingerprint != training_fingerprint(cohort):
                raise ValueError("Sensor preprocessing Train cohort fingerprint mismatch.")
        self.root = root.resolve()
        self.mode = mode
        self.preprocessing = preprocessing if mode != "image" else None
        self.augmentation = augmentation if split == "train" else None
        self.seed = seed
        selected = []
        for record in cohort.records:
            if record.split != split or not eligible(record):
                continue
            availability = []
            if mode != "sensor":
                availability.append(record.image_available)
            if self.preprocessing is not None:
                availability.append(window_available(record, self.preprocessing.config))
            if missing == "error" and not all(availability):
                raise ValueError(f"Sample {record.sample_id} lacks a requested modality.")
            if all(availability) or (missing == "preserve" and any(availability)):
                selected.append(record)
        self.records = tuple(selected)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> DatasetSample:
        record = self.records[index]
        width, height = record.image_size
        targets = tuple(Object(target.category, target.box) for target in record.targets)
        image, image_targets = None, None
        if self.mode != "sensor" and record.image_available:
            path, image = self._image(record)
            sample = Sample(path, width, height, targets)
            if self.augmentation is not None:
                image, sample, _ = augment(image, sample, self.augmentation, self.seed)
            image_targets = sample.objects
        sensor = None
        if self.preprocessing is not None and record.sequence is not None and record.capture_timestamp is not None:
            sensor = transform_sensor(record.sequence, record.capture_timestamp, self.preprocessing)
        return DatasetSample(
            record.sample_id, record.split, record.provider_split, record.source_group,
            image, sensor, (width, height), targets, image_targets,
            self.preprocessing.fingerprint if self.preprocessing is not None else None,
        )

    def _image(self, record: Record) -> tuple[Path, Image]:
        from PIL import Image

        path = (self.root / record.image_path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("The indexed image must resolve within the extracted dataset.")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != record.image_sha256:
            raise ValueError(f"Image bytes changed for {record.sample_id}. Rebuild the cohort index.")
        with Image.open(BytesIO(raw)) as image:
            if image.size != record.image_size or image.format != "JPEG":
                raise ValueError(f"Image metadata changed for {record.sample_id}. Rebuild the cohort index.")
            return path, image.convert("RGB")


def collate(samples: list[DatasetSample] | tuple[DatasetSample, ...]) -> Batch:
    if not samples:
        raise ValueError("Collation requires at least one sample.")
    if len({item.split for item in samples}) != 1:
        raise ValueError("A multimodal batch must contain only one split.")
    if len({item.preprocessing_fingerprint for item in samples}) != 1 or any(
        item.sensor is not None and item.sensor.preprocessing_fingerprint != item.preprocessing_fingerprint
        for item in samples
    ):
        raise ValueError("A multimodal batch must use the same frozen sensor preprocessing state.")
    windows = tuple(item.sensor for item in samples)
    lengths = tuple(len(window.values) if window is not None else 0 for window in windows)
    longest = max(lengths)
    return Batch(
        samples=tuple(samples),
        sensor_values=tuple((window.values if window is not None else ()) + (PAD_VALUES,) * (longest - length)
                            for window, length in zip(windows, lengths, strict=True)),
        sensor_observed=tuple((window.observed if window is not None else ()) + (PAD_OBSERVED,) * (longest - length)
                              for window, length in zip(windows, lengths, strict=True)),
        timestep_valid=tuple((True,) * length + (False,) * (longest - length) for length in lengths),
        lengths=lengths, modalities=tuple(item.modalities for item in samples),
    )
