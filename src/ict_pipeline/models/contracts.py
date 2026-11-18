"""Public contracts for local detector training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


if TYPE_CHECKING:
    from .dataset import DetectionData
    from .weights import VerifiedWeights


class TrainingError(Exception):
    """An actionable error safe to display or store."""


class ModelName(StrEnum):
    YOLO = "yolo"
    FASTER_RCNN = "faster_rcnn"
    RT_DETR = "rt_detr"


DataFormat = Literal["yolo", "coco"]
Progress = Callable[[str, int, int], None]


class CheckpointConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    source: str = Field(min_length=1)
    origin: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9:/._~+@-]*$")
    revision: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    license: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9 .+()/;-]*$")
    architecture: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    library_version: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
    format: Literal["state_dict", "safetensors", "ultralytics_pickle"]
    trusted_pickle: bool = False
    source_classes: int | None = Field(default=None, gt=0, le=10000)

    @model_validator(mode="after")
    def check_trust(self) -> Self:
        if self.trusted_pickle != (self.format == "ultralytics_pickle"):
            raise ValueError("Native Ultralytics pickle requires explicit trusted_pickle: true.")
        return self


class FineTuningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    optimizer: Literal["SGD", "Adam", "AdamW"] | None = None
    lr: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    weight_decay: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    momentum: float | None = Field(default=None, ge=0, lt=1, allow_inf_nan=False)


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    epochs: int = Field(default=50, gt=0)
    batch_size: int = Field(default=4, gt=0)
    image_size: int = Field(default=640, ge=64, multiple_of=32)
    optimizer: Literal["SGD", "Adam", "AdamW"] = "AdamW"
    lr: float = Field(default=0.001, gt=0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    device: Literal["auto", "cpu", "cuda", "rocm", "mps"] = "cpu"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    save_period: int = Field(default=0, ge=0)
    initialization: Literal["scratch", "pretrained"] = "scratch"
    checkpoint: CheckpointConfig | None = None
    freeze_backbone: bool = False
    fine_tuning: FineTuningConfig | None = None

    @model_validator(mode="after")
    def check_initialization(self) -> Self:
        if self.initialization == "pretrained" and self.checkpoint is None:
            raise ValueError("Pretrained initialization requires a checkpoint.")
        if self.initialization == "scratch" and (self.checkpoint or self.fine_tuning or self.freeze_backbone):
            raise ValueError("Checkpoint, fine-tuning and freezing settings require pretrained initialization.")
        return self

    @property
    def optimizer_name(self) -> str:
        return (self.fine_tuning.optimizer if self.fine_tuning else None) or self.optimizer

    @property
    def learning_rate(self) -> float:
        return (self.fine_tuning.lr if self.fine_tuning else None) or self.lr


@dataclass(frozen=True)
class Checkpoints:
    best: Path
    last: Path


@dataclass(frozen=True)
class TrainingRequest:
    config: TrainConfig
    data: DetectionData
    output: Path
    device: str
    progress: Progress
    resume: Checkpoints | None = None
    record_metrics: Callable[[Mapping[str, float]], None] = lambda _: None
    pretrained: VerifiedWeights | None = None
    record_transfer: Callable[[Mapping[str, object]], None] = lambda _: None


class ModelAdapter(Protocol):
    architecture: str

    def train(self, request: TrainingRequest) -> Checkpoints:
        """Initialize a detector or continue an earlier run using its saved state."""
        ...
