"""Public contracts for local detector training."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from .dataset import DetectionData


class TrainingError(Exception):
    """An actionable error safe to display or store."""


class ModelName(StrEnum):
    YOLO = "yolo"
    FASTER_RCNN = "faster_rcnn"
    RT_DETR = "rt_detr"


DataFormat = Literal["yolo", "coco"]
Progress = Callable[[str, int, int], None]


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


class ModelAdapter(Protocol):
    architecture: str

    def train(self, request: TrainingRequest) -> Checkpoints:
        """Train from scratch or continue an earlier run using its saved state."""
        ...
