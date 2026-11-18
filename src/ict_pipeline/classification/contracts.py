"""Versioned settings and artifacts for image-level classification."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1
LABEL_POLICY = "smt-target-defect-presence-v1"
IMAGE_SIZE = (8, 8)

Mode = Literal["sensor", "unimodal", "fusion"]
Modality = Literal["sensor", "image", "fusion"]
Algorithm = Literal["logistic", "centroid"]
Split = Literal["train", "validation"]
Measurements = tuple[float, float, float, float, float]


class ClassificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)

    mode: Mode = "sensor"
    sensor_model: Algorithm = "logistic"
    missing: Literal["available", "paired"] = "available"
    group_policy: Literal["session", "source_day", "source"] = "session"
    sensor_window_seconds: int = Field(default=0, ge=0, le=86400)
    sensor_value_atol: float = Field(default=0.001, ge=0, allow_inf_nan=False)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    epochs: int = Field(default=20, ge=1, le=1000)
    learning_rate: float = Field(default=0.05, gt=0, le=1)

    @property
    def modalities(self) -> tuple[Modality, ...]:
        if self.mode == "fusion":
            return ("sensor", "image", "fusion")
        if self.mode == "unimodal":
            return ("sensor", "image")
        return ("sensor",)


class Reading(BaseModel):
    timestamp: datetime
    values: Measurements


class Sample(BaseModel):
    sample_id: str
    split: Split
    label: Literal[0, 1]
    annotation_path: str
    image_path: str | None
    sensor_path: str | None
    sensor_source: Literal["json", "xlsx", "both"] | None
    sensor_image_id_offset: int | None
    capture_time: datetime | None
    sequence: tuple[Reading, ...]
    pixels: tuple[int, ...] | None
