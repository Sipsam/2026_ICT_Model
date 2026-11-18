"""Detector-independent predictions and evaluation settings."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from ..data.preprocessing.policy import NAMES
from ..models.contracts import ModelName
from ..models.dataset import Box, Sample

CLASSES = dict(enumerate(NAMES, 1))
MAX_DETECTIONS = 100  # COCO bbox AP, per image and category


class EvaluationError(Exception):
    """An actionable evaluation error safe to display."""


class EvaluationSplit(StrEnum):
    VALIDATION = "val"
    TEST = "test"


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", hide_input_in_errors=True)

    image_size: int = Field(default=640, ge=64, multiple_of=32)
    score_floor: float = Field(default=0.001, ge=0, lt=1)
    confidence: float = Field(default=0.25, ge=0, le=1)
    match_iou: float = Field(default=0.5, gt=0, le=1)
    nms_iou: float = Field(default=0.7, gt=0, le=1)
    warmup: int = Field(default=3, ge=0)
    device: Literal["cpu", "cuda", "rocm", "mps"] = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"


@dataclass(frozen=True)
class Preprocessing:
    original_size: tuple[int, int]  # width, height
    input_size: tuple[int, int]     # transformed tensor width, height
    resize: Literal["letterbox", "aspect_ratio", "stretch"]


@dataclass(frozen=True)
class Prediction:
    sample_id: str
    boxes: tuple[Box, ...]           # absolute original-image xyxy
    class_ids: tuple[int, ...]       # canonical one-based COCO IDs
    confidence_scores: tuple[float, ...]
    preprocessing: Preprocessing


class Predictor(Protocol):
    device: str
    hardware: dict[str, object]
    postprocessing: dict[str, object]
    libraries: dict[str, str]

    def predict(self, sample: Sample) -> Prediction: ...

    def synchronize(self) -> None: ...


def xywh_to_xyxy(box: Box) -> Box:
    x, y, width, height = box
    return x, y, x + width, y + height


def relative_cxcywh_to_xyxy(box: Box, width: int, height: int) -> Box:
    cx, cy, w, h = box
    return (cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height


def normalize_prediction(
    model: ModelName, sample: Sample, boxes: Iterable[Box], class_ids: Iterable[int],
    scores: Iterable[float], preprocessing: Preprocessing,
) -> Prediction:
    retained: list[tuple[Box, int, float]] = []
    for box, raw_id, score in zip(boxes, class_ids, scores, strict=True):
        if type(raw_id) is not int:
            raise EvaluationError("The detector returned a non-integer class ID.")
        category = raw_id if model == ModelName.FASTER_RCNN else raw_id + 1
        if category not in CLASSES:
            raise EvaluationError("The detector returned a class outside the configured mapping.")
        if not all(math.isfinite(value) for value in box) or not 0 <= score <= 1:
            raise EvaluationError("The detector returned invalid boxes or confidence scores.")
        x1, y1, x2, y2 = box
        if x2 < x1 or y2 < y1:
            raise EvaluationError("The detector returned reversed box coordinates.")
        x1, x2 = (min(max(x, 0), sample.width) for x in (x1, x2))
        y1, y2 = (min(max(y, 0), sample.height) for y in (y1, y2))
        if x2 > x1 and y2 > y1:
            retained.append(((x1, y1, x2, y2), category, score))
    return Prediction(
        sample.image.stem, tuple(row[0] for row in retained), tuple(row[1] for row in retained),
        tuple(row[2] for row in retained), preprocessing,
    )
