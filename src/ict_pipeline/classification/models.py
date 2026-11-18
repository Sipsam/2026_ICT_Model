"""Deterministic CPU model adapters for image-level binary classification."""

import math
from collections.abc import Callable, Sequence
from random import Random
from statistics import fmean, pstdev
from typing import Literal

from pydantic import BaseModel

from ..data.preprocessing.policy import SENSOR_FIELDS
from ..pipeline.model import StageFailure
from .contracts import IMAGE_SIZE, Algorithm, ClassificationConfig, Sample

Vector = tuple[float, ...]
Example = tuple[Vector, int]
Unimodal = Literal["sensor", "image"]
SENSOR_FEATURES = tuple(f"{channel}_{stat}" for channel in SENSOR_FIELDS for stat in ("mean", "std", "rate"))
IMAGE_FEATURES = tuple(f"rgb_{index}" for index in range(3 * math.prod(IMAGE_SIZE)))


def features(sample: Sample, modality: Unimodal, config: ClassificationConfig) -> Vector | None:
    if modality == "image":
        return tuple(value / 255.0 for value in sample.pixels) if sample.pixels is not None else None
    if not sample.sequence:
        return None
    start = sample.capture_time or sample.sequence[0].timestamp
    window = [
        step for step in sample.sequence
        if 0 <= (step.timestamp - start).total_seconds() <= config.sensor_window_seconds
    ]
    if not window:
        return None
    elapsed = (window[-1].timestamp - window[0].timestamp).total_seconds()
    result = tuple(
        summary
        for values in zip(*(step.values for step in window))
        for summary in (fmean(values), pstdev(values), (values[-1] - values[0]) / elapsed if elapsed else 0.0)
    )
    if not all(map(math.isfinite, result)):
        raise StageFailure("Sensor feature aggregation exceeded the numeric range. Check measurement values.")
    return result


def _probability(logit: float) -> float:
    return 1 / (1 + math.exp(-max(-700.0, min(logit, 700.0))))


class LinearModel(BaseModel):
    algorithm: Algorithm
    feature_names: tuple[str, ...]
    means: Vector
    scales: Vector
    weights: Vector
    bias: float

    def predict(self, values: Vector) -> float:
        return _probability(self.bias + sum(
            weight * (value - mean) / scale
            for value, mean, scale, weight in zip(values, self.means, self.scales, self.weights, strict=True)
        ))


def _logistic(
    examples: list[Example], width: int, config: ClassificationConfig, progress: Callable[[int], None],
) -> tuple[Vector, float]:
    weights = [0.0] * width
    bias = 0.0
    rng = Random(config.seed)
    ordering = list(range(len(examples)))
    for epoch in range(config.epochs):
        rng.shuffle(ordering)
        for index in ordering:
            vector, label = examples[index]
            logit = bias + sum(weight * value for weight, value in zip(weights, vector))
            step = config.learning_rate * (_probability(logit) - label)
            for column, value in enumerate(vector):
                weights[column] -= step * value
            bias -= step
        progress(epoch + 1)
    return tuple(weights), bias


def _centroid(
    examples: list[Example], progress: Callable[[int], None],
) -> tuple[Vector, float]:
    normal = tuple(map(fmean, zip(*(vector for vector, label in examples if label == 0))))
    defect = tuple(map(fmean, zip(*(vector for vector, label in examples if label == 1))))
    # The difference of squared distances to the two standardized centroids.
    weights = tuple(2 * (d - n) for n, d in zip(normal, defect))
    bias = sum(n * n - d * d for n, d in zip(normal, defect))
    progress(1)
    return weights, bias


def fit_model(
    modality: Unimodal, examples: Sequence[Example], config: ClassificationConfig, progress: Callable[[int], None],
) -> LinearModel:
    if {label for _, label in examples} != {0, 1}:
        raise StageFailure(f"The {modality} training cohort needs both image-level labels. Inspect alignment and coverage.")
    feature_names = SENSOR_FEATURES if modality == "sensor" else IMAGE_FEATURES
    algorithm = config.sensor_model if modality == "sensor" else "logistic"
    columns = tuple(zip(*(vector for vector, _ in examples), strict=True))
    means = tuple(map(fmean, columns))
    scales = tuple(pstdev(column) or 1.0 for column in columns)
    scaled = [
        (tuple((value - mean) / scale for value, mean, scale in zip(vector, means, scales)), label)
        for vector, label in examples
    ]
    if not all(math.isfinite(value) for vector, _ in scaled for value in vector):
        raise StageFailure(f"The {modality} training features exceeded the numeric range after standardization.")
    if algorithm == "logistic":
        weights, bias = _logistic(scaled, len(feature_names), config, progress)
    else:
        weights, bias = _centroid(scaled, progress)
    return LinearModel(
        algorithm=algorithm, feature_names=feature_names, means=means, scales=scales, weights=weights, bias=bias,
    )
