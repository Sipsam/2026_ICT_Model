"""Deterministic observed-only statistical features with explicit validity."""

import math
from statistics import fmean, pstdev

from ..classification.sensors import parse_timestamp
from ..inspection.contracts import Contract, SchemaVersion
from .contracts import CHANNELS, SensorSequence
from .preprocessing import SensorWindow

STATISTICS = ("mean", "std", "min", "max", "delta", "rate")
FEATURE_NAMES = tuple(f"{channel}_{name}" for channel in CHANNELS for name in STATISTICS)


class SensorFeatures(Contract):
    schema_version: SchemaVersion = 1
    values: tuple[float | None, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return FEATURE_NAMES

    @property
    def observed(self) -> tuple[bool, ...]:
        return tuple(value is not None for value in self.values)

    def model_post_init(self, context: object) -> None:
        if len(self.values) != len(FEATURE_NAMES):
            raise ValueError("Sensor feature vector has an incorrect dimension.")


def statistical_features(sequence: SensorSequence | SensorWindow) -> SensorFeatures:
    """Summarize raw values in source units or summarize an explicit transformed window."""
    if isinstance(sequence, SensorSequence):
        instants = [step.instant for step in sequence.steps]
        rows = [step.values for step in sequence.steps]
    else:
        instants = [parse_timestamp(value) for value in sequence.timestamps]
        rows = [tuple(value if valid else None for value, valid in zip(row, mask, strict=True))
                for row, mask in zip(sequence.values, sequence.observed, strict=True)]
    features: list[float | None] = []
    for index in range(len(CHANNELS)):
        measured = [(instant, row[index]) for instant, row in zip(instants, rows, strict=True)
                    if row[index] is not None]
        if not measured:
            features.extend([None] * len(STATISTICS))
            continue
        values = [value for _, value in measured]
        enough = len(values) > 1
        elapsed = (measured[-1][0] - measured[0][0]).total_seconds() if enough else 0.0
        delta = values[-1] - values[0] if enough else None
        features.extend((fmean(values), pstdev(values) if enough else None, min(values), max(values),
                         delta if elapsed > 0 else None, delta / elapsed if elapsed > 0 else None))
    if not all(value is None or math.isfinite(value) for value in features):
        raise ValueError("Sensor features exceed the numerical range.")
    return SensorFeatures(values=tuple(features))
