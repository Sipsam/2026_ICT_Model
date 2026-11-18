"""Fit pooled sensor statistics on eligible Train windows and freeze the result."""

import json
import math
from datetime import timedelta
from pathlib import Path
from statistics import fmean, median, pstdev, quantiles
from typing import Annotated, Any, Literal, Self

from pydantic import Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from ..classification.sensors import parse_timestamp
from ..inspection.contracts import Contract, SchemaVersion
from ..pipeline.store import Fingerprint, digest_json
from .audit import Cohort
from .contracts import CHANNELS, UNITS, Channels, Record, SensorSequence, SensorStep

WindowValue = Annotated[float, Field(ge=-1e150, le=1e150)]
FloatRow = tuple[WindowValue, WindowValue, WindowValue, WindowValue, WindowValue]
MaskRow = tuple[bool, bool, bool, bool, bool]
WINDOW_POLICY = "capture-through-decision-v1"
CAUSAL_WINDOW_POLICY = "at-or-before-capture-v1"


class PreprocessingConfig(Contract):
    schema_version: SchemaVersion = 1
    missing: Literal["mask", "train_median", "reject"] = "mask"
    scaling: Literal["none", "standard", "robust"] = "none"
    normalization: Literal["none", "l2"] = "none"
    window_seconds: int = Field(default=0, ge=0, le=86400)
    window_policy: Literal["capture-through-decision-v1", "at-or-before-capture-v1"] = WINDOW_POLICY

    @model_serializer(mode="wrap")
    def serialize_config(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.window_policy == WINDOW_POLICY:
            data.pop("window_policy", None)
        return data

    @model_validator(mode="after")
    def check_units(self) -> Self:
        if self.normalization == "l2" and self.scaling == "none":
            raise ValueError("Apply channel scaling before normalizing across source units.")
        return self

    @property
    def fingerprint(self) -> str:
        return digest_json([self.window_policy, CHANNELS, UNITS, self.model_dump(mode="json")])


class ChannelFit(Contract):
    count: int = Field(ge=0)
    center: float
    scale: Annotated[float, Field(gt=0)]
    fill: float | None

    @model_validator(mode="after")
    def check_missing(self) -> Self:
        if (self.count == 0) != (self.fill is None) or (
            self.count == 0 and (self.center != 0 or self.scale != 1)
        ):
            raise ValueError("Unobserved Train channels require identity scaling and no imputation.")
        return self


class PreprocessingState(Contract):
    schema_version: SchemaVersion = 1
    channels: Channels = CHANNELS
    units: Channels = UNITS
    config: PreprocessingConfig
    config_fingerprint: Fingerprint
    training_fingerprint: Fingerprint
    statistics: tuple[ChannelFit, ChannelFit, ChannelFit, ChannelFit, ChannelFit]

    @model_validator(mode="after")
    def compatible(self) -> Self:
        if self.channels != CHANNELS or self.units != UNITS or self.config.fingerprint != self.config_fingerprint:
            raise ValueError("Sensor preprocessing schema or configuration fingerprint mismatch.")
        return self

    @property
    def fingerprint(self) -> str:
        return digest_json(self.model_dump(mode="json"))

    def payload(self) -> dict[str, object]:
        return {"fingerprint": self.fingerprint, "state": self.model_dump(mode="json")}

    @classmethod
    def load(
        cls, path: Path, *, expected_config: PreprocessingConfig | None = None,
        expected_training_fingerprint: str | None = None,
    ) -> Self:
        saved = json.loads(path.read_text(encoding="utf-8"))
        if digest_json(saved["state"]) != saved["fingerprint"]:
            raise ValueError("Sensor preprocessing state fingerprint mismatch.")
        state = cls.model_validate_json(json.dumps(saved["state"]))
        if state.fingerprint != saved["fingerprint"]:
            raise ValueError("Sensor preprocessing state is not canonical.")
        if expected_config is not None and expected_config.fingerprint != state.config_fingerprint:
            raise ValueError("Sensor preprocessing configuration fingerprint mismatch.")
        if expected_training_fingerprint is not None and state.training_fingerprint != expected_training_fingerprint:
            raise ValueError("Sensor preprocessing Train cohort fingerprint mismatch.")
        return state


class SensorWindow(Contract):
    schema_version: SchemaVersion = 1
    preprocessing_fingerprint: Fingerprint
    timestamps: tuple[str, ...]
    values: tuple[FloatRow, ...]
    observed: tuple[MaskRow, ...]

    @model_validator(mode="after")
    def check_shape(self) -> Self:
        if len(self.timestamps) != len(self.values) or len(self.values) != len(self.observed):
            raise ValueError("Sensor windows require matching timestamp, value, and mask lengths.")
        instants = [parse_timestamp(timestamp) for timestamp in self.timestamps]
        if len({instant.tzinfo is None for instant in instants}) > 1 or any(
            left >= right for left, right in zip(instants, instants[1:])
        ):
            raise ValueError("Sensor window timestamps require a comparable clock and strictly increasing order.")
        return self

    @property
    def available(self) -> bool:
        return any(any(row) for row in self.observed)


def _window(sequence: SensorSequence, capture_timestamp: str, config: PreprocessingConfig) -> tuple[SensorStep, ...]:
    capture = parse_timestamp(capture_timestamp)
    if (capture.tzinfo is None) != (sequence.steps[0].instant.tzinfo is None):
        raise ValueError("Sensor timestamps and capture require comparable clocks.")
    horizon = timedelta(seconds=config.window_seconds)
    if config.window_policy == CAUSAL_WINDOW_POLICY:
        return tuple(step for step in sequence.steps if timedelta(0) <= capture - step.instant <= horizon)
    return tuple(step for step in sequence.steps if timedelta(0) <= step.instant - capture <= horizon)


def eligible(record: Record) -> bool:
    return record.detection_candidate and not record.quarantined and record.targets is not None


def window_available(record: Record, config: PreprocessingConfig) -> bool:
    return record.sequence is not None and record.capture_timestamp is not None and any(
        value is not None for step in _window(record.sequence, record.capture_timestamp, config)
        for value in step.values
    )


def _train_records(cohort: Cohort) -> tuple[Record, ...]:
    return tuple(record for record in cohort.records if record.split == "train" and eligible(record)
                 and record.sensor_available and record.capture_timestamp is not None)


def training_fingerprint(cohort: Cohort) -> str:
    return digest_json([
        [record.sample_id, record.provider_split, record.annotation_path, record.capture_timestamp,
         record.sequence.model_dump(mode="json")]
        for record in _train_records(cohort)
    ])


def _fit_channel(values: list[float], scaling: str) -> ChannelFit:
    if not values:
        return ChannelFit(count=0, center=0.0, scale=1.0, fill=None)
    fill = median(values)
    center, scale = 0.0, 1.0
    if scaling == "standard":
        center, scale = fmean(values), pstdev(values)
    elif scaling == "robust":
        center = fill
        if len(values) > 1:
            lower, _, upper = quantiles(values, n=4, method="inclusive")
            scale = upper - lower
    return ChannelFit(count=len(values), center=center, scale=scale or 1.0, fill=fill)


def fit_preprocessor(cohort: Cohort, config: PreprocessingConfig) -> PreprocessingState:
    windows = [
        step.values for record in _train_records(cohort)
        for step in _window(record.sequence, record.capture_timestamp, config)
    ]
    statistics = tuple(_fit_channel([row[index] for row in windows if row[index] is not None], config.scaling)
                       for index in range(len(CHANNELS)))
    if not any(channel.count for channel in statistics):
        raise ValueError("No observed sensor values in eligible Train windows.")
    if config.missing == "reject" and any(channel.count != len(windows) for channel in statistics):
        raise ValueError("Train sensor window contains missing measurements under the reject policy.")
    return PreprocessingState(
        config=config, config_fingerprint=config.fingerprint, training_fingerprint=training_fingerprint(cohort),
        statistics=statistics,
    )


def transform_sensor(
    sequence: SensorSequence, capture_timestamp: str, state: PreprocessingState,
) -> SensorWindow:
    steps = _window(sequence, capture_timestamp, state.config)
    values, masks = [], []
    for step in steps:
        observed = tuple(value is not None for value in step.values)
        if state.config.missing == "reject" and not all(observed):
            raise ValueError("Sensor window contains missing measurements under the reject policy.")
        row = []
        for measured, fit in zip(step.values, state.statistics, strict=True):
            value = fit.fill if measured is None and state.config.missing == "train_median" else measured
            row.append((value - fit.center) / fit.scale if value is not None else 0.0)
        if not all(math.isfinite(value) and abs(value) <= 1e150 for value in row):
            raise ValueError("Transformed sensor measurements exceed the numerical range.")
        if state.config.normalization == "l2":
            norm = math.hypot(*(value for value, valid in zip(row, observed, strict=True) if valid))
            if norm:
                row = [value / norm for value in row]
        values.append(tuple(row))
        masks.append(observed)
    return SensorWindow(
        preprocessing_fingerprint=state.fingerprint, timestamps=tuple(step.timestamp for step in steps),
        values=tuple(values), observed=tuple(masks),
    )
