"""Versioned SMT sequence, alignment, and target contracts."""

from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import Field, ValidationError, model_validator

from ..classification.sensors import SampleIssue, parse_timestamp
from ..data.preprocessing.policy import NAMES, SENSOR_FIELDS
from ..inspection.contracts import Contract, Identifier, SchemaVersion
from ..pipeline.store import Fingerprint
from ..streaming.contracts import SMT_UNITS

Split = Literal["train", "validation", "test"]
ProviderSplit = Literal["train", "validation"]
Decision = Literal["candidate", "hold_review", "exclude"]
AlignmentStatus = Literal["matched", "unmatched", "ambiguous"]
Channels = tuple[str, str, str, str, str]
CHANNELS: Channels = SENSOR_FIELDS
UNITS: Channels = tuple(SMT_UNITS[name] for name in CHANNELS)

# Numeric guardrail for safe downstream arithmetic. Source units are unchanged.
Value = Annotated[float, Field(ge=-1e30, le=1e30)]
Measurements = tuple[Value | None, Value | None, Value | None, Value | None, Value | None]


class SensorStep(Contract):
    timestamp: Annotated[str, Field(min_length=1, max_length=64)]
    values: Measurements

    @property
    def instant(self) -> datetime:
        return parse_timestamp(self.timestamp)

    @model_validator(mode="after")
    def check_timestamp(self) -> Self:
        try:
            _ = self.instant
        except (ValueError, OverflowError):
            raise ValueError("invalid_sensor_timestamp") from None
        return self


class SensorSequence(Contract):
    schema_version: SchemaVersion = 1
    channels: Channels = CHANNELS
    units: Channels = UNITS
    steps: tuple[SensorStep, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_schema(self) -> Self:
        if self.channels != CHANNELS or self.units != UNITS:
            raise ValueError("Sensor channel order or source units differ from the SMT schema.")
        instants = [step.instant for step in self.steps]
        if len({instant.tzinfo is None for instant in instants}) != 1 or any(
            left >= right for left, right in zip(instants, instants[1:])
        ):
            raise ValueError("non_increasing_or_incompatible_sensor_timestamps")
        return self

    @property
    def available(self) -> bool:
        return any(value is not None for step in self.steps for value in step.values)


def sensor_step(timestamp: object, values: tuple[object, ...]) -> SensorStep:
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()
    try:
        step = SensorStep(timestamp=timestamp, values=values)
    except ValidationError as exc:
        code = "invalid_sensor_value" if exc.errors()[0]["loc"][:1] == ("values",) else "invalid_sensor_timestamp"
        raise SampleIssue(code) from None
    return step


class Target(Contract):
    category: int = Field(ge=1, le=len(NAMES))  # COCO one-based
    box: tuple[float, float, float, float]  # original-image xywh

    @model_validator(mode="after")
    def check_box(self) -> Self:
        x, y, w, h = self.box
        if min(x, y) < 0 or min(w, h) <= 0:
            raise ValueError("Invalid detection box.")
        return self


class Record(Contract):
    schema_version: SchemaVersion = 1
    sample_id: Identifier
    split: Split
    provider_split: ProviderSplit
    decision: Decision
    detection_candidate: bool
    source_class: str
    source_group: str
    groups: tuple[tuple[str, str], ...] = ()
    annotation_path: str
    image_path: str
    sensor_path: str
    annotation_sha256: Fingerprint | None = None
    image_sha256: Fingerprint | None = None
    sensor_sha256: Fingerprint | None = None
    image_id: int | None = None
    sensor_image_id: int | None = None
    capture_timestamp: str | None = None
    image_size: tuple[int, int] | None = None  # (width, height) in source coordinates
    targets: tuple[Target, ...] | None = None
    image_available: bool = False
    sequence: SensorSequence | None = None
    alignment: AlignmentStatus = "unmatched"
    alignment_method: Literal["provider_record_json_xlsx"] | None = None
    issues: tuple[str, ...] = ()
    quarantined: bool = False

    @model_validator(mode="after")
    def check_modalities(self) -> Self:
        if self.alignment == "matched" and (self.sequence is None or self.alignment_method is None):
            raise ValueError("Matched samples require a verified sensor sequence.")
        if self.alignment != "matched" and (self.sequence is not None or self.alignment_method is not None):
            raise ValueError("Unmatched samples cannot carry attached measurements.")
        if self.image_available and (not self.image_path or not self.image_sha256 or not self.image_size):
            raise ValueError("An available image requires its path, fingerprint, and dimensions.")
        if self.targets is not None:
            if self.image_size is None or min(self.image_size) <= 0:
                raise ValueError("Detection targets require positive image dimensions.")
            width, height = self.image_size
            if any(x + w > width + 1e-6 or y + h > height + 1e-6 for x, y, w, h in
                   (target.box for target in self.targets)):
                raise ValueError("Detection target exceeds the original image.")
        return self

    @property
    def sensor_available(self) -> bool:
        return self.sequence is not None and self.sequence.available
