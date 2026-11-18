"""Input events and adapter interfaces independent of hardware drivers."""

from collections.abc import Iterator
from datetime import datetime
from threading import Event
from typing import Annotated, Literal, Protocol, Self, get_args, runtime_checkable

from pydantic import Field, model_validator

from ..classification.sensors import parse_timestamp, reading
from ..data.preprocessing.audit import XLSX_FIELDS
from ..data.preprocessing.policy import SENSOR_FIELDS
from ..inspection.contracts import Contract, Identifier, SensorInput
from ..inspection.service import MAX_IMAGE_BYTES

PublicID = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]
ImageKind = Literal["directory", "usb", "rtsp", "custom"]
SensorKind = Literal["csv", "json", "custom"]
SourceCode = Literal[
    "invalid_frame", "invalid_manifest", "invalid_sensor_data", "input_limit",
    "source_unavailable", "driver_unavailable", "disconnected", "timeout",
]
SOURCE_CODES = frozenset(get_args(SourceCode))
SMT_UNITS = dict(zip(SENSOR_FIELDS, ("°C", "%", "m/s²", "g", "dB"), strict=True))
SENSOR_COLUMNS = dict(zip(XLSX_FIELDS, SENSOR_FIELDS, strict=True))


class SourceIssue(Exception):
    def __init__(self, code: SourceCode) -> None:
        self.code = code if code in SOURCE_CODES else "source_unavailable"
        super().__init__(self.code)


class SampleReference(Contract):
    sample_id: Identifier | None = None
    production_id: Identifier | None = None
    timestamp: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def validate_timestamp(self) -> Self:
        if self.timestamp is not None:
            try:
                parse_timestamp(self.timestamp)
            except OverflowError:
                raise ValueError("Timestamp exceeds the supported time range.") from None
        return self

    @property
    def instant(self) -> datetime | None:
        return parse_timestamp(self.timestamp) if self.timestamp is not None else None


class InputEvent(SampleReference):
    source_id: PublicID
    sequence_id: Identifier


class CameraFrame(InputEvent):
    kind: ImageKind
    format: Literal["JPEG", "PNG"]
    data: bytes = Field(min_length=1, max_length=MAX_IMAGE_BYTES, repr=False)


class SensorObservation(InputEvent):
    kind: SensorKind
    timestamp: str = Field(max_length=64)
    values: dict[str, float | None] = Field(default_factory=dict, repr=False)
    units: dict[str, str] = Field(default_factory=lambda: SMT_UNITS.copy())

    @property
    def measurements(self) -> dict[str, float]:
        return {name: value for name, value in self.values.items() if value is not None}

    @model_validator(mode="after")
    def validate_measurements(self) -> Self:
        if (self.values.keys() - SMT_UNITS.keys() or self.units.keys() - SMT_UNITS.keys()
                or any(self.units.get(key) != SMT_UNITS[key] for key in self.values)
                or any(unit != SMT_UNITS[key] for key, unit in self.units.items())):
            raise ValueError("Use the canonical SMT measurement fields and units.")
        available = self.measurements
        if available:
            SensorInput(source_id=self.source_id, values=available)
        if available.keys() == SMT_UNITS.keys():
            reading(self.timestamp, tuple(available[name] for name in SENSOR_FIELDS))
        return self


class ImageSource(Protocol):
    source_id: str
    kind: ImageKind

    def frames(self, stop: Event) -> Iterator[CameraFrame | SourceIssue]:
        """Yield frames or recoverable issues. Raise SourceIssue on terminal failure."""
        ...


@runtime_checkable
class IndexedImageSource(ImageSource, Protocol):
    def references(self, stop: Event) -> Iterator[SampleReference]:
        """Expose the complete camera index before publishing any frames."""
        ...


class SensorSource(Protocol):
    source_id: str
    kind: SensorKind

    def observations(self, stop: Event) -> Iterator[SensorObservation]:
        """Yield a finite snapshot. Device adapters must bound reads and honor stop."""
        ...
