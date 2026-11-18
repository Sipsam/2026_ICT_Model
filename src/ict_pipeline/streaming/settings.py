"""Validated configuration for opt-in streaming sessions."""

from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from ..inspection.contracts import Contract
from .contracts import PublicID


class DirectorySettings(Contract):
    kind: Literal["directory"] = "directory"
    source_id: PublicID
    directory: Path
    manifest: Path | None = None
    order: Literal["name", "reverse_name", "timestamp"] = "name"
    frames_per_second: float = Field(default=0, ge=0, le=240)
    max_frames: int = Field(default=10_000, ge=1, le=100_000)

    @model_validator(mode="after")
    def validate_pacing(self) -> Self:
        if 0 < self.frames_per_second < 0.001:
            raise ValueError("Set replay speed to zero or at least 0.001 frames per second.")
        return self


class DeviceSettings(Contract):
    source_id: PublicID
    enabled: Literal[True]
    timeout_seconds: float = Field(default=5, ge=0.1, le=60)
    reconnect_attempts: int = Field(default=2, ge=0, le=20)
    reconnect_delay_seconds: float = Field(default=1, ge=0, le=60)

    @field_validator("enabled", mode="before")
    @classmethod
    def require_explicit_opt_in(cls, value: object) -> object:
        if value is not True:
            raise ValueError("Set enabled to true to open a physical camera.")
        return value


class USBSettings(DeviceSettings):
    kind: Literal["usb"] = "usb"
    device_index: int = Field(ge=0, le=255)


class RTSPSettings(DeviceSettings):
    kind: Literal["rtsp"] = "rtsp"
    url_env: str = Field(pattern=r"^ICT_RTSP_[A-Z0-9_]{1,64}$")


CameraSettings = Annotated[DirectorySettings | USBSettings | RTSPSettings, Field(discriminator="kind")]


class SensorReplaySettings(Contract):
    kind: Literal["csv", "json"]
    source_id: PublicID
    path: Path
    max_observations: int = Field(default=10_000, ge=1, le=100_000)


class AlignmentSettings(Contract):
    allow_image_only: bool = False
    max_skew_seconds: float = Field(default=1, ge=0, le=3600)
    shared_clock: PublicID | None = None


class ProcessingSettings(Contract):
    queue_capacity: int = Field(default=8, ge=1, le=128)
    workers: int = Field(default=1, ge=1, le=8)
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_delay_seconds: float = Field(default=0.1, ge=0, le=10)
    shutdown_seconds: float = Field(default=3, ge=0, le=60)


class StreamSettings(Contract):
    stream_id: PublicID
    model_id: PublicID = "mock-empty"
    camera: CameraSettings
    sensor: SensorReplaySettings | None = None
    alignment: AlignmentSettings = Field(default_factory=AlignmentSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)

    @model_validator(mode="after")
    def check_sources(self) -> Self:
        if self.sensor is None and not self.alignment.allow_image_only:
            raise ValueError("Configure a sensor source or explicitly allow image-only inspection.")
        if self.alignment.shared_clock and (self.sensor is None or not isinstance(self.camera, DirectorySettings)):
            raise ValueError("Timestamp alignment requires two replay sources with a verified shared clock.")
        return self
