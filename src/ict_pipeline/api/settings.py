"""Server-owned configuration for HTTP inspection."""

import os
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ..data.preprocessing.common import safe_component
from ..inspection.contracts import CheckpointKind, Contract, Device, InspectionThresholds, MockScenario
from ..inspection.service import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS
from ..models.contracts import ModelName
from ..pipeline.store import Fingerprint
from ..streaming.settings import StreamSettings

ModelID = Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")]


class MockModel(Contract):
    mode: Literal["mock"]
    scenario: MockScenario = "empty"
    thresholds: InspectionThresholds = Field(default_factory=InspectionThresholds)


class TrustedModel(Contract):
    mode: Literal["real"]
    model: ModelName
    run: Path
    trust_run: Literal[True]
    checkpoint: CheckpointKind = "best"
    expected_fingerprint: Fingerprint | None = None
    device: Device = "cpu"
    image_size: int | None = Field(default=None, ge=64, multiple_of=32)
    thresholds: InspectionThresholds = Field(default_factory=InspectionThresholds)

    @model_validator(mode="after")
    def require_relative_run(self) -> Self:
        if (self.trust_run is not True or self.run.anchor or ".." in self.run.parts
                or any(":" in part for part in self.run.parts)):
            raise ValueError("Explicitly trust a run relative to the configured model root.")
        return self


ModelSpec = Annotated[MockModel | TrustedModel, Field(discriminator="mode")]


class APISettings(Contract):
    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    storage_root: Path = Path("outputs/inspection-api")
    storage_backend: Literal["file", "sqlite"] = "file"
    database_name: str = "inspections.sqlite3"
    model_root: Path = Path("runs/training")
    image_root: Path | None = None
    models: dict[ModelID, ModelSpec] = Field(default_factory=lambda: {
        "mock-empty": MockModel(mode="mock"),
        "mock-defect": MockModel(mode="mock", scenario="defect"),
    }, min_length=1, max_length=16)
    max_loaded_models: int = Field(default=3, ge=0, le=8)
    max_request_bytes: int = Field(default=24 * 1024**2, ge=1, le=128 * 1024**2)
    max_image_bytes: int = Field(default=8 * 1024**2, ge=1, le=MAX_IMAGE_BYTES)
    max_width: int = Field(default=6000, ge=1, le=MAX_IMAGE_PIXELS)
    max_height: int = Field(default=6000, ge=1, le=MAX_IMAGE_PIXELS)
    max_pixels: int = Field(default=16_000_000, ge=1, le=MAX_IMAGE_PIXELS)
    max_batch_items: int = Field(default=8, ge=1, le=32)
    max_concurrent_requests: int = Field(default=4, ge=1, le=32)
    upload_timeout_seconds: float = Field(default=15.0, gt=0)
    inference_timeout_seconds: float = Field(default=30.0, gt=0)
    retention_seconds: int | None = Field(default=None, ge=1)
    input_stream: StreamSettings | None = None

    @model_validator(mode="after")
    def bound_models(self) -> Self:
        if sum(isinstance(model, TrustedModel) for model in self.models.values()) > self.max_loaded_models:
            raise ValueError("Real models exceed the configured startup capacity.")
        if not self.host or not self.host.isascii() or any(c.isspace() for c in self.host):
            raise ValueError("Supply a valid bind host.")
        if not safe_component(self.database_name) or not self.database_name.endswith(".sqlite3"):
            raise ValueError("Supply a local .sqlite3 filename within storage_root.")
        return self


def is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def load_settings() -> APISettings:
    """Load optional JSON settings. Paths are relative to the working directory."""
    config = os.environ.get("ICT_API_CONFIG")
    try:
        return APISettings.model_validate_json(Path(config).read_bytes()) if config else APISettings()
    except (OSError, ValueError):
        raise RuntimeError("Cannot read or validate ICT_API_CONFIG.") from None
