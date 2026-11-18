"""Label-free requests and content-bound receipts for product inspection."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from ..classification.sensors import parse_timestamp
from ..inspection.contracts import Contract, Device, InspectionInput, SchemaVersion
from ..pipeline.store import Fingerprint, digest_json
from .contracts import Channels, Record, SensorSequence, SensorStep

SourceID = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")]
CaptureTime = Annotated[str, Field(min_length=1, max_length=64)]


class InputSequence(SensorSequence):
    channels: Channels = Field()
    units: Channels = Field()
    steps: tuple[SensorStep, ...] = Field(min_length=1, max_length=4096)


class SensorEvidence(Contract):
    source_id: SourceID
    sequence: InputSequence


class ProductInspectionInput(InspectionInput):
    request_id: SourceID
    source_id: SourceID
    sample_id: SourceID
    capture_timestamp: CaptureTime | None = None
    sensor: None = None
    sensor_evidence: SensorEvidence | None = None

    @model_validator(mode="after")
    def check_capture(self) -> Self:
        if self.sensor_evidence is not None and self.capture_timestamp is None:
            raise ValueError("Sensor evidence requires a capture timestamp.")
        if self.capture_timestamp is not None:
            capture = capture_instant(self.capture_timestamp)
            if self.sensor_evidence is not None:
                check_sensor_clock(self.sensor_evidence.sequence, capture)
        return self


def capture_instant(timestamp: str) -> datetime:
    try:
        return parse_timestamp(timestamp)
    except (ValueError, OverflowError):
        raise ValueError("Use a valid ISO 8601 capture timestamp.") from None


def check_sensor_clock(sequence: SensorSequence, capture: datetime) -> None:
    if (capture.tzinfo is None) != (sequence.steps[0].instant.tzinfo is None):
        raise ValueError("Sensor timestamps and capture require comparable clocks.")


def causal_sequence(sequence: SensorSequence, capture: datetime) -> SensorSequence | None:
    check_sensor_clock(sequence, capture)
    steps = tuple(step for step in sequence.steps if step.instant <= capture)
    return SensorSequence(steps=steps) if steps else None


def sequence_fingerprint(sequence: SensorSequence | None) -> str:
    rows = [] if sequence is None else [[step.instant.isoformat(), step.values] for step in sequence.steps]
    return digest_json(rows)


class PairingReceipt(Contract):
    schema_version: SchemaVersion = 1
    sample_id: SourceID
    image_source_id: SourceID
    sensor_source_id: SourceID
    image_sha256: Fingerprint
    sensor_fingerprint: Fingerprint
    capture_timestamp: CaptureTime

    @model_validator(mode="after")
    def check_timestamp(self) -> Self:
        if capture_instant(self.capture_timestamp).isoformat() != self.capture_timestamp:
            raise ValueError("Use a canonical capture timestamp in pairing receipts.")
        return self

    @property
    def fingerprint(self) -> str:
        return digest_json(self.model_dump(mode="json"))

    @classmethod
    def from_audited_record(
        cls, record: Record, *, image_source_id: str, sensor_source_id: str,
    ) -> Self:
        if (record.quarantined or record.alignment != "matched" or not record.image_available
                or record.image_sha256 is None or record.sequence is None or record.capture_timestamp is None):
            raise ValueError("Pairing requires a verified, non-quarantined image and sensor record.")
        capture = capture_instant(record.capture_timestamp)
        sequence = causal_sequence(record.sequence, capture)
        if sequence is None:
            raise ValueError("A pairing receipt requires sensor evidence available at capture.")
        return cls(
            sample_id=record.sample_id, image_source_id=image_source_id, sensor_source_id=sensor_source_id,
            image_sha256=record.image_sha256, capture_timestamp=capture.isoformat(),
            sensor_fingerprint=sequence_fingerprint(sequence),
        )


class PairingRegistry(Contract):
    schema_version: SchemaVersion = 1
    receipts: tuple[PairingReceipt, ...]


class MockProductModel(Contract):
    kind: Literal["mock"]
    pairings: Path | None = None


class CheckpointProductModel(Contract):
    kind: Literal["checkpoint"]
    checkpoint: Path
    checkpoint_sha256: Fingerprint
    architecture_fingerprint: Fingerprint
    preprocessing: Path
    pairings: Path | None = None
    device: Device = "cpu"


ProductModelSpec = Annotated[MockProductModel | CheckpointProductModel, Field(discriminator="kind")]
