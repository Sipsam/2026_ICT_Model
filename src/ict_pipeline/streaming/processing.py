"""Turn aligned frames into insert-only inspection records."""

import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from threading import Event
from typing import Literal, Protocol, TypeVar
from uuid import UUID

from ..api.repository import InspectionRepository
from ..api.schemas import StoredInspection
from ..api.sqlite_schema import InspectionDatabaseBusy
from ..inspection.contracts import InspectionInterrupted, InspectionMetadataInput, SensorInput, TransientInspectionError
from ..inspection.service import InspectionService
from ..pipeline.store import digest_json
from .alignment import Alignment
from .contracts import CameraFrame, SMT_UNITS
from .settings import AlignmentSettings, ProcessingSettings

T = TypeVar("T")
ProcessingOutcome = Literal["persisted", "reused", "invalid_image", "inference_failed", "storage_failed", "cancelled"]
TRANSIENT_ERRORS = (InspectionDatabaseBusy, TransientInspectionError, BlockingIOError, TimeoutError)


class StreamRepository(InspectionRepository, Protocol):
    def read_record(self, inspection_id: UUID) -> StoredInspection | None:
        """Read an existing record without applying retention, for deduplication only."""
        ...


class ProcessingStopped(Exception):
    pass


class FrameProcessor:
    def __init__(
        self, *, stream_id: str, model_id: str, service: InspectionService, repository: StreamRepository,
        alignment: AlignmentSettings, settings: ProcessingSettings, limits: tuple[int, int, int, int],
    ) -> None:
        self.stream_id = stream_id
        self.model_id = model_id
        self.service = service
        self.repository = repository
        self.alignment = alignment
        self.settings = settings
        self.limits = limits

    def _retry(self, action: Callable[[], T], stop: Event, on_retry: Callable[[], None]) -> T:
        for attempt in range(self.settings.max_retries + 1):
            if stop.is_set():
                raise ProcessingStopped
            if attempt:
                on_retry()
            try:
                return action()
            except TRANSIENT_ERRORS:
                if attempt == self.settings.max_retries:
                    raise
                if stop.wait(self.settings.retry_delay_seconds):
                    raise ProcessingStopped from None
        raise AssertionError("Retry loop must return or raise.")

    def process(
        self, frame: CameraFrame, match: Alignment, stop: Event, on_retry: Callable[[], None],
    ) -> ProcessingOutcome:
        max_bytes, width, height, pixels = self.limits
        if len(frame.data) > max_bytes:
            return "invalid_image"
        sha256 = hashlib.sha256(frame.data).hexdigest()
        observation = match.observation
        measurements = observation.measurements if observation else {}
        sensor = SensorInput(source_id=observation.source_id, values=measurements) if observation and measurements else None
        metadata = {
            "streaming": {
                "schema_version": 1,
                "stream_id": self.stream_id,
                "camera": frame.model_dump(mode="json", exclude={"data"}),
                "sensor_observation": observation.model_dump(mode="json", exclude={"values"}) if observation else None,
                "missing_measurements": [name for name in SMT_UNITS if name not in measurements],
                "alignment": {
                    "sensor_status": "available" if sensor else "unavailable",
                    "method": match.method,
                    "confidence": ("bounded_timestamp" if match.method == "timestamp"
                                   else "explicit" if match.method else "none"),
                    "reason": match.reason,
                    "skew_seconds": match.skew_seconds,
                    "policy": self.alignment.model_dump(mode="json"),
                },
            },
        }
        request_id = digest_json({
            "metadata": metadata, "image_sha256": sha256, "measurements": measurements,
            "model_id": self.model_id, "provenance": self.service.provenance.model_dump(mode="json"),
            "thresholds": self.service.thresholds.model_dump(mode="json"), "limits": self.limits,
        })
        # Keep the public UUID4 shape. Verify the full fingerprint on a lookup.
        inspection_id = UUID(hex=request_id[:32], version=4)
        request = InspectionMetadataInput(
            request_id=request_id, source_id=frame.source_id, sensor=sensor, metadata=metadata,
        )

        def already_stored() -> bool:
            existing = self.repository.read_record(inspection_id)
            if existing is None:
                return False
            if (existing.result.request_id != request_id or existing.model_id != self.model_id
                    or existing.result.image is None or existing.result.image.sha256 != sha256):
                raise ValueError("Inspection ID conflicts with a stored record.")
            return True

        try:
            if self._retry(already_stored, stop, on_retry):
                return "reused"
        except (OSError, ValueError):
            return "storage_failed"
        try:
            result = self._retry(
                lambda: self.service.inspect_bytes(
                    request, frame.data, image_format=frame.format, max_width=width, max_height=height,
                    max_pixels=pixels, propagate_transient=True, cancelled=stop,
                ), stop, on_retry,
            )
        except InspectionInterrupted:
            return "cancelled"
        except TransientInspectionError:
            return "inference_failed"
        if result.outcome == "inference_failed":
            return "invalid_image" if result.reason in {"invalid_image", "dimension_mismatch"} else "inference_failed"
        if result.request_id != request_id:
            return "storage_failed"
        record = StoredInspection(
            inspection_id=inspection_id, created_at=datetime.now(UTC), model_id=self.model_id, result=result,
        )

        def persist() -> ProcessingOutcome:
            try:
                self.repository.add(record)
            except OSError:
                if already_stored():
                    return "reused"
                raise
            return "persisted"

        try:
            return self._retry(persist, stop, on_retry)
        except (OSError, ValueError):
            return "storage_failed"
