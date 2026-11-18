"""Versioned HTTP envelopes around the shared inspection contracts."""

from typing import Literal

from pydantic import AwareDatetime, UUID4

from ..inspection.contracts import Contract, InspectionInput, InspectionResult, MockScenario, SchemaVersion
from ..models.contracts import ModelName
from .settings import ModelID


class Health(Contract):
    status: Literal["ok", "degraded", "unavailable"]


class AvailableModel(Contract):
    model_id: ModelID
    mode: Literal["mock", "real"]
    model: ModelName | None = None
    mock_scenario: MockScenario | None = None
    available: bool


class ModelsResponse(Contract):
    models: tuple[AvailableModel, ...]


class SingleDetails(InspectionInput):
    model_id: ModelID


class BatchDetails(Contract):
    schema_version: SchemaVersion = 1
    model_id: ModelID
    items: tuple[InspectionInput, ...]


class StoredInspection(Contract):
    schema_version: SchemaVersion = 1
    inspection_id: UUID4
    created_at: AwareDatetime
    model_id: ModelID
    result: InspectionResult


class ErrorInfo(Contract):
    code: str
    message: str


class ErrorResponse(Contract):
    request_id: str
    error: ErrorInfo


class BatchItem(Contract):
    index: int
    status: int
    inspection: StoredInspection | None = None
    error: ErrorInfo | None = None


class BatchResponse(Contract):
    schema_version: SchemaVersion = 1
    items: tuple[BatchItem, ...]
