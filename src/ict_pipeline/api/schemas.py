"""Versioned HTTP envelopes around the shared inspection contracts."""

from typing import Any, Literal

from pydantic import AwareDatetime, ConfigDict, SerializerFunctionWrapHandler, UUID4, model_serializer

from ..inspection.contracts import Contract, InspectionInput, InspectionResult, MockScenario, SchemaVersion
from ..models.contracts import ModelName
from ..multimodal.inference_contracts import ProductInspectionInput
from .settings import ModelID


class Health(Contract):
    status: Literal["ok", "degraded", "unavailable"]


class AvailableModel(Contract):
    model_config = ConfigDict(json_schema_mode_override="validation")

    model_id: ModelID
    mode: Literal["mock", "real", "finalized"]
    model: ModelName | None = None
    mock_scenario: MockScenario | None = None
    available: bool
    input_kind: Literal["multimodal"] | None = None

    @model_serializer(mode="wrap")
    def serialize_model(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.input_kind is None:
            data.pop("input_kind", None)
        return data


class ModelsResponse(Contract):
    models: tuple[AvailableModel, ...]


class SingleDetails(InspectionInput):
    model_id: ModelID


class ProductDetails(ProductInspectionInput):
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


def decode_stored_inspection(data: bytes | str) -> StoredInspection:
    record = StoredInspection.model_validate_json(data)
    if ("schema_version" not in record.model_fields_set
            or "schema_version" not in record.result.model_fields_set):
        raise ValueError("A persisted inspection must declare both schema versions.")
    return record


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
