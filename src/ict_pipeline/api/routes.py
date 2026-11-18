"""Version 1 HTTP routes for the inspection service."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from pydantic import UUID4
from starlette.concurrency import run_in_threadpool

from ..streaming.runner import StreamRunner, StreamStatus
from ..multimodal.contracts import CHANNELS, UNITS
from .errors import APIError, ERRORS
from .history import (
    DefectStatistics,
    ExportQuery,
    HistoryPage,
    HistoryQuery,
    InspectionHistory,
    StatisticsFilter,
    StatisticsSummary,
    TrendStatistics,
)
from .history_export import write_history
from .inspection_images import original_image
from .schemas import (
    BatchDetails, BatchResponse, ErrorResponse, Health, ModelsResponse, ProductDetails, SingleDetails, StoredInspection,
)
from .services import InspectionAPI, Upload
from .uploads import read_form, read_image, split_form

router = APIRouter()
DETAILS_EXAMPLE = {"request_id": "request-1", "source_id": "camera-3",
                   "sensor": {"source_id": "session-19", "values": {"temperature": 23.4}}}


def _errors(*statuses: int) -> dict[int, dict[str, Any]]:
    responses: dict[int, dict[str, Any]] = {}
    for code, (status, message) in ERRORS.items():
        if status in statuses:
            response = responses.setdefault(status, {
                "model": ErrorResponse, "content": {"application/json": {"examples": {}}},
            })
            response["content"]["application/json"]["examples"][code] = {
                "value": {"request_id": "679de798cf784be6b55acd91d5ec1602",
                          "error": {"code": code, "message": message}},
            }
    return responses


AUTH_ERRORS = _errors(401, 403, 429, 500)
UPLOAD_ERRORS = _errors(400, 404, 408, 413, 415, 422, 503, 504)
HISTORY_ERRORS = {**AUTH_ERRORS, **_errors(422, 503)}


def _multipart_schema(*, batch: bool = False, product: bool = False) -> dict[str, Any]:
    field = "images" if batch else "image"
    binary = {"type": "string", "format": "binary"}
    details = {"model_id": "mock-defect"}
    details.update({"items": [DETAILS_EXAMPLE]} if batch else DETAILS_EXAMPLE)
    if product:
        timestamp = "2026-01-01T00:00:00Z"
        details = {"model_id": "product-fixture", "request_id": "request-1", "source_id": "station-1",
                   "sample_id": "part-1", "capture_timestamp": timestamp,
                   "sensor_evidence": {"source_id": "sensor-1", "sequence": {
                       "channels": CHANNELS, "units": UNITS,
                       "steps": [{"timestamp": timestamp, "values": [23, None, None, None, None]}],
                   }}}
    return {"requestBody": {"required": True, "content": {"multipart/form-data": {
        "schema": {"type": "object", "required": ["details"] if product else [field, "details"], "properties": {
            field: {"type": "array", "items": binary} if batch else binary,
            "details": {"type": "string", "description": "Version 1 metadata encoded as JSON.",
                        "examples": [json.dumps(details)]},
        }},
    }}}}


def _api(request: Request) -> InspectionAPI:
    return request.app.state.inspections


@contextmanager
def _history(request: Request) -> Iterator[InspectionHistory]:
    repository = _api(request).repository
    if not isinstance(repository, InspectionHistory):
        raise APIError("history_unavailable")
    try:
        yield repository
    except (OSError, ValueError):
        raise APIError("storage_unavailable") from None


@router.get("/health", responses={**AUTH_ERRORS, **_errors(503)})
async def health(request: Request) -> Health:
    status = _api(request).models.health()
    if status.status == "unavailable":
        raise APIError("model_unavailable")
    return status


@router.get("/v1/sources/status", responses=AUTH_ERRORS)
async def source_status(request: Request) -> StreamStatus:
    stream = request.app.state.input_stream
    return stream.status() if isinstance(stream, StreamRunner) else stream


@router.get("/v1/models", responses=AUTH_ERRORS)
async def models(request: Request) -> ModelsResponse:
    return ModelsResponse(models=_api(request).models.catalog)


@router.post("/v1/inspections", status_code=201, responses={**AUTH_ERRORS, **UPLOAD_ERRORS},
             openapi_extra=_multipart_schema(batch=False))
async def inspect_image(request: Request, response: Response) -> StoredInspection:
    api = _api(request)
    form = await read_form(request, api.settings, 1)
    try:
        metadata, (file,) = split_form(form, "image")
        try:
            details = SingleDetails.model_validate_json(metadata)
        except ValueError:
            raise APIError("invalid_request") from None
        upload = await read_image(file, details, api.settings)
    finally:
        await form.close()
    (result,) = await api.inspect(details.model_id, [upload])
    if result.inspection is None:
        assert result.error is not None
        raise APIError(result.error.code)
    response.headers["Location"] = f"/v1/inspections/{result.inspection.inspection_id}"
    return result.inspection


@router.post("/v1/inspections/multimodal", status_code=201, responses={**AUTH_ERRORS, **UPLOAD_ERRORS},
             openapi_extra=_multipart_schema(product=True))
async def inspect_product(request: Request, response: Response) -> StoredInspection:
    api = _api(request)
    form = await read_form(request, api.settings, 1)
    try:
        metadata, files = split_form(form, "image", optional_file=True)
        try:
            details = ProductDetails.model_validate_json(metadata)
        except ValueError:
            raise APIError("invalid_request") from None
        upload = await read_image(files[0], details, api.settings) if files else Upload(details, None, None)
    finally:
        await form.close()
    (result,) = await api.inspect(details.model_id, [upload], product=True)
    if result.inspection is None:
        assert result.error is not None
        raise APIError(result.error.code)
    response.headers["Location"] = f"/v1/inspections/{result.inspection.inspection_id}"
    return result.inspection


@router.post("/v1/inspections/batch", responses={**AUTH_ERRORS, **UPLOAD_ERRORS},
             openapi_extra=_multipart_schema(batch=True))
async def inspect_batch(request: Request) -> BatchResponse:
    api = _api(request)
    form = await read_form(request, api.settings, api.settings.max_batch_items)
    try:
        metadata, files = split_form(form, "images")
        try:
            details = BatchDetails.model_validate_json(metadata)
        except ValueError:
            raise APIError("invalid_request") from None
        if len(details.items) != len(files):
            raise APIError("invalid_request")
        uploads: list[Upload | APIError] = []
        for file, item in zip(files, details.items, strict=True):
            try:
                uploads.append(await read_image(file, item, api.settings))
            except APIError as exc:
                uploads.append(exc)
    finally:
        await form.close()
    return BatchResponse(items=await api.inspect(details.model_id, uploads))


@router.get("/v1/inspections", responses=HISTORY_ERRORS)
def list_inspections(request: Request, query: Annotated[HistoryQuery, Query()]) -> HistoryPage:
    with _history(request) as history:
        return history.list(query)


@router.get("/v1/inspections/export", response_model=HistoryPage, responses={
    **HISTORY_ERRORS, 200: {"content": {"text/csv": {"schema": {"type": "string"}}}},
})
def export_inspections(request: Request, query: Annotated[ExportQuery, Query()]) -> HistoryPage | Response:
    with _history(request) as history:
        page = history.list(query)
    headers = {
        "Content-Disposition": f'attachment; filename="history.{query.format}"',
        "X-Total-Count": str(page.total),
        "X-Limit": str(page.limit),
        "X-Offset": str(page.offset),
    }
    if query.format == "json":
        return Response(page.model_dump_json(), media_type="application/json", headers=headers)
    output = StringIO(newline="")
    write_history(output, page.items, "csv")
    return Response(output.getvalue(), media_type="text/csv", headers=headers)


@router.get("/v1/statistics/summary", responses=HISTORY_ERRORS)
def statistics_summary(request: Request, query: Annotated[StatisticsFilter, Query()]) -> StatisticsSummary:
    with _history(request) as history:
        return history.summary(query)


@router.get("/v1/statistics/defects", responses=HISTORY_ERRORS)
def statistics_defects(request: Request, query: Annotated[StatisticsFilter, Query()]) -> DefectStatistics:
    with _history(request) as history:
        return history.defects(query)


@router.get("/v1/statistics/trends", responses=HISTORY_ERRORS)
def statistics_trends(request: Request, query: Annotated[StatisticsFilter, Query()]) -> TrendStatistics:
    with _history(request) as history:
        return history.trends(query)


@router.get("/v1/inspections/{inspection_id}", responses={**AUTH_ERRORS, **_errors(404, 422, 503)})
async def get_inspection(request: Request, inspection_id: UUID4) -> StoredInspection:
    return await _api(request).get(inspection_id)


@router.get("/v1/inspections/{inspection_id}/image", response_class=Response, responses={
    **AUTH_ERRORS, **_errors(404, 422, 503),
    200: {"content": {kind: {"schema": {"type": "string", "format": "binary"}}
                      for kind in ("image/jpeg", "image/png")}},
})
async def get_inspection_image(request: Request, inspection_id: UUID4) -> Response:
    record = await _api(request).get(inspection_id)
    data, media_type = await run_in_threadpool(
        original_image, request.app.state.inspection_image_root, record.result.image,
    )
    return Response(data, media_type=media_type, headers={
        "X-Content-Type-Options": "nosniff",
        "Cross-Origin-Resource-Policy": "same-origin",
    })
