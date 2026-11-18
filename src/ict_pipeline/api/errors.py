"""Stable error codes and public messages for the HTTP interface."""

from fastapi.responses import JSONResponse

from .schemas import ErrorInfo, ErrorResponse

ERRORS: dict[str, tuple[int, str]] = {
    "malformed_request": (400, "Send a complete multipart form with the documented fields."),
    "unauthorized": (401, "Supply a valid bearer token."),
    "local_access_required": (403, "Use a loopback connection and a loopback Host header."),
    "not_found": (404, "The requested resource was not found."),
    "model_not_found": (404, "The selected model is not configured."),
    "inspection_not_found": (404, "The inspection is unknown or has expired."),
    "inspection_image_unavailable": (404, "The original inspection image is unavailable."),
    "method_not_allowed": (405, "The HTTP method is not supported for this resource."),
    "upload_timeout": (408, "The upload did not finish within the configured time."),
    "request_too_large": (413, "The multipart request exceeds the configured byte limit."),
    "image_too_large": (413, "The image exceeds the configured byte limit."),
    "unsupported_media_type": (415, "Use multipart/form-data with image/jpeg or image/png uploads."),
    "invalid_request": (422, "The request does not match the documented schema or batch limits."),
    "invalid_image": (422, "Send a valid single-frame JPEG or PNG matching its MIME type and image limits."),
    "dimension_mismatch": (422, "Image dimensions differ from expected_size."),
    "busy": (429, "The inspection API is busy. Retry later."),
    "internal_error": (500, "The request could not be completed."),
    "prediction_failed": (500, "The selected model could not complete the inspection."),
    "model_unavailable": (503, "The selected model could not be initialized."),
    "storage_unavailable": (503, "The inspection store could not complete the operation."),
    "history_unavailable": (503, "The configured inspection store does not support history queries."),
    "inference_timeout": (504, "Inspection exceeded its configured processing deadline."),
}


class APIError(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        self.status, message = ERRORS[code]
        self.info = ErrorInfo(code=code, message=message)
        super().__init__(code)

    def response(self, request_id: str, *, headers: dict[str, str] | None = None) -> JSONResponse:
        headers = dict(headers or {})
        if self.status == 401:
            headers["WWW-Authenticate"] = "Bearer"
        elif self.status == 429:
            headers["Retry-After"] = "1"
        return JSONResponse(
            status_code=self.status, headers=headers,
            content=ErrorResponse(request_id=request_id, error=self.info).model_dump(mode="json"),
        )
