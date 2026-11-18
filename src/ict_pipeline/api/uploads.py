"""Bounded, in-memory handling of multipart uploads."""

import asyncio
from collections.abc import AsyncGenerator
from typing import Literal

from fastapi import Request
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.datastructures import FormData, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.requests import ClientDisconnect

from ..inspection.contracts import InspectionInput
from .errors import APIError
from .services import Upload
from .settings import APISettings

MAX_DETAILS_BYTES = 32 * 1024
IMAGE_FORMATS: dict[str, Literal["JPEG", "PNG"]] = {"image/jpeg": "JPEG", "image/png": "PNG"}


class UploadParser(MultiPartParser):
    def __init__(self, request: Request, settings: APISettings, max_files: int) -> None:
        # The total body is bounded before it reaches the parser. Keep private
        # pixels and metadata out of the OS spool directory.
        self.spool_max_size = settings.max_request_bytes
        self.complete = False
        super().__init__(request.headers, _bounded_stream(request, settings.max_request_bytes),
                         max_files=max_files, max_fields=1, max_part_size=MAX_DETAILS_BYTES)

    def on_part_begin(self) -> None:
        super().on_part_begin()
        self._header_bytes = 0

    def _count_header_bytes(self, size: int) -> None:
        self._header_bytes += size
        if self._header_bytes > 8192:
            raise MultiPartException("Multipart headers exceed their size limit.")

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._count_header_bytes(end - start)
        super().on_header_field(data, start, end)

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._count_header_bytes(end - start)
        super().on_header_value(data, start, end)

    def on_end(self) -> None:
        self.complete = True


async def _bounded_stream(request: Request, limit: int) -> AsyncGenerator[bytes, None]:
    size = 0
    async for data in request.stream():
        size += len(data)
        if size > limit:
            raise APIError("request_too_large")
        for start in range(0, len(data), 64 * 1024):
            yield data[start:start + 64 * 1024]


async def read_form(request: Request, settings: APISettings, max_files: int) -> FormData:
    length = request.headers.get("content-length")
    if length is not None:
        if not length.isascii() or not length.isdecimal():
            raise APIError("malformed_request")
        if len(length) > 12 or int(length) > settings.max_request_bytes:
            raise APIError("request_too_large")
    try:
        media_type, options = parse_options_header(request.headers.get("content-type"))
        if media_type.lower() != b"multipart/form-data":
            raise APIError("unsupported_media_type")
        if not 1 <= len(options.get(b"boundary", b"")) <= 70:
            raise APIError("malformed_request")
        parser = UploadParser(request, settings, max_files)
        async with asyncio.timeout(settings.upload_timeout_seconds):
            form = await parser.parse()
    except TimeoutError:
        raise APIError("upload_timeout") from None
    except (MultiPartException, MultipartParseError, ClientDisconnect, ValueError):
        raise APIError("malformed_request") from None
    if not parser.complete:
        await form.close()
        raise APIError("malformed_request")
    return form


def split_form(form: FormData, file_field: str) -> tuple[str, list[UploadFile]]:
    details = form.get("details")
    files = form.getlist(file_field)
    if (set(form) != {"details", file_field} or not isinstance(details, str)
            or not files or any(not isinstance(file, UploadFile) for file in files)):
        raise APIError("malformed_request")
    return details, files


async def read_image(file: UploadFile, details: InspectionInput, settings: APISettings) -> Upload:
    image_format = IMAGE_FORMATS.get((file.content_type or "").lower())
    if image_format is None:
        raise APIError("unsupported_media_type")
    if file.size is not None and file.size > settings.max_image_bytes:
        raise APIError("image_too_large")
    data = await file.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        raise APIError("image_too_large")
    return Upload(details, data, image_format)
