"""FastAPI application factory. Model resources belong to the ASGI lifespan."""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from secrets import compare_digest
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders, URL
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..config.settings import Credentials
from ..evaluation.adapters import PredictorFactory
from .errors import APIError
from .repository import InspectionRepository
from .routes import router
from .services import InspectionAPI, ModelManager
from .settings import APISettings, is_loopback, load_settings
from .sqlite_schema import InspectionDatabaseError
from .storage import open_repository, storage_root

logger = logging.getLogger(__name__)


class RequestMiddleware:
    def __init__(self, app: ASGIApp, *, capacity: int, token: str) -> None:
        self.app = app
        self.capacity = capacity
        self.token = token.encode("ascii")
        self.active = 0

    def _authorize(self, scope: Scope) -> None:
        headers = Headers(scope=scope)
        if self.token:
            scheme, _, value = headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not compare_digest(value.encode("latin-1"), self.token):
                raise APIError("unauthorized")
        else:
            try:
                hostname = URL(scope=scope).hostname or ""
            except (UnicodeError, ValueError):
                hostname = ""
            peer = scope.get("client")
            host = headers.get("host")
            origin = headers.get("origin")
            if (not peer or not is_loopback(peer[0]) or not host or not is_loopback(hostname)
                    or (origin is not None and origin != f"{scope['scheme']}://{host}")):
                raise APIError("local_access_required")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id
        started = perf_counter()
        status = None
        admitted = False

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                headers["Cache-Control"] = "no-store"
            await send(message)

        try:
            self._authorize(scope)
            if self.active >= self.capacity:
                raise APIError("busy")
            self.active += 1
            admitted = True
            await self.app(scope, receive, send_with_id)
        except APIError as exc:
            await exc.response(request_id)(scope, receive, send_with_id)
        except Exception:
            if status is None:
                await APIError("internal_error").response(request_id)(scope, receive, send_with_id)
            else:
                # Keep exception contents out of server logs after a response starts.
                raise RuntimeError("Inspection response could not complete.") from None
        finally:
            if admitted:
                self.active -= 1
            route = scope.get("route")
            path = getattr(route, "path", None)
            logger.info(json.dumps({
                "event": "inspection_http_request", "request_id": request_id,
                "route": path, "status": status, "elapsed_ms": round((perf_counter() - started) * 1000, 1),
            }))


def create_app(
    settings: APISettings | None = None, *, repository: InspectionRepository | None = None,
    predictor_factory: PredictorFactory | None = None,
) -> FastAPI:
    """Configure an app without reading checkpoint files or loading weights."""
    settings = settings or load_settings()
    token = os.environ.get("ICT_API_TOKEN", "")
    if (token and (len(token) < 32 or not token.isascii() or not token.isprintable()
                   or any(character.isspace() for character in token))):
        raise RuntimeError("ICT_API_TOKEN must contain at least 32 printable ASCII characters without whitespace.")
    if not is_loopback(settings.host) and not token:
        raise RuntimeError("Set ICT_API_TOKEN before binding to a non-loopback host.")
    credentials = Credentials.load({}, os.environ)
    if token:
        credentials.add("ICT_API_TOKEN", token)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            root = storage_root(settings, credentials)
            root.mkdir(mode=0o700, parents=True, exist_ok=True)
            store = repository if repository is not None else await run_in_threadpool(
                open_repository, settings, credentials,
            )
            temporary = TemporaryDirectory(dir=root, prefix=".work-")
        except InspectionDatabaseError as exc:
            raise RuntimeError(f"Cannot initialize inspection storage. {exc}") from None
        except Exception:
            raise RuntimeError("Cannot initialize inspection storage. Check the configured roots and permissions.") from None
        with temporary as directory:
            scratch = Path(directory)
            models = await run_in_threadpool(ModelManager, settings, scratch, credentials, predictor_factory)
            inspections = InspectionAPI(settings, models, store, scratch)
            app.state.inspections = inspections
            try:
                yield
            finally:
                try:
                    await inspections.close()
                finally:
                    del app.state.inspections

    app = FastAPI(
        title="ICT Inspection API", version="1.0.0", lifespan=lifespan, redirect_slashes=False,
    )
    app.add_middleware(RequestMiddleware, capacity=settings.max_concurrent_requests, token=token)
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, _: RequestValidationError) -> JSONResponse:
        return APIError("invalid_request").response(request.state.request_id)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException) -> JSONResponse:
        codes = {404: "not_found", 405: "method_not_allowed"}
        return APIError(codes.get(exc.status_code, "internal_error")).response(
            request.state.request_id, headers=exc.headers,
        )

    return app
