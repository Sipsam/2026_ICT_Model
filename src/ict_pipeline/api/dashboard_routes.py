"""Fixed, read-only browser assets served by the inspection API."""

from functools import lru_cache
from importlib.resources import files

from fastapi import APIRouter, Request, Response

router = APIRouter(include_in_schema=False)
ASSETS = {
    "/dashboard": ("index.html", "text/html"),
    "/dashboard/dashboard.css": ("dashboard.css", "text/css"),
    "/dashboard/dashboard.mjs": ("dashboard.mjs", "text/javascript"),
    "/dashboard/client.mjs": ("client.mjs", "text/javascript"),
}
HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self' blob:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
}


@lru_cache(maxsize=len(ASSETS))
def _asset(filename: str) -> bytes:
    return files("ict_pipeline.api.dashboard").joinpath(filename).read_bytes()


async def _serve_asset(request: Request) -> Response:
    filename, media_type = ASSETS[request.scope["route"].path]
    return Response(_asset(filename), media_type=media_type, headers=HEADERS)


for asset_path in ASSETS:
    router.add_api_route(asset_path, _serve_asset, methods=["GET"])
