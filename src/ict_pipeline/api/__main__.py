"""Start a single local inspection server using the configured bind address."""

import logging

import uvicorn

from .app import create_app
from .settings import load_settings


def main() -> None:
    try:
        settings = load_settings()
        app = create_app(settings)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    logger = logging.getLogger("ict_pipeline.api")
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    uvicorn.run(
        app, host=settings.host, port=settings.port, workers=1, access_log=False, proxy_headers=False, lifespan="on",
    )


if __name__ == "__main__":
    main()
