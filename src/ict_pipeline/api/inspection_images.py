"""Read an explicitly staged original whose bytes match the recorded image."""

from pathlib import Path

from ..inspection.contracts import ImageSource
from ..inspection.service import ImageError, _read_image
from .errors import APIError

IMAGE_TYPES = {"JPEG": (".jpg", "image/jpeg"), "PNG": (".png", "image/png")}


def original_image(root: Path | None, source: ImageSource | None) -> tuple[bytes, str]:
    if root is None or source is None:
        raise APIError("inspection_image_unavailable")
    extension, media_type = IMAGE_TYPES[source.format]
    candidate = root / f"{source.sha256}{extension}"
    try:
        # The configured directory is trusted. Never follow linked files or directories.
        if candidate.is_symlink() or candidate.resolve(strict=True) != candidate:
            raise APIError("inspection_image_unavailable")
        decoded, data = _read_image(candidate)
    except (OSError, RuntimeError, ImageError):
        raise APIError("inspection_image_unavailable") from None
    if decoded != source:
        raise APIError("inspection_image_unavailable")
    return data, media_type
