"""Read filesystem capacity at a destination, including a path not yet created."""

from dataclasses import dataclass
from pathlib import Path
from shutil import disk_usage

MIB = 1024**2
RESERVE_BYTES = 64 * MIB


class StorageError(Exception):
    """A filesystem capacity check could not be satisfied."""


@dataclass(frozen=True)
class Volume:
    device: int
    path: Path
    free_bytes: int


def volume_at(path: Path) -> Volume:
    try:
        path = path.resolve()
        while not path.exists():
            path = path.parent
        return Volume(path.stat().st_dev, path, disk_usage(path).free)
    except OSError:
        raise StorageError("Cannot measure free disk space. Check the destination volume and permissions.") from None


def require_space(path: Path, write_bytes: int = 0, *, label: str) -> None:
    volume = volume_at(path)
    required = RESERVE_BYTES + write_bytes
    if volume.free_bytes < required:
        raise StorageError(
            f"Low disk space for {label} at {volume.path}. "
            f"Measured free bytes: {volume.free_bytes}; required bytes including reserve: {required}. "
            "Free space or choose another destination."
        )
