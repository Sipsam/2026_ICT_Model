"""Inspection persistence with an insert-only local implementation."""

import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from uuid import UUID

from .schemas import StoredInspection


class InspectionRepository(Protocol):
    def add(self, record: StoredInspection) -> None: ...

    def get(self, inspection_id: UUID) -> StoredInspection | None: ...


class FileInspectionRepository:
    def __init__(self, root: Path, *, retention_seconds: int | None = None) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.retention_seconds = retention_seconds

    def _path(self, inspection_id: UUID) -> Path:
        path = self.root / f"{inspection_id.hex}.json"
        if path.resolve() != path:
            raise OSError("Invalid inspection record location.")
        return path

    def add(self, record: StoredInspection) -> None:
        target = self._path(record.inspection_id)
        temporary = None
        try:
            with NamedTemporaryFile(dir=self.root, prefix=".pending-", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(record.model_dump_json().encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
            # A same-directory hard link publishes the complete record atomically
            # and fails if another writer already owns this ID.
            target.hardlink_to(temporary)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def get(self, inspection_id: UUID) -> StoredInspection | None:
        try:
            data = self._path(inspection_id).read_bytes()
        except FileNotFoundError:
            return None
        record = StoredInspection.model_validate_json(data)
        if record.inspection_id != inspection_id:
            raise ValueError("Invalid inspection record identity.")
        if (self.retention_seconds is not None
                and (datetime.now(UTC) - record.created_at).total_seconds() >= self.retention_seconds):
            return None
        return record
