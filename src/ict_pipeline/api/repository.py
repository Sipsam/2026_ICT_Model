"""Inspection persistence with an insert-only local implementation."""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from uuid import UUID

from .history import (
    DefectStatistics,
    HistoryFilter,
    HistoryPage,
    HistoryQuery,
    Order,
    StatisticsFilter,
    StatisticsSummary,
    TrendStatistics,
    count_days,
    count_defects,
    matches,
    retention_cutoff,
    summarize,
)
from .schemas import StoredInspection, decode_stored_inspection


class InspectionRepository(Protocol):
    def add(self, record: StoredInspection) -> None: ...

    def get(self, inspection_id: UUID) -> StoredInspection | None: ...


class FileInspectionRepository:
    def __init__(self, root: Path, *, retention_seconds: int | None = None, read_only: bool = False) -> None:
        self.root = root.resolve()
        if not read_only:
            self.root.mkdir(parents=True, mode=0o700, exist_ok=True)
        self._read_only = read_only
        self.retention_seconds = retention_seconds

    def _path(self, inspection_id: UUID) -> Path:
        path = self.root / f"{inspection_id.hex}.json"
        if path.resolve() != path:
            raise OSError("Invalid inspection record location.")
        return path

    def add(self, record: StoredInspection) -> None:
        if self._read_only:
            raise OSError("Inspection storage is read-only.")
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

    def read_record(self, inspection_id: UUID) -> StoredInspection | None:
        """Read a file without applying retention, including for explicit imports."""
        try:
            data = self._path(inspection_id).read_bytes()
        except FileNotFoundError:
            return None
        record = decode_stored_inspection(data)
        if record.inspection_id != inspection_id:
            raise ValueError("Invalid inspection record identity.")
        return record

    def get(self, inspection_id: UUID) -> StoredInspection | None:
        record = self.read_record(inspection_id)
        cutoff = retention_cutoff(self.retention_seconds)
        if record is not None and cutoff is not None and record.created_at <= cutoff:
            return None
        return record

    def record_ids(self) -> Iterator[UUID]:
        if not self.root.exists():
            return
        for path in self.root.iterdir():
            if path.suffix != ".json" or len(path.stem) != 32:
                continue
            try:
                inspection_id = UUID(hex=path.stem)
            except ValueError:
                continue
            if path.stem == inspection_id.hex:
                yield inspection_id

    def _matching(self, filters: HistoryFilter, cutoff: datetime | None) -> Iterator[StoredInspection]:
        ids = (filters.inspection_id,) if filters.inspection_id is not None else self.record_ids()
        for inspection_id in ids:
            record = self.read_record(inspection_id)
            if record is not None and matches(record, filters, cutoff):
                yield record

    @contextmanager
    def records(self, filters: HistoryFilter, order: Order) -> Iterator[Iterator[StoredInspection]]:
        records = sorted(
            self._matching(filters, retention_cutoff(self.retention_seconds)),
            key=lambda record: (record.created_at.astimezone(UTC), record.inspection_id),
            reverse=order == "newest",
        )
        yield iter(records)

    def list(self, query: HistoryQuery) -> HistoryPage:
        with self.records(query, query.order) as records:
            items = list(records)
        return HistoryPage(total=len(items), limit=query.limit, offset=query.offset,
                           items=tuple(items[query.offset:query.offset + query.limit]))

    def summary(self, filters: StatisticsFilter) -> StatisticsSummary:
        return summarize(self._matching(filters, retention_cutoff(self.retention_seconds)), filters.mode)

    def defects(self, filters: StatisticsFilter) -> DefectStatistics:
        return count_defects(self._matching(filters, retention_cutoff(self.retention_seconds)), filters.mode)

    def trends(self, filters: StatisticsFilter) -> TrendStatistics:
        return count_days(self._matching(filters, retention_cutoff(self.retention_seconds)), filters.mode)
