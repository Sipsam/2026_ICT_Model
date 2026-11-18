"""SQLite inspection storage with short, per-operation connections."""

import json
import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import closing, contextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Literal
from uuid import UUID

from ..config.settings import Credentials
from .history import (
    DailyCount,
    DefectFrequency,
    DefectStatistics,
    HistoryFilter,
    HistoryPage,
    HistoryQuery,
    Order,
    StatisticsFilter,
    StatisticsSummary,
    TrendStatistics,
    retention_cutoff,
    utc_timestamp,
)
from .schemas import StoredInspection, decode_stored_inspection
from .sqlite_schema import SCHEMA_VERSION, InspectionDatabaseBusy, InspectionDatabaseError, check_identity, initialize_schema

DuplicateStatus = Literal["new", "duplicate", "conflict"]
RECORD_COLUMNS = (
    "inspection_id", "created_at", "schema_version", "result_schema_version", "model_id",
    "detector_model", "status", "mode",
)


def _record_fields(record: StoredInspection) -> tuple[str, str, int, int, str, str | None, str, str]:
    return (
        record.inspection_id.hex, utc_timestamp(record.created_at), record.schema_version,
        record.result.schema_version, record.model_id, record.result.provenance.model,
        record.result.outcome, record.result.provenance.mode,
    )


def _decode(row: sqlite3.Row) -> StoredInspection:
    try:
        record = decode_stored_inspection(row["record_json"])
        if _record_fields(record) != tuple(row[column] for column in RECORD_COLUMNS):
            raise ValueError
        if record.created_at.utcoffset() != timedelta(0):
            raise ValueError
    except (TypeError, ValueError):
        raise InspectionDatabaseError("Stored inspection record is invalid.") from None
    return record


def prepare_record(record: StoredInspection, credentials: Credentials) -> tuple[StoredInspection, str]:
    """Validate the complete record and normalize the storage timestamp."""
    document = record.model_dump(mode="json")
    document["created_at"] = utc_timestamp(record.created_at)
    if credentials.redact(document) != document:
        raise ValueError("Inspection record contains unsanitized credential material.")
    payload = json.dumps(document, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return decode_stored_inspection(payload), payload


class SQLiteInspectionRepository:
    def __init__(
        self, path: Path, *, retention_seconds: int | None = None, timeout: float = 5.0,
        read_only: bool = False, credentials: Credentials | None = None,
    ) -> None:
        self.path = path.absolute()
        self.retention_seconds = retention_seconds
        self._timeout = timeout
        self._read_only = read_only
        self._credentials = credentials if credentials is not None else Credentials.load({}, os.environ)
        if credentials is None and (token := os.environ.get("ICT_API_TOKEN")):
            self._credentials.add("ICT_API_TOKEN", token)
        if not read_only:
            self.path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        with self._connect(create=not read_only) as connection:
            with connection:
                connection.execute("BEGIN" if read_only else "BEGIN IMMEDIATE")
                initialize_schema(connection, read_only=read_only)
            if not read_only and connection.execute("PRAGMA journal_mode = WAL").fetchone()[0] != "wal":
                raise InspectionDatabaseError("Inspection database requires WAL on a local filesystem.")

    def _check_paths(self) -> None:
        for suffix in ("", "-wal", "-shm", "-journal"):
            path = self.path.with_name(self.path.name + suffix)
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise InspectionDatabaseError("Use an inspection database and sidecars without filesystem links.")

    @contextmanager
    def _connect(self, *, create: bool = False) -> Iterator[sqlite3.Connection]:
        mode = "ro" if self._read_only else "rwc" if create else "rw"
        try:
            self._check_paths()
            with closing(sqlite3.connect(f"{self.path.as_uri()}?mode={mode}", uri=True,
                                         timeout=self._timeout)) as connection:
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA trusted_schema = OFF")
                if not self._read_only:
                    connection.execute("PRAGMA synchronous = FULL")
                yield connection
        except sqlite3.Error as exc:
            if getattr(exc, "sqlite_errorcode", 0) & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise InspectionDatabaseBusy("Inspection database is busy. Retry the operation.") from None
            raise InspectionDatabaseError("Inspection database operation failed. Check integrity and permissions.") from None
        except OSError as exc:
            if isinstance(exc, InspectionDatabaseError):
                raise
            raise InspectionDatabaseError("Cannot access inspection database files.") from None

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        with self._connect() as connection, connection:
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            version = check_identity(connection)
            if version == 0 or (version != SCHEMA_VERSION and not self._read_only):
                raise InspectionDatabaseError("Reopen the inspection store to check its schema.")
            yield connection

    def _insert(self, connection: sqlite3.Connection, record: StoredInspection, payload: str) -> None:
        connection.execute(
            """INSERT INTO inspections (
                inspection_id, created_at, schema_version, result_schema_version, model_id,
                detector_model, status, mode, record_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (*_record_fields(record), payload),
        )
        connection.executemany(
            """INSERT INTO detections (
                inspection_id, ordinal, class_id, class_name, source_category_id, confidence,
                x_min, y_min, x_max, y_max
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ((record.inspection_id.hex, ordinal, item.class_id, item.class_name, item.source_category_id,
              item.confidence, *item.bbox_xyxy) for ordinal, item in enumerate(record.result.detections)),
        )

    def add(self, record: StoredInspection) -> None:
        record, payload = prepare_record(record, self._credentials)
        with self._transaction(write=True) as connection:
            if self._find(connection, record.inspection_id) is not None:
                raise InspectionDatabaseError("Inspection ID already exists.")
            self._insert(connection, record, payload)

    def _find(
        self, connection: sqlite3.Connection, inspection_id: UUID, *, apply_retention: bool = False,
    ) -> StoredInspection | None:
        cutoff = retention_cutoff(self.retention_seconds) if apply_retention else None
        if cutoff is None:
            row = connection.execute("SELECT * FROM inspections WHERE inspection_id = ?",
                                     (inspection_id.hex,)).fetchone()
        else:
            row = connection.execute("SELECT * FROM inspections WHERE inspection_id = ? AND created_at > ?",
                                     (inspection_id.hex, utc_timestamp(cutoff))).fetchone()
        return _decode(row) if row is not None else None

    def get(self, inspection_id: UUID) -> StoredInspection | None:
        with self._transaction() as connection:
            return self._find(connection, inspection_id, apply_retention=True)

    def read_record(self, inspection_id: UUID) -> StoredInspection | None:
        """Read a record for internal deduplication, including retained expired IDs."""
        with self._transaction() as connection:
            return self._find(connection, inspection_id)

    def import_record(self, record: StoredInspection, *, dry_run: bool = False) -> DuplicateStatus:
        """Classify or insert a source record. Expired IDs still participate in deduplication."""
        record, payload = prepare_record(record, self._credentials)
        with self._transaction(write=not dry_run) as connection:
            existing = self._find(connection, record.inspection_id)
            if existing is not None:
                return "duplicate" if existing == record else "conflict"
            if not dry_run:
                self._insert(connection, record, payload)
        return "new"

    def _where(self, filters: HistoryFilter) -> tuple[str, tuple[str, ...]]:
        clauses, params = [], []
        cutoff = retention_cutoff(self.retention_seconds)
        for clause, value in (
            ("i.created_at > ?", utc_timestamp(cutoff) if cutoff is not None else None),
            ("i.created_at >= ?", utc_timestamp(filters.created_from) if filters.created_from is not None else None),
            ("i.created_at < ?", utc_timestamp(filters.created_before) if filters.created_before is not None else None),
            ("i.status = ?", filters.status),
            ("i.inspection_id = ?", filters.inspection_id.hex if filters.inspection_id is not None else None),
            ("i.mode = ?", filters.mode if filters.mode != "all" else None),
            ("i.model_id = ?", filters.model_id),
            ("i.detector_model = ?", filters.model),
            ("i.inspection_id IN (SELECT inspection_id FROM detections WHERE class_name = ?)", filters.defect_class),
        ):
            if value is not None:
                clauses.append(clause)
                params.append(value)
        return (" WHERE " + " AND ".join(clauses) if clauses else ""), tuple(params)

    @staticmethod
    def _ordering(order: Order) -> str:
        direction = "DESC" if order == "newest" else "ASC"
        return f" ORDER BY i.created_at {direction}, i.inspection_id {direction}"

    def list(self, query: HistoryQuery) -> HistoryPage:
        where, params = self._where(query)
        ordering = self._ordering(query.order)
        with self._transaction() as connection:
            count = connection.execute(f"SELECT COUNT(*) FROM inspections i{where}", params).fetchone()[0]
            rows = connection.execute(f"SELECT i.* FROM inspections i{where}{ordering} LIMIT ? OFFSET ?",
                                      (*params, query.limit, query.offset))
            return HistoryPage(total=count, limit=query.limit, offset=query.offset,
                               items=tuple(_decode(row) for row in rows))

    @contextmanager
    def records(self, filters: HistoryFilter, order: Order) -> Iterator[Iterator[StoredInspection]]:
        where, params = self._where(filters)
        with self._transaction() as connection:
            rows = connection.execute(f"SELECT i.* FROM inspections i{where}{self._ordering(order)}", params)
            yield map(_decode, rows)

    def summary(self, filters: StatisticsFilter) -> StatisticsSummary:
        where, params = self._where(filters)
        with self._transaction() as connection:
            counts = {}
            for field in ("status", "model_id"):
                counts[field] = dict(connection.execute(
                    f"SELECT i.{field}, COUNT(*) FROM inspections i{where} GROUP BY i.{field} ORDER BY i.{field}",
                    params,
                ))
            detections = connection.execute(
                f"""SELECT COUNT(*) FROM inspections i JOIN detections d ON d.inspection_id = i.inspection_id
                    {where}""", params,
            ).fetchone()[0]
            return StatisticsSummary(
                mode=filters.mode, total_inspections=sum(counts["status"].values()), total_detections=detections,
                by_status=counts["status"], by_model=counts["model_id"],
            )

    def defects(self, filters: StatisticsFilter) -> DefectStatistics:
        where, params = self._where(filters)
        with self._transaction() as connection:
            rows = connection.execute(
                f"""SELECT d.class_name, COUNT(DISTINCT i.inspection_id), COUNT(*)
                    FROM inspections i JOIN detections d ON d.inspection_id = i.inspection_id
                    {where} GROUP BY d.class_name ORDER BY d.class_name""", params,
            )
            return DefectStatistics(mode=filters.mode, items=tuple(
                DefectFrequency(class_name=name, inspection_count=inspections, detection_count=detections)
                for name, inspections, detections in rows
            ))

    def trends(self, filters: StatisticsFilter) -> TrendStatistics:
        where, params = self._where(filters)
        with self._transaction() as connection:
            rows = connection.execute(
                f"""SELECT substr(i.created_at, 1, 10) AS day, COUNT(*)
                    FROM inspections i{where} GROUP BY day ORDER BY day""", params,
            )
            return TrendStatistics(mode=filters.mode, items=tuple(
                DailyCount(date=date.fromisoformat(day), inspection_count=count) for day, count in rows
            ))

    def purge_expired(self, *, approve: bool = False) -> int:
        """Preview or delete expired database rows. Foreign keys remove detection rows."""
        if self.retention_seconds is None:
            raise ValueError("Configure retention before running a purge.")
        cutoff = retention_cutoff(self.retention_seconds)
        if cutoff is None:
            return 0
        params = (utc_timestamp(cutoff),)
        with self._transaction(write=approve) as connection:
            if approve:
                return connection.execute("DELETE FROM inspections WHERE created_at <= ?", params).rowcount
            return connection.execute("SELECT COUNT(*) FROM inspections WHERE created_at <= ?", params).fetchone()[0]
