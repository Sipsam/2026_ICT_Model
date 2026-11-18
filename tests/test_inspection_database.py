import csv
import json
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest
from typer.testing import CliRunner

from ict_pipeline.api.history import HistoryQuery, StatisticsFilter
from ict_pipeline.api.history_import import import_files
from ict_pipeline.api.repository import FileInspectionRepository
from ict_pipeline.api.schemas import StoredInspection
from ict_pipeline.api.sqlite_repository import SQLiteInspectionRepository
from ict_pipeline.api.sqlite_schema import APPLICATION_ID, MIGRATIONS, SCHEMA_VERSION, InspectionDatabaseError
from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.inspection.contracts import Detection, ImageSource, InspectionResult, InspectionThresholds, ModelProvenance
from ict_pipeline.models.contracts import ModelName

NOW = datetime(2025, 1, 2, 0, 0, 0, 123, tzinfo=UTC)
FINGERPRINT = "1" * 64
COLD_SOLDER = Detection(class_id=1, class_name="cold_solder", source_category_id=23,
                        bbox_xyxy=(1, 1, 2, 2), confidence=0.9)
SHORT = Detection(class_id=3, class_name="short", source_category_id=25,
                  bbox_xyxy=(2, 2, 3, 3), confidence=0.8)


def inspection(
    *, mode: Literal["real", "mock"] = "real", created: datetime = NOW,
    detections: tuple[Detection, ...] = (COLD_SOLDER,), identifier: UUID | None = None,
) -> StoredInspection:
    provenance = ModelProvenance(
        mode="real", model=ModelName("yolo"), checkpoint_kind="best", checkpoint_fingerprint=FINGERPRINT,
        run_metadata_fingerprint=FINGERPRINT, device="cpu", image_size=640,
    ) if mode == "real" else ModelProvenance(mode="mock", mock_scenario="defect" if detections else "empty")
    return StoredInspection(
        inspection_id=identifier or uuid4(), created_at=created, model_id="simulator" if mode == "mock" else "line-3",
        result=InspectionResult(
            request_id="=request", source_id="camera-3", sensor_source_id="session-19", metadata={"lot": "A37"},
            image=ImageSource(format="PNG", sha256=FINGERPRINT, width=8, height=8),
            provenance=provenance, thresholds=InspectionThresholds(), detections=detections,
            outcome="defect_detected" if mode == "real" and detections else "review_required",
            reason="simulated_prediction" if mode == "mock" else "detections_retained" if detections else "no_detections",
        ),
    )


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(path)) as database, database:
        yield database


@pytest.mark.parametrize("backend", ["file", "sqlite"])
def test_history_filters_order_pagination_and_statistics(tmp_path: Path, backend: str) -> None:
    repo = (SQLiteInspectionRepository(tmp_path / "history.sqlite3") if backend == "sqlite"
            else FileInspectionRepository(tmp_path))
    first = inspection(
        created=NOW.astimezone(timezone(timedelta(hours=-5))), detections=(COLD_SOLDER, COLD_SOLDER, SHORT),
        identifier=UUID("00000000-0000-4000-8000-000000000001"),
    )
    second = inspection(detections=(), identifier=UUID("00000000-0000-4000-8000-000000000002"))
    mock = inspection(mode="mock", created=NOW - timedelta(microseconds=1))
    for record in (second, first, mock):
        repo.add(record)

    page = repo.list(HistoryQuery(limit=1, offset=1))
    assert (page.total, page.limit, page.offset, page.items) == (3, 1, 1, (first,))
    assert repo.list(HistoryQuery(order="oldest")).items == (mock, first, second)
    assert repo.list(HistoryQuery(offset=2**63 - 1)).items == ()
    assert repo.list(HistoryQuery(
        created_from=NOW.isoformat(), created_before=NOW + timedelta(microseconds=1),
        status="defect_detected", model="yolo", model_id="line-3", defect_class="short", mode="real",
    )).items == (first,)
    assert repo.list(HistoryQuery(created_before=NOW)).items == (mock,)
    assert repo.list(HistoryQuery(mode="mock", model="yolo")).total == 0
    assert repo.list(HistoryQuery(status="review_required")).items == (second, mock)

    summary = repo.summary(StatisticsFilter())
    assert (summary.mode, summary.total_inspections, summary.total_detections) == ("real", 2, 3)
    assert summary.by_status == {"defect_detected": 1, "review_required": 1}
    assert summary.by_model == {"line-3": 2}
    assert repo.summary(StatisticsFilter(mode="all")).by_model == {"line-3": 2, "simulator": 1}
    assert repo.summary(StatisticsFilter(mode="mock")).total_inspections == 1
    assert repo.summary(StatisticsFilter(model_id="unknown")).total_inspections == 0
    defects = repo.defects(StatisticsFilter(defect_class="cold_solder")).items
    assert [(item.class_name, item.inspection_count, item.detection_count) for item in defects] == [
        ("cold_solder", 1, 2), ("short", 1, 1),
    ]
    assert repo.defects(StatisticsFilter(model_id="unknown")).items == ()
    trends = repo.trends(StatisticsFilter(mode="all")).items
    assert [(str(item.date), item.inspection_count) for item in trends] == [("2025-01-02", 3)]
    assert repo.trends(StatisticsFilter(model_id="unknown")).items == ()


def test_persistence_with_concurrent_writes_and_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "a file #1.sqlite3"
    repo = SQLiteInspectionRepository(path)
    records = [inspection() for _ in range(20)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(repo.add, records))
        duplicate = inspection()
        inserts = [pool.submit(repo.add, duplicate) for _ in range(2)]
        outcomes = [insert.exception() for insert in inserts]
    assert sum(outcome is None for outcome in outcomes) == 1
    assert any(isinstance(outcome, InspectionDatabaseError) for outcome in outcomes)
    reopened = SQLiteInspectionRepository(path)
    assert reopened.list(HistoryQuery()).total == 21
    assert reopened.get(records[0].inspection_id) == records[0]
    with connect(path) as database:
        assert database.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert database.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert database.execute("PRAGMA foreign_key_check").fetchall() == []
        assert database.execute("SELECT count(*) FROM detections").fetchone()[0] == 21
        assert database.execute("SELECT DISTINCT created_at FROM inspections").fetchall() == [("2025-01-02T00:00:00.000123Z",)]


def test_database_lock_and_atomic_rollback(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    repo = SQLiteInspectionRepository(path, timeout=0.02)
    record = inspection(detections=(COLD_SOLDER, SHORT))
    with connect(path) as database:
        database.execute("BEGIN IMMEDIATE")
        assert repo.list(HistoryQuery()).total == 0
        with pytest.raises(InspectionDatabaseError, match="busy"):
            repo.add(record)
    with connect(path) as database:
        database.execute("""CREATE TRIGGER reject_box BEFORE INSERT ON detections
                            WHEN NEW.ordinal = 1 BEGIN SELECT RAISE(ABORT, 'test'); END""")
    with pytest.raises(InspectionDatabaseError, match="operation failed"):
        repo.add(record)
    assert repo.get(record.inspection_id) is None
    with connect(path) as database:
        assert database.execute("SELECT COUNT(*) FROM detections").fetchone()[0] == 0
        database.execute("DROP TRIGGER reject_box")
    repo.add(record)
    assert repo.get(record.inspection_id) == record


def test_retention_hides_records_until_approved_purge(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    repo = SQLiteInspectionRepository(path, retention_seconds=60)
    expired = inspection(created=datetime.now(UTC) - timedelta(minutes=3))
    active = inspection(created=datetime.now(UTC))
    repo.add(expired)
    repo.add(active)
    assert repo.get(expired.inspection_id) is None
    assert repo.list(HistoryQuery()).items == (active,)
    assert repo.summary(StatisticsFilter()).total_detections == 1
    assert repo.defects(StatisticsFilter()).items[0].detection_count == 1
    assert repo.trends(StatisticsFilter()).items[0].inspection_count == 1
    assert repo.purge_expired() == 1
    assert SQLiteInspectionRepository(path, read_only=True).get(expired.inspection_id) == expired
    assert repo.purge_expired(approve=True) == 1
    assert repo.purge_expired() == 0
    with connect(path) as database:
        assert database.execute("SELECT DISTINCT inspection_id FROM detections").fetchall() == [(active.inspection_id.hex,)]


def test_upgrade_preserves_records_and_read_only_does_not_upgrade(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    record = inspection()
    SQLiteInspectionRepository(path).add(record)
    with connect(path) as database:
        for index in MIGRATIONS[1]:
            database.execute(f"DROP INDEX {index}")
        database.execute("PRAGMA user_version = 1")
    assert SQLiteInspectionRepository(path, read_only=True).get(record.inspection_id) == record
    with connect(path) as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == 1
    upgraded = SQLiteInspectionRepository(path)
    assert upgraded.get(record.inspection_id) == record
    with connect(path) as database:
        assert database.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        assert database.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID


@pytest.mark.parametrize("kind", ["foreign", "future", "invalid_layout", "corrupt"])
def test_rejects_incompatible_database(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "history.sqlite3"
    if kind == "corrupt":
        path.write_bytes(b"not a database or a secret to include in errors")
    elif kind == "foreign":
        with connect(path) as database:
            database.execute("CREATE TABLE unrelated (value TEXT)")
    else:
        SQLiteInspectionRepository(path)
        with connect(path) as database:
            if kind == "future":
                database.execute("PRAGMA user_version = 999")
            else:
                database.execute("DROP INDEX inspections_time")
    before = path.read_bytes()
    with pytest.raises(InspectionDatabaseError):
        SQLiteInspectionRepository(path)
    assert path.read_bytes() == before


def test_corrupt_payload_and_known_credentials(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    credentials = Credentials.load({}, {"AIHUB_API_KEY": "known-secret"})
    repo = SQLiteInspectionRepository(path, credentials=credentials)
    record = inspection()
    record.result.metadata["secret_value"] = "known-secret"
    with pytest.raises(ValueError, match="unsanitized credential"):
        repo.add(record)
    assert repo.list(HistoryQuery()).total == 0
    record.result.metadata.clear()
    repo.add(record)
    with connect(path) as database:
        database.execute("UPDATE inspections SET record_json = '{}'")
    with pytest.raises(InspectionDatabaseError, match="record is invalid"):
        repo.get(record.inspection_id)
    with pytest.raises(InspectionDatabaseError, match="record is invalid"):
        repo.list(HistoryQuery())


def test_import_resumes_and_checks_conflicts_and_versions(tmp_path: Path) -> None:
    source = FileInspectionRepository(tmp_path / "files")
    path = tmp_path / "history.sqlite3"
    existing, conflict, new = (inspection() for _ in range(3))
    for record in (existing, conflict, new):
        source.add(record)
    bad = inspection()
    (source.root / f"{bad.inspection_id.hex}.json").write_text(
        bad.model_dump_json().replace('"schema_version":1', '"schema_version":2'), encoding="utf-8",
    )
    (source.root / ".pending-unpublished").write_text("partial", encoding="utf-8")
    snapshots = {path.name: path.read_bytes() for path in source.root.iterdir()}
    interrupted = SQLiteInspectionRepository(path)
    interrupted.add(existing)
    interrupted.add(inspection(detections=(), identifier=conflict.inspection_id))

    def progress(_stage: str, _done: int, _total: int) -> None:
        pass

    read_only = SQLiteInspectionRepository(path, read_only=True)
    preview = import_files(source.root, read_only, dry_run=True, credentials=Credentials(), progress=progress)
    assert (preview.scanned, preview.would_insert, preview.duplicates) == (4, 1, 1)
    assert {(issue.inspection_id, issue.reason) for issue in preview.issues} == {
        (str(bad.inspection_id), "invalid_record"), (str(conflict.inspection_id), "id_conflict"),
    }
    assert interrupted.get(new.inspection_id) is None
    report = import_files(source.root, interrupted, dry_run=False, credentials=Credentials(), progress=progress)
    assert (report.inserted, report.duplicates) == (1, 1)
    assert interrupted.get(conflict.inspection_id).result.detections == ()
    repeated = import_files(source.root, interrupted, dry_run=False, credentials=Credentials(), progress=progress)
    assert (repeated.inserted, repeated.duplicates) == (0, 2)
    assert {path.name: path.read_bytes() for path in source.root.iterdir()} == snapshots


def test_history_cli_preview_import_list_export_and_purge(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "records"
    config = tmp_path / "api.json"
    config.write_text(json.dumps({
        "storage_backend": "sqlite", "storage_root": str(root), "database_name": "local.sqlite3",
    }), encoding="utf-8")
    monkeypatch.setenv("ICT_API_CONFIG", str(config))
    source = FileInspectionRepository(tmp_path / "files")
    record = inspection()
    source.add(record)
    cli = CliRunner()
    preview = cli.invoke(app, ["history", "import-files", str(source.root)])
    assert preview.exit_code == 0, preview.output
    assert json.loads(preview.stdout)["would_insert"] == 1
    assert not root.exists()
    applied = cli.invoke(app, ["history", "import-files", str(source.root), "--apply"])
    assert applied.exit_code == 0, applied.output
    assert json.loads(applied.stdout)["inserted"] == 1
    page = cli.invoke(app, ["history", "--model-id", "line-3", "list", "--limit", "1"])
    assert page.exit_code == 0, page.output
    assert json.loads(page.stdout)["total"] == 1
    for format in ("json", "csv"):
        output = tmp_path / format
        command = ["history", "--mode", "real", "export", str(output), "--format", format]
        exported = cli.invoke(app, command)
        assert exported.exit_code == 0, exported.output
        with (output / f"history.{format}").open(encoding="utf-8", newline="") as stream:
            if format == "csv":
                (row,) = list(csv.DictReader(stream))
                assert row["request_id"] == "'=request"
                assert StoredInspection.model_validate_json(row["record_json"]) == record
            else:
                (item,) = json.load(stream)["items"]
                assert item["inspection_id"] == str(record.inspection_id)
        assert cli.invoke(app, command).exit_code == 1
    contents = json.loads(config.read_text(encoding="utf-8"))
    config.write_text(json.dumps({**contents, "retention_seconds": 1}), encoding="utf-8")
    preview_purge = cli.invoke(app, ["history", "purge"])
    assert preview_purge.exit_code == 0, preview_purge.output
    assert json.loads(preview_purge.stdout) == {"expired": 1, "deleted": 0}
    purge = cli.invoke(app, ["history", "purge", "--approve-purge"])
    assert purge.exit_code == 0, purge.output
    assert json.loads(purge.stdout) == {"expired": 1, "deleted": 1}
    assert source.get(record.inspection_id) == record
