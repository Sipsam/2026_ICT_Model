"""Explicit, restartable migration of file-backed inspection records."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..config.settings import Credentials
from ..models.contracts import Progress
from .repository import FileInspectionRepository
from .sqlite_repository import SQLiteInspectionRepository, prepare_record


@dataclass(frozen=True)
class ImportIssue:
    inspection_id: str
    reason: Literal["invalid_record", "id_conflict"]


@dataclass
class ImportReport:
    dry_run: bool
    scanned: int = 0
    inserted: int = 0
    would_insert: int = 0
    duplicates: int = 0
    issues: list[ImportIssue] = field(default_factory=list)


def import_files(
    source: Path, destination: SQLiteInspectionRepository | None, *, dry_run: bool,
    credentials: Credentials, progress: Progress,
) -> ImportReport:
    if not source.is_dir():
        raise ValueError("Select the directory containing file-backed inspection records.")
    if destination is None and not dry_run:
        raise ValueError("Select an SQLite destination for the import.")
    files = FileInspectionRepository(source, read_only=True)
    inspection_ids = sorted(files.record_ids())
    total = len(inspection_ids)
    report = ImportReport(dry_run=dry_run)
    stage = "preview inspection import" if dry_run else "import inspections"
    progress(stage, 0, total)
    for inspection_id in inspection_ids:
        try:
            record = files.read_record(inspection_id)
            if record is None:
                raise ValueError
        except (OSError, ValueError):
            report.issues.append(ImportIssue(str(inspection_id), "invalid_record"))
        else:
            try:
                if destination is None:
                    prepare_record(record, credentials)
                    status = "new"
                else:
                    status = destination.import_record(record, dry_run=dry_run)
            except ValueError:
                report.issues.append(ImportIssue(str(inspection_id), "invalid_record"))
            else:
                if status == "duplicate":
                    report.duplicates += 1
                elif status == "conflict":
                    report.issues.append(ImportIssue(str(inspection_id), "id_conflict"))
                elif dry_run:
                    report.would_insert += 1
                else:
                    report.inserted += 1
        report.scanned += 1
        progress(stage, report.scanned, total)
    return report
