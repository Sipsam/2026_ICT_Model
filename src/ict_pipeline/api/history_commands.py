"""Local inspection history commands that do not start inference."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from time import monotonic

import typer
from pydantic import ValidationError

from ..config.settings import Credentials, SetupError
from ..config.storage import SetupFiles
from ..data.preprocessing.common import require_separate_output
from ..data.preprocessing.console import fresh_output
from ..inspection.contracts import InspectionError, Modality, Outcome
from ..inspection.service import local_path
from ..models.contracts import ModelName
from .history import ExportFormat, HistoryQuery, Mode, Order
from .history_export import write_history
from .history_import import import_files
from .schemas import StoredInspection
from .settings import APISettings, load_settings
from .sqlite_repository import SQLiteInspectionRepository
from .sqlite_schema import InspectionDatabaseError
from .storage import database_path, open_repository, storage_root

history_app = typer.Typer(help="List, export, import, and manage stored inspections.", no_args_is_help=True)


@history_app.callback()
def history_options(
    ctx: typer.Context,
    created_from: str | None = typer.Option(None, help="Inclusive timestamp with a UTC offset."),
    created_before: str | None = typer.Option(None, help="Exclusive timestamp with a UTC offset."),
    status: Outcome | None = None,
    defect_class: str | None = None,
    model_id: str | None = None,
    model: ModelName | None = None,
    mode: Mode = "all",
    order: Order = "newest",
    modality: Modality | None = None,
) -> None:
    """Filters and ordering for list and export."""
    try:
        ctx.obj = HistoryQuery.model_validate({
            "created_from": created_from, "created_before": created_before, "status": status,
            "defect_class": defect_class, "model_id": model_id, "model": model, "mode": mode, "order": order,
            "modality": modality,
        })
    except ValidationError:
        raise typer.BadParameter("Invalid history filters. Supply ordered timestamps with UTC offsets and canonical IDs.") from None


@contextmanager
def _operation() -> Iterator[tuple[APISettings, Credentials]]:
    try:
        settings = load_settings()
        credentials = SetupFiles.read(Path.cwd(), None).credentials(os.environ)
        if token := os.environ.get("ICT_API_TOKEN"):
            credentials.add("ICT_API_TOKEN", token)
        yield settings, credentials
    except (InspectionDatabaseError, SetupError, InspectionError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except (OSError, ValueError, RuntimeError):
        typer.echo("History operation failed. Check the configuration, inputs, and storage permissions.", err=True)
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        typer.echo("History operation interrupted.", err=True)
        raise typer.Exit(130) from None


@history_app.command("list")
def list_history(
    ctx: typer.Context, limit: int = typer.Option(100, min=1, max=500),
    offset: int = typer.Option(0, min=0, max=2**63 - 1),
) -> None:
    """Print one JSON page and its total matching count."""
    with _operation() as (settings, credentials):
        repository = open_repository(settings, credentials, read_only=True)
        page = repository.list(ctx.obj.model_copy(update={"limit": limit, "offset": offset}))
        typer.echo(page.model_dump_json())


def _observed_records(records: Iterator[StoredInspection]) -> Iterator[StoredInspection]:
    started = last = monotonic()
    count = 0
    typer.echo("export inspection history", err=True)
    for count, record in enumerate(records, 1):
        yield record
        now = monotonic()
        if now - last >= 1:
            typer.echo(f"export inspection history {count} records   elapsed {now - started:.1f}s", err=True)
            last = now
    typer.echo(f"export inspection history {count} records   elapsed {monotonic() - started:.1f}s", err=True)


@history_app.command("export")
def export_history(ctx: typer.Context, output_dir: Path, format: ExportFormat = "json") -> None:
    """Export all matching inspections to a fresh output directory."""
    with _operation() as (settings, credentials):
        output = local_path(output_dir, Path.cwd(), credentials)
        require_separate_output(settings.model_root, output)
        root = storage_root(settings, credentials)
        repository = open_repository(settings, credentials, read_only=True)
        target = f"history.{format}"
        with fresh_output(output, root) as working:
            with repository.records(ctx.obj, ctx.obj.order) as records, (
                (working / target).open("x", encoding="utf-8", newline="")
            ) as stream:
                write_history(stream, _observed_records(records), format)
        typer.echo(str(output / target))


def _maintenance_path(ctx: typer.Context, settings: APISettings, credentials: Credentials) -> Path:
    if ctx.obj != HistoryQuery():
        raise typer.BadParameter("History filters and ordering apply to list and export.")
    if settings.storage_backend != "sqlite":
        raise typer.BadParameter("Set storage_backend to sqlite in ICT_API_CONFIG.")
    return database_path(settings, credentials)


@history_app.command("import-files")
def import_history(
    ctx: typer.Context, source: Path,
    apply: bool = typer.Option(False, "--apply/--dry-run", help="Insert new records or preview the import."),
) -> None:
    """Validate and import versioned JSON records from the file repository."""
    started = last = monotonic()

    def progress(stage: str, done: int, total: int) -> None:
        nonlocal last
        now = monotonic()
        if done in (0, total) or now - last >= 1:
            elapsed = now - started
            eta = f"{elapsed * (total - done) / done:.1f}s" if done else "unknown"
            typer.echo(f"{stage} {done}/{total}   elapsed {elapsed:.1f}s   ETA {eta}", err=True)
            last = now

    with _operation() as (settings, credentials):
        path = _maintenance_path(ctx, settings, credentials)
        source = local_path(source, Path.cwd(), credentials)
        if not source.is_dir():
            raise typer.BadParameter("Select the directory containing file-backed inspection records.")
        typer.echo(f"{'Import' if apply else 'Preview'} inspection files from {source} into {path}", err=True)
        destination = None
        if apply or path.exists() or path.is_symlink():
            destination = SQLiteInspectionRepository(path, read_only=not apply, credentials=credentials)
        report = import_files(source, destination, dry_run=not apply, credentials=credentials, progress=progress)
        typer.echo(json.dumps(asdict(report)))
        if report.issues:
            raise typer.Exit(1)


@history_app.command("purge")
def purge_history(
    ctx: typer.Context,
    approve_purge: bool = typer.Option(False, "--approve-purge", help="Delete expired rows in the configured database."),
) -> None:
    """Preview physical cleanup of logically expired SQLite inspections."""
    with _operation() as (settings, credentials):
        path = _maintenance_path(ctx, settings, credentials)
        preview = SQLiteInspectionRepository(path, retention_seconds=settings.retention_seconds, read_only=True)
        expired = preview.purge_expired()
        typer.echo(f"{expired} expired inspection rows in {path}", err=True)
        deleted = 0
        if approve_purge and expired:
            repository = SQLiteInspectionRepository(path, retention_seconds=settings.retention_seconds)
            deleted = repository.purge_expired(approve=True)
        typer.echo(json.dumps({"expired": expired, "deleted": deleted}))
