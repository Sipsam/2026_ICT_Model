"""CLI presentation and configuration for local dataset ingestion."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Annotated, NoReturn

import typer

from ..config.settings import Credentials, SetupError, merge_settings, validate_settings
from ..pipeline.model import PipelineError
from .ingestion import Acquisition, has_dataset, publish_dataset, storage_error, verify_dataset
from .model import DataError, Progress, Selection, load_selection
from .providers import AIHubProvider
from .stages import limited_progress

data_app = typer.Typer(help="Acquire and verify the selected AI Hub SMT dataset.", no_args_is_help=True)


def dataset_paths(
    workspace: Path, selection: Selection, credentials: Credentials,
    raw_path: Path | None, data_path: Path | None,
) -> tuple[Path, Path]:
    raw_root = (workspace / (raw_path or selection.raw_root)).resolve()
    dataset = (workspace / (data_path or selection.extracted_root)).resolve()
    credentials.protect_path(str(raw_root), "raw path")
    credentials.protect_path(str(dataset), "dataset path")
    if raw_root.is_relative_to(dataset) or dataset.is_relative_to(raw_root):
        raise DataError("Use separate, non-overlapping directories for archives and extracted data.")
    return raw_root, dataset


def provider_for(credentials: Credentials, environ: Mapping[str, str]) -> AIHubProvider:
    return AIHubProvider(
        credentials.values.get("AIHUB_API_KEY"), windows=os.name == "nt",
        distribution=environ.get("ICT_WSL_DISTRIBUTION"),
    )


def show_download_plan(acquisition: Acquisition, missing: tuple[str, ...]) -> None:
    for split in acquisition.selection.splits:
        if split.name in missing:
            typer.echo(
                f"AI Hub {acquisition.selection.dataset_key} {split.name}   {len(split.file_keys)} selected file keys"
                f"   up to {split.max_download_bytes / 1024**3:g} GiB"
            )
    if missing:
        typer.echo(f"Archive destination   {acquisition.cache}")


def _progress(label: str, units: str = "bytes") -> Progress:
    started = monotonic()

    def report(done: int, total: int | None) -> None:
        elapsed = monotonic() - started
        eta = f"{elapsed * (total - done) / done:.1f}s" if total is not None and done else "unknown"
        expected = total if total is not None else "?"
        typer.echo(f"{label}   {done}/{expected} {units}   elapsed {elapsed:.1f}s   ETA {eta}")

    return limited_progress(report)


@dataclass(frozen=True)
class DataSession:
    workspace: Path
    dataset: Path
    acquisition: Acquisition
    provider: AIHubProvider
    existing_only: bool

    def existing(self) -> bool:
        return self.existing_only or has_dataset(self.dataset, self.acquisition.selection)


def _fail(exc: Exception) -> NoReturn:
    if isinstance(exc, OSError):
        exc = storage_error(exc)
    elif isinstance(exc, (RuntimeError, ValueError)):
        exc = DataError("Cannot resolve dataset paths. Check the configured directories.")
    typer.echo(f"Dataset operation failed. {exc}", err=True)
    raise typer.Exit(code=1) from None


def _cancel() -> NoReturn:
    typer.echo("Dataset operation cancelled. Rerun the command to continue.", err=True)
    raise typer.Exit(code=130) from None


@data_app.callback()
def configure(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(help="Read setup settings from this YAML file.")] = None,
    data_path: Annotated[Path | None, typer.Option(help="Existing dataset or new extraction destination.")] = None,
    raw_dir: Annotated[Path | None, typer.Option(help="Root for legacy ZIPs and new archive checkpoints.")] = None,
) -> None:
    """Resolve ingestion paths and credentials."""
    from ..config.storage import SetupFiles

    try:
        root = Path.cwd()
        files = SetupFiles.read(root, config)
        credentials = files.credentials(os.environ)
        settings_values = merge_settings(files.config_text, os.environ, {})
        settings_values.setdefault("dataset_source", "aihub")
        selection = load_selection()
        if data_path is not None or settings_values.get("data_path") is None:
            settings_values["data_path"] = str(data_path or selection.extracted_root)
        settings = validate_settings(settings_values, credentials)
        raw_root, dataset = dataset_paths(root, selection, credentials, raw_dir, Path(settings.data_path) if settings.data_path is not None else None)
        ctx.obj = DataSession(
            root, dataset, Acquisition(selection, raw_root), provider_for(credentials, os.environ),
            settings.dataset_source == "existing",
        )
    except (DataError, SetupError, OSError, RuntimeError, ValueError) as exc:
        _fail(exc)


@data_app.command(help="Show the selected dataset and local archive availability.")
def status(ctx: typer.Context) -> None:
    session: DataSession = ctx.obj
    acquisition = session.acquisition
    try:
        typer.echo(f"Dataset {acquisition.selection.dataset_key}   {session.dataset}")
        if session.existing():
            report = verify_dataset(session.dataset, acquisition.selection)
            _show_dataset(report.samples)
            return
        archives = acquisition.find()
        for split in acquisition.selection.splits:
            state = "available" if split.name in archives.by_split else "missing or incomplete"
            typer.echo(f"{split.name} archives   {state}")
    except (DataError, OSError) as exc:
        _fail(exc)


@data_app.command(help="Verify local inputs and preview or approve missing transfers.")
def download(
    ctx: typer.Context,
    approve: Annotated[bool, typer.Option("--approve-download", help="Authorize the listed AI Hub transfers.")] = False,
) -> None:
    session: DataSession = ctx.obj
    acquisition = session.acquisition
    try:
        if session.existing():
            _show_existing(session)
            return
        if not approve:
            missing = acquisition.missing(acquisition.find(crc_local=True))
            if missing:
                show_download_plan(acquisition, missing)
                typer.echo("Preview only. Pass --approve-download to authorize transfer.")
                return
            typer.echo("Selected archives verified.")
            return
        acquisition.acquire(
            session.provider, lambda split: _progress(f"Download {split}"), approved=True,
            show_plan=lambda missing: show_download_plan(acquisition, missing),
        )
        typer.echo("Selected archives verified.")
    except KeyboardInterrupt:
        _cancel()
    except (DataError, OSError, PipelineError) as exc:
        _fail(exc)


@data_app.command(help="Extract verified local archives to a new English-layout directory.")
def extract(ctx: typer.Context) -> None:
    session: DataSession = ctx.obj
    acquisition = session.acquisition
    try:
        if session.existing():
            _show_existing(session)
            return
        archives = acquisition.find()
        missing = acquisition.missing(archives)
        if missing:
            raise DataError(f"Selected archives are missing or incomplete for {', '.join(missing)}.")
        typer.echo(f"Extraction destination   {session.dataset}")
        report = publish_dataset(session.workspace, session.dataset, archives, acquisition.selection, _progress("Extract"))
        _show_dataset(report.samples)
    except KeyboardInterrupt:
        _cancel()
    except (DataError, OSError, PipelineError) as exc:
        _fail(exc)


@data_app.command(help="Verify extracted samples or the selected local archives.")
def verify(ctx: typer.Context) -> None:
    session: DataSession = ctx.obj
    acquisition = session.acquisition
    try:
        if session.existing():
            _show_existing(session, show_hash=True)
            return
        missing = acquisition.missing(acquisition.find(crc_local=True))
        if missing:
            raise DataError(f"Selected archives are missing or incomplete for {', '.join(missing)}.")
        typer.echo("Selected archives verified.")
    except (DataError, OSError) as exc:
        _fail(exc)


def _show_dataset(samples: dict[str, int]) -> None:
    typer.echo("Dataset verified   " + "   ".join(f"{name} {count} samples" for name, count in samples.items()))


def _show_existing(session: DataSession, *, show_hash: bool = False) -> None:
    report = verify_dataset(
        session.dataset, session.acquisition.selection, checksum=True,
        progress=_progress("Verify dataset", "folders"),
    )
    _show_dataset(report.samples)
    if show_hash:
        typer.echo(f"Dataset SHA-256   {report.fingerprint}")
