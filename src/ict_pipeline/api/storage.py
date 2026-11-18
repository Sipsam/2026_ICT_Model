"""Open the configured inspection store without initializing any models."""

from pathlib import Path

from ..config.settings import Credentials
from ..data.preprocessing.common import require_separate_output
from ..data.preprocessing.console import protect_source_data
from ..streaming.settings import DirectorySettings
from .repository import FileInspectionRepository
from .settings import APISettings, TrustedModel
from .sqlite_repository import SQLiteInspectionRepository


def storage_root(settings: APISettings, credentials: Credentials) -> Path:
    root = settings.storage_root.resolve()
    credentials.protect_path(str(root), "inspection storage")
    protect_source_data(root, Path.cwd())
    if settings.storage_backend == "sqlite" or any(isinstance(spec, TrustedModel) for spec in settings.models.values()):
        require_separate_output(settings.model_root, root)
    if settings.input_stream is not None and isinstance(settings.input_stream.camera, DirectorySettings):
        require_separate_output(settings.input_stream.camera.directory, root)
    return root


def database_path(settings: APISettings, credentials: Credentials) -> Path:
    path = storage_root(settings, credentials) / settings.database_name
    credentials.protect_path(str(path), "inspection database")
    protect_source_data(path, Path.cwd())
    require_separate_output(settings.model_root, path)
    if stream := settings.input_stream:
        sources = [stream.sensor.path] if stream.sensor is not None else []
        if isinstance(stream.camera, DirectorySettings) and stream.camera.manifest is not None:
            sources.append(stream.camera.manifest)
        database_files = {path.with_name(path.name + suffix).resolve() for suffix in ("", "-wal", "-shm", "-journal")}
        if any(source.resolve() in database_files for source in sources):
            raise ValueError("Choose a database path separate from replay inputs.")
    return path


def open_repository(
    settings: APISettings, credentials: Credentials, *, read_only: bool = False,
) -> FileInspectionRepository | SQLiteInspectionRepository:
    if settings.storage_backend == "sqlite":
        return SQLiteInspectionRepository(database_path(settings, credentials),
                                          retention_seconds=settings.retention_seconds,
                                          read_only=read_only, credentials=credentials)
    return FileInspectionRepository(storage_root(settings, credentials),
                                    retention_seconds=settings.retention_seconds, read_only=read_only)
