"""Open the configured inspection store without initializing any models."""

from pathlib import Path

from ..config.settings import Credentials
from ..data.preprocessing.common import require_separate_output
from ..data.preprocessing.console import protect_source_data
from .repository import FileInspectionRepository
from .settings import APISettings, TrustedModel
from .sqlite_repository import SQLiteInspectionRepository


def storage_root(settings: APISettings, credentials: Credentials) -> Path:
    root = settings.storage_root.resolve()
    credentials.protect_path(str(root), "inspection storage")
    protect_source_data(root, Path.cwd())
    if settings.storage_backend == "sqlite" or any(isinstance(spec, TrustedModel) for spec in settings.models.values()):
        require_separate_output(settings.model_root, root)
    return root


def database_path(settings: APISettings, credentials: Credentials) -> Path:
    path = storage_root(settings, credentials) / settings.database_name
    credentials.protect_path(str(path), "inspection database")
    protect_source_data(path, Path.cwd())
    require_separate_output(settings.model_root, path)
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
