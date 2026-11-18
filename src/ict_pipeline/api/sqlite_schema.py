"""Transactional schema upgrades for the SQLite inspection store."""

from sqlite3 import Connection

APPLICATION_ID = 0x49435449  # ICTI
MIGRATIONS = (
    {
        "inspections": """CREATE TABLE inspections (
    inspection_id TEXT NOT NULL PRIMARY KEY,
    created_at TEXT NOT NULL,
    schema_version INTEGER NOT NULL CHECK (schema_version = 1),
    result_schema_version INTEGER NOT NULL CHECK (result_schema_version = 1),
    model_id TEXT NOT NULL,
    detector_model TEXT,
    status TEXT NOT NULL CHECK (status IN ('pass', 'defect_detected', 'review_required', 'inference_failed')),
    mode TEXT NOT NULL CHECK (mode IN ('real', 'mock')),
    record_json TEXT NOT NULL
)""",
        "detections": """CREATE TABLE detections (
    inspection_id TEXT NOT NULL REFERENCES inspections (inspection_id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    class_id INTEGER NOT NULL,
    class_name TEXT NOT NULL,
    source_category_id INTEGER NOT NULL,
    confidence REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    x_min REAL NOT NULL CHECK (x_min >= 0),
    y_min REAL NOT NULL CHECK (y_min >= 0),
    x_max REAL NOT NULL CHECK (x_max > x_min),
    y_max REAL NOT NULL CHECK (y_max > y_min),
    PRIMARY KEY (inspection_id, ordinal)
)""",
    },
    {
        "inspections_time": "CREATE INDEX inspections_time ON inspections (created_at, inspection_id)",
        "inspections_mode": "CREATE INDEX inspections_mode ON inspections (mode, created_at, inspection_id)",
        "inspections_status": "CREATE INDEX inspections_status ON inspections (status, created_at, inspection_id)",
        "inspections_model_id": "CREATE INDEX inspections_model_id ON inspections (model_id, created_at, inspection_id)",
        "inspections_detector": "CREATE INDEX inspections_detector ON inspections (detector_model, created_at, inspection_id)",
        "detections_class": "CREATE INDEX detections_class ON detections (class_name, inspection_id)",
    },
)
SCHEMA_VERSION = len(MIGRATIONS)


class InspectionDatabaseError(OSError):
    """A storage error safe to display without SQLite exception details."""


def check_identity(connection: Connection) -> int:
    application_id = connection.execute("PRAGMA application_id").fetchone()[0]
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if application_id == APPLICATION_ID and 1 <= version <= SCHEMA_VERSION:
        return version
    if application_id == version == 0:
        return 0
    raise InspectionDatabaseError("Unsupported inspection database identity or schema version.")


def initialize_schema(connection: Connection, *, read_only: bool) -> None:
    version = check_identity(connection)
    expected = {name: sql for migration in MIGRATIONS[:version] for name, sql in migration.items()}
    actual = {row[0]: row[1] for row in connection.execute(
        "SELECT name, sql FROM sqlite_schema WHERE name NOT GLOB 'sqlite_*'",
    )}
    if actual != expected or (version == 0 and read_only):
        raise InspectionDatabaseError("Invalid inspection database schema.")
    if not read_only:
        for number, migration in enumerate(MIGRATIONS[version:], start=version + 1):
            for statement in migration.values():
                connection.execute(statement)
            connection.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {number}")
