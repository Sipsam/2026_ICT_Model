"""Checkpoint completed leakage observations in an analysis-owned local database."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import Any

from ...pipeline.store import digest_json
from .common import require_separate_output
from .console import protect_source_data

APPLICATION_ID = 0x4943544C


class ObservationCache:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def load(self, index: int) -> dict[str, Any] | None:
        record = self.connection.execute(
            "SELECT payload, fingerprint FROM observations WHERE sample_index = ?", (index,),
        ).fetchone()
        if record is None:
            return None
        payload = json.loads(record[0])
        if digest_json(payload) != record[1]:
            raise ValueError("An observation checkpoint is corrupt. Use a fresh checkpoint path.")
        return payload

    def save(self, index: int, payload: dict[str, Any]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO observations VALUES (?, ?, ?)",
            (index, json.dumps(payload, separators=(",", ":")), digest_json(payload)),
        )
        if index % 32 == 31:
            self.connection.commit()

    def stage(self, status: str) -> None:
        self.connection.execute("UPDATE analysis SET status = ?", (status,))
        self.connection.commit()


@contextmanager
def open_observation_cache(path: Path | None, source: Path, identity: str) -> Iterator[ObservationCache | None]:
    if path is None:
        yield None
        return
    if path.is_symlink():
        raise ValueError("Use a checkpoint file that is not a link.")
    path = path.resolve()
    require_separate_output(source, path)
    protect_source_data(path, Path.cwd())
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb"):
            pass
        fresh = True
    except FileExistsError:
        fresh = False

    connection = sqlite3.connect(path)
    try:
        if fresh:
            connection.executescript(
                f"PRAGMA application_id = {APPLICATION_ID};"
                "CREATE TABLE analysis (identity TEXT NOT NULL, status TEXT NOT NULL);"
                "CREATE TABLE observations (sample_index INTEGER PRIMARY KEY, payload TEXT NOT NULL,"
                " fingerprint TEXT NOT NULL);"
            )
            connection.execute("INSERT INTO analysis VALUES (?, ?)", (identity, "pending"))
            connection.commit()
        elif (
            connection.execute("PRAGMA application_id").fetchone() != (APPLICATION_ID,)
            or connection.execute("SELECT identity FROM analysis").fetchall() != [(identity,)]
        ):
            raise ValueError("The checkpoint does not match this manifest, analysis configuration, or implementation.")
        cache = ObservationCache(connection)
        try:
            yield cache
        except BaseException:
            # These are complete, individually checksummed observations.
            with suppress(sqlite3.Error):
                cache.stage("interrupted")
            raise
    except sqlite3.Error:
        raise ValueError("Cannot use the observation checkpoint. Check the path or choose a fresh file.") from None
    finally:
        connection.close()
