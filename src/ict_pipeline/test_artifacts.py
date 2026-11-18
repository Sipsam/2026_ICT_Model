"""Lease and retire disposable pytest sessions in an isolated storage root."""

import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from getpass import getuser
from pathlib import Path
from shutil import rmtree
from time import time
from typing import Literal
from uuid import uuid4

from .disk import StorageError, require_space
from .pipeline.model import PipelineError
from .pipeline.store import RunStore, _is_link, _reject_link


Status = Literal["running", "failed", "passed", "interrupted"]
_MANIFEST = "session.json"
_ID = re.compile(r"[a-f0-9]{32}")


def artifact_root(parent: Path | None = None) -> Path:
    if parent is None:
        configured = os.environ.get("ICT_TEST_ARTIFACT_ROOT") or os.environ.get("PYTEST_DEBUG_TEMPROOT")
        parent = Path(configured) if configured else Path(tempfile.gettempdir())
        if not configured and os.name != "nt":
            parent /= f"ict-uid-{os.getuid()}"
            _reject_link(parent)
    root = parent.resolve() / "ict-test-artifacts-v1"
    _reject_link(root)
    _reject_link(root / "sessions")
    return root


def _owner() -> str:
    if os.name == "nt":
        return f"{os.environ.get('USERDOMAIN', '')}\\{getuser()}"
    return f"uid:{os.getuid()}"


@dataclass(frozen=True)
class Candidate:
    path: Path
    owner: str
    status: Status
    logical_bytes: int | None
    eligible: bool
    detail: str


@dataclass
class TestSession:
    __test__ = False
    store: RunStore
    status: Status = "running"
    keep_failed: bool = False

    @property
    def payload(self) -> Path:
        return self.store.directory / "files"

    def record(self) -> None:
        self.store.write_json(_MANIFEST, {
            "schema": 1, "purpose": "ict-pytest", "id": self.store.run_id,
            "owner": _owner(), "status": self.status, "updated": int(time()),
        })

    def discard_test(self, path: Path) -> None:
        if path.parent != self.payload / "pytest":
            raise PipelineError("The test directory is outside this session.")
        _discard_payload(path)


def _regular_files(directory: Path) -> Iterator[tuple[Path, os.stat_result]]:
    for path in directory.iterdir():
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            # pytest's *current pointers and fixture symlinks are leaves.
            continue
        if _is_link(info):
            raise PipelineError("Temporary storage contains a junction or other reparse point.")
        if stat.S_ISDIR(info.st_mode):
            yield from _regular_files(path)
        elif stat.S_ISREG(info.st_mode):
            yield path, info
        else:
            raise PipelineError("Temporary storage contains a special file.")


def _check_tree(directory: Path) -> int:
    _reject_link(directory)
    size = 0
    for path, info in _regular_files(directory):
        if path.name == "run.lock":
            with RunStore(path.parent.parent, path.parent.name).inspect() as active:
                if active:
                    raise PipelineError("A nested ICT job holds a run lock.")
        size += info.st_size
    return size


def _discard_payload(directory: Path) -> None:
    if directory.exists() or directory.is_symlink():
        _check_tree(directory)
        rmtree(directory)


@contextmanager
def open_test_session(root: Path, *, keep_failed: bool, limit: int = 8) -> Iterator[TestSession]:
    """Hold an OS lease until pytest has finished writing to this session."""
    if limit < 1:
        raise ValueError("The test session limit must be positive.")
    sessions = root / "sessions"
    try:
        require_space(root, label="pytest artifacts")
    except StorageError as exc:
        raise PipelineError(str(exc)) from None
    control = RunStore(root, "control")
    lease = RunStore(sessions, uuid4().hex)
    with ExitStack() as held:
        with control.lock(wait_seconds=5):
            sessions.mkdir(mode=0o700, exist_ok=True)
            if sum(1 for path in sessions.iterdir() if _ID.fullmatch(path.name)) >= limit:
                raise PipelineError("Test session limit reached. Preview retained sessions with ict clean or choose another root.")
            lease.directory.mkdir(mode=0o700)
            held.enter_context(lease.lock())
            session = TestSession(lease, keep_failed=keep_failed)
            # Publish ownership before creating disposable data, even if startup is interrupted.
            session.record()
        try:
            session.payload.mkdir()
            try:
                yield session
            finally:
                if session.status == "running":
                    session.status = "interrupted"
                if session.status == "passed" or not session.keep_failed:
                    try:
                        _discard_payload(session.payload)
                    except (OSError, PipelineError):
                        session.record()
                        raise
                else:
                    session.record()
        finally:
            # Serialize removal of the small lease directory with explicit cleanup.
            with control.lock(wait_seconds=5):
                held.close()
                if not session.payload.exists() and not session.payload.is_symlink():
                    (lease.directory / _MANIFEST).unlink(missing_ok=True)
                    lease.lock_path.unlink(missing_ok=True)
                    lease.directory.rmdir()


def _candidate(store: RunStore, *, active: bool, age_hours: float) -> Candidate | None:
    try:
        manifest = store.directory / _MANIFEST
        _reject_link(manifest)
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            not isinstance(record, dict) or record.get("schema") != 1 or record.get("purpose") != "ict-pytest"
            or record.get("id") != store.run_id or record.get("owner") != _owner()
            or record.get("status") not in ("running", "passed", "failed", "interrupted")
            or type(record.get("updated")) is not int
            or not store.lock_path.is_file()
        ):
            return None
    except (OSError, ValueError, PipelineError):
        return None
    status = record["status"]
    if active:
        return Candidate(store.directory, record["owner"], status, None, False, "active session")
    if status == "running":
        status = "interrupted"
    try:
        payload = store.directory / "files"
        _reject_link(payload)
        size = _check_tree(payload) if payload.exists() else 0
        entries = {path.name for path in store.directory.iterdir()}
        if entries - {_MANIFEST, "run.lock", "files"}:
            raise PipelineError("Unexpected entries beside session payload.")
    except (OSError, ValueError, PipelineError) as exc:
        detail = str(exc) if isinstance(exc, PipelineError) else "session tree is inaccessible"
        return Candidate(store.directory, record["owner"], status, None, False, detail)
    eligible = time() - record["updated"] >= age_hours * 3600
    return Candidate(store.directory, record["owner"], status, size, eligible, "eligible" if eligible else "too recent")


def preview_cleanup(root: Path, *, age_hours: float) -> Iterator[Candidate]:
    sessions = root / "sessions"
    _reject_link(sessions)
    if not sessions.is_dir():
        return
    with RunStore(root, "control").inspect() as busy:
        if busy:
            raise PipelineError("Test session registration is busy. Retry the preview.")
        for path in sorted(sessions.iterdir()):
            if not _ID.fullmatch(path.name):
                continue
            try:
                store = RunStore(sessions, path.name)
                with store.inspect() as active:
                    candidate = _candidate(store, active=active, age_hours=age_hours)
            except (OSError, PipelineError):
                continue
            if candidate:
                yield candidate


def delete_session(root: Path, session_id: str, *, age_hours: float) -> None:
    if not _ID.fullmatch(session_id):
        raise PipelineError("Specify an exact session ID from ict clean.")
    store = RunStore(root / "sessions", session_id)
    if not store.lock_path.is_file():
        raise PipelineError("No registered session with that ID.")
    with RunStore(root, "control").lock(wait_seconds=5):
        with store.inspect() as active:
            candidate = _candidate(store, active=active, age_hours=age_hours)
            if candidate is None or not candidate.eligible:
                detail = candidate.detail if candidate else "ownership could not be verified"
                raise PipelineError(f"Cannot delete session {session_id}. {detail}.")
            _discard_payload(store.directory / "files")
        (store.directory / _MANIFEST).unlink()
        store.lock_path.unlink()
        store.directory.rmdir()


def measure_test_files(directory: Path) -> tuple[int, int, int, int]:
    """Return logical bytes, unique file bytes, checkpoint names and checkpoint bytes."""
    logical = checkpoints = checkpoint_bytes = 0
    unique: dict[tuple[int, int] | Path, int] = {}
    for path, info in _regular_files(directory):
        logical += info.st_size
        unique[(info.st_dev, info.st_ino) if info.st_ino else path] = info.st_size
        if path.suffix == ".pt":
            checkpoints += 1
            checkpoint_bytes += info.st_size
    return logical, sum(unique.values()), checkpoints, checkpoint_bytes
