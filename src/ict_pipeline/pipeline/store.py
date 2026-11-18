"""Per-run locking, atomic state snapshots, and artifact fingerprints."""

import errno
import hashlib
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Annotated, BinaryIO, Literal

from pydantic import BaseModel, Field, ValidationError

from .model import PipelineError, valid_id

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


AttemptId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
Fingerprint = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Status = Literal["running", "completed", "failed", "interrupted"]


class StageRecord(BaseModel):
    status: Status
    attempt: AttemptId
    started_at: str
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    fingerprint: Fingerprint | None = None
    config_fingerprint: Fingerprint | None = None
    output_fingerprint: Fingerprint | None = None
    error: str | None = None


class RunState(BaseModel):
    schema_version: Literal[1] = 1
    run_id: str
    created_at: str
    updated_at: str
    status: Status = "running"
    selection: list[str]
    pending_force: list[str]
    stages: dict[str, StageRecord] = Field(default_factory=dict)


def _try_lock(stream: BinaryIO) -> bool:
    try:
        stream.seek(0)
        if os.name == "nt":
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        if exc.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
            return False
        raise
    return True


def _is_link(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def _reject_link(path: Path) -> None:
    try:
        linked = _is_link(path.lstat())
    except FileNotFoundError:
        return
    if linked:
        raise PipelineError("Run storage contains a symbolic link or junction. Use a regular run directory.")


class RunStore:
    def __init__(self, runs_dir: Path, run_id: str) -> None:
        if not valid_id(run_id):
            raise PipelineError("Run IDs use up to 64 lowercase letters, digits, underscores, or hyphens.")
        self.run_id = run_id
        self.directory = runs_dir / run_id
        self.state_path = self.directory / "state.json"
        self.lock_path = self.directory / "run.lock"
        _reject_link(runs_dir)
        _reject_link(self.directory)

    @contextmanager
    def lock(self) -> Iterator[None]:
        self.directory.mkdir(parents=True, exist_ok=True)
        _reject_link(self.lock_path)
        # Leave the lock file in place. The OS releases the lock when the
        # descriptor closes, including when the owning process is killed.
        with self.lock_path.open("a+b") as stream:
            if not _try_lock(stream):
                raise PipelineError(f"Run {self.run_id} is active. Inspect it with ict status.")
            yield

    def read(self) -> RunState | None:
        _reject_link(self.state_path)
        try:
            raw = self.state_path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            state = RunState.model_validate_json(raw, strict=True)
            names = [*state.stages, *state.selection, *state.pending_force]
            if (
                state.run_id != self.run_id
                or any(not valid_id(name) for name in names)
                or not state.selection
                or not set(state.pending_force).issubset(state.selection)
            ):
                raise ValueError
        except (ValidationError, ValueError):
            raise PipelineError("Run state is invalid. Restore a valid state.json or start a new run.") from None
        return state

    def status(self) -> RunState | None:
        # Read a stable snapshot after probing the lock, so a just-finished
        # invocation is not reported as abandoned.
        _reject_link(self.lock_path)
        try:
            stream = self.lock_path.open("r+b")
        except FileNotFoundError:
            state = self.read()
            active = False
        else:
            with stream:
                active = not _try_lock(stream)
                state = self.read()
        if state is not None and not active:
            mark_interrupted(state)
        return state

    def save(self, state: RunState) -> None:
        state.updated_at = timestamp()
        with NamedTemporaryFile(dir=self.directory, prefix="state-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(state.model_dump_json(indent=2).encode("utf-8") + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                # On Windows a concurrent reader can briefly deny replacement.
                for retry in range(5):
                    try:
                        temporary.replace(self.state_path)
                        break
                    except PermissionError:
                        if os.name != "nt" or retry == 4:
                            raise
                        sleep(0.02 * (retry + 1))
            finally:
                stream.close()
                temporary.unlink(missing_ok=True)

    def log(self, event: dict[str, object]) -> None:
        path = self.directory / "events.jsonl"
        _reject_link(path)
        with path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event) + "\n")

    def artifact_dir(self, name: str, attempt: str) -> Path:
        return self._attempt_dir("artifacts", name, attempt)

    def working_dir(self, name: str, attempt: str) -> Path:
        return self._attempt_dir("work", name, attempt)

    def _attempt_dir(self, kind: str, name: str, attempt: str) -> Path:
        parent = self.directory / kind
        _reject_link(parent)
        _reject_link(parent / name)
        return parent / name / attempt


def mark_interrupted(state: RunState) -> None:
    if state.status == "running":
        state.status = "interrupted"
    for record in state.stages.values():
        if record.status == "running":
            record.status = "interrupted"
            record.error = "Execution stopped before committing the stage. Resume to retry it."


class FingerprintError(OSError):
    """A file tree is unreadable, changing, or contains unsupported entries."""


def digest_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fingerprint_path(path: Path) -> str:
    digest = hashlib.sha256()

    def visit(entry: Path, relative: str) -> None:
        before = entry.lstat()
        if _is_link(before):
            raise FingerprintError
        if stat.S_ISDIR(before.st_mode):
            digest.update(json.dumps(["directory", relative]).encode())
            for child in sorted(entry.iterdir(), key=lambda p: p.name):
                visit(child, f"{relative}/{child.name}")
        elif stat.S_ISREG(before.st_mode):
            with entry.open("rb") as stream:
                content = hashlib.file_digest(stream, "sha256").hexdigest()
            digest.update(json.dumps(["file", relative, content]).encode())
        else:
            raise FingerprintError
        after = entry.lstat()
        if (
            before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns
        ) != (
            after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise FingerprintError

    try:
        visit(path, ".")
    except (OSError, ValueError, RecursionError):
        raise FingerprintError from None
    return digest.hexdigest()
