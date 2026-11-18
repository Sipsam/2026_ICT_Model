"""Record aggregate training and evaluation execution alongside native artifacts."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..pipeline.store import RunStore, timestamp
from .metadata import runtime_snapshot

ATTEMPT_FILE = "attempt.json"


@contextmanager
def track_attempt(
    store: RunStore, workspace: Path, kind: str, configuration: dict[str, object],
    packages: tuple[str, ...], safe_errors: tuple[type[Exception], ...],
) -> Iterator[None]:
    """Record an invocation while the caller holds its run lock."""
    state = {
        "schema_version": 1, "run_id": store.run_id, "kind": kind,
        "started_at": timestamp(), "finished_at": None, "status": "running", "failure_reason": None,
        "configuration": store.credentials.redact(configuration),
        "runtime": {},
    }
    store.write_json(ATTEMPT_FILE, state)
    try:
        state["runtime"] = store.credentials.redact(runtime_snapshot(workspace, packages))
        store.write_json(ATTEMPT_FILE, state)
        yield
    except (GeneratorExit, KeyboardInterrupt):
        state.update(status="interrupted", failure_reason="Execution stopped before the run finished.")
        raise
    except Exception as exc:
        detail = str(exc) if isinstance(exc, safe_errors) else "Check the inputs, runtime, permissions and storage."
        state.update(status="failed", failure_reason=store.credentials.redact_text(detail))
        raise
    else:
        state["status"] = "completed"
    finally:
        state["finished_at"] = timestamp()
        store.write_json(ATTEMPT_FILE, state)
