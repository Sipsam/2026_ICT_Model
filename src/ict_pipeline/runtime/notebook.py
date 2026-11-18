"""Opt-in notebook entry point for an already-provisioned runtime."""

from pathlib import Path

from . import JobRequest, LocalExecutor, WorkerRequest
from .jobs import Progress
from .model import ExecutorError


def preprocess(
    dataset: Path, workspace: Path, job: str, *, on_event: Progress | None = None,
) -> dict[str, object]:
    """Process an existing dataset on the notebook host and return the job state."""
    executor = LocalExecutor()
    task = JobRequest(dataset=str(dataset.resolve()))
    result = executor.execute(
        WorkerRequest(action="run", workspace=str(workspace.resolve()), job=job, task=task), on_event,
    )
    if result["status"] != "completed":
        raise ExecutorError(str(result.get("reason", "Preprocessing did not complete. Inspect the job state.")))
    return result
