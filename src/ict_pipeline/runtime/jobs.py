"""Target-side job lifecycle around the existing pipeline and model contracts."""

import json
import os
import platform
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from shutil import disk_usage
from subprocess import DEVNULL, Popen
from time import monotonic

from ..config.settings import Credentials, SetupError
from ..data.ingestion import has_dataset
from ..data.model import DataError, load_selection
from ..data.preprocessing.console import protect_source_data
from ..environment.doctor import collect_environment
from ..evaluation.contracts import EvaluationConfig, EvaluationError, EvaluationSplit
from ..models.contracts import TrainingError
from ..models.registry import REGISTRY
from ..pipeline.model import PipelineError
from ..pipeline.runner import StageEvent, StageRunner
from ..pipeline.stages import pipeline_stages
from ..pipeline.store import RunStore, _reject_link, timestamp
from .model import ExecutorError, JobRequest, WorkerRequest

PREPROCESS = ("audit", "manifest", "convert", "validate_detection")
_SAFE_ERRORS = (PipelineError, SetupError, DataError, TrainingError, EvaluationError)
_MAX_REPORT = 384 * 1024
Progress = Callable[[dict[str, object]], None]


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ExecutorError("Use an absolute path on the executor's filesystem.")
    return path.resolve()


def require_disjoint_paths(source: Path, job: Path) -> None:
    if source.is_relative_to(job) or job.is_relative_to(source):
        raise ExecutorError("Use separate, non-overlapping input and job directories.")


def job_store(request: WorkerRequest, credentials: Credentials) -> RunStore:
    workspace = absolute_path(request.workspace)
    credentials.protect_path(str(workspace), "workspace")
    credentials.protect_path(request.job, "job ID")
    if not workspace.is_dir():
        raise ExecutorError("Create the job workspace on the target first.")
    store = RunStore(workspace, request.job, credentials)
    protect_source_data(store.directory, Path.cwd().resolve())
    return store


def preflight(store: RunStore, task: JobRequest) -> dict[str, object]:
    for value in (task.dataset, task.model_run, task.config_dir):
        if value is None:
            continue
        store.credentials.protect_path(value, "input path")
        source = absolute_path(value)
        store.credentials.protect_path(str(source), "resolved input path")
        require_disjoint_paths(source, store.directory)
        if not source.is_dir() or not os.access(source, os.R_OK | os.X_OK):
            raise ExecutorError("An input directory is missing or unreadable on the target.")
    if not os.access(store.directory.parent, os.W_OK | os.X_OK):
        raise ExecutorError("The target job workspace is not writable.")

    dataset = absolute_path(task.dataset)
    if task.workflow == "preprocess":
        if not has_dataset(dataset, load_selection()):
            raise ExecutorError("Point the dataset path at an existing extracted SMT dataset.")
    else:
        data_format = REGISTRY[task.model].data_format if task.model else "coco"
        data = dataset / data_format
        if not data.is_dir() or not data.resolve().is_relative_to(dataset):
            raise ExecutorError("Point the dataset path at a generated detection directory.")
        if task.model_run is not None and not (absolute_path(task.model_run) / "run.json").is_file():
            raise ExecutorError("Point the model run path at an ICT training attempt.")

    report: dict[str, object] = {
        "ready": True, "os": platform.system(), "architecture": platform.machine(),
        "python": platform.python_version(), "free_bytes": disk_usage(store.directory.parent).free,
        "device": task.device, "workflow": task.workflow, "dataset": str(dataset),
        "job_directory": str(store.directory),
    }
    if task.workflow == "preprocess":
        from importlib.util import find_spec

        if any(find_spec(name) is None for name in ("PIL", "openpyxl")):
            report.update(ready=False, reason="Install the preprocessing extra on the target after approval.")
        return report
    # Hardware inventory can be inspected before installing PyTorch. A tensor
    # operation in the requested backend is required before launching work.
    environment = collect_environment(store.directory.parent)
    report.update(
        torch=environment.torch.version, cuda_build=environment.torch.cuda_build,
        rocm_build=environment.torch.rocm_build,
        nvidia_driver=environment.graphics.nvidia_driver,
        graphics=[asdict(device) for device in environment.graphics.devices],
    )
    if task.model and task.device not in REGISTRY[task.model].devices:
        report.update(ready=False, reason="The model does not support the requested device.")
    elif getattr(environment.torch, task.device).readiness != "available":
        report.update(
            ready=False, reason="The requested PyTorch device failed readiness checks. Check the target runtime."
        )
    elif task.model:
        try:
            REGISTRY[task.model].versions()
            from ..models.training import plan_training

            plan_training(
                store.directory, (task.model,), dataset, store.directory / "runs" / "training",
                absolute_path(task.config_dir) if task.config_dir else None,
                absolute_path(task.model_run) if task.model_run else None, device=task.device,
            )
        except TrainingError as exc:
            report.update(ready=False, reason=store.credentials.redact_text(str(exc)))
    elif task.model_run is not None:
        from ..evaluation.runner import inspect_checkpoint

        try:
            checkpoint = inspect_checkpoint(absolute_path(task.model_run), "best", store.credentials)
            backend = REGISTRY[checkpoint.name]
            if task.device not in backend.devices:
                raise EvaluationError("The model does not support the requested device.")
            backend.versions()
        except (EvaluationError, TrainingError) as exc:
            report.update(ready=False, reason=store.credentials.redact_text(str(exc)))
    return report


def _read_json(path: Path) -> dict[str, object] | None:
    _reject_link(path)
    try:
        with path.open("rb") as stream:
            raw = stream.read(_MAX_REPORT + 1)
    except FileNotFoundError:
        return None
    if len(raw) > _MAX_REPORT:
        raise ExecutorError("Job metadata exceeds the size limit. Inspect it on the target.")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ExecutorError("Job metadata is invalid.")
    return value


def _state(store: RunStore) -> dict[str, object]:
    state = _read_json(store.directory / "job.json")
    if state is None:
        raise ExecutorError("No job state found. Check the workspace and job ID.")
    return state


def status(store: RunStore) -> dict[str, object]:
    with store.inspect() as active:
        state = _state(store)
    return state if active else _inactive(state)


def _saved_task(store: RunStore) -> JobRequest:
    value = _read_json(store.directory / "request.json")
    if value is None:
        raise ExecutorError("No saved job request found.")
    return JobRequest.model_validate(value)


def logs(store: RunStore, after: int) -> dict[str, object]:
    state = status(store)
    path = store.directory / "job-events.jsonl"
    _reject_link(path)
    events = []
    if path.exists():
        with path.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if after > size:
                raise ExecutorError("Log cursor is beyond the end of the job log.")
            if after:
                stream.seek(after - 1)
                if stream.read(1) != b"\n":
                    raise ExecutorError("Log cursor must follow a complete event.")
            while stream.tell() - after < 32 * 1024:
                offset = stream.tell()
                line = stream.readline(8192)
                if not line.endswith(b"\n"):
                    stream.seek(offset)
                    break
                events.append(json.loads(line))
            after = stream.tell()
    return {"status": state["status"], "events": events, "next": after}


def snapshot(store: RunStore) -> dict[str, object]:
    with store.inspect() as active:
        if active:
            raise ExecutorError("Wait for the job to stop before synchronizing results.")
        return _snapshot(store, _inactive(_state(store)))


def _snapshot(store: RunStore, state: dict[str, object]) -> dict[str, object]:
    task = _saved_task(store)
    root = store.directory
    if task.workflow == "preprocess":
        pipeline = RunStore(root / "runs", "preprocess")
        paths = [pipeline.state_path]
        run = pipeline.read()
        summaries = {
            "audit": "summary.json", "manifest": "manifest_summary.json",
            "convert": "detection/conversion_summary.json", "validate_detection": "summary.json",
        }
        if run:
            paths.extend(
                pipeline.artifact_dir(stage, record.attempt) / summaries[stage]
                for stage, record in run.stages.items() if stage in summaries and record.status == "completed"
            )
    elif task.workflow == "train":
        paths = sorted((root / "runs" / "training").glob("*/*/run.json"))
    else:
        directory = root / "outputs" / "evaluation"
        paths = sorted(path for pattern in ("*/??-*.json", "*/evaluation.json", "*/attempt.json")
                       for path in directory.glob(pattern))
    reports = {}
    total = 0
    for path in paths:
        relative = path.relative_to(root)
        for parent in relative.parents:
            _reject_link(root / parent)
        value = _read_json(path)
        if value is None:
            continue
        total += len(json.dumps(value))
        if total > _MAX_REPORT:
            raise ExecutorError("The result snapshot exceeds its size limit. Inspect reports on the target.")
        reports[relative.as_posix()] = value
    return {"schema_version": 1, "job": state, "reports": reports}


def _execute(
    store: RunStore, task: JobRequest, state: dict[str, object], on_event: Progress | None,
) -> dict[str, object]:
    job_dir = store.directory
    start = monotonic()
    events = job_dir / "job-events.jsonl"
    _reject_link(events)
    result_paths: list[str] = []
    last_progress: dict[str, float] = {}
    stage_starts: dict[str, float] = {}

    def event(
        stage: str, finished: int | None, total: int | None, status: str = "progress", *,
        stage_event: StageEvent | None = None,
    ) -> None:
        now = monotonic()
        if status == "progress" and finished not in (0, total) and now - last_progress.get(stage, -1e9) < 1:
            return
        last_progress[stage] = now
        elapsed = now - start
        stage_elapsed = now - stage_starts.setdefault(stage, now)
        eta = stage_event.eta_seconds if stage_event is not None else None
        if stage_event is None and total is not None and finished:
            eta = stage_elapsed * (total - finished) / finished
        value = {
            "stage": store.credentials.redact_text(stage)[:160], "status": status, "completed": finished,
            "total": total, "elapsed_seconds": round(elapsed, 2), "timestamp": timestamp(),
            "eta_seconds": round(eta, 2) if eta is not None else None,
        }
        if stage_event is not None:
            value.update(stages_completed=stage_event.stages_completed, stages_total=stage_event.stages_total)
        with events.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value) + "\n")
        if on_event is not None:
            on_event(value)

    def pipeline_event(progress: StageEvent) -> None:
        event(progress.stage, progress.completed, progress.total, progress.status, stage_event=progress)

    try:
        dataset = absolute_path(task.dataset)
        model_run = absolute_path(task.model_run) if task.model_run else None
        event(task.workflow, 0, None, "running")
        if task.workflow == "preprocess":
            runner = StageRunner(
                pipeline_stages(dataset_root=dataset), job_dir, {"execution_mode": "local", "data_path": str(dataset)},
                on_event=pipeline_event, credentials=store.credentials,
            )
            # The runner revalidates input and output fingerprints on every retry.
            run = runner.run("preprocess", PREPROCESS)
            converted = RunStore(job_dir / "runs", "preprocess").artifact_dir("convert", run.stages["convert"].attempt)
            result_paths.append((converted / "detection").relative_to(job_dir).as_posix())
        elif task.workflow == "train":
            from ..models.training import execute_training, plan_training

            assert task.model is not None
            plan = plan_training(
                job_dir, (task.model,), dataset, job_dir / "runs" / "training",
                absolute_path(task.config_dir) if task.config_dir else None, model_run, device=task.device,
            )
            for name, _ in execute_training(plan, event):
                result_paths.append((plan.destination / name).relative_to(job_dir).as_posix())
        else:
            from ..evaluation.runner import evaluate_runs

            assert model_run is not None
            for path in evaluate_runs(
                job_dir, (model_run,), dataset, EvaluationSplit.VALIDATION, EvaluationConfig(device=task.device),
                job_dir / "outputs" / "evaluation", "best", False, None, None, event,
            ):
                result_paths.append(path.relative_to(job_dir).as_posix())
        state.update(status="completed", results=result_paths)
    except KeyboardInterrupt:
        state.update(status="interrupted", reason="Worker was interrupted.")
    except _SAFE_ERRORS as exc:
        state.update(status="failed", reason=store.credentials.redact_text(str(exc)))
    except Exception:
        state.update(status="failed", reason="Execution failed. Check target dependencies, permissions, and storage.")
    state.update(finished_at=timestamp(), elapsed_seconds=round(monotonic() - start, 2))
    event(task.workflow, None, None, str(state["status"]))
    store.write_json("job.json", state)
    return state


def dispatch(request: WorkerRequest, credentials: Credentials, on_event: Progress | None = None) -> dict[str, object]:
    store = job_store(request, credentials)
    action, task = request.action, request.task
    if action == "status":
        return status(store)
    if action == "logs":
        return logs(store, request.after)
    if action == "snapshot":
        return snapshot(store)
    if action == "work":
        # The detached launcher passes its still-locked descriptor. Keeping it
        # open for the worker lifetime avoids a gap and race between launchers.
        return _execute(store, _saved_task(store), _state(store), on_event)
    if action == "preflight":
        assert task is not None
        return preflight(store, task)
    if action not in {"submit", "resume", "run"}:
        raise ExecutorError("Unsupported worker operation.")
    if action in {"submit", "resume"} and platform.system() != "Linux":
        raise ExecutorError("Detached execution requires a Linux target.")
    if action == "resume":
        task = _saved_task(store)
        if task.workflow != "preprocess":
            raise ExecutorError("Start a new training job with its saved model run, or resume evaluation on the target.")
    assert task is not None
    task.require_approval()
    # A repeated submit after a lost acknowledgement is a status check.
    if action == "submit" and (store.directory / "job.json").exists():
        if task != _saved_task(store):
            raise ExecutorError("This job ID already belongs to another request.")
        return status(store)
    result = preflight(store, task)
    if result["ready"] is not True:
        raise ExecutorError(str(result["reason"]))

    with store.lock() as lease:
        previous = _read_json(store.directory / "job.json")
        saved = _read_json(store.directory / "request.json")
        if saved is not None and task != JobRequest.model_validate(saved):
            raise ExecutorError("This job ID already belongs to another request.")
        if previous is not None and action != "resume":
            if action == "submit":
                return _inactive(previous)
            raise ExecutorError("This job ID already exists. Use a new ID.")
        if previous is not None and previous["status"] == "completed":
            return previous
        if saved is None:
            store.write_json("request.json", task.model_dump(mode="json"))
        state = {
            "schema_version": 1, "job_id": store.run_id, "workflow": task.workflow, "status": "running",
            "started_at": timestamp(), "attempt": int(str(previous["attempt"])) + 1 if previous else 1,
        }
        store.write_json("job.json", state)
        if action == "run":
            return _execute(store, task, state, on_event)
        try:
            Popen(
                [sys.executable, "-m", "ict_pipeline.runtime.worker", str(lease.fileno()),
                 str(store.directory.parent), store.run_id],
                stdin=DEVNULL, stdout=DEVNULL, stderr=DEVNULL, start_new_session=True,
                pass_fds=(lease.fileno(),), cwd=store.directory,
            )
        except OSError:
            state.update(status="failed", finished_at=timestamp(), reason="Could not start the detached worker.")
            store.write_json("job.json", state)
            raise ExecutorError(str(state["reason"])) from None
        return state


def _inactive(state: dict[str, object]) -> dict[str, object]:
    if state["status"] == "running":
        return {**state, "status": "interrupted", "reason": "Worker exited before publishing its result."}
    return state
