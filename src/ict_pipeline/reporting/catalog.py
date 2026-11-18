"""Read pipeline, training and evaluation snapshots without loading checkpoints."""

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..evaluation.contracts import EvaluationError
from ..pipeline.model import PipelineError, valid_id
from ..pipeline.store import RunStore, digest_json
from .attempt import ATTEMPT_FILE

Run = dict[str, Any]
_STATUSES = {"running", "completed", "failed", "interrupted"}


def _object(path: Path) -> dict[str, Any] | None:
    if path.is_symlink():
        raise ValueError
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not isinstance(document, dict):
        raise ValueError
    json.dumps(document, allow_nan=False)
    return document


def _path(path: Path, workspace: Path) -> str:
    return (path.relative_to(workspace) if path.is_relative_to(workspace) else path).as_posix()


def _logs(directory: Path, workspace: Path) -> list[str]:
    log = directory / "events.jsonl"
    return [_path(log, workspace)] if log.is_file() and not log.is_symlink() else []


def _last_event(directory: Path) -> Run:
    path = directory / "events.jsonl"
    event: Run = {}
    if path.is_file() and not path.is_symlink():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    event = record
    return event


def _directories(root: Path) -> Iterator[Path]:
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if valid_id(path.name) and path.is_dir() and not path.is_symlink():
                yield path


def _pipeline(directory: Path, workspace: Path) -> Run | None:
    state = RunStore(directory.parent, directory.name).status()
    if state is None:
        return None
    run = state.model_dump(mode="json")
    run.update(
        kind="pipeline", started_at=state.started_at or state.created_at,
        finished_at=state.finished_at or (state.updated_at if state.status in {"completed", "failed"} else None),
        logs=_logs(directory, workspace),
    )
    if not run["failure_reason"] and state.status in {"failed", "interrupted"}:
        run["failure_reason"] = next((
            state.stages[name].error for name in reversed(state.selection)
            if name in state.stages and state.stages[name].error
        ), None)
    return run


def _training(directory: Path, workspace: Path, active: bool) -> list[Run]:
    models = []
    for name in ("yolo", "faster_rcnn", "rt_detr"):
        folder = directory / name
        RunStore(directory, name)  # Reject links and junctions before reading.
        model = _object(folder / "run.json")
        if model is None:
            continue
        if model.get("schema_version") != 1 or model.get("model") != name or model.get("status") not in _STATUSES:
            raise ValueError
        if model["status"] == "running" and not active:
            with RunStore(directory, name).inspect() as model_active:
                model = _object(folder / "run.json") or model
                if model["status"] == "running" and not model_active:
                    model.update(status="interrupted", error="Execution stopped before recording a final status.")
        model["logs"] = _logs(folder, workspace)
        models.append(model)
    return models


def _evaluation(directory: Path) -> tuple[Run, list[Run], bool]:
    from ..evaluation.runner import read_report

    marker = _object(directory / "evaluation.json")
    if marker is None:
        return {}, [], False
    identity = marker.pop("request_fingerprint")
    if (
        digest_json(marker) != identity or not isinstance(marker["models"], list) or not marker["models"]
        or not all(isinstance(marker.get(key), dict) for key in ("dataset", "protocol", "code"))
    ):
        raise ValueError
    models = []
    complete = True
    for index, record in enumerate(marker["models"], 1):
        name = record["model"]
        if name not in {"yolo", "faster_rcnn", "rt_detr"} or not isinstance(record.get("checkpoint"), dict):
            raise ValueError
        path = directory / f"{index:02d}-{name}.json"
        try:
            if not path.is_file() or path.is_symlink():
                raise EvaluationError
            result = read_report(path)
            json.dumps(result, allow_nan=False)
            if (result.get("request_fingerprint") != identity or result.get("model_index") != index
                    or result.get("dataset") != marker["dataset"] or result.get("protocol") != marker["protocol"]
                    or result.get("code_fingerprints") != marker["code"]
                    or not all(isinstance(result.get(key), dict) for key in ("metrics", "speed", "libraries"))
                    or not isinstance(result["speed"].get("hardware"), dict)
                    or not isinstance(result["speed"].get("latency_ms", {}), dict)
                    or not isinstance(result.get("python"), str)
                    or any(result.get(key) != value for key, value in record.items())):
                raise EvaluationError
            models.append(result)
        except (EvaluationError, ValueError):
            complete = False
            models.append({**record, "model_index": index, "status": "incomplete"})
    return marker, models, complete


def _experiment(directory: Path, workspace: Path, kind: str) -> Run | None:
    store = RunStore(directory.parent, directory.name)
    with store.inspect() as active:
        attempt = _object(directory / ATTEMPT_FILE)
        if attempt is not None and (
            attempt.get("schema_version") != 1 or attempt.get("kind") != kind
            or attempt.get("run_id") != directory.name or attempt.get("status") not in _STATUSES
            or any(attempt.get(key) is not None and not isinstance(attempt[key], str)
                   for key in ("started_at", "finished_at"))
        ):
            raise ValueError
        if kind == "training":
            models = _training(directory, workspace, active)
            marker = {}
            expected = (attempt or {}).get("configuration", {}).get("models", {})
            complete = bool(models) and {model["model"] for model in models}.issuperset(expected) and all(
                model["status"] == "completed" for model in models
            )
        else:
            marker, models, complete = _evaluation(directory)
    if not (attempt or marker or models):
        return None
    run = {
        "kind": kind, **(attempt or {}), "run_id": f"{kind}/{directory.name}",
        "logs": _logs(directory, workspace), "models": models,
    }
    if marker:
        run.update(dataset=marker["dataset"], protocol=marker["protocol"])
    if attempt is None:
        failures = [model for model in models if model["status"] == "failed"]
        run.update(
            status="running" if active else "failed" if failures else "completed" if complete else "incomplete",
            started_at=min((model["started_at"] for model in models if model.get("started_at")), default=None),
            finished_at=max((model["finished_at"] for model in models if model.get("finished_at")), default=None),
            failure_reason=failures[-1].get("error") if failures else None,
        )
        if kind == "evaluation" and run["status"] == "incomplete":
            event = _last_event(directory)
            if event.get("status") in ("failed", "interrupted"):
                run.update(
                    status=event["status"], finished_at=event.get("time"),
                    failure_reason="Execution stopped during " + str(event.get("stage", "evaluation")) + ".",
                )
    elif run["status"] == "running" and not active:
        run.update(status="interrupted", failure_reason="Execution stopped before recording a final status.")
    elif run["status"] == "completed" and not complete:
        run.update(status="incomplete", failure_reason="Expected model records are missing or incomplete.")
    return run


def list_runs(workspace: Path, training_root: Path, evaluation_root: Path) -> list[Run]:
    """Read the default pipeline root and the selected training and evaluation roots."""
    from ..config.storage import SetupFiles

    workspace = workspace.resolve()
    credentials = SetupFiles.read(workspace, None).credentials(os.environ)
    runs = []
    for kind, root in (("pipeline", Path("runs")), ("training", training_root), ("evaluation", evaluation_root)):
        credentials.protect_path(str(root), "Run root")
        for directory in _directories(workspace / root):
            try:
                run = _pipeline(directory, workspace) if kind == "pipeline" else _experiment(directory, workspace, kind)
            except (OSError, PipelineError, ValueError, KeyError, TypeError, AttributeError):
                run = {
                    "run_id": directory.name if kind == "pipeline" else f"{kind}/{directory.name}",
                    "kind": kind, "status": "incomplete",
                    "failure_reason": "Run metadata is unreadable, corrupt, or unsupported.",
                }
            if run:
                run["directory"] = _path(directory, workspace)
                runs.append(credentials.redact(run))
    return sorted(
        runs, key=lambda run: (run.get("updated_at") or run.get("finished_at") or run.get("started_at") or "",
                               run["run_id"]), reverse=True,
    )
