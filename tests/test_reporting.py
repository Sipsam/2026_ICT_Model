import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.pipeline.store import RunStore, digest_json
from ict_pipeline.reporting import metadata
from ict_pipeline.reporting.attempt import track_attempt
from ict_pipeline.reporting.catalog import list_runs

CLIENT = CliRunner()
STAMP = "2026-01-01T00:00:00+00:00"


def save_evaluation(root: Path, run_id: str, model: str, split: str, metric: float) -> Path:
    dataset = {"split": split, "role": "evaluation", "image_count": 2,
               "image_fingerprint": split + "-images", "annotation_fingerprint": "sha"}
    protocol = {"configuration": {"image_size": 64}, "ap": "COCO"}
    code = {"evaluate.py": "sha"}
    entry = {"model": model, "checkpoint": {"path": f"{model}/best.pt", "fingerprint": "sha"}}
    request = {"dataset": dataset, "protocol": protocol, "code": code, "models": [entry]}
    identity = digest_json(request)
    store = RunStore(root, run_id)
    with store.lock():
        store.write_json("evaluation.json", {"request_fingerprint": identity, **request})
        result = {
            **entry, "schema_version": 1, "status": "completed", "finished_at": STAMP,
            "request_fingerprint": identity, "model_index": 1, "dataset": dataset, "protocol": protocol,
            "code_fingerprints": code, "python": "3.11", "libraries": {"torch": "fake"},
            "metrics": {"map50": metric, "map50_95": 0, "per_class": {"scratch": {"ap": None}}},
            "speed": {"hardware": {"cpu": "fake", "device": "cpu"}, "fps": 2, "latency_ms": {"p95": 500}},
        }
        store.write_json(f"01-{model}.json", {"fingerprint": digest_json(result), "result": result})
    return store.directory


def invoke(*args: str) -> str:
    result = CLIENT.invoke(app, list(args))
    assert result.exit_code == 0, result.output
    return result.stdout.strip()


def test_runtime_snapshot_preserves_git_and_package_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "b" * 40

    def git(command: list[str]) -> tuple[str, str]:
        assert command == ["git", "-C", str(tmp_path), "rev-parse", "--verify", "HEAD"]
        return "available", commit

    monkeypatch.setattr(metadata, "run_probe", git)
    snapshot = metadata.runtime_snapshot(tmp_path, ("nonexistent-ict-fixture-package",))
    assert snapshot["git_commit"] == commit
    assert snapshot["python"]["version"]
    assert snapshot["os"]["os"]
    assert snapshot["hardware"]["resources"]["cpu_model"] == "Fixture CPU"
    assert snapshot["packages"]["ict-pipeline"]
    assert snapshot["packages"]["nonexistent-ict-fixture-package"] is None


def test_comparison_and_all_formats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "custom-evals"
    save_evaluation(root, "first", "yolo", "val", 0.75)
    save_evaluation(root, "second", "rt_detr", "val", 0.875)
    save_evaluation(root, "test", "yolo", "test", 0.5)
    incomplete = save_evaluation(root, "corrupt", "yolo", "val", 0.99)
    (incomplete / "01-yolo.json").write_text("{}", encoding="utf-8")
    options = ("--evaluation-root", "custom-evals")
    comparison = json.loads(invoke("compare", *options))
    groups = {group["dataset"]["split"]: group for group in comparison["comparison_groups"]}
    assert len(groups["val"]["models"]) == 2
    assert groups["val"]["matching_speed_environment"] is True
    assert len(groups["test"]["models"]) == 1
    assert "evaluation/corrupt   evaluation   incomplete" in invoke("runs", "list", *options)
    assert json.loads(invoke("runs", "show", "evaluation/first", *options))["models"][0]["metrics"]["map50"] == 0.75
    assert "rt_detr   completed" in invoke("status", "--run-id", "evaluation/second", *options)

    json_path = Path(invoke("report", "--format", "json", *options))
    assert json.loads(json_path.read_text(encoding="utf-8")) == comparison
    html = Path(invoke("report", *options)).read_text(encoding="utf-8")
    assert "0.875" in html and "incomplete" in html
    first_csv = Path(invoke("report", "--format", "csv", *options))
    second_csv = Path(invoke("report", "--format", "csv", *options))
    assert first_csv != second_csv
    with first_csv.open(newline="", encoding="utf-8") as stream:
        rows = {row["run_id"]: row for row in csv.DictReader(stream)}
    assert rows["evaluation/first"]["map50_95"] == "0"
    assert rows["evaluation/corrupt"]["map50"] == ""
    assert rows["evaluation/test"]["comparison_group"] != rows["evaluation/first"]["comparison_group"]


def test_at_rest_redaction_and_failed_or_abandoned_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIHUB_API_KEY", "known-sensitive-value")
    monkeypatch.setenv("UNRELATED_ENV", "must-not-be-collected")
    monkeypatch.setattr("ict_pipeline.reporting.attempt.runtime_snapshot", lambda *_: {"git_commit": "fake-commit"})
    credentials = Credentials.load({}, {"AIHUB_API_KEY": "known-sensitive-value"})
    store = RunStore(tmp_path / "runs/training", "failed", credentials)
    raw_reason = "<script> password='unknown secret' known-sensitive-value Bearer hidden-bearer =test"
    with store.lock(), pytest.raises(ValueError):
        with track_attempt(
            store, tmp_path, "training",
            {"nested": [{"apiKey": "hidden-key", "url": "https://user:hidden-pass@example.test"}]},
            (), (ValueError,),
        ):
            store.log({"detail": raw_reason})
            raise ValueError(raw_reason)
    attempt = (store.directory / "attempt.json").read_text()
    logs = (store.directory / "events.jsonl").read_text()
    assert "[REDACTED]" in attempt and "[REDACTED]" in logs
    for value in ("unknown secret", "known-sensitive-value", "hidden-bearer", "hidden-key", "hidden-pass",
                  "must-not-be-collected"):
        assert value not in attempt + logs

    abandoned = RunStore(tmp_path / "outputs/evaluation", "abandoned")
    with abandoned.lock():
        abandoned.write_json("attempt.json", {
            "schema_version": 1, "run_id": "abandoned", "kind": "evaluation", "status": "running",
            "started_at": STAMP, "finished_at": None,
        })
        runs = list_runs(tmp_path, Path("runs/training"), Path("outputs/evaluation"))
        assert any(run["run_id"] == "evaluation/abandoned" and run["status"] == "running" for run in runs)
    status = json.loads(invoke("runs", "show", "evaluation/abandoned"))
    assert status["status"] == "interrupted"
    assert status["finished_at"] is None
    failed = json.loads(invoke("runs", "show", "training/failed"))
    assert failed["status"] == "failed" and failed["finished_at"]
    assert failed["logs"] == ["runs/training/failed/events.jsonl"]
    html = Path(invoke("report")).read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "secret'" not in html and "known-sensitive-value" not in html


def test_report_refuses_source_output_and_supports_empty_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert invoke("runs", "list") == "No saved runs."
    assert not list(tmp_path.iterdir())
    result = CLIENT.invoke(app, ["report", "--output-dir", "data/raw"])
    assert result.exit_code == 1 and not list(tmp_path.iterdir())
    assert json.loads(Path(invoke("report", "--format", "json")).read_text())["runs"] == []


def test_mlflow_export_is_opt_in_and_sanitizes_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    uri = "https://user:fake-password@example.test"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    calls: list[Any] = []

    class FakeClient:
        def __init__(self, tracking_uri: str) -> None:
            calls.append(("init", tracking_uri))

        def create_run(self, experiment_id: str, run_name: str) -> SimpleNamespace:
            calls.append(("create", experiment_id))
            return SimpleNamespace(info=SimpleNamespace(run_id="mlflow-id"))

        def log_dict(self, run_id: str, document: dict[str, object], filename: str) -> None:
            calls.append(("artifact", run_id, document, filename))
            if len(calls) > 4:
                raise RuntimeError(uri)

        def set_terminated(self, run_id: str, status: str) -> None:
            calls.append((run_id, status))

    monkeypatch.setitem(sys.modules, "mlflow", SimpleNamespace(MlflowClient=FakeClient))
    invoke("report")
    assert calls == []
    invoke("report", "--mlflow", "--experiment-id", "4")
    assert calls == [
        ("init", uri), ("create", "4"),
        ("artifact", "mlflow-id", {"schema_version": 1, "runs": [], "comparison_groups": []}, "comparison.json"),
        ("mlflow-id", "FINISHED"),
    ]
    failed = CLIENT.invoke(app, ["report", "--mlflow"])
    assert failed.exit_code == 1 and uri not in failed.output
    assert Path(failed.stdout.strip()).is_file()
    assert calls[-1] == ("mlflow-id", "FAILED")
