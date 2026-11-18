"""Exercise executor contracts without external services or accelerators."""

import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from time import monotonic, sleep
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials, SetupError
from ict_pipeline.environment.model import Backend, Torch
from ict_pipeline.models import training
from ict_pipeline.models.contracts import ModelName, Progress
from ict_pipeline.models.registry import Backend as ModelBackend
from ict_pipeline.pipeline.store import fingerprint_path
from ict_pipeline.runtime import ContainerExecutor, ExecutorError, JobRequest, LocalExecutor, RemoteExecutor, WorkerRequest
from ict_pipeline.runtime import executors, jobs
from test_preprocessing import TRAIN_MIXED, sample


@pytest.fixture
def target(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "work space"
    workspace.mkdir()
    data = tmp_path / "data, extracted"
    sample(data, "train", TRAIN_MIXED, "cold_solder", (23,), color=16)
    sample(data, "validation", "SD_DEF_ST_ACME_20240101_006", "tombstone", (30,), color=80)
    return workspace, data


def test_local_workflow_and_readonly_sync(target: tuple[Path, Path], tmp_path: Path) -> None:
    workspace, dataset = target
    client = CliRunner()
    cli = ["runtime", "--workspace", str(workspace)]
    args = ["fixture", "--dataset", str(dataset)]
    original = fingerprint_path(dataset)
    preview = client.invoke(app, [*cli, "preflight", *args])
    assert preview.exit_code == 0, preview.output
    assert json.loads(preview.output)["ready"]
    assert not list(workspace.iterdir())

    result = client.invoke(app, [*cli, "run", *args])
    assert result.exit_code == 0, result.output
    state = json.loads(result.output.splitlines()[-1])
    assert state["status"] == "completed"
    detection = workspace / "fixture" / state["results"][0]
    assert (detection / "coco" / "annotations" / "instances_val.json").is_file()
    assert fingerprint_path(dataset) == original

    log = json.loads(client.invoke(app, [*cli, "logs", "fixture"]).output)
    assert {item["stage"] for item in log["events"]}.issuperset(jobs.PREPROCESS)
    tail = client.invoke(app, [*cli, "logs", "fixture", "--after", str(log["next"])])
    assert json.loads(tail.output)["events"] == []

    output = tmp_path / "snapshot.json"
    sync = [*cli, "sync", "fixture", "--output", str(output)]
    assert client.invoke(app, sync).exit_code == 0
    data = output.read_bytes()
    reports = json.loads(data)["reports"]
    assert reports["runs/preprocess/state.json"]["status"] == "completed"
    assert any(name.endswith("/conversion_summary.json") for name in reports)
    assert client.invoke(app, sync).exit_code != 0
    assert output.read_bytes() == data
    assert client.invoke(app, [*cli, "run", *args]).exit_code != 0


def test_device_failure_and_approval_before_execution(
    target: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, dataset = target
    (dataset / "yolo").mkdir()
    monkeypatch.setattr(jobs, "collect_environment", lambda _path: SimpleNamespace(
        torch=Torch("available", version="2.6", cpu=Backend("available", "available")),
        graphics=SimpleNamespace(nvidia_driver="535.10", devices=()),
    ))
    executor = LocalExecutor()
    task = JobRequest(workflow="train", dataset=str(dataset), device="cuda", model="yolo")
    request = WorkerRequest(action="preflight", job="gpu", workspace=str(workspace), task=task)
    preflight = executor.execute(request)
    assert not preflight["ready"]
    assert preflight["device"] == "cuda"
    assert preflight["nvidia_driver"] == "535.10"
    with pytest.raises(ExecutorError, match="--approve-train"):
        executor.execute(request.model_copy(update={"action": "run"}))
    with pytest.raises(ExecutorError, match="failed readiness"):
        executor.execute(request.model_copy(update={
            "action": "run", "task": task.model_copy(update={"approve_train": True}),
        }))
    assert not list(workspace.iterdir())


def test_training_reuses_config_and_explicit_device(
    target: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, dataset = target
    (dataset / "yolo").mkdir()
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "yolo.yaml").write_text("epochs: 1\ndevice: cpu\n", encoding="utf-8")
    monkeypatch.setattr(jobs, "collect_environment", lambda _path: SimpleNamespace(
        torch=Torch("available", version="2.6", cuda=Backend("available", "available")),
        graphics=SimpleNamespace(nvidia_driver="535.10", devices=()),
    ))
    monkeypatch.setattr(ModelBackend, "versions", lambda _self: {})
    plans = []

    def train(plan: training.TrainingPlan, progress: Progress) -> list[tuple[ModelName, None]]:
        plans.append(plan)
        progress("training", 1, 1)
        return [(ModelName.YOLO, None)]

    monkeypatch.setattr(training, "execute_training", train)
    task = JobRequest(workflow="train", dataset=str(dataset), device="cuda", model="yolo",
                      config_dir=str(configs), approve_train=True)
    request = WorkerRequest(action="run", workspace=str(workspace), job="train", task=task)
    result = LocalExecutor().execute(request)
    assert result["status"] == "completed"
    assert len(plans) == 1
    assert plans[0].models[ModelName.YOLO].device == "cuda"
    assert plans[0].models[ModelName.YOLO].epochs == 1


def test_destination_cannot_contain_a_writable_alias_of_source(
    target: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _ = target
    nested = workspace / "existing" / "input"
    nested.mkdir(parents=True)
    monkeypatch.setattr(executors, "_command", lambda *_args, **_kwargs: pytest.fail("No Docker command expected."))
    request = WorkerRequest(
        action="preflight", workspace=str(workspace), job="existing", task=JobRequest(dataset=str(nested)),
    )
    with pytest.raises(ExecutorError, match="non-overlapping"):
        ContainerExecutor("ict-pipeline:test").execute(request)


def test_docker_translates_mounts_and_pins_daemon(
    target: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, dataset = target
    calls = []

    def command(argv: list[str], **options: Any) -> str:
        calls.append((argv, options))
        if "context" in argv:
            return "unix:///test-docker.sock\n"
        if "info" in argv:
            return "linux\n"
        return '{"schema_version":1,"result":{"ready":true,"status":"completed"}}'

    monkeypatch.delenv("DOCKER_HOST", raising=False)
    monkeypatch.delenv("DOCKER_CONTEXT", raising=False)
    monkeypatch.setattr(executors, "_command", command)
    executor = ContainerExecutor("ict-pipeline:test")
    request = WorkerRequest(action="preflight", workspace=str(workspace), job="test", task=JobRequest(dataset=str(dataset)))
    result = executor.execute(request)
    assert result["dataset"] == str(dataset)
    assert not (workspace / "test").exists()
    assert executor.execute(request.model_copy(update={"action": "run"}))["status"] == "completed"

    argv, options = calls[-1]
    assert argv[:3] == ["docker", "--host", "unix:///test-docker.sock"]
    assert "--network=none" in argv and "--pull=never" in argv and "--gpus" not in argv
    assert "HTTPS_PROXY=" in argv and "https_proxy=" in argv
    mounts = [next(csv.reader([argv[index + 1]])) for index, flag in enumerate(argv) if flag == "--mount"]
    assert mounts == [
        ["type=bind", f"source={dataset}", "target=/ict/dataset", "readonly", "bind-recursive=disabled"],
        ["type=bind", f"source={workspace / 'test'}", "target=/ict/jobs/test"],
    ]
    mounted = WorkerRequest.model_validate_json(options["payload"])
    assert mounted.task and mounted.task.dataset == "/ict/dataset"
    assert mounted.workspace == "/ict/jobs"
    assert dataset.name not in options["payload"]

    monkeypatch.setenv("DOCKER_HOST", "ssh://another-machine")
    with pytest.raises(ExecutorError, match="local Docker daemon"):
        executor.execute(request)
    assert next(csv.reader([executors._bind(dataset, PurePosixPath("/ict/input"), readonly=True)]))[1] == f"source={dataset}"


def test_ssh_uses_fixed_command_and_keeps_credentials_out_of_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    credentials = Credentials.load({}, {"AIHUB_API_KEY": "test-secret-value"})
    monkeypatch.setenv("AIHUB_API_KEY", "test-secret-value")
    monkeypatch.setattr(executors, "which", lambda _name: "ssh")

    def run(argv: list[str], **options: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, options))
        return subprocess.CompletedProcess(argv, 0, stdout=(
            'banner\n{"schema_version":1,"result":{"reason":"test-secret-value"}}\n'
        ))

    monkeypatch.setattr(executors.subprocess, "run", run)
    interpreter = "/opt/ICT environment/bin/python"
    executor = RemoteExecutor("ict-gpu", user="runner", python=interpreter, credentials=credentials)
    task = JobRequest(dataset="/srv/data with spaces/$(ignored);x")
    request = WorkerRequest(action="preflight", workspace="/srv/jobs with spaces", job="trial", task=task)
    assert executor.execute(request) == {"reason": "[REDACTED]"}
    argv, options = calls[0]
    assert "StrictHostKeyChecking=yes" in argv and "ForwardAgent=no" in argv
    assert argv[-2] == "ict-gpu"
    assert shlex.split(argv[-1]) == [interpreter, "-m", "ict_pipeline.runtime.worker", "--ssh"]
    assert task.dataset not in argv[-1] and task.dataset in options["input"]
    assert "test-secret-value" not in options["input"]
    assert "AIHUB_API_KEY" not in options["env"]
    assert options["shell"] is False and options["stderr"] == subprocess.DEVNULL
    with pytest.raises(ExecutorError, match="valid SSH host"):
        RemoteExecutor("-oProxyCommand=unsafe")
    with pytest.raises(ExecutorError, match="absolute Linux paths"):
        executor.execute(request.model_copy(update={"workspace": r"C:\jobs"}))
    with pytest.raises(SetupError, match="contains a credential") as failure:
        executor.execute(request.model_copy(update={"workspace": "/srv/test-secret-value"}))
    assert "test-secret-value" not in str(failure.value)


def test_remote_retry_reuses_saved_request(
    target: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, dataset = target
    monkeypatch.setattr(jobs.platform, "system", lambda: "Linux")
    launches = []
    monkeypatch.setattr(jobs, "Popen", lambda argv, **options: launches.append(options))
    submit = WorkerRequest(action="submit", workspace=str(workspace), job="retry", task=JobRequest(dataset=str(dataset)))
    inspect = WorkerRequest(action="status", workspace=str(workspace), job="retry")
    credentials = Credentials()
    assert jobs.dispatch(submit, credentials)["status"] == "running"
    assert launches[0]["start_new_session"] and launches[0]["pass_fds"]
    assert jobs.dispatch(inspect, credentials)["status"] == "interrupted"
    assert jobs.dispatch(submit, credentials)["status"] == "interrupted"
    assert len(launches) == 1
    resumed = jobs.dispatch(inspect.model_copy(update={"action": "resume"}), credentials)
    assert resumed["attempt"] == 2 and len(launches) == 2


@pytest.mark.skipif(sys.platform != "linux", reason="Detached worker requires Linux file-descriptor inheritance.")
def test_detached_linux_worker_survives_submitter_exit(target: tuple[Path, Path]) -> None:
    workspace, dataset = target
    credentials = Credentials()
    request = WorkerRequest(action="submit", workspace=str(workspace), job="detached", task=JobRequest(dataset=str(dataset)))
    process = subprocess.run(
        [sys.executable, "-m", "ict_pipeline.runtime.worker", "--ssh"],
        input=request.model_dump_json(), capture_output=True, text=True, check=True, timeout=30,
    )
    assert json.loads(process.stdout)["result"]["status"] == "running"
    status = WorkerRequest(action="status", workspace=str(workspace), job="detached")
    deadline = monotonic() + 30
    while (result := jobs.dispatch(status, credentials))["status"] == "running" and monotonic() < deadline:
        sleep(0.05)
    assert result["status"] == "completed", result
    assert result["attempt"] == 1


def test_saved_docker_or_remote_mode_cannot_accidentally_run_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ICT_EXECUTION_MODE", "remote")
    result = CliRunner().invoke(app, ["run", "--dry-run"])
    assert result.exit_code == 1
    assert "ict runtime" in result.output
