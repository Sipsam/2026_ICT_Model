import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest
from typer.testing import CliRunner

from ict_pipeline import cli
from ict_pipeline.pipeline import PipelineError, Stage, StageContext, StageInput, StageResult, StageRunner
from ict_pipeline.pipeline.runner import StageEvent
from ict_pipeline.pipeline.store import RunStore

PAYLOAD = Path("payload.txt")
SECRET = "synthetic-secret-for-runner-test"


@dataclass
class MockDag:
    root: Path
    calls: Counter[str] = field(default_factory=Counter)
    events: list[StageEvent] = field(default_factory=list)
    failure: str | None = None

    def source(self, context: StageContext) -> StageResult:
        self.calls["source"] += 1
        (context.output_dir / PAYLOAD).write_bytes(context.inputs["input"].read_bytes())
        if self.failure == "change input":
            context.inputs["input"].write_text("changed", encoding="utf-8")
        return StageResult.SUCCESS

    def transform(self, context: StageContext) -> StageResult:
        self.calls["transform"] += 1
        context.progress(1, 2)
        if self.failure != "missing output":
            contents = context.inputs["source"].read_text(encoding="utf-8")
            (context.output_dir / PAYLOAD).write_text(contents + context.settings["model"], encoding="utf-8")
        if self.failure == "exception":
            raise RuntimeError(SECRET)
        if self.failure == "disk":
            raise OSError(SECRET)
        if self.failure == "cancel":
            raise KeyboardInterrupt
        if self.failure == "failed":
            return StageResult.FAILED
        if self.failure == "invalid data":
            return StageResult.INVALID_DATA
        if self.failure == "missing requirement":
            return StageResult.MISSING_REQUIREMENT
        if self.failure == "invalid result":
            return cast(StageResult, SECRET)
        assert not context.artifact_dir.exists()
        return StageResult.SUCCESS

    def side(self, context: StageContext) -> StageResult:
        self.calls["side"] += 1
        (context.output_dir / PAYLOAD).write_bytes(context.inputs["source"].read_bytes())
        return StageResult.SUCCESS

    def report(self, context: StageContext) -> StageResult:
        self.calls["report"] += 1
        contents = b" ".join(path.read_bytes() for path in context.inputs.values())
        (context.output_dir / PAYLOAD).write_bytes(contents)
        return StageResult.SUCCESS

    def stages(self, version: str = "1") -> tuple[Stage, ...]:
        source = StageInput("source", PAYLOAD, "source")
        report_inputs = tuple(StageInput(name, PAYLOAD, name) for name in ("transform", "side"))
        return (
            Stage("source", (), (StageInput("input", Path("input.txt")),), (PAYLOAD,), (), "1", self.source),
            Stage("transform", ("source",), (source,), (PAYLOAD,), ("model",), version, self.transform),
            Stage("side", ("source",), (source,), (PAYLOAD,), (), "1", self.side),
            Stage("report", ("transform", "side"), report_inputs, (PAYLOAD,), (), "1", self.report),
        )

    def runner(self, mode: str = "A", version: str = "1") -> StageRunner:
        return StageRunner(self.stages(version), self.root, {"model": mode, "unused": SECRET},
                           on_event=self.events.append)


@pytest.fixture
def dag(tmp_path: Path) -> MockDag:
    (tmp_path / "input.txt").write_text("fixture", encoding="utf-8")
    return MockDag(tmp_path)


def test_dag_selection_dry_run_and_cache(dag: MockDag) -> None:
    runner = dag.runner()
    graph = runner.graph
    assert graph.select("transform", "side", ()) == ("transform", "side")
    assert graph.select(None, None, ("side", "transform", "side")) == ("transform", "side")
    with pytest.raises(PipelineError, match="by itself"):
        graph.select("source", None, ("side",))
    with pytest.raises(PipelineError, match="Unknown stage"):
        graph.select(None, "missing", ())
    with pytest.raises(PipelineError, match="must precede"):
        graph.select("side", "source", ())
    assert [step.action for step in runner.plan(graph.order, run_id="first")] == ["run"] * 4
    assert not runner.runs_dir.exists()
    assert not dag.calls

    state = runner.run("first", graph.order)
    assert state.status == "completed"
    store = RunStore(runner.runs_dir, "first")
    artifact = store.artifact_dir("report", state.stages["report"].attempt) / PAYLOAD
    assert artifact.read_text() == "fixtureA fixture"
    assert all(record.finished_at and record.elapsed_seconds is not None for record in state.stages.values())
    assert all(record.fingerprint and record.config_fingerprint for record in state.stages.values())
    progress = next(event for event in dag.events if event.status == "progress")
    assert (progress.completed, progress.total) == (1, 2)
    assert progress.eta_seconds is not None
    assert dag.events[-1].stages_completed == dag.events[-1].stages_total == 4
    assert [step.action for step in runner.plan(graph.order, run_id="first")] == ["reuse"] * 4
    again = runner.resume()
    assert again.stages == state.stages
    assert dag.calls == {"source": 1, "transform": 1, "side": 1, "report": 1}
    events = (store.directory / "events.jsonl").read_text()
    assert all("timestamp" in json.loads(line) for line in events.splitlines())
    assert SECRET not in events
    assert SECRET not in store.state_path.read_text()
    assert state.configuration == {"model": "A"}
    assert state.started_at and state.finished_at
    assert state.runtime["git_commit"] is None


@pytest.mark.parametrize(
    ("kind", "changed"),
    [
        ("settings", {"transform", "report"}),
        ("version", {"transform", "report"}),
        ("source bytes", {"source", "transform", "side", "report"}),
        ("corrupt artifact", {"transform", "report"}),
        ("missing artifact", {"transform", "report"}),
        ("unexpected artifact", {"transform", "report"}),
        ("force", {"source", "transform", "side", "report"}),
    ],
)
def test_invalidates_only_affected_work(dag: MockDag, kind: str, changed: set[str]) -> None:
    runner = dag.runner()
    initial = runner.run("cache", runner.graph.order)
    transform = RunStore(runner.runs_dir, "cache").artifact_dir("transform", initial.stages["transform"].attempt)
    if kind == "settings":
        runner = dag.runner(mode="B")
    elif kind == "version":
        runner = dag.runner(version="2")
    elif kind == "source bytes":
        (dag.root / "input.txt").write_text("new source")
    elif kind == "corrupt artifact":
        (transform / PAYLOAD).write_text("corrupt!")
    elif kind == "missing artifact":
        (transform / PAYLOAD).unlink()
    elif kind == "unexpected artifact":
        (transform / "extra.txt").write_text("unexpected")
    refreshed = runner.run("cache", runner.graph.order, force=(kind == "force"))
    assert {name for name in initial.stages if initial.stages[name].attempt != refreshed.stages[name].attempt} == changed
    assert transform.is_dir()


def test_partial_runs_require_valid_transitive_dependencies(dag: MockDag) -> None:
    runner = dag.runner()
    with pytest.raises(PipelineError, match="Prerequisite source"):
        runner.plan(("report",))
    with pytest.raises(PipelineError, match="Prerequisite source"):
        runner.run("partial", ("report",))
    assert not dag.calls
    initial = runner.run("partial", runner.graph.order)
    plan = runner.plan(("report",), run_id="partial")
    assert [step.action for step in plan] == ["reuse prerequisite"] * 3 + ["reuse"]
    runner.run("partial", ("report",), force=True)
    assert dag.calls["report"] == 2

    with pytest.raises(PipelineError, match="Prerequisite transform"):
        runner.run("partial", ("source", "report"), force=True)
    assert dag.calls["source"] == 1
    assert runner.status("partial").stages["source"] == initial.stages["source"]
    (dag.root / "input.txt").write_text("changed")
    with pytest.raises(PipelineError, match="Prerequisite source"):
        runner.resume("partial")


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        ("exception", "raised an exception"),
        ("disk", "I/O error"),
        ("cancel", "cancelled"),
        ("failed", "reported failure"),
        ("missing requirement", "missing tool"),
        ("invalid data", "invalid data"),
        ("invalid result", "invalid result"),
        ("missing output", "missing or unreadable outputs"),
    ],
)
def test_failure_leaves_partial_unpublished_then_resumes(dag: MockDag, failure: str, message: str) -> None:
    runner = dag.runner()
    dag.failure = failure
    error = KeyboardInterrupt if failure == "cancel" else PipelineError
    with pytest.raises(error) as exc_info:
        runner.run("recovery", runner.graph.order)
    assert SECRET not in str(exc_info.value)
    state = runner.status()
    status = "interrupted" if failure == "cancel" else "failed"
    assert state.status == state.stages["transform"].status == status
    bad = state.stages["transform"]
    assert bad.error and message in bad.error
    assert bad.output_fingerprint is None
    store = RunStore(runner.runs_dir, "recovery")
    assert not store.artifact_dir("transform", bad.attempt).exists()
    assert store.working_dir("transform", bad.attempt).is_dir()
    assert SECRET not in store.state_path.read_text()
    assert SECRET not in (store.directory / "events.jsonl").read_text()

    dag.failure = None
    final = runner.resume()
    assert final.status == "completed"
    assert final.stages["source"] == state.stages["source"]
    assert final.stages["transform"].attempt != bad.attempt
    assert dag.calls == {"source": 1, "transform": 2, "side": 1, "report": 1}


def test_input_change_during_stage_and_missing_input(dag: MockDag) -> None:
    runner = dag.runner()
    dag.failure = "change input"
    with pytest.raises(PipelineError, match="Inputs changed"):
        runner.run("unstable", runner.graph.order)
    assert runner.status().stages["source"].status == "failed"
    (dag.root / "input.txt").unlink()
    with pytest.raises(PipelineError, match="Cannot verify source input input"):
        runner.resume()
    (dag.root / "input.txt").write_text("stable")
    dag.failure = None
    assert runner.resume().status == "completed"


def test_publish_failure_keeps_state_retryable(dag: MockDag, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = dag.runner()
    with monkeypatch.context() as patch:
        def fail_rename(self: Path, target: Path) -> Path:
            raise PermissionError(SECRET)

        patch.setattr(Path, "rename", fail_rename)
        with pytest.raises(PipelineError, match="Check permissions and storage"):
            runner.run("publish", runner.graph.order)
    assert runner.status().stages["source"].status == "failed"
    assert not list((runner.runs_dir / "publish" / "artifacts" / "source").iterdir())
    assert runner.resume().status == "completed"


def test_runner_preserves_an_existing_publication_destination(dag: MockDag) -> None:
    source = dag.stages()[0]

    def collide(context: StageContext) -> StageResult:
        context.artifact_dir.mkdir(parents=True)
        (context.artifact_dir / PAYLOAD).write_text("existing output")
        return dag.source(context)

    runner = StageRunner((replace(source, execute=collide),), dag.root, {})
    with pytest.raises(PipelineError, match="destination already exists"):
        runner.run("collision", ("source",))
    failed = runner.status().stages["source"]
    assert failed.status == "failed"
    store = RunStore(runner.runs_dir, "collision")
    existing = store.artifact_dir("source", failed.attempt) / PAYLOAD
    assert existing.read_text() == "existing output"
    assert StageRunner((source,), dag.root, {}).resume().status == "completed"
    assert existing.read_text() == "existing output"


def test_pending_force_survives_cancellation(dag: MockDag) -> None:
    runner = dag.runner()
    original = runner.run("force", runner.graph.order)
    dag.failure = "cancel"
    # These branches are independent of one another and depend only on the
    # cached source, so both can be forced in a partial selection.
    with pytest.raises(KeyboardInterrupt):
        runner.run("force", ("transform", "side"), force=True)
    interrupted = runner.status()
    assert interrupted.pending_force == ["transform", "side"]
    assert interrupted.stages["side"] == original.stages["side"]
    dag.failure = None
    recovered = runner.resume()
    assert recovered.pending_force == []
    assert recovered.stages["side"].attempt != original.stages["side"].attempt
    assert recovered.stages["source"] == original.stages["source"]


@pytest.mark.parametrize("after_rename", [False, True])
def test_process_loss_recovers_lock_and_retries_uncommitted_generation(tmp_path: Path, after_rename: bool) -> None:
    worker = dedent("""\
        import os
        import sys
        from pathlib import Path

        from ict_pipeline.pipeline import Stage, StageResult, StageRunner
        from ict_pipeline.pipeline.store import RunStore

        def action(context):
            (context.output_dir / "payload.txt").write_text("uncommitted")
            if sys.argv[1] == "before":
                os._exit(17)
            return StageResult.SUCCESS

        save = RunStore.save
        def abort_before_commit(self, state):
            if state.stages.get("source") and state.stages["source"].status == "completed":
                os._exit(17)
            save(self, state)

        RunStore.save = abort_before_commit
        stage = Stage("source", (), (), (Path("payload.txt"),), (), "1", action)
        StageRunner((stage,), Path.cwd(), {}).run("lost", ("source",))
    """)
    crashed = subprocess.run(
        [sys.executable, "-c", worker, "after" if after_rename else "before"],
        cwd=tmp_path, capture_output=True, text=True, timeout=20, check=False,
    )
    assert crashed.returncode == 17, crashed.stderr

    def recover(context: StageContext) -> StageResult:
        (context.output_dir / PAYLOAD).write_text("committed")
        return StageResult.SUCCESS

    runner = StageRunner((Stage("source", (), (), (PAYLOAD,), (), "1", recover),), tmp_path, {})
    store = RunStore(runner.runs_dir, "lost")
    state = runner.status()
    uncommitted = state.stages["source"]
    assert uncommitted.status == state.status == "interrupted"
    assert json.loads(store.state_path.read_text())["status"] == "running"
    assert store.artifact_dir("source", uncommitted.attempt).exists() == after_rename
    result = runner.resume()
    assert result.stages["source"].attempt != uncommitted.attempt
    assert (store.artifact_dir("source", result.stages["source"].attempt) / PAYLOAD).read_text() == "committed"
    assert "Execution stopped before committing" in (store.directory / "events.jsonl").read_text()


def test_concurrent_writer_denied_while_status_remains_readable(dag: MockDag) -> None:
    runner = dag.runner()
    runner.run("locked", ("source",))
    store = RunStore(runner.runs_dir, "locked")
    worker = dedent("""\
        from pathlib import Path
        from ict_pipeline.pipeline import PipelineError, Stage, StageRunner

        stage = Stage("source", (), (), (Path("payload.txt"),), (), "1")
        runner = StageRunner((stage,), Path.cwd(), {})
        assert runner.status("locked").status == "running"
        try:
            runner.resume("locked")
        except PipelineError as exc:
            assert "is active" in str(exc)
        else:
            raise AssertionError("A second writer acquired the run lock")
    """)
    with store.lock():
        state = store.read()
        assert state is not None
        state.status = "running"
        store.save(state)
        result = subprocess.run([sys.executable, "-c", worker], cwd=dag.root, capture_output=True,
                                text=True, timeout=20, check=False)
        assert result.returncode == 0, result.stderr
        assert runner.run("independent", ("source",)).status == "completed"
    assert runner.resume("locked").status == "completed"


def test_bad_state_and_invalid_id_do_not_expose_raw_contents(dag: MockDag) -> None:
    runner = dag.runner()
    runner.run("state", ("source",))
    store = RunStore(runner.runs_dir, "state")
    store.state_path.write_text(SECRET)
    with pytest.raises(PipelineError, match="state is invalid") as exc_info:
        runner.resume()
    assert SECRET not in str(exc_info.value)
    assert store.state_path.read_text() == SECRET
    for run_id in ("../escape", "CON", "com1", "a/b"):
        with pytest.raises(PipelineError, match="Run IDs"):
            runner.run(run_id, ("source",))


def test_dag_rejects_cycles_and_invalid_artifact_contracts(dag: MockDag) -> None:
    source, transform, *_ = dag.stages()
    assert StageRunner(reversed(dag.stages()), dag.root, {}).graph.order == dag.runner().graph.order
    with pytest.raises(PipelineError, match="Dependency cycle"):
        StageRunner((replace(source, dependencies=("transform",)), transform), dag.root, {})
    with pytest.raises(PipelineError, match="unknown dependency"):
        StageRunner((transform,), dag.root, {})
    with pytest.raises(PipelineError, match="portable output paths"):
        StageRunner((replace(source, outputs=(Path("../escape"),)),), dag.root, {})
    with pytest.raises(PipelineError, match="declared dependency output"):
        StageRunner((replace(source, outputs=(Path("other"),)), transform), dag.root, {})


def test_cli_production_preview_and_missing_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    client = CliRunner()
    assert "No saved run found" in client.invoke(cli.app, ["status"]).stderr
    preview = client.invoke(cli.app, ["run", "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert [line.split()[0] for line in preview.stdout.splitlines()] == [
        "audit", "manifest", "convert", "validate_detection",
    ]
    assert preview.stdout.count("adapter unavailable") == 0
    full = client.invoke(cli.app, ["run", "--from", "download", "--until", "report", "--dry-run"])
    assert full.exit_code == 0, full.output
    assert full.stdout.count("adapter unavailable") == 2
    assert all(f"{stage}   run" in full.stdout for stage in ("train", "validate_models", "evaluate", "report"))
    assert list(tmp_path.iterdir()) == []
    result = client.invoke(cli.app, ["run", "--run-id", "production"])
    assert result.exit_code == 1
    assert "Cannot verify audit input dataset" in result.stderr
    state = client.invoke(cli.app, ["status"])
    assert state.exit_code == 0
    assert "Run production   failed" in state.stdout
    assert "audit   failed" in state.stdout
    assert not (tmp_path / "runs" / "production" / "artifacts").exists()


def test_cli_partial_force_status_and_resume(dag: MockDag, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(dag.root)
    monkeypatch.setattr(cli, "pipeline_stages", lambda **_: dag.stages())
    client = CliRunner()
    profile = dag.root / "profile.yaml"
    profile.write_text("model: rt_detr\n", encoding="utf-8")
    run = ["run", "--run-id", "fixture", "--config", str(profile)]

    dag.failure = "cancel"
    cancelled = client.invoke(cli.app, [*run, "--until", "transform"])
    assert cancelled.exit_code == 130, cancelled.output
    assert "elapsed" in cancelled.stdout and "ETA" in cancelled.stdout
    interrupted = client.invoke(cli.app, ["status", "--run-id", "fixture"])
    assert "transform   interrupted" in interrupted.stdout
    dag.failure = None
    resumed = client.invoke(cli.app, ["resume", "--run-id", "fixture", "--config", str(profile)])
    assert resumed.exit_code == 0, resumed.output
    assert "source reused" in resumed.stdout
    selected = client.invoke(cli.app, [*run, "--from", "side", "--until", "report"])
    assert selected.exit_code == 0, selected.output
    preview = client.invoke(cli.app, [*run, "--only", "report", "--dry-run", "--force"])
    assert preview.exit_code == 0, preview.output
    assert "transform   reuse prerequisite" in preview.stdout
    assert "report   run" in preview.stdout
    forced = client.invoke(cli.app, [*run, "--only", "report", "--force"])
    assert forced.exit_code == 0, forced.output
    assert dag.calls == {"source": 1, "transform": 2, "side": 1, "report": 2}


def test_cli_does_not_accept_a_credential_as_a_run_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIHUB_API_KEY", SECRET)
    result = CliRunner().invoke(cli.app, ["run", "--run-id", SECRET])
    assert result.exit_code == 1
    assert SECRET not in result.output
    assert not (tmp_path / "runs").exists()
