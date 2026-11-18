"""Run dependency-ordered stages and verify committed artifacts on reuse."""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from types import MappingProxyType
from uuid import uuid4

from .model import PipelineError, Stage, StageContext, StageGraph, StageResult, valid_id
from .store import (
    FingerprintError, RunState, RunStore, StageRecord, digest_json, fingerprint_path,
    mark_interrupted, timestamp,
)

_FAILURE_DETAILS = {
    StageResult.FAILED: "reported failure. Check its adapter and input data",
    StageResult.INVALID_DATA: "reported invalid data. Check the stage inputs",
    StageResult.MISSING_REQUIREMENT: "requires a missing tool, setting, or resource. Check its prerequisites",
}


@dataclass(frozen=True)
class PlanStep:
    stage: str
    action: str


@dataclass(frozen=True)
class StageEvent:
    stage: str
    attempt: str
    status: str
    stages_completed: int
    stages_total: int
    elapsed_seconds: float
    completed: int | None = None
    total: int | None = None
    eta_seconds: float | None = None
    detail: str | None = None


class StageRunner:
    def __init__(
        self,
        stages: Iterable[Stage],
        workspace: Path,
        settings: Mapping[str, str],
        *,
        on_event: Callable[[StageEvent], None] | None = None,
    ) -> None:
        self.graph = StageGraph(stages)
        self.workspace = workspace.resolve()
        self.runs_dir = self.workspace / "runs"
        self.settings = dict(settings)
        self.on_event = on_event

    def plan(
        self, selection: tuple[str, ...], *, run_id: str | None = None, force: bool = False
    ) -> tuple[PlanStep, ...]:
        selection = self._ordered_selection(selection)
        store = RunStore(self.runs_dir, run_id if run_id is not None else uuid4().hex)
        state = store.status() if run_id is not None else None
        forced = set(selection) if force else set()
        return self._plan(store, state, selection, forced)

    def run(
        self, run_id: str, selection: tuple[str, ...], *, force: bool = False
    ) -> RunState:
        selection = self._ordered_selection(selection)
        store = RunStore(self.runs_dir, run_id)
        with store.lock():
            now = timestamp()
            state = store.read() or RunState(
                run_id=run_id, created_at=now, updated_at=now, selection=list(selection), pending_force=[]
            )
            forced = set(selection) if force else set()
            self._preflight(store, state, selection, forced)
            return self._execute(store, state, selection, forced)

    def resume(self, run_id: str | None = None) -> RunState:
        store = self._existing_store(run_id)
        with store.lock():
            state = store.read()
            if state is None:
                raise PipelineError("Run state is missing. Start a run with ict run.")
            selection = self._ordered_selection(tuple(state.selection))
            forced = set(state.pending_force)
            self._preflight(store, state, selection, forced)
            return self._execute(store, state, selection, forced)

    def status(self, run_id: str | None = None) -> RunState:
        state = self._existing_store(run_id).status()
        if state is None:
            raise PipelineError("Run state is missing. Start a run with ict run.")
        return state

    def _existing_store(self, run_id: str | None) -> RunStore:
        if run_id is None:
            latest: tuple[int, str] | None = None
            if self.runs_dir.is_dir():
                for directory in self.runs_dir.iterdir():
                    if not valid_id(directory.name):
                        continue
                    try:
                        candidate = ((directory / "state.json").stat().st_mtime_ns, directory.name)
                    except (FileNotFoundError, NotADirectoryError):
                        continue
                    latest = max(latest, candidate) if latest else candidate
            if latest is None:
                raise PipelineError("No saved run found. Start one with ict run.")
            run_id = latest[1]
        store = RunStore(self.runs_dir, run_id)
        if not store.state_path.is_file():
            raise PipelineError("Run state is missing. Check --run-id or start a run with ict run.")
        return store

    def _ordered_selection(self, selection: tuple[str, ...]) -> tuple[str, ...]:
        if not selection or any(name not in self.graph.stages for name in selection):
            raise PipelineError("The run selection references missing stages. Review the DAG or start a new run.")
        return tuple(name for name in self.graph.order if name in selection)

    def _required_order(self, selection: tuple[str, ...]) -> tuple[str, ...]:
        required = set(selection)
        for name in reversed(self.graph.order):
            if name in required:
                required.update(self.graph.stages[name].dependencies)
        return tuple(name for name in self.graph.order if name in required)

    def _preflight(
        self, store: RunStore, state: RunState, selection: tuple[str, ...], forced: set[str]
    ) -> None:
        # Only partial selections that exclude prerequisites need simulation
        # before execution. The full runner checks caches as it goes.
        if self._required_order(selection) != selection:
            self._plan(store, state, selection, forced)

    def _plan(
        self, store: RunStore, state: RunState | None, selection: tuple[str, ...], forced: set[str]
    ) -> tuple[PlanStep, ...]:
        reusable: dict[str, StageRecord] = {}
        steps: list[PlanStep] = []
        for name in self._required_order(selection):
            stage = self.graph.stages[name]
            record = state.stages.get(name) if state else None
            if name not in forced and self._reusable(store, stage, record, reusable):
                assert record is not None
                reusable[name] = record
                action = "reuse" if name in selection else "reuse prerequisite"
            elif name not in selection:
                raise self._dependency_error(name)
            else:
                action = "run" if stage.execute else "adapter unavailable"
            steps.append(PlanStep(name, action))
        return tuple(steps)

    @staticmethod
    def _dependency_error(name: str) -> PipelineError:
        return PipelineError(
            f"Prerequisite {name} has no reusable result for this selection. "
            "Include it and any missing prerequisites in the run."
        )

    def _stage_settings(self, stage: Stage) -> dict[str, str]:
        return {key: self.settings[key] for key in stage.config_keys if key in self.settings}

    def _inputs(
        self, store: RunStore, stage: Stage, available: Mapping[str, StageRecord]
    ) -> tuple[dict[str, Path], str]:
        paths: dict[str, Path] = {}
        fingerprints: dict[str, tuple[str, str]] = {}
        for item in stage.inputs:
            try:
                if item.producer is None:
                    path = (self.workspace / item.path).resolve()
                    if path.is_relative_to(self.runs_dir) or self.runs_dir.is_relative_to(path):
                        raise FingerprintError
                    identity = path.as_posix()
                else:
                    path = store.artifact_dir(item.producer, available[item.producer].attempt) / item.path
                    identity = item.path.as_posix()
                fingerprints[item.name] = (identity, fingerprint_path(path))
            except (OSError, RuntimeError, ValueError):
                raise PipelineError(
                    f"Cannot verify {stage.name} input {item.name}. "
                    "Check that it exists, is readable and stable, and contains regular files."
                ) from None
            paths[item.name] = path
        fingerprint = digest_json({
            "stage": stage.name,
            "version": stage.version,
            "config": digest_json(self._stage_settings(stage)),
            "inputs": fingerprints,
            "outputs": [path.as_posix() for path in stage.outputs],
            "dependencies": {
                name: (available[name].attempt, available[name].fingerprint, available[name].output_fingerprint)
                for name in stage.dependencies
            },
        })
        return paths, fingerprint

    def _reusable(
        self, store: RunStore, stage: Stage, record: StageRecord | None, available: Mapping[str, StageRecord]
    ) -> bool:
        if (
            record is None or record.status != "completed" or record.output_fingerprint is None
            or any(name not in available for name in stage.dependencies)
        ):
            return False
        try:
            _, current = self._inputs(store, stage, available)
            return record.fingerprint == current and record.output_fingerprint == self._verify_outputs(
                stage, store.artifact_dir(stage.name, record.attempt)
            )
        except (FingerprintError, PipelineError):
            return False

    @staticmethod
    def _verify_outputs(stage: Stage, directory: Path) -> str:
        if not directory.is_dir() or any(not (directory / path).exists() for path in stage.outputs):
            raise FingerprintError
        return fingerprint_path(directory)

    def _emit(self, store: RunStore, event: StageEvent) -> None:
        store.log({"timestamp": timestamp(), **asdict(event)})
        if self.on_event is not None:
            self.on_event(event)

    def _execute(
        self, store: RunStore, state: RunState, selection: tuple[str, ...], forced: set[str]
    ) -> RunState:
        abandoned = [(name, record) for name, record in state.stages.items() if record.status == "running"]
        mark_interrupted(state)
        state.status = "running"
        state.selection = list(selection)
        state.pending_force = [name for name in selection if name in forced]
        store.save(state)
        available: dict[str, StageRecord] = {}
        completed = 0
        try:
            for name, record in abandoned:
                self._emit(store, StageEvent(name, record.attempt, "interrupted", 0, len(selection), 0,
                                             detail=record.error))
            for name in self._required_order(selection):
                stage = self.graph.stages[name]
                record = state.stages.get(name)
                checked_at = monotonic()
                if name in selection and name not in forced and record and record.status == "completed":
                    self._emit(store, StageEvent(name, record.attempt, "checking cache", completed, len(selection), 0))
                if name not in forced and self._reusable(store, stage, record, available):
                    assert record is not None
                    available[name] = record
                    if name in selection:
                        completed += 1
                        self._emit(store, StageEvent(
                            name, record.attempt, "reused", completed, len(selection), monotonic() - checked_at,
                        ))
                elif name not in selection:
                    raise self._dependency_error(name)
                else:
                    available[name] = self._run_stage(store, state, stage, completed, len(selection), available)
                    completed += 1
            state.status = "completed"
            store.save(state)
        except KeyboardInterrupt:
            mark_interrupted(state)
            store.save(state)
            raise
        except (PipelineError, OSError):
            state.status = "failed"
            store.save(state)
            raise
        return state

    def _run_stage(
        self, store: RunStore, state: RunState, stage: Stage, completed: int, total: int,
        available: Mapping[str, StageRecord],
    ) -> StageRecord:
        name, attempt = stage.name, uuid4().hex
        started = monotonic()
        settings = self._stage_settings(stage)
        record = StageRecord(
            status="running", attempt=attempt, started_at=timestamp(), config_fingerprint=digest_json(settings),
        )
        state.stages[name] = record

        def emit(
            status: str, *, units: int | None = None, expected: int | None = None, detail: str | None = None
        ) -> None:
            elapsed = monotonic() - started
            eta = elapsed * (expected - units) / units if expected is not None and units else None
            self._emit(store, StageEvent(
                name, attempt, status, completed + int(status == "completed"), total, elapsed, units, expected,
                eta, detail,
            ))

        def progress(units: int, expected: int | None) -> None:
            if (
                type(units) is not int or units < 0
                or (expected is not None and (type(expected) is not int or expected < units))
            ):
                raise ValueError("Invalid stage progress.")
            emit("progress", units=units, expected=expected)

        try:
            store.save(state)
            emit("running")
            if stage.execute is None:
                raise PipelineError(f"Stage {name} requires an adapter. See the pipeline phases in PLANS.md.")
            inputs, record.fingerprint = self._inputs(store, stage, available)
            temporary = store.working_dir(name, attempt)
            final = store.artifact_dir(name, attempt)
            temporary.mkdir(parents=True)
            context = StageContext(MappingProxyType(inputs), MappingProxyType(settings), temporary, final, progress)
            try:
                result = stage.execute(context)
            except OSError:
                raise PipelineError(
                    f"Stage {name} encountered an I/O error. Check file access and available storage, then resume."
                ) from None
            except Exception:
                raise PipelineError(
                    f"Stage {name} raised an exception. Check its input data and adapter setup, then resume."
                ) from None
            if result is not StageResult.SUCCESS:
                reason = "returned an invalid result. Check its adapter contract"
                if isinstance(result, StageResult):
                    reason = _FAILURE_DETAILS[result]
                raise PipelineError(f"Stage {name} {reason}, then resume.")
            try:
                output_fingerprint = self._verify_outputs(stage, temporary)
            except FingerprintError:
                raise PipelineError(f"Stage {name} has missing or unreadable outputs. Check its output contract.") from None
            if self._inputs(store, stage, available)[1] != record.fingerprint:
                raise PipelineError(f"Inputs changed while {name} was running. Stabilize them and resume.")
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists() or final.is_symlink():
                raise PipelineError(f"Artifact destination already exists for {name}. Resume to allocate a new attempt.")
            temporary.rename(final)
        except (KeyboardInterrupt, OSError, PipelineError) as exc:
            record.status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            if isinstance(exc, KeyboardInterrupt):
                record.error = "Stage cancelled. Resume to retry it."
            elif isinstance(exc, PipelineError):
                record.error = str(exc)
            else:
                record.error = f"Could not write run files for {name}. Check permissions and storage, then resume."
            record.finished_at = timestamp()
            record.elapsed_seconds = monotonic() - started
            store.save(state)
            emit(record.status, detail=record.error)
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise PipelineError(record.error) from None

        record.status = "completed"
        record.output_fingerprint = output_fingerprint
        record.finished_at = timestamp()
        record.elapsed_seconds = monotonic() - started
        if name in state.pending_force:
            state.pending_force.remove(name)
        store.save(state)
        emit("completed")
        return record
