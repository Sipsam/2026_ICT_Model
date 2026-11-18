"""Source lifecycle, backpressure, and a read-only operational snapshot."""

from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Literal

from pydantic import Field, TypeAdapter

from ..inspection.contracts import Contract
from .alignment import Alignment, SampleAligner
from .contracts import CameraFrame, ImageKind, ImageSource, IndexedImageSource, PublicID, SampleReference
from .contracts import SensorKind, SensorObservation, SensorSource, SourceCode, SourceIssue
from .processing import FrameProcessor, ProcessingStopped

StreamState = Literal["disabled", "running", "completed", "degraded", "failed", "stopping", "stopped"]
SourceState = Literal["waiting", "reading", "recovering", "exhausted", "failed", "stopped"]
SOURCE_ID = TypeAdapter(PublicID)


class SourceStatus(Contract):
    source_id: str
    kind: ImageKind | SensorKind
    state: SourceState = "waiting"
    last_error: SourceCode | None = None
    errors: dict[SourceCode, int] = Field(default_factory=dict)


class StreamStatus(Contract):
    schema_version: Literal[1] = 1
    state: StreamState = "disabled"
    stream_id: str | None = None
    model_id: str | None = None
    mode: Literal["mock", "real"] | None = None
    sources: tuple[SourceStatus, ...] = ()
    error: Literal["model_unavailable", "storage_unavailable", "configuration_error"] | None = None
    counters: dict[str, int] = Field(default_factory=dict)
    elapsed_seconds: float = 0
    frames_per_second: float = 0
    inspections_per_second: float = 0
    queue_capacity: int = 0
    queue_depth: int = 0
    active_workers: int = 0
    mean_latency_ms: float | None = None
    max_latency_ms: float | None = None
    frames_completed: int = 0
    total_frames: int | None = None
    estimated_seconds_remaining: float | None = None


@dataclass(frozen=True)
class WorkItem:
    frame: CameraFrame
    alignment: Alignment
    received: float


class StreamRunner:
    """Run a single source pair once. Replay again with a fresh runner to resume."""

    def __init__(
        self, camera: ImageSource, sensor: SensorSource | None, processor: FrameProcessor,
        *, max_observations: int = 10_000, max_camera_references: int = 100_000,
    ) -> None:
        if processor.alignment.shared_clock and (sensor is None or not isinstance(camera, IndexedImageSource)):
            raise ValueError("Timestamp alignment requires a complete camera index and a sensor snapshot.")
        if sensor is None and not processor.alignment.allow_image_only:
            raise ValueError("Image-only inspection requires explicit configuration.")
        if not (1 <= max_observations <= 100_000 and 1 <= max_camera_references <= 100_000):
            raise ValueError("Input index limits must be between 1 and 100000.")
        self.camera = camera
        self.sensor = sensor
        self.processor = processor
        self.max_observations = max_observations
        self.max_camera_references = max_camera_references
        self._queue: Queue[WorkItem] = Queue(processor.settings.queue_capacity)
        self._stop = Event()
        self._input_done = Event()
        self._lock = Lock()
        self._counts: Counter[str] = Counter()
        self._sources = {"camera": SourceStatus(source_id=SOURCE_ID.validate_python(camera.source_id), kind=camera.kind)}
        if sensor is not None:
            self._sources["sensor"] = SourceStatus(source_id=SOURCE_ID.validate_python(sensor.source_id), kind=sensor.kind)
        self._threads: list[Thread] = []
        self._started = 0.0
        self._ended: float | None = None
        self._remaining = 0
        self._total_frames: int | None = None
        self._active = 0
        self._latency_count = 0
        self._latency_sum = 0.0
        self._latency_max = 0.0

    def _count(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counts[name] += amount

    def _source_state(self, role: str, state: SourceState, code: SourceCode | None = None) -> None:
        with self._lock:
            prior = self._sources[role]
            errors = prior.errors.copy()
            if code is not None:
                errors[code] = errors.get(code, 0) + 1
            self._sources[role] = prior.model_copy(update={
                "state": state, "last_error": code or prior.last_error, "errors": errors,
            })

    def _snapshot_sensor(self) -> list[SensorObservation] | None:
        if self.sensor is None or self._stop.is_set():
            return None
        self._source_state("sensor", "reading")
        observations = []
        try:
            for item in self.sensor.observations(self._stop):
                if self._stop.is_set():
                    break
                if len(observations) >= self.max_observations:
                    raise SourceIssue("input_limit")
                if (not isinstance(item, SensorObservation) or item.source_id != self.sensor.source_id
                        or item.kind != self.sensor.kind):
                    raise SourceIssue("invalid_sensor_data")
                observations.append(item)
                self._count("observations_received")
        except SourceIssue as issue:
            self._source_state("sensor", "failed", issue.code)
            return None
        except Exception:
            self._source_state("sensor", "failed", "source_unavailable")
            return None
        self._source_state("sensor", "stopped" if self._stop.is_set() else "exhausted")
        return observations

    def _submit(self, item: WorkItem) -> None:
        if self._queue.full():
            self._count("backpressure_events")
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except Full:
                continue
        self._count("cancelled")

    def _read_frames(self, aligner: SampleAligner | None) -> None:
        frames: Iterator[CameraFrame | SourceIssue] = iter(self.camera.frames(self._stop))
        try:
            for item in frames:
                if self._stop.is_set():
                    break
                if isinstance(item, SourceIssue):
                    self._source_state("camera", "recovering", item.code)
                    self._count("source_events")
                    if item.code in {"invalid_frame", "input_limit"}:
                        self._count("rejected_source_frames")
                    continue
                if not isinstance(item, CameraFrame) or item.source_id != self.camera.source_id or item.kind != self.camera.kind:
                    raise SourceIssue("invalid_frame")
                received = monotonic()
                self._source_state("camera", "reading")
                self._count("frames_received")
                alignment = (aligner.match(item) if aligner is not None else Alignment(
                    reason="no_sensor_source" if self.sensor is None else "sensor_unavailable",
                ))
                if alignment.reason is not None:
                    self._count(f"alignment_{alignment.reason}")
                    if not self.processor.alignment.allow_image_only:
                        self._count("skipped_no_sensor")
                        continue
                    self._count("image_only")
                else:
                    self._count("aligned")
                self._submit(WorkItem(item, alignment, received))
        finally:
            close = getattr(frames, "close", None)
            if close is not None:
                close()

    def _produce(self) -> None:
        aligner = None
        try:
            observations = self._snapshot_sensor()
            if self._stop.is_set():
                return
            if observations is None and self.sensor is not None and not self.processor.alignment.allow_image_only:
                return
            self._source_state("camera", "reading")
            references: list[SampleReference] = []
            if isinstance(self.camera, IndexedImageSource):
                for reference in self.camera.references(self._stop):
                    if self._stop.is_set():
                        return
                    if len(references) >= self.max_camera_references:
                        raise SourceIssue("input_limit")
                    references.append(reference)
                with self._lock:
                    self._total_frames = len(references)
            if observations is not None:
                aligner = SampleAligner(self.processor.alignment, observations, references)
                self._count("duplicate_observations", aligner.duplicate_observations)
            self._read_frames(aligner)
            self._source_state("camera", "stopped" if self._stop.is_set() else "exhausted")
        except SourceIssue as issue:
            self._source_state("camera", "failed", issue.code)
        except Exception:
            self._source_state("camera", "failed", "source_unavailable")
        finally:
            if aligner is not None:
                self._count("unused_observations", aligner.unused_observations)
            self._input_done.set()
            if self._stop.is_set():
                self._discard_pending()

    def _discard_pending(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return
            self._count("cancelled")
            self._queue.task_done()

    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.05)
            except Empty:
                if self._input_done.is_set() and self._queue.empty():
                    return
                continue
            with self._lock:
                self._active += 1
            try:
                outcome = self.processor.process(item.frame, item.alignment, self._stop, lambda: self._count("retries"))
            except ProcessingStopped:
                outcome = "cancelled"
            except Exception:
                outcome = "processing_failed"
            finally:
                latency = (monotonic() - item.received) * 1000
                with self._lock:
                    self._counts[outcome] += 1
                    self._active -= 1
                    self._latency_count += 1
                    self._latency_sum += latency
                    self._latency_max = max(self._latency_max, latency)
                self._queue.task_done()

    def _run_thread(self, function: Callable[[], None]) -> None:
        try:
            function()
        finally:
            with self._lock:
                self._remaining -= 1
                if not self._remaining:
                    self._ended = monotonic()

    def start(self) -> None:
        if self._threads or self._stop.is_set():
            raise RuntimeError("Create a fresh stream runner for another session.")
        self._started = monotonic()
        self._remaining = 1 + self.processor.settings.workers
        self._threads = [Thread(target=self._run_thread, args=(self._produce,), daemon=True, name="ict-input")]
        self._threads.extend(Thread(target=self._run_thread, args=(self._work,), daemon=True, name=f"ict-stream-{index}")
                             for index in range(self.processor.settings.workers))
        for thread in self._threads:
            thread.start()

    def wait(self, timeout: float) -> bool:
        deadline = monotonic() + timeout
        for thread in self._threads:
            thread.join(max(0, deadline - monotonic()))
        return not any(thread.is_alive() for thread in self._threads)

    def close(self) -> bool:
        self._stop.set()
        stopped = self.wait(self.processor.settings.shutdown_seconds)
        self._discard_pending()
        return stopped

    def status(self) -> StreamStatus:
        with self._lock:
            running = bool(self._remaining)
            elapsed = (self._ended or monotonic()) - self._started if self._started else 0.0
            sources = tuple(self._sources.values())
            counts = dict(self._counts)
            failures = sum(counts.get(key, 0) for key in (
                "invalid_image", "inference_failed", "storage_failed", "processing_failed", "skipped_no_sensor",
            ))
            inspections = counts.get("persisted", 0)
            completed = inspections + counts.get("reused", 0) + failures + counts.get("rejected_source_frames", 0)
            eta = None
            if running and not self._stop.is_set() and self._total_frames is not None and completed:
                eta = round(max(0, self._total_frames - completed) * elapsed / completed, 3)
            if self._stop.is_set():
                state = "stopping" if running else "stopped"
            elif any(source.state == "failed" for source in sources):
                state = "degraded" if running or inspections or counts.get("reused") else "failed"
            elif failures or any(source.errors for source in sources):
                state = "degraded"
            else:
                state = "running" if running else "completed" if self._started else "disabled"
            credentials = self.processor.service.credentials
            return StreamStatus(
                state=state, stream_id=credentials.redact_text(self.processor.stream_id),
                model_id=credentials.redact_text(self.processor.model_id),
                mode=self.processor.service.provenance.mode,
                sources=tuple(source.model_copy(update={"source_id": credentials.redact_text(source.source_id)})
                              for source in sources),
                counters=counts, elapsed_seconds=round(elapsed, 3),
                frames_per_second=round(counts.get("frames_received", 0) / elapsed, 3) if elapsed else 0,
                inspections_per_second=round(inspections / elapsed, 3) if elapsed else 0,
                queue_capacity=self._queue.maxsize, queue_depth=self._queue.qsize(), active_workers=self._active,
                mean_latency_ms=round(self._latency_sum / self._latency_count, 3) if self._latency_count else None,
                max_latency_ms=round(self._latency_max, 3) if self._latency_count else None,
                frames_completed=completed, total_frames=self._total_frames, estimated_seconds_remaining=eta,
            )
