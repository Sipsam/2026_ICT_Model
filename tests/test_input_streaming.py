import json
import socket
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Barrier, Event
from time import monotonic, sleep
from uuid import UUID

import pytest
from PIL import Image
from pydantic import ValidationError

from ict_pipeline.api.history import HistoryQuery
from ict_pipeline.api.repository import FileInspectionRepository
from ict_pipeline.api.schemas import StoredInspection
from ict_pipeline.api import sqlite_repository
from ict_pipeline.api.sqlite_repository import SQLiteInspectionRepository
from ict_pipeline.api.sqlite_schema import InspectionDatabaseBusy
from ict_pipeline.evaluation.contracts import Prediction
from ict_pipeline.inspection.contracts import TransientInspectionError
from ict_pipeline.inspection.service import InspectionService, MockPredictor
from ict_pipeline.models.dataset import Sample
from ict_pipeline.streaming import camera as camera_module
from ict_pipeline.streaming.alignment import SampleAligner
from ict_pipeline.streaming.camera import OpenCVCamera
from ict_pipeline.streaming.contracts import CameraFrame, SampleReference, SensorObservation, SourceIssue
from ict_pipeline.streaming.processing import FrameProcessor
from ict_pipeline.streaming.replay import DirectoryReplay, SensorReplay
from ict_pipeline.streaming.runner import StreamRunner
from ict_pipeline.streaming.settings import AlignmentSettings, DirectorySettings, ProcessingSettings, RTSPSettings
from ict_pipeline.streaming.settings import SensorReplaySettings, USBSettings


STAMP = "2026-10-30T10:00:00+09:00"
LIMITS = (16_384, 100, 100, 10_000)


def images(root: Path, count: int) -> Path:
    directory = root / "images"
    directory.mkdir()
    for n in range(count):
        Image.new("RGB", (16, 16), (n, 0, 0)).save(directory / f"{n}.png")
    return directory


def observation(
    sequence: str = "S1", timestamp: str = STAMP, **references: str,
) -> SensorObservation:
    return SensorObservation(source_id="sensor", sequence_id=sequence, kind="json", timestamp=timestamp,
                             values={"temperature": 0}, **references)


def until(condition: Callable[[], bool], timeout: float = 5) -> None:
    deadline = monotonic() + timeout
    while not condition():
        assert monotonic() < deadline, "Timed out waiting for the stream"
        sleep(0.005)


def test_directory_order_and_interruptible_pacing(tmp_path: Path) -> None:
    directory = images(tmp_path, 2)
    manifest = tmp_path / "frames.json"
    manifest.write_text(json.dumps({
        "0.png": {"timestamp": STAMP, "sample_id": "part-0"},
        "1.png": {"timestamp": "2026-10-30T00:59:00Z", "sample_id": "part-1"},
    }), encoding="utf-8")

    def replay(order: str, rate: float = 0) -> DirectoryReplay:
        settings = DirectorySettings(source_id="replay-camera", directory=directory, manifest=manifest,
                                     order=order, frames_per_second=rate)
        return DirectoryReplay(settings, max_image_bytes=LIMITS[0])

    forward = list(replay("name").frames(Event()))
    assert [frame.sample_id for frame in forward] == ["part-0", "part-1"]
    assert list(replay("reverse_name").frames(Event())) == forward[::-1]
    assert list(replay("timestamp").frames(Event())) == forward[::-1]
    assert forward[0].timestamp == STAMP
    stop = Event()
    paced = replay("name", 0.001).frames(stop)
    assert next(paced) == forward[0]
    stop.set()
    assert list(paced) == []


def test_replay_csv_and_json_preserve_units_missing_and_zero(tmp_path: Path) -> None:
    csv_path = tmp_path / "sensor.csv"
    csv_path.write_text(f"sequence_id,timestamp,sample_id,temperature(°C),vibration\nS1,{STAMP},part-0,0,\n",
                        encoding="utf-8")
    json_path = tmp_path / "sensor.json"
    json_path.write_text(json.dumps([{
        "sequence_id": "S1", "timestamp": STAMP, "sample_id": "part-0",
        "values": {"temperature": 0, "vibration": None},
    }]), encoding="utf-8")
    for kind, path in (("csv", csv_path), ("json", json_path)):
        [item] = SensorReplay(SensorReplaySettings(kind=kind, source_id="sensor", path=path)).observations(Event())
        assert item.timestamp == STAMP
        assert item.values == {"temperature": 0.0, "vibration": None}
        assert item.measurements == {"temperature": 0.0}
        assert item.units["temperature"] == "°C"


@pytest.mark.parametrize("row", [
    {"values": {"defect_detected": 1}},
    {"values": {"temperature": True}},
    {"values": {"temperature": float("inf")}},
    {"values": {"temperature": 1}, "units": {"temperature": "°F"}},
    {"timestamp": "yesterday"},
])
def test_malformed_measurements_are_rejected(tmp_path: Path, row: dict[str, object]) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([{"sequence_id": "S1", "timestamp": STAMP, **row}]), encoding="utf-8")
    with pytest.raises(SourceIssue, match="invalid_sensor_data"):
        list(SensorReplay(SensorReplaySettings(kind="json", source_id="sensor", path=path)).observations(Event()))


def test_known_identity_wins_and_conflicting_duplicate_and_late_records_are_unavailable() -> None:
    frame = SampleReference(sample_id="part", production_id="item", timestamp=STAMP)
    near = observation("near", timestamp=STAMP)
    correct = observation(sample_id="part", production_id="item")
    settings = AlignmentSettings(shared_clock="clock", max_skew_seconds=1)
    aligner = SampleAligner(settings, [near, correct], [frame])
    assert aligner.match(frame).observation == correct
    assert aligner.match(frame).reason == "already_matched"
    assert aligner.unused_observations == 1
    assert SampleAligner(settings, [near], [frame]).match(frame).reason == "unmatched"
    assert SampleAligner(settings, [correct], [frame, frame]).match(frame).reason == "ambiguous"
    duplicate = observation(sample_id="elsewhere")
    assert SampleAligner(settings, [correct, duplicate]).duplicate_observations == 2
    assert SampleAligner(settings, [correct, duplicate]).match(frame).reason == "ambiguous"
    assert SampleAligner(settings, [correct, correct.model_copy(update={"sequence_id": "S2"})]).match(
        frame,
    ).reason == "ambiguous"
    late = observation(timestamp="2026-10-30T01:00:01.000001Z", sample_id="part")
    result = SampleAligner(settings, [late]).match(frame)
    assert result.reason == "outside_time_window" and result.skew_seconds == 1.000001
    conflict = observation(sample_id="other", production_id="item")
    assert SampleAligner(settings, [conflict]).match(frame).reason == "conflicting_identifiers"
    missing = correct.model_copy(update={"values": {"temperature": None}})
    assert SampleAligner(settings, [missing]).match(frame).reason == "missing_measurements"
    assert SampleAligner(settings, [correct]).match(SampleReference(production_id="item")).method == "production_id"


def test_timestamp_requires_a_shared_clock_and_bidirectional_uniqueness() -> None:
    frame = SampleReference(timestamp=STAMP)
    boundary = observation(timestamp="2026-10-30T01:00:01Z")
    settings = AlignmentSettings(shared_clock="verified", max_skew_seconds=1)
    result = SampleAligner(settings, [boundary], [frame]).match(frame)
    assert (result.method, result.skew_seconds) == ("timestamp", 1.0)
    assert SampleAligner(settings, [boundary], [frame, frame]).match(frame).reason == "ambiguous"
    assert SampleAligner(settings, [boundary, observation("second")], [frame]).match(frame).reason == "ambiguous"
    assert SampleAligner(AlignmentSettings(), [boundary], [frame]).match(frame).reason == "missing_reference"
    naive = SampleReference(timestamp="2026-10-30T01:00:00")
    assert SampleAligner(settings, [boundary], [naive]).match(naive).reason == "missing_reference"


class BusyOnceRepository:
    def __init__(self, backing: SQLiteInspectionRepository) -> None:
        self.backing = backing
        self.attempts = 0

    def read_record(self, inspection_id: UUID) -> StoredInspection | None:
        return self.backing.read_record(inspection_id)

    def add(self, record: StoredInspection) -> None:
        self.attempts += 1
        if self.attempts == 1:
            raise InspectionDatabaseBusy("busy")
        self.backing.add(record)

    def get(self, inspection_id: UUID) -> StoredInspection | None:
        return self.backing.get(inspection_id)


def test_complete_path_retries_and_reuses_expired_sqlite_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = images(tmp_path, 2)
    (directory / "1.png").write_bytes(b"corrupt")
    manifest = tmp_path / "frames.json"
    manifest.write_text(json.dumps({"0.png": {"sample_id": "a", "timestamp": STAMP}}), encoding="utf-8")
    csv_path = tmp_path / "sensor.csv"
    csv_path.write_text(f"sequence_id,timestamp,sample_id,temperature\nreading-23,{STAMP},a,19.625\n", encoding="utf-8")
    database = tmp_path / "db.sqlite3"
    store = SQLiteInspectionRepository(database)
    repo = BusyOnceRepository(store)
    original = MockPredictor.predict
    predict_calls = 0

    def transient_predict(self: MockPredictor, sample: Sample) -> Prediction:
        nonlocal predict_calls
        predict_calls += 1
        if predict_calls == 1:
            raise TransientInspectionError("transient payload")
        return original(self, sample)

    monkeypatch.setattr(MockPredictor, "predict", transient_predict)

    def run(store: BusyOnceRepository | SQLiteInspectionRepository) -> StreamRunner:
        processor = FrameProcessor(
            stream_id="smoke", model_id="mock-defect", service=InspectionService.mock("defect"), repository=store,
            alignment=AlignmentSettings(allow_image_only=True), settings=ProcessingSettings(retry_delay_seconds=0),
            limits=LIMITS,
        )
        camera = DirectoryReplay(DirectorySettings(source_id="camera", directory=directory, manifest=manifest),
                                 max_image_bytes=LIMITS[0])
        sensor = SensorReplay(SensorReplaySettings(kind="csv", source_id="sensor", path=csv_path))
        stream = StreamRunner(camera, sensor, processor)
        stream.start()
        assert stream.wait(5)
        return stream

    status = run(repo).status()
    assert status.state == "degraded"
    assert status.frames_completed == status.total_frames == 2
    assert status.estimated_seconds_remaining is None
    assert status.counters["aligned"] == status.counters["persisted"] == status.counters["invalid_image"] == 1
    assert status.counters["retries"] == repo.attempts == predict_calls == 2
    [record] = store.list(HistoryQuery()).items
    assert record.result.outcome == "review_required"
    assert record.result.reason == "simulated_prediction"
    assert record.result.sensor_source_id == "sensor"
    assert record.result.metadata["streaming"]["alignment"]["method"] == "sample_id"
    assert "19.625" not in record.model_dump_json()
    assert "transient payload" not in status.model_dump_json()
    assert record.inspection_id.version == 4
    expired = SQLiteInspectionRepository(database, retention_seconds=1)
    monkeypatch.setattr(sqlite_repository, "retention_cutoff", lambda seconds: record.created_at if seconds else None)
    assert expired.get(record.inspection_id) is None
    assert run(expired).status().counters["reused"] == 1
    assert store.list(HistoryQuery()).total == 1
    assert predict_calls == 2


def test_backpressure_interruption_and_restart(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = images(tmp_path, 6)
    repo = FileInspectionRepository(tmp_path / "records")
    active = Event()
    unblocked = Event()
    original = MockPredictor.predict
    attempts = 0

    def blocked_predict(self: MockPredictor, sample: Sample) -> Prediction:
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            active.set()
            assert unblocked.wait(5)
        return original(self, sample)

    monkeypatch.setattr(MockPredictor, "predict", blocked_predict)

    def runner() -> StreamRunner:
        camera = DirectoryReplay(DirectorySettings(source_id="queue", directory=directory), max_image_bytes=LIMITS[0])
        processor = FrameProcessor(
            stream_id="queue", model_id="mock-empty", service=InspectionService.mock(), repository=repo,
            alignment=AlignmentSettings(allow_image_only=True),
            settings=ProcessingSettings(queue_capacity=1, shutdown_seconds=0.02), limits=LIMITS,
        )
        return StreamRunner(camera, None, processor)

    interrupted = runner()
    interrupted.start()
    try:
        assert active.wait(5)
        until(lambda: interrupted.status().counters.get("frames_received", 0) == 4)
        status = interrupted.status()
        assert status.queue_depth == status.active_workers == 1
        assert status.counters["backpressure_events"] >= 1
        assert not interrupted.close()
        assert interrupted.status().state == "stopping"
    finally:
        unblocked.set()
    assert interrupted.wait(5)
    assert len(list(repo.record_ids())) == 1
    restarted = runner()
    restarted.start()
    assert restarted.wait(5)
    assert restarted.status().counters["persisted"] == 5
    assert restarted.status().counters["reused"] == 1
    first = repo.get(next(repo.record_ids()))
    assert first is not None
    assert first.result.metadata["streaming"]["alignment"]["reason"] == "no_sensor_source"


def _stalled_child(channel: socket.socket, settings: USBSettings, limits: tuple[int, ...]) -> None:
    with channel:
        channel.sendall(b"O")
        channel.recv(1)
        channel.sendall(b"F\x00")  # Partial frame header followed by a native read stall.
        sleep(30)


def test_physical_device_requires_explicit_enable() -> None:
    config = {"kind": "usb", "source_id": "camera", "device_index": 0}
    for enabled in ({}, {"enabled": False}, {"enabled": 1}, {"enabled": "true"}):
        with pytest.raises(ValidationError):
            USBSettings.model_validate({**config, **enabled})


def test_rtsp_invalid_url_fails_before_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = RTSPSettings(kind="rtsp", source_id="bench", enabled=True, url_env="ICT_RTSP_TEST_INVALID",
                            timeout_seconds=5, reconnect_attempts=0)
    monkeypatch.setenv(settings.url_env, "rtsp://synthetic-user:synthetic-secret@[invalid-host")
    adapter = OpenCVCamera(settings, max_image_bytes=LIMITS[0], max_width=LIMITS[1],
                           max_height=LIMITS[2], max_pixels=LIMITS[3])
    with pytest.raises(SourceIssue, match="^source_unavailable$"):
        list(adapter.frames(Event()))


def test_camera_timeout_and_disconnect_recovery_without_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = USBSettings(kind="usb", enabled=True, device_index=3, source_id="usb",
                           reconnect_attempts=1, timeout_seconds=0.5, reconnect_delay_seconds=0)
    adapter = OpenCVCamera(settings, max_image_bytes=LIMITS[0], max_width=LIMITS[1],
                           max_height=LIMITS[2], max_pixels=LIMITS[3])
    monkeypatch.setattr(camera_module, "_capture", _stalled_child)
    started = monotonic()
    with pytest.raises(SourceIssue, match="timeout"):
        frames = adapter.frames(Event())
        assert next(frames).code == "timeout"
        next(frames)
    assert monotonic() - started < 5

    attempts = 0

    def session(self: OpenCVCamera, stop: Event) -> Iterator[CameraFrame]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise SourceIssue("disconnected")
        yield CameraFrame(source_id="usb", kind="usb", sequence_id="capture-1", format="JPEG", data=b"a")

    monkeypatch.setattr(OpenCVCamera, "_session", session)
    disconnected, frame = adapter.frames(Event())
    assert disconnected.code == "disconnected" and frame.sequence_id == "capture-1"


def test_malformed_sensor_snapshot_never_supplies_a_partial_join(tmp_path: Path) -> None:
    directory = images(tmp_path, 1)
    manifest = tmp_path / "frames.json"
    manifest.write_text('{"0.png":{"sample_id":"linked"}}', encoding="utf-8")
    sensor_path = tmp_path / "sensor.json"
    sensor_path.write_text(json.dumps([{
        "sequence_id": "S1", "timestamp": STAMP, "sample_id": "linked", "values": {"temperature": 0},
    }, {"timestamp": "invalid"}]), encoding="utf-8")
    repo = FileInspectionRepository(tmp_path / "records")
    camera = DirectoryReplay(DirectorySettings(source_id="replay", directory=directory, manifest=manifest),
                             max_image_bytes=LIMITS[0])
    sensor = SensorReplay(SensorReplaySettings(kind="json", source_id="sensor", path=sensor_path))
    stream = StreamRunner(camera, sensor, FrameProcessor(
        stream_id="malformed", model_id="mock-empty", service=InspectionService.mock(), repository=repo,
        alignment=AlignmentSettings(allow_image_only=True), settings=ProcessingSettings(), limits=LIMITS,
    ))
    stream.start()
    assert stream.wait(5)
    assert stream.status().sources[1].last_error == "invalid_sensor_data"
    [identifier] = repo.record_ids()
    record = repo.get(identifier)
    assert record is not None
    result = record.result
    assert result.sensor_source_id is None
    assert result.metadata["streaming"]["alignment"]["reason"] == "sensor_unavailable"


def test_concurrent_replay_publishes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = images(tmp_path, 1)
    store = SQLiteInspectionRepository(tmp_path / "records.sqlite3")
    rendezvous = Barrier(2)
    original = store.add

    def overlapping_add(record: StoredInspection) -> None:
        rendezvous.wait(5)
        original(record)

    monkeypatch.setattr(store, "add", overlapping_add)
    streams = []
    for _ in range(2):
        camera = DirectoryReplay(DirectorySettings(source_id="camera", directory=directory), max_image_bytes=LIMITS[0])
        processor = FrameProcessor(
            stream_id="concurrent", model_id="mock-empty", service=InspectionService.mock(), repository=store,
            alignment=AlignmentSettings(allow_image_only=True), settings=ProcessingSettings(workers=2), limits=LIMITS,
        )
        stream = StreamRunner(camera, None, processor)
        streams.append(stream)
        stream.start()
    assert all(stream.wait(10) for stream in streams)
    assert sorted(stream.status().counters.get("persisted", 0) for stream in streams) == [0, 1]
    assert sorted(stream.status().counters.get("reused", 0) for stream in streams) == [0, 1]
    assert store.list(HistoryQuery()).total == 1
