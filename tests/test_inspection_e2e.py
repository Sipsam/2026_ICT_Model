import asyncio
import json
import logging
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import BinaryIO
from uuid import UUID

import pytest

pytest.importorskip("fastapi", minversion="0.139")
pytest.importorskip("httpx")
pytest.importorskip("python_multipart", minversion="0.0.27")

from starlette import formparsers
from starlette.requests import Request
from starlette.types import Message
from typer.testing import CliRunner

from ict_pipeline.api import services
from ict_pipeline.api.errors import APIError
from ict_pipeline.api.settings import APISettings
from ict_pipeline.api.uploads import read_form
from ict_pipeline.cli import app as cli
from ict_pipeline.evaluation.contracts import Prediction, Preprocessing
from ict_pipeline.inspection.service import MockPredictor
from ict_pipeline.models.dataset import Sample
from scripts.inspection_e2e import HISTORY, SOURCE_STATUS, client_for, make_inputs, replay_status, run_demo

from test_rest_api import assert_error, single

pytestmark = pytest.mark.e2e


@pytest.fixture(autouse=True)
def clear_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ICT_API_TOKEN", raising=False)


@pytest.mark.parametrize("with_sensor", [True, False])
def test_demo_round_trips_over_shared_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, with_sensor: bool,
) -> None:
    predict = MockPredictor.predict
    predictions: list[tuple[Path, bytes]] = []

    def capture(self: MockPredictor, sample: Sample) -> Prediction:
        assert type(sample) is Sample
        assert vars(sample).keys() == {"image", "width", "height", "objects"}
        assert (sample.width, sample.height, sample.objects) == (80, 40, ())
        predictions.append((sample.image, sample.image.read_bytes()))
        return predict(self, sample)

    monkeypatch.setattr(MockPredictor, "predict", capture)
    root, passed = run_demo(tmp_path, with_sensor=with_sensor)
    report = json.loads((root / "integration-report.json").read_bytes())
    assert passed, report
    assert report["status"] == "passed" and report["inference_mode"] == "mock"
    assert report["real_model_inference"] == "not_run"
    assert all(check["status"] == "passed" for check in report["checks"].values())
    assert report["checks"]["sqlite_restore"]["restored_records"] == 4
    assert report["checks"]["restart_and_replay_deduplication"]["replay"]["reused"] == 3
    originals = {path.read_bytes() for path in (root / "inputs" / "images").glob("*.png")}
    assert len(predictions) == 5  # CLI, three replay frames, HTTP. Restore and replay reuse skip inference.
    assert {data for _, data in predictions} == originals
    assert all(not path.parent.exists() for path, _ in predictions)


def test_strict_alignment_never_attaches_ambiguous_or_unavailable_sensor(
    tmp_path: Path,
) -> None:
    settings, _ = make_inputs(tmp_path, True)
    stream = settings.input_stream
    assert stream is not None and stream.sensor is not None
    strict_stream = stream.model_copy(update={"alignment": stream.alignment.model_copy(update={"allow_image_only": False})})
    strict = settings.model_copy(update={"input_stream": strict_stream})
    with client_for(strict) as client:
        status = replay_status(client)
        assert status["state"] == "degraded"
        assert status["frames_completed"] == status["total_frames"] == 3
        assert status["counters"]["persisted"] == 1
        assert status["counters"]["skipped_no_sensor"] == 2
        assert status["counters"]["alignment_ambiguous"] == status["counters"]["alignment_unmatched"] == 1
        [linked] = client.get(HISTORY).json()["items"]
        assert linked["result"]["metadata"]["streaming"]["camera"]["sample_id"] == "linked"
    unavailable = stream.sensor.model_copy(update={"path": tmp_path / "unavailable-private-sensor.json"})
    updated = strict.model_copy(update={"input_stream": strict_stream.model_copy(update={"sensor": unavailable})})
    with client_for(updated) as client:
        status = replay_status(client)
        assert status["state"] == "failed"
        assert status["sources"][1]["last_error"] == "source_unavailable"
        assert "unavailable-private-sensor" not in json.dumps(status)
        assert client.get(HISTORY).json()["items"] == [linked]


def test_all_canonical_defect_labels_survive_sqlite_and_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, image = make_inputs(tmp_path, False)
    settings = settings.model_copy(update={"input_stream": None})
    classes = [(1, "cold_solder", 23), (2, "misalignment", 24), (3, "short", 25),
               (4, "missing_component", 27), (5, "tombstone", 30)]
    box = (7.0, 5.0, 74.0, 30.0)

    def predict(self: MockPredictor, sample: Sample) -> Prediction:
        size = (sample.width, sample.height)
        return Prediction(sample.image.stem, (box,) * 5, (1, 2, 3, 4, 5), (0.9,) * 5,
                          Preprocessing(size, size, "stretch"))

    monkeypatch.setattr(MockPredictor, "predict", predict)
    with client_for(settings) as client:
        upload = single(client, data=image.read_bytes())
        assert upload.status_code == 201
        record = upload.json()
        detections = record["result"]["detections"]
        assert record["result"]["outcome"] == "review_required"
        assert [(row["class_id"], row["class_name"], row["source_category_id"]) for row in detections] == classes
        assert all(row["bbox_xyxy"] == list(box) for row in detections)
        for _, name, _ in classes:
            assert client.get(HISTORY, params={"mode": "mock", "defect_class": name}).json()["items"] == [record]
        exported = client.get(f"{HISTORY}/export", params={"mode": "mock"})
        assert exported.status_code == 200 and exported.json()["items"] == [record]


def test_failed_requests_and_uuid_collision_preserve_committed_sqlite_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    settings, image = make_inputs(tmp_path, False)
    settings = settings.model_copy(update={"input_stream": None})
    token = "fixture-api-token-for-inspection-regression"
    monkeypatch.setenv("ICT_API_TOKEN", token)
    data = image.read_bytes()
    caplog.set_level(logging.INFO, logger="ict_pipeline.api")
    with client_for(settings) as client:
        assert_error(client.get(HISTORY, headers={"Authorization": "Bearer incorrect"}), 401, "unauthorized")
        first = single(client, data=data, source_id=token)
        second = single(client, data=data, source_id=token)
        assert first.status_code == second.status_code == 201
        assert first.json()["result"] == second.json()["result"]
        assert first.json()["inspection_id"] != second.json()["inspection_id"]
        assert first.json()["result"]["source_id"] == "[REDACTED]"
        assert_error(single(client, data=data, image_path=str(image)), 422, "invalid_request")
        assert_error(single(client, data=data, mime="image/jpeg"), 422, "invalid_image")
        assert_error(single(client, data=data, model_id=str(image)), 422, "invalid_request")
        assert_error(single(client, data=data, model_id="outside-catalog"), 404, "model_not_found")
        details = {
            "model_id": "mock-defect",
            "items": [{"request_id": name, "source_id": "mixed-batch"} for name in ("good", "corrupt")],
        }
        response = client.post(f"{HISTORY}/batch", data={"details": json.dumps(details)}, files=[
            ("images", ("good.png", data, "image/png")),
            ("images", ("bad.png", b"not an image", "image/png")),
        ])
        assert response.status_code == 200
        good, corrupt = response.json()["items"]
        assert good["status"] == 201 and corrupt["status"] == 422 and corrupt["inspection"] is None
        assert corrupt["error"]["code"] == "invalid_image"
        monkeypatch.setattr(services, "uuid4", lambda: UUID(first.json()["inspection_id"]))
        conflict = single(client, data=data, request_id="different")
        assert_error(conflict, 503, "storage_unavailable")
        assert client.get(first.headers["Location"]).json() == first.json()
        history = client.get(HISTORY).json()
        assert history["total"] == 3
        assert token not in first.text + second.text + conflict.text + json.dumps(history) + caplog.text
        assert str(image) not in caplog.text
    with client_for(settings) as restarted:
        assert restarted.get(HISTORY).json() == history
    assert token.encode() not in (settings.storage_root / settings.database_name).read_bytes()


def test_cli_batch_exports_survive_missing_and_corrupt_images(tmp_path: Path) -> None:
    _, image = make_inputs(tmp_path, False)
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(b"not an image")
    manifest = tmp_path / "batch.jsonl"
    manifest.write_text("\n".join(
        json.dumps({"request_id": str(index), "source_id": "cli", "image_path": str(path)})
        for index, path in enumerate((image, corrupt, tmp_path / "missing.png"))
    ), encoding="utf-8")
    result = CliRunner().invoke(cli, ["inspect", "--batch", str(manifest), "--mock", "defect",
                                     "--output-dir", str(tmp_path / "reports")])
    assert result.exit_code == 1, result.output
    output = Path(result.stdout.strip())
    records = json.loads((output / "inspection.json").read_bytes())["records"]
    assert [row["reason"] for row in records] == ["simulated_prediction", "invalid_image", "missing_image"]
    assert len(records[0]["detections"]) == 1 and records[1]["detections"] == records[2]["detections"] == []
    assert (output / "inspection.csv").is_file()


def test_http_timeout_while_replay_owns_predictor_under_queue_pressure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings, image = make_inputs(tmp_path, False)
    settings = settings.model_copy(update={"inference_timeout_seconds": 0.2})
    entered, release = Event(), Event()
    predict = MockPredictor.predict
    started: list[Path] = []
    data = image.read_bytes()

    def block_first(self: MockPredictor, sample: Sample) -> Prediction:
        started.append(sample.image)
        if len(started) == 1:
            entered.set()
            assert release.wait(5)
        return predict(self, sample)

    monkeypatch.setattr(MockPredictor, "predict", block_first)
    with client_for(settings) as client:
        try:
            assert entered.wait(5)
            deadline = monotonic() + 5
            while True:
                status = client.get(SOURCE_STATUS).json()
                if status["counters"].get("backpressure_events", 0):
                    break
                assert monotonic() < deadline
                sleep(0.01)
            assert status["queue_depth"] == status["queue_capacity"] == status["active_workers"] == 1
            assert_error(single(client, data=data), 504, "inference_timeout")
            assert len(started) == 1 and client.get(HISTORY).json()["total"] == 0
        finally:
            release.set()
        assert replay_status(client)["counters"]["persisted"] == 3
        assert len(started) == 3 and single(client, data=data).status_code == 201
        assert client.get(HISTORY).json()["total"] == 4
    assert all(not path.parent.exists() for path in started)


@pytest.mark.parametrize(("second_chunk", "limit", "expected"), [
    (b"", 1024, "malformed_request"),
    (b"x" * 1024, 1024, "request_too_large"),
    (None, 1024, "upload_timeout"),
])
def test_multipart_abort_closes_incomplete_file(
    monkeypatch: pytest.MonkeyPatch, second_chunk: bytes | None, limit: int, expected: str,
) -> None:
    created: list[BinaryIO] = []
    spool = formparsers.SpooledTemporaryFile

    def track(*args: object, **kwargs: object) -> BinaryIO:
        file = spool(*args, **kwargs)
        created.append(file)
        return file

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", track)
    prefix = b'--a\r\nContent-Disposition: form-data; name="image"; filename="part.png"\r\n\r\npixels'
    calls = 0

    async def receive() -> Message:
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "http.request", "body": prefix, "more_body": True}
        if second_chunk is None:
            await asyncio.Event().wait()
        return {"type": "http.request", "body": second_chunk, "more_body": False}

    request = Request({"type": "http", "headers": [(b"content-type", b"multipart/form-data; boundary=a")]}, receive)
    settings = APISettings(max_request_bytes=limit, upload_timeout_seconds=0.05)
    with pytest.raises(APIError) as raised:
        asyncio.run(read_form(request, settings, 1))
    assert raised.value.code == expected
    assert len(created) == 1 and created[0].closed
