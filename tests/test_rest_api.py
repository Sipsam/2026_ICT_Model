import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from typing import Any
from uuid import uuid4

import pytest
from PIL import Image

pytest.importorskip("fastapi", minversion="0.139")
pytest.importorskip("httpx")
pytest.importorskip("python_multipart", minversion="0.0.27")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from pydantic import ValidationError

from ict_pipeline.api.app import create_app
from ict_pipeline.api.repository import FileInspectionRepository
from ict_pipeline.api.schemas import StoredInspection
from ict_pipeline.api.settings import APISettings, MockModel, TrustedModel
from ict_pipeline.inspection.service import MockPredictor
from ict_pipeline.models.contracts import ModelName
from ict_pipeline.models.dataset import Sample
from ict_pipeline.evaluation.contracts import Prediction
from ict_pipeline.pipeline.store import fingerprint_path

from test_inspection import AdapterFixture, make_run

SINGLE = "/v1/inspections"
BATCH = f"{SINGLE}/batch"
DETAILS = {"request_id": "part-1", "source_id": "camera-3"}


@pytest.fixture(autouse=True)
def clear_api_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ICT_API_TOKEN", raising=False)
    monkeypatch.delenv("ICT_API_CONFIG", raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> APISettings:
    return APISettings(storage_root=tmp_path / "records", model_root=tmp_path / "models")


def image_bytes(format: str = "PNG", size: tuple[int, int] = (80, 40)) -> bytes:
    output = BytesIO()
    Image.new("RGB", size).save(output, format=format)
    return output.getvalue()


def client_for(app: FastAPI) -> TestClient:
    return TestClient(app, base_url="http://localhost", client=("127.0.0.1", 55000))


def single(
    client: TestClient, *, data: bytes | None = None, mime: str = "image/png", **overrides: object,
) -> Response:
    return client.post(SINGLE, data={"details": json.dumps({
        "model_id": "mock-defect", **DETAILS, **overrides,
    })}, files={"image": ("../../private.png", image_bytes() if data is None else data, mime)})


def assert_error(response: Response, status: int, code: str) -> None:
    assert response.status_code == status, response.text
    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["error"]["code"] == code and body["error"]["message"]
    assert response.headers["Cache-Control"] == "no-store"


def test_mock_result_persists_and_openapi_uses_shared_schema(settings: APISettings) -> None:
    app = create_app(settings)
    assert not settings.storage_root.exists()
    with client_for(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/v1/models").json()["models"][1]["mode"] == "mock"
        response = single(client, sensor={"source_id": "session-19", "values": {"temperature": 23.4}})
        assert response.status_code == 201, response.text
        record = StoredInspection.model_validate_json(response.content)
        result = record.result
        assert result.provenance.mode == "mock" and result.provenance.mock_scenario == "defect"
        assert (result.outcome, result.reason) == ("review_required", "simulated_prediction")
        assert result.sensor_source_id == "session-19" and result.image and result.image.format == "PNG"
        assert result.detections[0].bbox_xyxy == (20, 10, 60, 30)
        assert (result.detections[0].class_name, result.detections[0].class_id) == ("cold_solder", 1)
        assert "temperature" not in response.text and "private.png" not in response.text
        path = response.headers["Location"]
        assert client.get(path).json() == response.json()
        schema = client.get("/openapi.json").json()
        operation = schema["paths"][SINGLE]["post"]
        assert "multipart/form-data" in operation["requestBody"]["content"]
        assert "504" in operation["responses"]
        assert "InspectionResult" in schema["components"]["schemas"]
    assert len(list(settings.storage_root.iterdir())) == 1
    with client_for(create_app(settings)) as restarted:
        assert restarted.get(path).json() == response.json()
        assert_error(restarted.get(f"{SINGLE}/{uuid4()}"), 404, "inspection_not_found")
        assert_error(restarted.get(f"{SINGLE}/server-secret"), 422, "invalid_request")


def test_batch_order_with_validation_and_per_image_failures(settings: APISettings) -> None:
    files = [
        ("images", ("first.jpg", image_bytes("JPEG"), "image/jpeg")),
        ("images", ("broken.png", b"bad PNG", "image/png")),
        ("images", ("third.png", image_bytes(), "application/octet-stream")),
    ]
    details = {"model_id": "mock-empty", "items": [{**DETAILS, "request_id": str(i)} for i in range(3)]}
    with client_for(create_app(settings)) as client:
        response = client.post(BATCH, files=files, data={"details": json.dumps(details)})
        assert response.status_code == 200
        items = response.json()["items"]
        assert [(row["index"], row["status"]) for row in items] == [(0, 201), (1, 422), (2, 415)]
        assert items[0]["inspection"]["result"]["provenance"]["mock_scenario"] == "empty"
        assert items[0]["inspection"]["result"]["image"]["format"] == "JPEG"
        assert items[1]["error"]["code"] == "invalid_image"
        assert items[2]["error"]["code"] == "unsupported_media_type"
        assert client.get(f'{SINGLE}/{items[0]["inspection"]["inspection_id"]}').status_code == 200
        details["items"].pop()
        assert_error(client.post(BATCH, files=files, data={"details": json.dumps(details)}), 422, "invalid_request")
        details["items"][0]["sensor"] = {"source_id": "session", "values": {"defect_detected": 1}}
        assert_error(client.post(BATCH, files=files[:2], data={"details": json.dumps(details)}), 422, "invalid_request")
    assert len(list(settings.storage_root.glob("*.json"))) == 1


@pytest.mark.parametrize(("overrides", "status", "code"), [
    ({"schema_version": True}, 422, "invalid_request"),
    ({"sensor": {"source_id": "session", "values": {"temperature": float("inf")}}}, 422, "invalid_request"),
    ({"expected_size": {"width": 40, "height": 80}}, 422, "dimension_mismatch"),
    ({"image_path": "../../sensitive"}, 422, "invalid_request"),
    ({"checkpoint": "../../sensitive"}, 422, "invalid_request"),
    ({"model_id": "unknown"}, 404, "model_not_found"),
])
def test_rejects_invalid_details(settings: APISettings, overrides: dict[str, Any], status: int, code: str) -> None:
    with client_for(create_app(settings)) as client:
        assert_error(single(client, **overrides), status, code)
    assert not list(settings.storage_root.iterdir())


def test_rejects_malformed_and_unsupported_uploads(settings: APISettings) -> None:
    with client_for(create_app(settings)) as client:
        assert_error(client.post(SINGLE, json=DETAILS), 415, "unsupported_media_type")
        assert_error(client.post(SINGLE, headers={"Content-Type": "multipart/form-data"}), 400, "malformed_request")
        truncated = b'--a\r\nContent-Disposition: form-data; name="details"\r\n\r\n{}\r\n--a\r\n'
        assert_error(client.post(SINGLE, headers={"Content-Type": "multipart/form-data; boundary=a"},
                                 content=truncated), 400, "malformed_request")
        assert_error(client.post(SINGLE, files={"image": ("a", b"a", "image/png")},
                                 data={"details": "{"}), 422, "invalid_request")
        assert_error(single(client, data=b""), 422, "invalid_image")
        assert_error(single(client, mime="image/gif"), 415, "unsupported_media_type")
        assert_error(single(client, data=image_bytes("GIF")), 422, "invalid_image")
        assert_error(single(client, mime="image/jpeg"), 422, "invalid_image")
        assert_error(single(client, data=image_bytes()[:60]), 422, "invalid_image")


def test_request_image_and_dimension_limits(settings: APISettings) -> None:
    settings = settings.model_copy(update={"max_request_bytes": 1024, "max_image_bytes": 120, "max_width": 90,
                                           "max_height": 50, "max_pixels": 3300})
    with client_for(create_app(settings)) as client:
        assert_error(single(client, data=b"x" * 1024), 413, "request_too_large")
        assert_error(single(client, data=b"x" * 121), 413, "image_too_large")
        assert_error(single(client, data=image_bytes(size=(100, 4))), 422, "invalid_image")
        assert_error(single(client, data=image_bytes(size=(4, 51))), 422, "invalid_image")
        assert_error(single(client, data=image_bytes(size=(80, 42))), 422, "invalid_image")
        prefix = b'--a\r\nContent-Disposition: form-data; name="image"; filename="part.png"\r\n\r\n'
        response = client.post(SINGLE, content=iter([prefix + b"a" * 512, b"x" * 513]),
                               headers={"Content-Type": "multipart/form-data; boundary=a"})
        assert_error(response, 413, "request_too_large")


@pytest.mark.parametrize("model", tuple(ModelName))
def test_trusted_models_load_once_in_lifespan(settings: APISettings, model: ModelName) -> None:
    run, checkpoint = make_run(settings.model_root, model)
    predictor = AdapterFixture(model)
    loads = []

    def factory(name: ModelName, snapshot: Path, *_: object) -> AdapterFixture:
        assert name == model and snapshot.is_relative_to(settings.storage_root)
        assert snapshot.read_bytes() == checkpoint.read_bytes()
        loads.append(name)
        return predictor

    spec = TrustedModel(mode="real", model=model, run=run.relative_to(settings.model_root),
                        trust_run=True, expected_fingerprint=fingerprint_path(checkpoint))
    app = create_app(settings.model_copy(update={"models": {"approved": spec}}), predictor_factory=factory)
    assert loads == []
    with client_for(app) as client:
        assert loads == [model]
        assert client.get("/v1/models").json()["models"] == [
            {"model_id": "approved", "mode": "real", "model": model, "mock_scenario": None, "available": True},
        ]
        for _ in range(2):
            response = single(client, model_id="approved")
            assert response.status_code == 201, response.text
            result = response.json()["result"]
            assert result["outcome"] == "defect_detected" and result["provenance"]["mode"] == "real"
            assert result["detections"][0]["class_name"] == "tombstone"
            assert str(run) not in response.text
        assert loads == [model] and predictor.calls == 2


def test_untrusted_and_incompatible_model_configuration(settings: APISettings) -> None:
    run, _ = make_run(settings.model_root, ModelName.YOLO)
    with pytest.raises(ValidationError):
        TrustedModel(mode="real", model=ModelName.YOLO, run=run, trust_run=True)
    with pytest.raises(ValidationError):
        TrustedModel(mode="real", model=ModelName.YOLO, run=Path("yolo"), trust_run=False)
    wrong_model = TrustedModel(mode="real", model=ModelName.RT_DETR, run=Path("yolo"), trust_run=True)
    wrong_pin = TrustedModel(mode="real", model=ModelName.YOLO, run=Path("yolo"), trust_run=True,
                             expected_fingerprint="0" * 64)
    settings = settings.model_copy(update={"models": {"wrong-model": wrong_model, "wrong-pin": wrong_pin}})
    with client_for(create_app(settings, predictor_factory=lambda *_: pytest.fail("Unexpected load"))) as client:
        assert_error(client.get("/health"), 503, "model_unavailable")
        assert all(not model["available"] for model in client.get("/v1/models").json()["models"])
        for model_id in ("wrong-model", "wrong-pin"):
            assert_error(single(client, model_id=model_id), 503, "model_unavailable")


def test_backend_load_failure_does_not_fall_back(
    settings: APISettings, caplog: pytest.LogCaptureFixture,
) -> None:
    make_run(settings.model_root, ModelName.YOLO)
    spec = TrustedModel(mode="real", model=ModelName.YOLO, run=Path("yolo"), trust_run=True)
    settings = settings.model_copy(update={"models": {**settings.models, "real": spec}})

    def broken_backend(*_: object) -> AdapterFixture:
        raise RuntimeError("internal model path and backend details")

    with client_for(create_app(settings, predictor_factory=broken_backend)) as client:
        assert client.get("/health").json() == {"status": "degraded"}
        response = single(client, model_id="real")
        assert_error(response, 503, "model_unavailable")
        assert "internal model path" not in response.text + caplog.text
        assert single(client).json()["result"]["provenance"]["mode"] == "mock"


def test_prediction_timeout_keeps_worker_slot_and_recovers(
    settings: APISettings, monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings.model_copy(update={"inference_timeout_seconds": 0.2})
    predict = MockPredictor.predict
    entered, release = Event(), Event()
    attempts = 0

    def blocking_predict(self: MockPredictor, sample: Sample) -> Prediction:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            entered.set()
            assert release.wait(5)
        elif attempts == 2:
            raise RuntimeError("credential=must-not-leak")
        return predict(self, sample)

    monkeypatch.setattr(MockPredictor, "predict", blocking_predict)
    with client_for(create_app(settings)) as client, ThreadPoolExecutor(max_workers=1) as pool:
        slow = pool.submit(single, client)
        try:
            assert entered.wait(5)
            assert client.get("/health").status_code == 200
            assert_error(single(client), 429, "busy")
            assert_error(slow.result(timeout=5), 504, "inference_timeout")
            assert_error(single(client), 429, "busy")
            assert not list(settings.storage_root.glob("*.json"))
        finally:
            release.set()
        deadline = monotonic() + 5
        while (response := single(client)).status_code == 429 and monotonic() < deadline:
            sleep(0.01)
        assert_error(response, 500, "prediction_failed")
        assert "must-not-leak" not in response.text
        assert single(client).status_code == 201
    assert len(list(settings.storage_root.glob("*.json"))) == 1


def test_repository_atomic_insert_corruption_and_expiry(settings: APISettings) -> None:
    with client_for(create_app(settings)) as client:
        response = single(client)
        assert response.status_code == 201
    record = StoredInspection.model_validate_json(response.content)
    repository = FileInspectionRepository(settings.storage_root, retention_seconds=60)
    duplicate = record.model_copy(update={"inspection_id": uuid4()})
    with ThreadPoolExecutor(max_workers=2) as pool:
        writes = [pool.submit(repository.add, duplicate) for _ in range(2)]
    assert sum(future.exception() is None for future in writes) == 1
    assert repository.get(duplicate.inspection_id) == duplicate
    expired = record.model_copy(update={"inspection_id": uuid4(), "created_at": datetime.now(UTC) - timedelta(hours=1)})
    repository.add(expired)
    assert repository.get(expired.inspection_id) is None
    assert (repository.root / f"{expired.inspection_id.hex}.json").is_file()
    broken = uuid4()
    (repository.root / f"{broken.hex}.json").write_text("malformed", encoding="utf-8")
    with client_for(create_app(settings, repository=repository)) as client:
        assert_error(client.get(f"{SINGLE}/{broken}"), 503, "storage_unavailable")
        assert_error(client.get(f"{SINGLE}/{expired.inspection_id}"), 404, "inspection_not_found")
        assert client.get(response.headers["Location"]).status_code == 200


def test_local_access_and_bearer_auth(settings: APISettings, monkeypatch: pytest.MonkeyPatch,
                                      caplog: pytest.LogCaptureFixture) -> None:
    app = create_app(settings)
    with TestClient(app, base_url="http://localhost", client=("192.0.2.1", 55000)) as client:
        assert_error(client.get("/health"), 403, "local_access_required")
    with client_for(app) as client:
        assert_error(client.get("/health", headers={"Host": "other.example"}), 403, "local_access_required")
        assert_error(client.get("/health", headers={"Origin": "https://other.example"}), 403, "local_access_required")
    remote = settings.model_copy(update={"host": "0.0.0.0"})
    with pytest.raises(RuntimeError, match="ICT_API_TOKEN"):
        create_app(remote)
    token = "example-only-for-testing-0123456789"
    monkeypatch.setenv("ICT_API_TOKEN", token)
    with client_for(create_app(remote)) as client:
        assert_error(client.get("/docs"), 401, "unauthorized")
        client.headers["Authorization"] = f"Bearer {token}"
        assert client.get("/health").status_code == 200
        with caplog.at_level(logging.INFO, logger="ict_pipeline.api"):
            response = single(client, request_id=token)
        assert response.status_code == 201 and token not in response.text and token not in caplog.text
        assert response.json()["result"]["request_id"] == "[REDACTED]"
        assert any(response.headers["X-Request-ID"] in line.message for line in caplog.records)


def test_json_configuration_and_model_bound(settings: APISettings, tmp_path: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "api.json"
    configured = settings.model_copy(update={"port": 8013, "models": {"simulated": MockModel(mode="mock")}})
    config_path.write_text(configured.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("ICT_API_CONFIG", str(config_path))
    with client_for(create_app()) as client:
        assert single(client, model_id="simulated").status_code == 201
    with pytest.raises(ValidationError):
        APISettings(max_loaded_models=0, models={
            "real": TrustedModel(mode="real", model=ModelName.YOLO, run=Path("yolo"), trust_run=True),
        })
