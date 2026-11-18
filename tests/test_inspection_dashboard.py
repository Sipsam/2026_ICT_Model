import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from test_inspection_database import COLD_SOLDER, SHORT, inspection
from test_rest_api import SINGLE, assert_error, client_for, image_bytes, single

from ict_pipeline.api.app import create_app
from ict_pipeline.api.schemas import StoredInspection
from ict_pipeline.api.settings import APISettings
from ict_pipeline.api.storage import open_repository
from ict_pipeline.config.settings import Credentials


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> APISettings:
    monkeypatch.delenv("ICT_API_TOKEN", raising=False)
    return APISettings(storage_root=tmp_path / "records", model_root=tmp_path / "models", max_concurrent_requests=1)


def test_dashboard_serves_only_fixed_read_only_assets(settings: APISettings) -> None:
    with client_for(create_app(settings)) as client:
        for suffix, mime, marker in (
            ("", "text/html", "Inspection dashboard"),
            ("/dashboard.css", "text/css", ".image-stage"),
            ("/dashboard.mjs", "text/javascript", "openDetail"),
            ("/client.mjs", "text/javascript", "boxPercent"),
        ):
            response = client.get("/dashboard" + suffix)
            assert response.status_code == 200
            assert response.headers["content-type"].startswith(mime)
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["cache-control"] == "no-store"
            assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
            assert marker in response.text
        assert_error(client.get("/dashboard/private.png"), 404, "not_found")
        assert_error(client.post("/dashboard"), 405, "method_not_allowed")
        assert_error(client.get("/dashboard", headers={"Host": "example.test"}), 403, "local_access_required")
        assert_error(client.get("/dashboard", headers={"Origin": "https://example.test"}), 403, "local_access_required")


def test_dashboard_token_and_non_loopback_binding(settings: APISettings, monkeypatch: pytest.MonkeyPatch) -> None:
    token = "test-only-token-" + "0" * 32
    monkeypatch.setenv("ICT_API_TOKEN", token)
    auth = {"Authorization": "Bearer " + token}
    with client_for(create_app(settings)) as client:
        assert client.get("/dashboard").status_code == 200
        assert_error(client.get(SINGLE), 401, "unauthorized")
        assert_error(client.get(f"{SINGLE}/{uuid4()}/image"), 401, "unauthorized")
        assert client.get(SINGLE, headers=auth).status_code == 200
    with client_for(create_app(settings.model_copy(update={"host": "0.0.0.0"}))) as client:
        assert_error(client.get("/dashboard", headers=auth), 403, "local_access_required")
        assert client.get(SINGLE, headers=auth).status_code == 200


@pytest.mark.parametrize("backend", ["file", "sqlite"])
def test_dashboard_filters_exports_and_statistics(settings: APISettings, backend: str) -> None:
    settings = settings.model_copy(update={"storage_backend": backend})
    store = open_repository(settings, Credentials())
    day = datetime(2026, 10, 20, tzinfo=UTC)
    first = inspection(created=day, detections=(COLD_SOLDER, COLD_SOLDER, SHORT),
                       identifier=UUID("00000000-0000-4000-8000-000000000001"))
    second = inspection(created=day, detections=(), identifier=UUID("00000000-0000-4000-8000-000000000002"))
    failed_data = inspection(created=day + timedelta(days=1), detections=()).model_dump(mode="json")
    failed_data["result"].update(outcome="inference_failed", reason="missing_image", image=None)
    failed = StoredInspection.model_validate_json(json.dumps(failed_data))
    mock = inspection(mode="mock", created=day)
    for record in (mock, second, failed, first):
        store.add(record)

    with client_for(create_app(settings, repository=store)) as client:
        history = {"mode": "real", "limit": 1, "order": "oldest"}
        assert client.get(SINGLE, params=history).json()["items"] == [first.model_dump(mode="json")]
        assert client.get(SINGLE, params={**history, "offset": 1}).json()["items"][0]["inspection_id"] == str(second.inspection_id)
        assert client.get(SINGLE, params={**history, "status": "inference_failed"}).json()["items"][0]["inspection_id"] == str(failed.inspection_id)
        assert_error(client.get(f"{SINGLE}/{failed.inspection_id}/image"), 404, "inspection_image_unavailable")
        assert client.get("/v1/statistics/summary").json()["total_inspections"] == 3
        assert client.get("/v1/statistics/summary", params={"mode": "mock"}).json()["total_inspections"] == 1

        filters = {"mode": "real", "model_id": "line-3", "model": "yolo", "defect_class": "short",
                   "created_from": day.isoformat(), "created_before": (day + timedelta(days=1)).isoformat(),
                   "inspection_id": str(first.inspection_id), "status": "defect_detected"}
        page = client.get(SINGLE, params={**history, **filters}).json()
        assert page["total"] == 1 and page["items"][0]["inspection_id"] == str(first.inspection_id)
        assert client.get(f"{SINGLE}/export", params={**history, **filters, "format": "json"}).json() == page
        csv = client.get(f"{SINGLE}/export", params={**history, **filters, "format": "csv"})
        assert csv.status_code == 200 and str(first.inspection_id) in csv.text
        summary = client.get("/v1/statistics/summary", params=filters).json()
        assert (summary["total_inspections"], summary["total_detections"]) == (1, 3)
        assert summary["by_status"] == {"defect_detected": 1} and summary["by_model"] == {"line-3": 1}
        assert client.get("/v1/statistics/defects", params=filters).json()["items"][0] == {
            "class_name": "cold_solder", "inspection_count": 1, "detection_count": 2,
        }
        assert client.get("/v1/statistics/trends", params=filters).json()["items"] == [
            {"date": "2026-10-20", "inspection_count": 1},
        ]
        assert client.get(SINGLE, params={**filters, "mode": "mock"}).json()["total"] == 0
        assert_error(client.get(SINGLE, params={"inspection_id": "../private"}), 422, "invalid_request")


@pytest.mark.parametrize(("format", "suffix", "mime"), [("PNG", ".png", "image/png"), ("JPEG", ".jpg", "image/jpeg")])
def test_image_endpoint_requires_exact_staged_bytes(
    settings: APISettings, tmp_path: Path, format: str, suffix: str, mime: str,
) -> None:
    root = tmp_path / "images"
    root.mkdir()
    settings = settings.model_copy(update={"image_root": root})
    data = image_bytes(format)
    with client_for(create_app(settings)) as client:
        submission = single(client, data=data, mime=mime)
        assert submission.status_code == 201
        path = submission.headers["Location"] + "/image"
        source = submission.json()["result"]["image"]
        staged = root / (source["sha256"] + suffix)
        assert_error(client.get(path), 404, "inspection_image_unavailable")
        staged.write_bytes(image_bytes(format, (10, 10)))
        assert_error(client.get(path), 404, "inspection_image_unavailable")
        staged.write_bytes(data)
        response = client.get(path, params={"path": str(tmp_path / "secret")})
        assert response.status_code == 200 and response.content == data
        assert response.headers["content-type"] == mime
        assert response.headers["cross-origin-resource-policy"] == "same-origin"
        assert_error(client.get(f"{SINGLE}/{uuid4()}/image"), 404, "inspection_not_found")
        assert_error(client.get(f"{SINGLE}/not-a-uuid/image"), 422, "invalid_request")

    with client_for(create_app(settings.model_copy(update={"image_root": None}))) as client:
        assert_error(client.get(path), 404, "inspection_image_unavailable")


def test_image_route_respects_retention(settings: APISettings, tmp_path: Path) -> None:
    settings = settings.model_copy(update={"retention_seconds": 60, "image_root": tmp_path})
    store = open_repository(settings, Credentials())
    old = inspection(created=datetime.now(UTC) - timedelta(minutes=2))
    store.add(old)
    with client_for(create_app(settings, repository=store)) as client:
        assert_error(client.get(f"{SINGLE}/{old.inspection_id}/image"), 404, "inspection_not_found")
