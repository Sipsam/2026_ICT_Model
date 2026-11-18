import json
from pathlib import Path

import pytest

from ict_pipeline.config.settings import Credentials
from ict_pipeline.inspection.service import InspectionService

from test_product_inspection import InputFixture, checkpoint, fixture_inputs, fixture_model, frozen  # noqa: F401

pytest.importorskip("fastapi", minversion="0.139")
pytest.importorskip("httpx")
pytest.importorskip("python_multipart", minversion="0.0.27")

from ict_pipeline.api.app import create_app  # noqa: E402
from ict_pipeline.api.schemas import StoredInspection  # noqa: E402
from ict_pipeline.api.settings import APISettings, MockModel, ProductModel  # noqa: E402
from test_rest_api import client_for, single  # noqa: E402

ENDPOINT = "/v1/inspections/multimodal"


@pytest.mark.parametrize("backend", ["file", "sqlite"])
def test_allowlisted_product_api_and_history(
    tmp_path: Path, inputs: InputFixture, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    monkeypatch.delenv("ICT_API_TOKEN", raising=False)
    root = tmp_path / "models"
    root.mkdir()
    spec = fixture_model(root, inputs)
    originals = tmp_path / "originals"
    originals.mkdir()
    (originals / f"{inputs.receipt.image_sha256}.png").write_bytes(inputs.image.read_bytes())
    settings = APISettings(
        model_root=root, storage_root=tmp_path / "records", storage_backend=backend,
        image_root=originals,
        models={"product-fixture": ProductModel(mode="multimodal", product=spec),
                "mock-defect": MockModel(mode="mock", scenario="defect")},
    )
    data = inputs.request.model_dump(mode="json")
    data["model_id"] = "product-fixture"
    image = {"image": ("../../private-name.png", inputs.image.read_bytes(), "image/png")}
    with client_for(create_app(settings)) as client:
        assert client.get("/v1/models").json()["models"][0]["input_kind"] == "multimodal"
        post = client.get("/openapi.json").json()["paths"][ENDPOINT]["post"]
        assert post["requestBody"]["content"]["multipart/form-data"]["schema"]["required"] == ["details"]
        for modality in ("sensor", "image", "paired"):
            payload = dict(data)
            files = {} if modality == "sensor" else dict(image)
            if modality == "image":
                payload.pop("sensor_evidence")
            files["details"] = (None, json.dumps(payload))
            response = client.post(ENDPOINT, files=files)
            assert response.status_code == 201, response.text
            record = StoredInspection.model_validate_json(response.content)
            product = record.result.multimodal
            assert product and product.input_mode == product.branch == modality
            assert record.result.outcome == "review_required" and product.weight_status == "simulated"
            assert product.alignment == ("verified_receipt" if modality == "paired" else "not_applicable")
            assert "private-name" not in response.text and "verified-pairs.json" not in response.text
            detail_url = response.headers["Location"]
            assert client.get(detail_url).json() == response.json()
            original_image = client.get(detail_url + "/image")
            if modality == "sensor":
                assert original_image.status_code == 404
            else:
                assert original_image.status_code == 200 and original_image.content == inputs.image.read_bytes()
            history = client.get("/v1/inspections", params={"modality": modality}).json()
            assert history["total"] == 1 and history["items"][0] == response.json()
            assert client.get("/v1/statistics/summary", params={"modality": modality}).json()["total_inspections"] == 0
            assert client.get("/v1/statistics/summary", params={"modality": modality, "mode": "mock"}).json()[
                "total_inspections"
            ] == 1
        for overrides, status in (
            ({"model_id": "unknown"}, 404),
            ({"checkpoint": str(root / "secret.pt")}, 422),
            ({"source_id": "unrelated-camera"}, 422),
            ({"sensor_evidence": None, "expected_size": {"width": 10, "height": 10}}, 422),
        ):
            response = client.post(ENDPOINT, data={"details": json.dumps(data | overrides)}, files=image)
            assert response.status_code == status and "secret.pt" not in response.text
        assert client.post(ENDPOINT, files={"details": (None, json.dumps(data | {"sensor_evidence": None}))}).status_code == 422
        assert client.post(ENDPOINT, files={"details": (None, json.dumps(data | {"model_id": "mock-defect"}))}).status_code == 422
        original = single(client, data=inputs.image.read_bytes())
        assert original.status_code == 201 and "multimodal" not in original.json()["result"]
        assert client.get("/dashboard").status_code == 200
    with client_for(create_app(settings)) as restarted:
        assert restarted.get("/v1/inspections", params={"modality": "paired"}).json()["total"] == 1


def test_untrained_checkpoint_api_excludes_default_statistics(
    tmp_path: Path, inputs: InputFixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ICT_API_TOKEN", raising=False)
    root = tmp_path / "models"
    root.mkdir()
    spec = checkpoint(root, frozen(root))
    settings = APISettings(
        model_root=root, storage_root=tmp_path / "records", storage_backend="sqlite",
        models={"product-checkpoint": ProductModel(mode="multimodal", product=spec)},
    )
    loaded = InspectionService.from_product(spec, root, credentials=Credentials())
    expected = loaded.inspect_product(inputs.request).multimodal
    with client_for(create_app(settings)) as client:
        payload = inputs.request.model_dump(mode="json") | {"model_id": "product-checkpoint"}
        response = client.post(ENDPOINT, files={"details": (None, json.dumps(payload))})
        assert response.status_code == 201, response.text
        result = StoredInspection.model_validate_json(response.content).result
        assert result.multimodal == expected
        assert result.multimodal and result.multimodal.weight_status == "untrained"
        assert client.get("/v1/inspections", params={"modality": "sensor", "mode": "real"}).json()["total"] == 1
        assert client.get("/v1/statistics/summary").json()["total_inspections"] == 0
        assert client.get("/v1/statistics/summary", params={"mode": "all"}).json()["total_inspections"] == 1
