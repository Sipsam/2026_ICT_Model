import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner
from PIL import Image

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.inspection.contracts import (
    InspectionError, InspectionRequest, InspectionResult, InspectionThresholds, ModelProvenance,
)
from ict_pipeline.inspection.service import InspectionService, MockPredictor
from ict_pipeline.models.contracts import ModelName
from ict_pipeline.multimodal.contracts import CHANNELS, UNITS, Record, SensorStep
from ict_pipeline.multimodal.inference_contracts import (
    CheckpointProductModel, InputSequence, MockProductModel, PairingReceipt, PairingRegistry,
    ProductInspectionInput, SensorEvidence,
)
from ict_pipeline.multimodal.preprocessing import (
    CAUSAL_WINDOW_POLICY, ChannelFit, PreprocessingConfig, PreprocessingState, transform_sensor,
)

CAPTURE = "2026-01-01T00:00:05+00:00"
POLICY = "smt-target-defect-presence-v1"
FINGERPRINT = "a" * 64


@dataclass(frozen=True)
class InputFixture:
    image: Path
    request: ProductInspectionInput
    record: Record
    receipt: PairingReceipt


@pytest.fixture(name="inputs")
def fixture_inputs(tmp_path: Path) -> InputFixture:
    image = tmp_path / "private-image.png"
    Image.new("RGB", (40, 24), (10, 40, 20)).save(image)
    sequence = InputSequence(
        channels=CHANNELS, units=UNITS, steps=(
            SensorStep(timestamp="2026-01-01T00:00:04Z", values=(10., None, 1., None, None)),
            SensorStep(timestamp=CAPTURE, values=(11., 0., 2., None, None)),
            SensorStep(timestamp="2026-01-01T00:00:06Z", values=(200., 0., 4., None, None)),
        ),
    )
    request = ProductInspectionInput(
        request_id="check-1", sample_id="part-1", source_id="camera-1", capture_timestamp=CAPTURE,
        sensor_evidence=SensorEvidence(source_id="station-1", sequence=sequence),
    )
    row = Record(
        sample_id=request.sample_id, split="validation", provider_split="validation", decision="candidate",
        detection_candidate=True, source_class="normal", source_group="fixture", annotation_path="annotations.json",
        image_path=image.name, sensor_path="sensor.xlsx", annotation_sha256=FINGERPRINT,
        image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(), image_size=(40, 24), image_available=True,
        targets=(), sequence=sequence, capture_timestamp=CAPTURE, alignment="matched",
        alignment_method="provider_record_json_xlsx",
    )
    return InputFixture(image, request, row, PairingReceipt.from_audited_record(
        row, image_source_id="camera-1", sensor_source_id="station-1",
    ))


def frozen(tmp_path: Path, *, lookahead: bool = False, train_hash: str = FINGERPRINT) -> PreprocessingState:
    config = (PreprocessingConfig(window_seconds=2) if lookahead else
              PreprocessingConfig(window_seconds=2, window_policy=CAUSAL_WINDOW_POLICY))
    state = PreprocessingState(
        config=config, config_fingerprint=config.fingerprint, training_fingerprint=train_hash,
        statistics=(ChannelFit(count=3, center=0., scale=1., fill=0.),) * len(CHANNELS),
    )
    with (tmp_path / "preprocessor.json").open("x", encoding="utf-8") as stream:
        json.dump(state.payload(), stream)
    return state


def checkpoint(tmp_path: Path, state: PreprocessingState) -> CheckpointProductModel:
    pytest.importorskip("torch")
    from ict_pipeline.multimodal.checkpoint import save_checkpoint
    from ict_pipeline.multimodal.model import MultimodalModel
    from ict_pipeline.multimodal.model_config import ArchitectureConfig, FusionConfig, SensorEncoderConfig

    model = MultimodalModel(ArchitectureConfig(
        preprocessing_fingerprint=state.fingerprint, fusion=FusionConfig(kind="gated"),
        sensor=SensorEncoderConfig(kind="gru", min_steps=2),
    ), state).eval()
    path = tmp_path / "product.pt"
    save_checkpoint(path, model)
    return CheckpointProductModel(
        kind="checkpoint", checkpoint=Path("product.pt"), preprocessing=Path("preprocessor.json"),
        architecture_fingerprint=model.architecture_fingerprint,
        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def without_sensor(request: ProductInspectionInput) -> ProductInspectionInput:
    return request.model_copy(update={"sensor_evidence": None})


def fixture_model(root: Path, inputs: InputFixture) -> MockProductModel:
    registry = root / "verified-pairs.json"
    registry.write_text(PairingRegistry(receipts=(inputs.receipt,)).model_dump_json(), encoding="utf-8")
    return MockProductModel(kind="mock", pairings=Path(registry.name))


def test_legacy_image_and_mock_product_cli(tmp_path: Path, inputs: InputFixture) -> None:
    mock = InspectionService.mock("defect")
    old = mock.inspect(InspectionRequest(request_id="r", source_id="camera", image_path=inputs.image))
    legacy_json = old.model_dump_json()
    assert "multimodal" not in legacy_json and "product_only" not in legacy_json
    assert InspectionResult.model_validate_json(legacy_json) == old
    assert old.detections[0].bbox_xyxy == (10, 6, 30, 18)

    spec = tmp_path / "mock.json"
    spec.write_text(fixture_model(tmp_path, inputs).model_dump_json(), encoding="utf-8")
    request = tmp_path / "request.json"
    request.write_text(inputs.request.model_dump_json(), encoding="utf-8")
    output = tmp_path / "records"
    result = CliRunner().invoke(app, [
        "inspect", "--product-model", str(spec), "--product-request", str(request), "--output-dir", str(output),
    ])
    assert result.exit_code == 0, result.output
    row = json.loads(next(output.glob("*/inspection.json")).read_bytes())["records"][0]
    assert row["outcome"] == "review_required" and row["reason"] == "simulated_prediction"
    assert row["image"] is None and row["detections"] == []
    assert row["multimodal"]["branch"] == "sensor" and row["multimodal"]["weight_status"] == "simulated"
    assert str(inputs.image) not in json.dumps(row)


def test_checkpoint_cpu_modalities_fallback_and_future_exclusion(tmp_path: Path, inputs: InputFixture) -> None:
    state = frozen(tmp_path)
    spec = checkpoint(tmp_path, state)
    pairing = fixture_model(tmp_path, inputs).pairings
    spec = spec.model_copy(update={"pairings": pairing})
    service = InspectionService.from_product(spec, tmp_path, credentials=Credentials())

    for request, image, branch in (
        (without_sensor(inputs.request), inputs.image, "image"),
        (inputs.request, None, "sensor"),
        (inputs.request, inputs.image, "paired"),
    ):
        result = service.inspect_product(request, image)
        assert result.outcome == "review_required" and not result.detections
        product = result.multimodal
        assert product is not None and product.branch == product.input_mode == branch
        assert product.score_kind == "uncalibrated_sigmoid" and product.weight_status == "untrained"
        assert product.model and product.model.checkpoint_sha256 == spec.checkpoint_sha256
        assert product.model.preprocessing_fingerprint == state.fingerprint
        assert InspectionResult.model_validate_json(result.model_dump_json()) == result

    evidence = inputs.request.sensor_evidence
    assert evidence is not None
    revised = evidence.sequence.model_copy(update={"steps": (*evidence.sequence.steps[:2], SensorStep(
        timestamp="2026-01-01T00:00:06Z", values=(9000., 9000., 9000., 9000., 9000.),
    ))})
    later = inputs.request.model_copy(update={"sensor_evidence": evidence.model_copy(update={"sequence": revised})})
    assert service.inspect_product(later, inputs.image) == service.inspect_product(inputs.request, inputs.image)
    window = transform_sensor(revised, CAPTURE, state)
    assert window.timestamps == ("2026-01-01T00:00:04Z", CAPTURE)
    assert window.observed[1] == (True, True, True, False, False)

    wrong_pair = inputs.request.model_copy(update={"source_id": "camera-2"})
    rejected = service.inspect_product(wrong_pair, inputs.image)
    assert rejected.reason == "invalid_request" and rejected.multimodal is None
    changed_measurement = evidence.sequence.model_copy(update={
        "steps": (evidence.sequence.steps[0].model_copy(update={"values": (12., None, 1., None, None)}),
                  *evidence.sequence.steps[1:]),
    })
    wrong_sensor = inputs.request.model_copy(update={
        "sensor_evidence": evidence.model_copy(update={"sequence": changed_measurement}),
    })
    assert service.inspect_product(wrong_sensor, inputs.image).reason == "invalid_request"
    other_image = tmp_path / "other.png"
    Image.new("RGB", (40, 24), (20, 50, 10)).save(other_image)
    assert service.inspect_product(inputs.request, other_image).reason == "invalid_request"
    assert service.inspect_product(without_sensor(inputs.request)).reason == "invalid_request"

    short = evidence.sequence.model_copy(update={"steps": evidence.sequence.steps[1:]})
    one_row = inputs.request.model_copy(update={"sensor_evidence": evidence.model_copy(update={"sequence": short})})
    unavailable = service.inspect_product(one_row)
    product = unavailable.multimodal
    assert unavailable.reason == "product_unavailable" and product and not product.scores
    assert product.branch is None and product.sensor_status == "insufficient_steps"

    no_measurements = evidence.sequence.model_copy(update={
        "steps": (SensorStep(timestamp=CAPTURE, values=(None,) * 5),),
    })
    no_sensor = inputs.request.model_copy(update={"sensor_evidence": evidence.model_copy(update={"sequence": no_measurements})})
    no_sensor_record = inputs.record.model_copy(update={"sequence": no_measurements})
    receipt = PairingReceipt.from_audited_record(no_sensor_record, image_source_id="camera-1", sensor_source_id="station-1")
    path = tmp_path / "fallback-pairs.json"
    path.write_text(PairingRegistry(receipts=(receipt,)).model_dump_json(), encoding="utf-8")
    fallback = InspectionService.from_product(spec.model_copy(update={"pairings": Path(path.name)}),
                                              tmp_path, credentials=Credentials()).inspect_product(no_sensor, inputs.image)
    product = fallback.multimodal
    assert product and product.input_mode == "paired" and product.branch == "image"
    assert product.alignment == "verified_receipt" and product.sensor_status == "no_observations"


def test_checkpoint_mismatches_and_lookahead_are_rejected(tmp_path: Path, inputs: InputFixture) -> None:
    spec = checkpoint(tmp_path, frozen(tmp_path, lookahead=True))
    credentials = Credentials()
    service = InspectionService.from_product(spec, tmp_path, credentials=credentials)
    assert service.inspect_product(inputs.request).reason == "invalid_request"
    assert service.inspect_product(without_sensor(inputs.request), inputs.image).outcome == "review_required"

    for field in ("checkpoint_sha256", "architecture_fingerprint"):
        with pytest.raises(InspectionError, match="mismatch"):
            InspectionService.from_product(spec.model_copy(update={field: FINGERPRINT}), tmp_path, credentials=credentials)
    other = tmp_path / "other"
    other.mkdir()
    frozen(other, train_hash="b" * 64)
    with pytest.raises(InspectionError, match="frozen preprocessing"):
        InspectionService.from_product(spec.model_copy(update={"preprocessing": Path("other/preprocessor.json")}),
                                       tmp_path, credentials=credentials)


def test_companion_detection_keeps_original_coordinates(tmp_path: Path, inputs: InputFixture) -> None:
    spec = checkpoint(tmp_path, frozen(tmp_path))
    credentials = Credentials()
    detector = InspectionService(
        MockPredictor("defect"), ModelProvenance(
            mode="real", model=ModelName.YOLO, checkpoint_kind="best", checkpoint_fingerprint=FINGERPRINT,
            run_metadata_fingerprint=FINGERPRINT, device="cpu", image_size=640,
        ), InspectionThresholds(), credentials,
    )
    service = InspectionService.from_product(spec, tmp_path, credentials=credentials, detector=detector)
    result = service.inspect_product(without_sensor(inputs.request), inputs.image)
    assert result.detections[0].bbox_xyxy == (10, 6, 30, 18)
    assert result.outcome == "review_required" and result.multimodal and result.multimodal.weight_status == "untrained"
    sensor = service.inspect_product(inputs.request)
    assert sensor.image is None and sensor.detections == ()


@pytest.mark.parametrize("change", ["order", "units", "missing_units", "future_clock", "target_feature", "missing_capture"])
def test_sensor_payload_validation(inputs: InputFixture, change: str) -> None:
    document = inputs.request.model_dump(mode="json")
    sequence = document["sensor_evidence"]["sequence"]
    if change == "order":
        sequence["channels"].reverse()
    elif change == "units":
        sequence["units"] = ["invalid"] * len(UNITS)
    elif change == "missing_units":
        sequence.pop("units")
    elif change == "future_clock":
        document["capture_timestamp"] = "2026-01-01T00:00:05"
    elif change == "target_feature":
        sequence["steps"][0]["defect_detected"] = 1
    else:
        document.pop("capture_timestamp")
    with pytest.raises(ValidationError):
        ProductInspectionInput.model_validate_json(json.dumps(document))
