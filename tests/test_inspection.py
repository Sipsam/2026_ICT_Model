import csv
import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.data.preprocessing.policy import label_policy
from ict_pipeline.evaluation.contracts import EvaluationConfig, Prediction, Preprocessing, normalize_prediction
from ict_pipeline.inspection import records, service as implementation
from ict_pipeline.inspection.contracts import (
    ImageSize, InspectionError, InspectionRequest, InspectionResult, InspectionThresholds, MockScenario,
    ModelProvenance, SensorInput,
)
from ict_pipeline.inspection.service import InspectionService, MockPredictor
from ict_pipeline.models.contracts import ModelName, TrainConfig
from ict_pipeline.models.dataset import Sample
from ict_pipeline.models.registry import REGISTRY
from ict_pipeline.models.training import checkpoint_paths
from ict_pipeline.pipeline.store import fingerprint_path


@pytest.fixture
def image(tmp_path: Path) -> Path:
    path = tmp_path / "part.png"
    Image.new("RGB", (80, 40)).save(path)
    return path


def request_for(path: Path) -> InspectionRequest:
    return InspectionRequest(request_id="request-1", source_id="source-1", image_path=path)


def make_run(root: Path, model: ModelName) -> tuple[Path, Path]:
    run = root / model
    checkpoint = checkpoint_paths(model, run).best
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture checkpoint")
    (run / "run.json").write_text(json.dumps({
        "schema_version": 1, "model": model, "status": "completed", "label_policy": label_policy(),
        "checkpoint_format": REGISTRY[model].checkpoint_format,
        "configuration": TrainConfig(image_size=128).model_dump(mode="json"),
        "split_fingerprints": {},
        "checkpoints": {"best": {
            "path": checkpoint.relative_to(run).as_posix(), "fingerprint": fingerprint_path(checkpoint),
        }},
    }), encoding="utf-8")
    return run, checkpoint


class AdapterFixture:
    device = "cpu"
    hardware: dict[str, object] = {}
    postprocessing: dict[str, object] = {}
    libraries = {"torch": "fixture"}

    def __init__(self, model: ModelName, *, empty: bool = False) -> None:
        self.model = model
        self.empty = empty
        self.calls = 0

    def synchronize(self) -> None:
        pass

    def predict(self, sample: Sample) -> Prediction:
        self.calls += 1
        original = (sample.width, sample.height)
        assert original == (80, 40)
        with Image.open(sample.image) as source:
            assert source.size == original
        preprocessing = Preprocessing(original, (128, 128), "stretch")
        if self.empty:
            return Prediction(sample.image.stem, (), (), (), preprocessing)
        labels = (5, 5, 1, 2) if self.model == ModelName.FASTER_RCNN else (4, 4, 0, 1)
        return normalize_prediction(
            self.model, sample, ((-5, 4, 90, 40), (0, 4, 80, 40), (0, 4, 80, 40), (0, 0, 8, 8)),
            labels, (0.91, 0.8, 0.85, 0.2), preprocessing,
        )


@pytest.mark.parametrize("model", tuple(ModelName))
def test_real_factory_loads_once_and_reuses_class_mapping(tmp_path: Path, image: Path, model: ModelName) -> None:
    run, checkpoint = make_run(tmp_path, model)
    predictor = AdapterFixture(model)
    loads: list[Path] = []

    def factory(name: ModelName, local: Path, train: TrainConfig, config: EvaluationConfig, scratch: Path) -> AdapterFixture:
        assert name == model and local != checkpoint and local.is_relative_to(scratch)
        assert local.read_bytes() == checkpoint.read_bytes()
        assert config.image_size == train.image_size == 128
        assert config.device == "cpu" and config.nms_iou == 0.5
        loads.append(local)
        return predictor

    service = InspectionService.from_run(
        run, trust_run=True, factory=factory, expected_fingerprint=fingerprint_path(checkpoint),
        thresholds=InspectionThresholds(confidence=0.25, nms_iou=0.5),
    )
    result = service.inspect(request_for(image))
    assert service.inspect(request_for(image)) == result
    assert len(loads) == 1 and predictor.calls == 2 and not loads[0].exists()
    assert result.outcome == "defect_detected"
    assert result.provenance.mode == "real" and result.provenance.model == model
    assert result.provenance.checkpoint_fingerprint == fingerprint_path(checkpoint)
    assert result.provenance.run_metadata_fingerprint == fingerprint_path(run / "run.json")
    assert [(row.class_id, row.class_name, row.source_category_id) for row in result.detections] == [
        (5, "tombstone", 30), (1, "cold_solder", 23),
    ]
    assert result.detections[0].bbox_xyxy == (0, 4, 80, 40)
    assert result.image is not None and result.image.sha256 == hashlib.sha256(image.read_bytes()).hexdigest()
    assert InspectionResult.model_validate_json(result.model_dump_json()) == result


def test_empty_real_prediction_requires_review(tmp_path: Path, image: Path) -> None:
    run, _ = make_run(tmp_path, ModelName.YOLO)
    service = InspectionService.from_run(
        run, trust_run=True, factory=lambda *_: AdapterFixture(ModelName.YOLO, empty=True),
    )
    result = service.inspect(request_for(image))
    assert (result.outcome, result.reason, result.detections) == ("review_required", "no_detections", ())
    with pytest.raises(ValidationError):
        InspectionResult.model_validate_json(result.model_copy(update={
            "outcome": "pass", "reason": "verified_good",
        }).model_dump_json())
    verified = InspectionResult.model_validate_json(result.model_copy(update={
        "outcome": "pass", "reason": "verified_good", "verification_reference": "qa-123",
    }).model_dump_json())
    assert verified.outcome == "pass"


@pytest.mark.parametrize(("scenario", "confidence", "count"), [("empty", 0.25, 0), ("defect", 0.95, 1), ("defect", 1.0, 0)])
def test_mock_predictions_are_simulated(image: Path, scenario: MockScenario, confidence: float, count: int) -> None:
    service = InspectionService.mock(scenario, thresholds=InspectionThresholds(confidence=confidence))
    result = service.inspect(request_for(image))
    assert result.provenance.mode == "mock" and result.provenance.mock_scenario == scenario
    assert result.outcome == "review_required" and result.reason == "simulated_prediction"
    assert len(result.detections) == count
    assert result.provenance.checkpoint_fingerprint is None and result.provenance.model is None
    if count:
        assert result.detections[0].bbox_xyxy == (20, 10, 60, 30)


@pytest.mark.parametrize("case", ("untrusted", "policy", "format", "checkpoint", "pin", "incomplete", "path", "missing_metadata"))
def test_provenance_errors_prevent_deserialization(tmp_path: Path, case: str) -> None:
    run, checkpoint = make_run(tmp_path, ModelName.YOLO)
    marker = run / "run.json"
    metadata = json.loads(marker.read_text())
    if case == "policy":
        metadata["label_policy"] = {}
    elif case == "format":
        metadata["checkpoint_format"] = "wrong-format"
    elif case == "checkpoint":
        checkpoint.write_bytes(b"modified")
    elif case == "incomplete":
        metadata["status"] = "running"
    elif case == "path":
        metadata["checkpoints"]["best"]["path"] = "../outside.pt"
    marker.write_text(json.dumps(metadata), encoding="utf-8")
    if case == "missing_metadata":
        run = tmp_path / "missing"

    def forbidden(*_: object) -> AdapterFixture:
        pytest.fail("The predictor must never load the rejected checkpoint.")

    with pytest.raises(InspectionError):
        InspectionService.from_run(
            run, trust_run=case != "untrusted", factory=forbidden,
            expected_fingerprint="0" * 64 if case == "pin" else None,
        )


def test_checkpoint_change_during_snapshot_prevents_loading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = make_run(tmp_path, ModelName.YOLO)

    def substitution(source: Path, destination: Path) -> None:
        destination.write_bytes(b"unexpected checkpoint")

    monkeypatch.setattr(implementation, "copyfile", substitution)
    with pytest.raises(InspectionError, match="changed"):
        InspectionService.from_run(
            run, trust_run=True, factory=lambda *_: pytest.fail("Snapshot must be verified before loading"),
        )


def test_image_dimensions_formats_and_corrupt_data(tmp_path: Path, image: Path) -> None:
    service = InspectionService.mock()
    request = request_for(image)
    bad_size = request.model_copy(update={"expected_size": ImageSize(width=40, height=80)})
    assert service.inspect(bad_size).reason == "dimension_mismatch"
    assert service.inspect(request_for(tmp_path / "absent.png")).reason == "missing_image"

    unsupported = tmp_path / "unsupported.png"
    Image.new("RGB", (80, 40)).save(unsupported, format="GIF")
    corrupt = tmp_path / "corrupt.png"
    corrupt.write_bytes(image.read_bytes()[:60])
    animated = tmp_path / "animated.png"
    first, second = Image.new("RGB", (80, 40)), Image.new("RGB", (80, 40), "red")
    first.save(animated, save_all=True, append_images=[second])
    for path in (tmp_path, unsupported, corrupt, animated):
        result = service.inspect(request_for(path))
        assert (result.outcome, result.reason) == ("inference_failed", "invalid_image")
    assert service.inspect(request_for(Path("https://example.invalid/image.png"))).reason == "invalid_request"

    jpeg = tmp_path / "header-detected.dat"
    first.save(jpeg, format="JPEG")
    descriptor = service.inspect(request_for(jpeg)).image
    assert descriptor is not None and descriptor.format == "JPEG"


@pytest.mark.parametrize("limit", ("MAX_IMAGE_BYTES", "MAX_IMAGE_PIXELS"))
def test_image_limits_fail_before_prediction(image: Path, monkeypatch: pytest.MonkeyPatch, limit: str) -> None:
    monkeypatch.setattr(implementation, limit, 1)
    result = InspectionService.mock("defect").inspect(request_for(image))
    assert (result.outcome, result.reason, result.detections) == ("inference_failed", "invalid_image", ())


@pytest.mark.parametrize("raises", (False, True))
def test_bad_prediction_returns_safe_failure(image: Path, capsys: pytest.CaptureFixture[str], raises: bool) -> None:
    class BadPredictor(MockPredictor):
        def predict(self, sample: Sample) -> Prediction:
            print("sensitive backend diagnostic")
            if raises:
                raise RuntimeError("another sensitive diagnostic")
            return Prediction(sample.image.stem, ((1, 2, 3, 4),), (99,), (0.95,),
                              Preprocessing((80, 40), (80, 40), "stretch"))

    service = InspectionService(BadPredictor("defect"), ModelProvenance(mode="mock", mock_scenario="defect"),
                                InspectionThresholds(), Credentials())
    result = service.inspect(request_for(image))
    assert (result.outcome, result.reason) == ("inference_failed", "prediction_failed")
    assert "sensitive" not in result.model_dump_json() and not result.detections
    assert "sensitive" not in str(capsys.readouterr())


def test_reject_incompatible_contract_values(image: Path) -> None:
    with pytest.raises(ValidationError):
        SensorInput(source_id="sensor", values={"defect_detected": 1.0})
    with pytest.raises(ValidationError):
        InspectionRequest(request_id="r", source_id="s", image_path=image, metadata={"n": [float("nan")]})
    with pytest.raises(ValidationError):
        ModelProvenance(mode="mock", mock_scenario="empty", model=ModelName.YOLO)
    with pytest.raises(ValidationError):
        ModelProvenance(mode="real")
    for value in (True, 1.0, "1", 2):
        with pytest.raises(ValidationError):
            InspectionRequest.model_validate_json(json.dumps({
                **request_for(image).model_dump(mode="json"), "schema_version": value,
            }))


def test_single_cli_exports_stable_mock_records(tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    args = ["inspect", image.name, "--mock", "defect", "--source-id", "=1+1", "--output-dir", "records"]
    result, repeat = (CliRunner().invoke(app, args) for _ in range(2))
    assert result.exit_code == repeat.exit_code == 0, result.output
    output, another_output = (Path(run.stdout.strip()) for run in (result, repeat))
    assert output.parent == another_output.parent == tmp_path / "records" and output != another_output
    for name in ("inspection.json", "inspection.csv"):
        assert (output / name).read_bytes() == (another_output / name).read_bytes()
    item = json.loads((output / "inspection.json").read_bytes())["records"][0]
    assert item["source_id"] == "=1+1" and item["provenance"]["mode"] == "mock"
    with (output / "inspection.csv").open(encoding="utf-8", newline="") as stream:
        row, = csv.DictReader(stream)
    assert row["source_id"] == "'=1+1" and row["outcome"] == "review_required"
    assert json.loads(row["detections_json"]) == item["detections"]


def test_batch_cli_keeps_order_and_publishes_failures(tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AIHUB_API_KEY", "sensitive-batch-secret")
    manifest = tmp_path / "batch.jsonl"
    request = {
        "schema_version": 1, "request_id": "q1", "source_id": "camera-1", "image_path": image.name,
        "expected_size": {"width": 80, "height": 40},
        "sensor": {"source_id": "sensor-2", "values": {"temperature": 21.5}},
        "metadata": {"lot": 4, "password": "sensitive-batch-secret"},
    }
    manifest.write_text("\n".join([
        json.dumps(request), "{bad", json.dumps({**request, "schema_version": 2}),
        json.dumps({**request, "request_id": "q2", "image_path": "missing.png"}),
    ]), encoding="utf-8")
    result = CliRunner().invoke(app, ["inspect", "--batch", manifest.name, "--mock", "empty", "--output-dir", "reports"])
    assert result.exit_code == 1, result.output
    assert "inspect images 0/4" in result.stderr and "inspect images 4/4" in result.stderr
    output = Path(result.stdout.strip())
    document = (output / "inspection.json").read_text(encoding="utf-8")
    rows = json.loads(document)["records"]
    assert [row["input_line"] for row in rows] == [1, 2, 3, 4]
    assert [row["reason"] for row in rows] == ["simulated_prediction", "invalid_request", "invalid_request", "missing_image"]
    assert rows[0]["sensor_source_id"] == "sensor-2" and rows[0]["metadata"]["lot"] == 4
    assert "sensitive-batch-secret" not in document


@pytest.mark.parametrize("changed_image", (False, True))
def test_batch_resume_validates_image_bytes_and_rejects_changed_options(
    tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch, changed_image: bool,
) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "batch.jsonl"
    row = request_for(Path(image.name)).model_dump_json()
    manifest.write_text(f"{row}\n{row}\n", encoding="utf-8")
    calls: list[str] = []
    predict = MockPredictor.predict

    def interrupt(self: MockPredictor, sample: Sample) -> Prediction:
        calls.append(sample.image.name)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return predict(self, sample)

    monkeypatch.setattr(MockPredictor, "predict", interrupt)
    args = ["inspect", "--batch", manifest.name, "--mock", "defect", "--output-dir", "records"]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 130, first.output
    attempt, = (tmp_path / "records").iterdir()
    assert not (attempt / "reports").exists()
    resume = args + ["--resume-from", str(attempt)]
    mismatch = CliRunner().invoke(app, resume + ["--confidence", "0.5"])
    assert mismatch.exit_code == 1 and "original manifest" in mismatch.stderr
    if changed_image:
        Image.new("RGB", (80, 40), "red").save(image)
    second = CliRunner().invoke(app, resume)
    assert second.exit_code == 0, second.output
    assert len(calls) == (4 if changed_image else 3)
    assert Path(second.stdout.strip()) == attempt / "reports"
    assert CliRunner().invoke(app, resume).exit_code == 1


def test_resume_rejects_corrupt_recovery_records(tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    manifest = tmp_path / "batch.jsonl"
    manifest.write_text(f"{request_for(Path(image.name)).model_dump_json()}\n", encoding="utf-8")

    publish = records.write_result

    def interrupt_after_checkpoint(path: Path, payload: dict[str, object]) -> None:
        publish(path, payload)
        raise KeyboardInterrupt

    monkeypatch.setattr(records, "write_result", interrupt_after_checkpoint)
    args = ["inspect", "--batch", manifest.name, "--mock", "empty", "--output-dir", "records"]
    assert CliRunner().invoke(app, args).exit_code == 130
    attempt, = (tmp_path / "records").iterdir()
    journal, = (attempt / "recovery").iterdir()
    original_document = json.loads(journal.read_bytes())
    original_document["result"]["source_id"] = "altered"
    journal.write_text(json.dumps(original_document), encoding="utf-8")
    result = CliRunner().invoke(app, args + ["--resume-from", str(attempt)])
    assert result.exit_code == 1 and "recovery record is invalid" in result.stderr
    assert not (attempt / "reports").exists()


def test_cli_only_loads_trusted_real_runs(tmp_path: Path, image: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, _ = make_run(tmp_path, ModelName.FASTER_RCNN)
    monkeypatch.chdir(tmp_path)
    predictor = AdapterFixture(ModelName.FASTER_RCNN)
    monkeypatch.setattr(implementation, "load_predictor", lambda *_: predictor)
    args = ["inspect", image.name, "--run", str(run)]
    assert CliRunner().invoke(app, args).exit_code == 2
    assert CliRunner().invoke(app, args + ["--trust-run", "--output-dir", str(run)]).exit_code == 1
    assert predictor.calls == 0
    result = CliRunner().invoke(app, args + ["--trust-run"])
    assert result.exit_code == 0, result.output
    document = json.loads((Path(result.stdout.strip()) / "inspection.json").read_bytes())
    assert document["records"][0]["provenance"]["mode"] == "real"
    assert document["records"][0]["outcome"] == "defect_detected"
