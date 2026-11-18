import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.data.preprocessing.policy import label_policy
from ict_pipeline.evaluation import runner
from ict_pipeline.evaluation.contracts import (
    CLASSES, EvaluationConfig, EvaluationError, EvaluationSplit, Prediction, Preprocessing,
    normalize_prediction, relative_cxcywh_to_xyxy, xywh_to_xyxy,
)
from ict_pipeline.evaluation.metrics import detection_metrics
from ict_pipeline.models.contracts import ModelName, TrainConfig
from ict_pipeline.models.dataset import Object, Sample, inspect_data
from ict_pipeline.models.registry import REGISTRY
from ict_pipeline.models.training import checkpoint_paths
from ict_pipeline.pipeline.store import fingerprint_path


def write_split(root: Path, split: str, *, has_labels: bool = True, duplicate_val: bool = False) -> None:
    images = root / "coco" / "images" / split
    images.mkdir(parents=True)
    name = "val" if duplicate_val else split
    rows = []
    for image_id, stem in enumerate((name, f"{name}-normal"), 1):
        image = images / f"{stem}.png"
        image.write_bytes(stem.encode())
        rows.append({"id": image_id, "file_name": image.name, "width": 100, "height": 80})
    annotations = root / "coco" / "annotations"
    annotations.mkdir(exist_ok=True)
    (annotations / f"instances_{split}.json").write_text(json.dumps({
        "images": rows, "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 40], "iscrowd": 0},
        ] if has_labels else [],
        "categories": [{"id": category, "name": label} for category, label in CLASSES.items()],
    }), encoding="utf-8")


def write_run(workspace: Path, model: ModelName, references: dict[str, str] | None = None) -> Path:
    run = workspace / "training" / model
    checkpoint = checkpoint_paths(model, run).best
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(model.encode())
    (run / "run.json").write_text(json.dumps({
        "schema_version": 1, "model": model, "status": "completed", "label_policy": label_policy(),
        "checkpoint_format": REGISTRY[model].checkpoint_format, "configuration": TrainConfig().model_dump(),
        "split_fingerprints": references or {},
        "checkpoints": {"best": {
            "path": checkpoint.relative_to(run).as_posix(), "fingerprint": fingerprint_path(checkpoint),
        }},
    }), encoding="utf-8")
    return run


class FakePredictor:
    device = "cpu"
    hardware = {"device": "cpu", "cpu": "fixture CPU", "synchronization": "synchronous CPU"}
    libraries = {"torch": "fixture", "torchvision": "fixture"}
    postprocessing = {"nms": "fixture"}

    def __init__(self, model: ModelName) -> None:
        self.model = model
        self.events: list[str] = []

    def synchronize(self) -> None:
        self.events.append("sync")

    def predict(self, sample: Sample) -> Prediction:
        self.events.append(sample.image.stem)
        prediction = not sample.image.stem.endswith("-normal")
        boxes = ((10, 20, 40, 60),) if prediction else ()
        ids = (int(self.model == ModelName.FASTER_RCNN),) if prediction else ()
        return normalize_prediction(
            self.model, sample, boxes, ids, (0.9,) if prediction else (),
            Preprocessing((sample.width, sample.height), (64, 64), "stretch"),
        )


@pytest.mark.parametrize(("model", "native_id"), [(ModelName.YOLO, 4), (ModelName.FASTER_RCNN, 5), (ModelName.RT_DETR, 4)])
def test_normalize_class_mapping_and_bounds(model: ModelName, native_id: int) -> None:
    sample = Sample(Path("sample.png"), 100, 80, ())
    preprocess = Preprocessing((100, 80), (64, 64), "stretch")
    pred = normalize_prediction(model, sample, [(-1, 5, 110, 80)], [native_id], [0.9], preprocess)
    assert pred == Prediction("sample", ((0, 5, 100, 80),), (5,), (0.9,), preprocess)
    with pytest.raises(EvaluationError, match="class"):
        normalize_prediction(model, sample, [pred.boxes[0]], [native_id + 1], [0.9], preprocess)


def test_box_conversion() -> None:
    assert xywh_to_xyxy((10, 20, 30, 40)) == (10, 20, 40, 60)
    assert relative_cxcywh_to_xyxy((0.5, 0.5, 0.5, 0.25), 200, 80) == (50, 30, 150, 50)


def test_coco_ap_interpolation_and_operating_metrics() -> None:
    a = Sample(Path("a.png"), 100, 100, (Object(1, (0, 0, 20, 20)), Object(1, (50, 50, 20, 20))))
    normal = Sample(Path("normal.png"), 100, 100, ())
    preprocess = Preprocessing((100, 100), (64, 64), "stretch")
    predictions = {
        "a": Prediction("a", ((0, 0, 15, 20), (0, 0, 15, 20), (50, 50, 70, 70)), (1, 1, 2),
                        (0.9, 0.8, 0.7), preprocess),
        "normal": Prediction("normal", ((0, 0, 15, 20),), (1,), (0.99,), preprocess),
    }
    metrics = detection_metrics((normal, a), predictions, EvaluationConfig())
    assert metrics["map50"] == pytest.approx(0.5 * 51 / 101)
    assert metrics["map50_95"] == pytest.approx(0.6 * 0.5 * 51 / 101)
    assert metrics["per_class"][1]["ap50"] is None
    assert metrics["precision"] == 0.25
    assert metrics["recall"] == 0.5
    assert metrics["f1"] == pytest.approx(1 / 3)
    matrix = metrics["confusion_matrix"]["counts"]
    assert matrix[0] == [1, 1, 0, 0, 0, 0]
    assert matrix[-1] == [2, 0, 0, 0, 0, 0]
    with pytest.raises(EvaluationError, match="exactly once"):
        detection_metrics((normal, a), {"a": predictions["a"]}, EvaluationConfig())


def test_empty_predictions_and_coco_detection_cap() -> None:
    sample = Sample(Path("s.png"), 100, 100, (Object(1, (50, 50, 20, 20)),))
    preprocess = Preprocessing((100, 100), (64, 64), "stretch")
    empty = Prediction("s", (), (), (), preprocess)
    capped = Prediction("s", ((0, 0, 5, 5),) * 100 + ((50, 50, 70, 70),), (1,) * 101, (0.9,) * 100 + (0.8,), preprocess)
    assert detection_metrics((sample,), {"s": empty}, EvaluationConfig())["map50"] == 0
    assert detection_metrics((sample,), {"s": capped}, EvaluationConfig())["recall"] == 0


def test_cli_refuses_missing_checkpoint_and_ground_truth(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run = write_run(tmp_path, ModelName.YOLO)
    result = CliRunner().invoke(app, ["evaluate", "--run", str(run), "--checkpoint", "last"])
    assert result.exit_code == 1
    assert "missing checkpoint" in result.stderr

    data = tmp_path / "data"
    write_split(data, "val", has_labels=False)
    result = CliRunner().invoke(app, ["evaluate", "--run", str(run), "--dataset", str(data)])
    assert result.exit_code == 1
    assert "ground-truth" in result.stderr
    assert not (tmp_path / "outputs").exists()

    checkpoint_paths(ModelName.YOLO, run).best.write_bytes(b"changed")
    result = CliRunner().invoke(app, ["evaluate", "--run", str(run), "--dataset", str(data)])
    assert result.exit_code == 1
    assert "fingerprint differs" in result.stderr


def test_cross_adapter_evaluation_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    data = tmp_path / "detection"
    write_split(data, "val")
    models = tuple(ModelName)
    runs = tuple(write_run(tmp_path, name) for name in models)
    loaded = {}
    ticks = iter(i / 4 for i in range(100))
    monkeypatch.setattr(runner, "perf_counter", lambda: next(ticks))

    def factory(name: ModelName, checkpoint: Path, training: TrainConfig, cfg: EvaluationConfig, out: Path) -> FakePredictor:
        if name == models[1] and name not in loaded:
            loaded[name] = None
            raise KeyboardInterrupt
        predictor = FakePredictor(name)
        loaded[name] = predictor
        return predictor

    config = EvaluationConfig(warmup=1, image_size=64)

    def run(resume: Path | None = None) -> tuple[Path, ...]:
        return tuple(runner.evaluate_runs(
            tmp_path, runs, data, EvaluationSplit.VALIDATION, config, tmp_path / "results",
            "best", False, None, resume, lambda *_: None, factory,
        ))

    with pytest.raises(KeyboardInterrupt):
        run()
    directory, = (tmp_path / "results").iterdir()
    events = [json.loads(row) for row in (directory / "events.jsonl").read_text().splitlines()]
    assert {"model_index": 2, "stage": "load checkpoint", "status": "interrupted"}.items() <= events[-1].items()
    first = loaded[models[0]]
    paths = run(directory)
    assert loaded[models[0]] is first
    assert first.events == ["val", "sync", "sync", "val", "sync", "sync", "val-normal", "sync"]
    result = CliRunner().invoke(app, ["compare", *(str(path) for path in paths)])
    assert result.exit_code == 0, result.stderr
    comparison = json.loads(result.stdout)
    assert comparison["dataset"]["role"] == "validation_for_tuning"
    assert comparison["matching_speed_environment"]
    reports = comparison["models"]
    assert all(item["metrics"] == reports[0]["metrics"] for item in reports)
    assert all(item["metrics"]["map50_95"] == 1 for item in reports)
    assert all(item["speed"]["fps"] == 4 for item in reports)
    assert all(item["speed"]["latency_ms"]["p95"] == 250 for item in reports)
    assert {item["model"] for item in reports} == set(models)


def test_held_out_requires_provenance_and_rejects_validation_comparison(tmp_path: Path) -> None:
    data = tmp_path / "detection"
    for split in ("train", "val", "test"):
        write_split(data, split)
    references = inspect_data(data, "coco", Credentials()).split_fingerprints
    run = write_run(tmp_path, ModelName.YOLO, references)

    def evaluate(split: EvaluationSplit, held_out: bool) -> Path:
        return next(runner.evaluate_runs(
            tmp_path, (run,), data, split, EvaluationConfig(warmup=0), tmp_path / "results",
            "best", held_out, None, None, lambda *_: None, lambda name, *_: FakePredictor(name),
        ))

    validation = evaluate(EvaluationSplit.VALIDATION, False)
    with pytest.raises(EvaluationError, match="attest"):
        evaluate(EvaluationSplit.TEST, False)
    test = evaluate(EvaluationSplit.TEST, True)
    assert runner.read_report(test)["dataset"]["role"] == "held_out_test"
    with pytest.raises(EvaluationError, match="same labeled split"):
        runner.compare_reports(tmp_path, (validation, test))
    (data / "coco" / "images" / "test" / "test.png").write_bytes(b"val")
    with pytest.raises(EvaluationError, match="shares image"):
        evaluate(EvaluationSplit.TEST, True)
