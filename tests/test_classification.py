"""Fixture coverage for the opt-in SMT classification pipeline."""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline.classification import stages
from ict_pipeline.classification.contracts import LABEL_POLICY, ClassificationConfig
from ict_pipeline.classification.models import SENSOR_FEATURES, features, fit_model
from ict_pipeline.classification.prepare import prepare_samples
from ict_pipeline.classification.sensors import SampleIssue, aligned_sequence, json_readings
from ict_pipeline.cli import app
from ict_pipeline.data.preprocessing.audit import XLSX_FIELDS
from ict_pipeline.data.preprocessing.common import read_csv
from ict_pipeline.data.preprocessing.policy import SENSOR_FIELDS
from ict_pipeline.pipeline.store import RunStore, fingerprint_path


def _sample(
    root: Path, split: str, index: int, label: int, measurement: float, *,
    json_sensor: bool = True, xlsx: bool = False, image: bool = True,
    session: str | None = None, problem: str = "", source: str = "ACME",
) -> str:
    capture = datetime(2025, 8, 7, 8, index)
    status = "DEF" if label else "NOR"
    stem = f"SD_{status}_CC_{source}_{capture:%Y%m%d-%H%M%S}_{index:05}"
    suffix = "soldering_" + ("defect" if label else "normal") + "_cold_solder"
    folders = {
        kind: root / f"{split}_{kind}_{suffix}"
        for kind in ("annotations", "images", "sensors")
    }
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    filename = stem + ".JPG"
    if image:
        Image.new("RGB", (16, 16), (index * 11, index * 7, 0)).save(folders["images"] / filename)
    times = [capture + timedelta(seconds=step) for step in range(2)]
    values = [(measurement, 40, 0.2, 0.4, 35), (measurement + 1000, 41, 0.3, 0.5, 36)]
    group = {"image_id": 638, "file_name": stem + ".xlsx"}
    if json_sensor:
        group["sensor_sequence"] = [
            {"timestamp": t.isoformat(), **dict(zip(SENSOR_FIELDS, v)), "defect_detected": label,
             "defect_detected_copy": 10000 * label}
            for t, v in zip(times, values)
        ]
        if problem == "label":
            group["sensor_sequence"][0]["defect_detected"] = 1 - label
    if problem == "filename":
        group["file_name"] = "different.xlsx"
    data = {
        "images": [{"id": 8, "width": 16, "height": 16, "file_name": filename}],
        "annotations": [{"image_id": 8, "category_id": 23, "bbox": [1, 1, 2, 2]}] if label else [],
        "sensor_data": [group],
    }
    if session:
        data["production_session_id"] = session
    (folders["annotations"] / f"{stem}.json").write_text(json.dumps(data), encoding="utf-8")
    if xlsx:
        book = Workbook()
        book.active.append(["timestamp", *XLSX_FIELDS, "defect_detected"])
        for t, v in zip(times, values):
            if problem == "timestamp":
                t += timedelta(seconds=1)
            if problem == "value":
                v = (v[0] + 1, *v[1:])
            book.active.append([t, *v, label])
        book.save(folders["sensors"] / f"{stem}.xlsx")
        book.close()
    return stem


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


@pytest.mark.e2e
def test_classification_cli_resume_and_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "extracted"
    normal = _sample(root, "train", 1, 0, 10, xlsx=True)
    _sample(root, "train", 2, 1, 20)
    _sample(root, "train", 3, 0, 30, json_sensor=False, xlsx=True, image=False)
    _sample(root, "validation", 4, 0, 900)
    _sample(root, "validation", 5, 1, 901, xlsx=True)
    sensor_only = _sample(root, "validation", 6, 0, 902, image=False)
    image_only = _sample(root, "validation", 7, 1, 903, json_sensor=False)
    before = fingerprint_path(root)
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "classification.yaml"
    config.write_text(
        "dataset_source: existing\ndata_path: extracted\nclassification:\n"
        "  mode: fusion\n  sensor_model: centroid\n  epochs: 2\n", encoding="utf-8",
    )
    client = CliRunner()
    args = ["--config", str(config), "--run-id", "classifier"]
    dry = client.invoke(app, ["run", *args, "--dry-run"])
    assert dry.exit_code == 0, dry.output
    assert "classification_train" in dry.output and "validate_detection" not in dry.output
    assert not (tmp_path / "runs").exists()

    denied = client.invoke(app, ["run", *args])
    assert denied.exit_code == 1 and "--approve-train" in denied.output
    store = RunStore(tmp_path / "runs", "classifier")
    stopped = store.read()
    prepare = stopped.stages["classification_prepare"]
    assert prepare.status == "completed"
    assert stopped.stages["classification_train"].status == "failed"

    evaluate = stages.ClassificationStages.evaluate

    def interrupt(self: stages.ClassificationStages, context: stages.StageContext) -> stages.StageResult:
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(stages.ClassificationStages, "evaluate", interrupt)
        interrupted = client.invoke(app, ["resume", *args, "--approve-train"])
    assert interrupted.exit_code == 130, interrupted.output
    interrupted_state = store.read()
    assert interrupted_state.stages["classification_evaluate"].status == "interrupted"
    assert stages.ClassificationStages.evaluate is evaluate

    resumed = client.invoke(app, ["resume", *args])
    assert resumed.exit_code == 0, resumed.output
    state = store.read()
    assert state.status == "completed"
    assert state.stages["classification_prepare"].attempt == prepare.attempt
    assert state.stages["classification_train"].attempt == interrupted_state.stages["classification_train"].attempt

    prepared = store.artifact_dir("classification_prepare", prepare.attempt)
    records = list(stages.read_samples(prepared / "samples.jsonl"))
    assert {sample.sensor_source for sample in records} == {"both", "json", "xlsx", None}
    normal_record = next(sample for sample in records if sample.sample_id == normal)
    assert normal_record.sensor_image_id_offset == 630
    assert "defect_detected" not in (prepared / "samples.jsonl").read_text(encoding="utf-8")
    assert normal_record.sequence[-1].values[0] >= 1000
    assert features(normal_record, "sensor", ClassificationConfig())[:3] == (10.0, 0.0, 0.0)

    checkpoints = store.artifact_dir("classification_train", state.stages["classification_train"].attempt)
    checkpoint = _json(checkpoints / "checkpoint.json")
    assert checkpoint["label_policy"] == LABEL_POLICY
    assert checkpoint["models"]["sensor"]["feature_names"] == list(SENSOR_FEATURES)
    assert checkpoint["models"]["sensor"]["means"][0] == 20.0
    assert checkpoint["train_counts"]["sensor"] == {"0": 2, "1": 1}
    assert checkpoint["train_counts"]["image"] == {"0": 1, "1": 1}

    outputs = store.artifact_dir("classification_evaluate", state.stages["classification_evaluate"].attempt)
    comparison = _json(outputs / "comparison.json")
    assert comparison["split"] == "validation"
    assert comparison["coverage"] == {"paired": 2, "sensor_only": 1, "image_only": 1}
    assert {r["modality"]: r["samples"] for r in comparison["comparisons"] if r["cohort"] == "paired"} == {
        "sensor": 2, "image": 2, "fusion": 2,
    }
    rows = read_csv(outputs / "predictions.csv")
    for sample_id, single in ((sensor_only, "sensor"), (image_only, "image")):
        assert {row["modality"] for row in rows if row["sample_id"] == sample_id} == {single, "fusion"}
        assert len({row["score"] for row in rows if row["sample_id"] == sample_id}) == 1
    assert fingerprint_path(root) == before


def test_alignment_group_policy_and_quarantine(tmp_path: Path) -> None:
    root = tmp_path / "extracted"
    train = _sample(root, "train", 1, 0, 10, session="shared")
    validation = _sample(root, "validation", 2, 1, 20, session="shared")
    for index, problem in enumerate(("value", "timestamp", "filename", "label"), 3):
        _sample(root, "validation", index, 1, 20, xlsx=True, problem=problem)
    _sample(root, "train", 7, 0, 10)
    _sample(root, "validation", 8, 1, 20)
    fingerprint = fingerprint_path(root)
    output = tmp_path / "result"
    output.mkdir()
    prepare_samples(root, output, ClassificationConfig(), fingerprint, lambda *_: None)
    report = _json(output / "alignment.json")
    assert report["retained"] == 2
    assert report["group_audit"]["source_day"]["cross_split_groups"] == 1
    assert report["group_audit"]["session"]["cross_split_rows"] == {"train": 1, "validation": 1}
    issues = {row["sample_id"]: row["reason"] for row in read_csv(output / "alignment_issues.csv")}
    assert issues[train] == issues[validation] == "cross_split_session"
    assert set(issues.values()) == {
        "cross_split_session", "sensor_value_mismatch", "sensor_timestamp_mismatch",
        "sensor_filename_mismatch", "sensor_label_conflict",
    }
    strict = tmp_path / "strict"
    strict.mkdir()
    prepare_samples(root, strict, ClassificationConfig(group_policy="source_day"), fingerprint, lambda *_: None)
    assert _json(strict / "alignment.json")["retained"] == 0


@pytest.mark.e2e
@pytest.mark.parametrize(("mode", "missing", "image"), [("sensor", "available", False), ("unimodal", "paired", True)])
def test_single_modality_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, missing: str, image: bool,
) -> None:
    root = tmp_path / "extracted"
    _sample(root, "train", 1, 0, 10, image=image)
    _sample(root, "train", 2, 1, 30, image=image)
    _sample(root, "validation", 3, 0, 12, image=image)
    _sample(root, "validation", 4, 1, 28, image=image)
    _sample(root, "validation", 5, 0, 24, image=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ICT_DATASET_SOURCE", "skip")
    monkeypatch.setenv("ICT_DATA_PATH", str(root))
    monkeypatch.setenv("ICT_CLASSIFICATION", json.dumps({"mode": mode, "missing": missing, "epochs": 2}))
    outcome = CliRunner().invoke(app, ["run", "--run-id", "baseline", "--approve-train"])
    assert outcome.exit_code == 0, outcome.output
    store = RunStore(tmp_path / "runs", "baseline")
    state = store.read()
    report = _json(store.artifact_dir("classification_evaluate", state.stages["classification_evaluate"].attempt)
                   / "comparison.json")
    available = {row["modality"]: row["samples"] for row in report["comparisons"] if row["cohort"] == "available"}
    assert available == ({"sensor": 3} if mode == "sensor" else {"sensor": 2, "image": 2})


def test_invalid_readings_and_deterministic_logistic() -> None:
    row = dict.fromkeys(SENSOR_FIELDS, 10)
    row["timestamp"] = "2025-08-07 11:12:13"
    readings = json_readings([row, row], 0)
    with pytest.raises(SampleIssue, match="non_increasing_sensor_timestamps"):
        aligned_sequence(readings, (), None, 0.001)
    with pytest.raises(SampleIssue, match="invalid_sensor_value"):
        aligned_sequence(json_readings([{**row, "noise": float("nan")}], 0), (), None, 0.001)
    with pytest.raises(SampleIssue, match="sensor_capture_timestamp_mismatch"):
        aligned_sequence(readings[:1], (), datetime(2025, 8, 7, 11, 12, 12), 0.001)

    cfg = ClassificationConfig(epochs=3, seed=17)
    examples = [((0.0,) * len(SENSOR_FEATURES), 0), ((1.0,) * len(SENSOR_FEATURES), 1)]
    first = fit_model("sensor", examples, cfg, lambda _: None)
    second = fit_model("sensor", examples, cfg, lambda _: None)
    assert first == second
    assert first.predict(examples[0][0]) < first.predict(examples[1][0])
