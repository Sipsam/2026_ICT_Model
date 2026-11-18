"""CPU-only integration checks for the SMT preprocessing DAG."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import pytest
import yaml
from openpyxl import Workbook
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline import cli
from ict_pipeline.data.model import load_selection
from ict_pipeline.data.preprocessing import stages as adapters
from ict_pipeline.data.preprocessing.audit import XLSX_FIELDS
from ict_pipeline.data.preprocessing.common import read_csv
from ict_pipeline.data.preprocessing.policy import SENSOR_FIELDS
from ict_pipeline.pipeline import PipelineError, StageRunner
from ict_pipeline.pipeline.model import StageContext, StageResult
from ict_pipeline.pipeline.stages import pipeline_stages
from ict_pipeline.pipeline.store import RunState, RunStore, fingerprint_path

STAGES = ("audit", "manifest", "convert", "validate_detection")
TRAIN_MIXED = "SD_DEF_CC_ACME_20240101_001"
TRAIN_NORMAL = "SD_NOR_CC_ACME_20240101_002"
VAL_SENSOR = "SD_DEF_ST_ACME_20240101_008"
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def sample(
    dataset: Path, split: str, stem: str, defect: str, categories: tuple[int, ...], *,
    color: int, offset: int = 600, bad_box: bool = False, mismatch: str = "",
) -> None:
    status = "normal" if stem.split("_")[1] == "NOR" else "defect"
    group = f"soldering_{status}_{defect}"
    folders = {kind: dataset / f"{split}_{kind}_{group}" for kind in ("images", "annotations", "sensors")}
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
    image = f"{stem}.JPG"
    Image.new("RGB", (32, 24), (color, 0, 0)).save(folders["images"] / image)
    time = datetime(2024, 1, 1, 12)
    measurements = [25.0, 40.0, 0.2, 0.4, 30.0]
    label = int(status == "defect")
    sequence_row = {
        "timestamp": str(time), **dict(zip(SENSOR_FIELDS, measurements)), "defect_detected": label,
    }
    data = {
        "images": [{"id": 7, "file_name": image, "width": 32, "height": 24}],
        "annotations": [
            {"image_id": 7, "category_id": cid, "bbox": [2, 3, 60 if bad_box else 8, 6]}
            for cid in categories
        ],
        "sensor_data": [{"image_id": 7 + offset, "sensor_sequence": [sequence_row]}],
    }
    (folders["annotations"] / f"{stem}.json").write_text(json.dumps(data), encoding="utf-8")
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["timestamp", *XLSX_FIELDS, "defect_detected"])
    row_time = time.replace(second=1) if mismatch == "timestamp" else time
    readings = [24.0, *measurements[1:]] if mismatch == "value" else measurements
    # The workbook label is never compared as a sensor measurement.
    sheet.append([row_time, *readings, 1 - label])
    workbook.save(folders["sensors"] / f"{stem}.xlsx")
    workbook.close()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data" / "extracted"
    sample(root, "train", TRAIN_MIXED, "cold_solder", (7, 23), color=16)
    sample(root, "train", TRAIN_NORMAL, "cold_solder", (7,), color=32, offset=0)
    sample(root, "train", "SD_DEF_ML_ACME_20240101_003", "misalignment", (24,), color=48)
    sample(root, "train", "SD_DEF_SH_ACME_20240101_004", "short", (25,), color=64)
    sample(root, "validation", "SD_DEF_MS_ACME_20240101_005", "missing_component", (27,), color=80)
    sample(root, "validation", "SD_DEF_ST_ACME_20240101_006", "tombstone", (30,), color=96)
    sample(root, "train", "SD_DEF_CC_ACME_20240101_007", "cold_solder", (23, 17), color=112)
    sample(root, "validation", VAL_SENSOR, "tombstone", (30,), color=128, mismatch="timestamp")
    sample(root, "validation", "SD_DEF_ST_ACME_20240101_009", "tombstone", (30,), color=144, mismatch="value")
    sample(root, "train", "SD_DEF_CC_ACME_20240101_010", "cold_solder", (23,), color=160, bad_box=True)
    return root


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def artifact(workspace: Path, state: RunState, stage: str) -> Path:
    return RunStore(workspace / "runs", state.run_id).artifact_dir(stage, state.stages[stage].attempt)


def runner(workspace: Path, dataset: Path) -> StageRunner:
    return StageRunner(pipeline_stages(dataset_root=dataset), workspace, {})


def test_cli_preprocessing_policy_and_cache(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ICT_DATASET_SOURCE", "skip")
    monkeypatch.setenv("ICT_DATA_PATH", str(dataset))
    client = CliRunner()
    before = fingerprint_path(dataset)
    command = ["run", "--from", "audit", "--until", "validate_detection", "--run-id", "fixture"]
    result = client.invoke(cli.app, command)
    assert result.exit_code == 0, result.output
    state = runner(tmp_path, dataset).status("fixture")

    audit_dir = artifact(tmp_path, state, "audit")
    audit = read_json(audit_dir / "summary.json")
    assert audit["total_samples"] == 10
    assert audit["sensor_id_offsets"] == {"0": 1, "600": 9}
    issues = {issue["sample"]: issue["issue"] for issue in read_csv(audit_dir / "issues.csv")}
    assert issues[TRAIN_MIXED] == "mixed_normal_defect_boxes"
    assert issues[VAL_SENSOR] == "sensor_timestamp_mismatch"
    assert issues["SD_DEF_ST_ACME_20240101_009"] == "sensor_value_mismatch"
    assert "sensor_timestamp_mismatch" in audit["issue_counts"]
    assert "sensor_value_mismatch" in audit["issue_counts"]

    manifest_dir = artifact(tmp_path, state, "manifest")
    manifest = read_json(manifest_dir / "manifest_summary.json")
    assert manifest["by_decision"] == {"candidate": 6, "hold_review": 3, "exclude": 1}
    assert manifest["by_split_decision"] == {
        "train/candidate": 4, "train/exclude": 1, "train/hold_review": 1,
        "validation/candidate": 2, "validation/hold_review": 2,
    }
    assert manifest["source_fingerprint"] == audit["source_fingerprint"] == before
    assert manifest["split_fingerprint"] == audit["split_fingerprint"]
    assert manifest["audit_fingerprints"]["issues.csv"] == fingerprint_path(audit_dir / "issues.csv")
    assert manifest["label_policy"] == audit["label_policy"]
    assert manifest["path_base"] == "dataset_root"
    rows = {row["sample_id"]: row for row in read_csv(manifest_dir / "manifest.csv")}
    mixed = rows[TRAIN_MIXED]
    assert (mixed["audit_status"], mixed["decision"], mixed["decision_reason"]) == (
        "review_required", "candidate", "mixed_objects_allowed",
    )
    assert mixed["target_box_count"] == "1"
    assert mixed["normal_box_count"] == "1"
    assert mixed["sensor_image_id_offset"] == "600"
    assert mixed["use_for_sensor"] == "pending_label_policy"
    assert "defect_detected" not in mixed
    assert "defect_detected" not in manifest["label_policy"]["sensor_fields_checked"]
    assert (dataset / mixed["annotation_path"]).is_file()
    assert not Path(mixed["annotation_path"]).is_absolute()
    assert rows["SD_DEF_CC_ACME_20240101_007"]["decision_reason"] == "other_defect_class"

    detection = artifact(tmp_path, state, "convert") / "detection"
    yolo = detection / "yolo"
    config = yaml.safe_load((yolo / "data.yaml").read_text(encoding="utf-8"))
    assert Path(config["path"]) == yolo
    assert config["train"] == "images/train" and config["val"] == "images/val"
    assert list(config["names"]) == list(range(5))
    assert (yolo / "labels" / "train" / f"{TRAIN_NORMAL}.txt").read_text() == ""
    for split, yolo_ids, coco_ids in (("train", [0, 1, 2], [1, 2, 3]), ("val", [3, 4], [4, 5])):
        coco = read_json(detection / "coco" / "annotations" / f"instances_{split}.json")
        labels = list((yolo / "labels" / split).glob("*.txt"))
        assert sorted(int(line.split()[0]) for file in labels for line in file.read_text().splitlines()) == yolo_ids
        assert sorted(box["category_id"] for box in coco["annotations"]) == coco_ids
    conversion = read_json(detection / "conversion_summary.json")
    assert conversion["class_mapping_yolo_zero_based"] == {"23": 0, "24": 1, "25": 2, "27": 3, "30": 4}
    assert conversion["class_mapping_coco_one_based"] == {"23": 1, "24": 2, "25": 3, "27": 4, "30": 5}
    assert conversion["image_transfer"] == {"copied": 12}
    assert not os.path.samefile(yolo / "images" / "train" / f"{TRAIN_MIXED}.JPG", dataset / mixed["image_path"])
    validation = read_json(artifact(tmp_path, state, "validate_detection") / "summary.json")
    assert validation["issue_count"] == 0 and validation["passed"]
    assert (validation["counts"]["objects_train"], validation["counts"]["objects_val"]) == (3, 2)
    assert fingerprint_path(dataset) == before

    repeat = client.invoke(cli.app, command)
    assert repeat.exit_code == 0, repeat.output
    assert all(f"{stage} reused" in repeat.stdout for stage in STAGES)
    assert runner(tmp_path, dataset).status("fixture").stages == state.stages


def test_cache_invalidates_corrupt_exports_and_changed_source(tmp_path: Path, dataset: Path) -> None:
    pipeline = runner(tmp_path, dataset)
    original = pipeline.run("cache", STAGES)
    label = artifact(tmp_path, original, "convert") / "detection" / "yolo" / "labels" / "train" / f"{TRAIN_MIXED}.txt"
    label.write_text("corrupt label", encoding="utf-8")
    repaired = pipeline.resume("cache")
    assert repaired.stages["audit"] == original.stages["audit"]
    assert repaired.stages["manifest"] == original.stages["manifest"]
    for stage in ("convert", "validate_detection"):
        assert repaired.stages[stage].attempt != original.stages[stage].attempt
    assert label.read_text() == "corrupt label"

    changed_annotation = next(dataset.rglob(f"{TRAIN_MIXED}.json"))
    changed_annotation.write_text(changed_annotation.read_text() + "\n", encoding="utf-8")
    refreshed = pipeline.resume("cache")
    assert all(refreshed.stages[stage].attempt != repaired.stages[stage].attempt for stage in STAGES)


def test_cli_from_existing_ingestion_through_validation(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    selection = load_selection()
    fixture_selection = replace(
        selection, folders={name: suffixes for name, suffixes in selection.folders.items() if (dataset / name).is_dir()},
    )
    monkeypatch.setattr(cli, "load_selection", lambda: fixture_selection)
    monkeypatch.setenv("ICT_DATASET_SOURCE", "existing")
    monkeypatch.setenv("ICT_DATA_PATH", str(dataset))
    monkeypatch.chdir(tmp_path)
    client = CliRunner()
    result = client.invoke(cli.app, ["run", "--until", "validate_detection", "--run-id", "ingestion"])
    assert result.exit_code == 0, result.output
    state = runner(tmp_path, dataset).status("ingestion")
    extracted = artifact(tmp_path, state, "extract") / "dataset"
    manifest = read_json(artifact(tmp_path, state, "manifest") / "manifest_summary.json")
    assert manifest["source_root"] == str(extracted)
    validation = read_json(artifact(tmp_path, state, "validate_detection") / "summary.json")
    assert validation["passed"]


def test_failed_validation_keeps_reports_and_retries(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert = adapters.convert

    def corrupt_export(context: StageContext) -> StageResult:
        result = convert(context)
        label = context.output_dir / "detection/yolo/labels/train" / f"{TRAIN_MIXED}.txt"
        label.write_text("corrupt label", encoding="utf-8")
        return result

    monkeypatch.setattr(adapters, "convert", corrupt_export)
    with pytest.raises(PipelineError, match="Detection validation found"):
        runner(tmp_path, dataset).run("failure", STAGES)
    state = runner(tmp_path, dataset).status("failure")
    assert state.status == "failed"
    record = state.stages["validate_detection"]
    work = RunStore(tmp_path / "runs", "failure").working_dir("validate_detection", record.attempt)
    assert read_json(work / "summary.json")["issue_count"] > 0
    assert not artifact(tmp_path, state, "validate_detection").exists()

    monkeypatch.setattr(adapters, "convert", convert)
    pipeline = runner(tmp_path, dataset)
    repaired = pipeline.run("failure", ("convert", "validate_detection"), force=True)
    assert repaired.status == "completed"
    assert repaired.stages["manifest"] == state.stages["manifest"]


def test_resume_reuses_completed_preprocessing_after_interruption(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    convert = adapters.convert

    def interrupt(context: StageContext) -> StageResult:
        convert(context)
        raise KeyboardInterrupt

    monkeypatch.setattr(adapters, "convert", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner(tmp_path, dataset).run("interrupted", STAGES)
    state = runner(tmp_path, dataset).status("interrupted")
    assert state.stages["convert"].status == "interrupted"
    assert not artifact(tmp_path, state, "convert").exists()

    monkeypatch.setattr(adapters, "convert", convert)
    resumed = runner(tmp_path, dataset).resume("interrupted")
    assert resumed.status == "completed"
    for stage in ("audit", "manifest"):
        assert resumed.stages[stage] == state.stages[stage]
    assert resumed.stages["convert"].attempt != state.stages["convert"].attempt


def test_legacy_wrappers_on_tiny_fixture(tmp_path: Path, dataset: Path) -> None:
    def invoke(script: str, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(SCRIPTS / f"{script}.py"), *args], cwd=tmp_path,
            capture_output=True, text=True, encoding="utf-8", timeout=30,
        )

    before = fingerprint_path(dataset)
    commands = [
        ("validate_dataset", "--out", "outputs/data_audit_hash", "--deep", "--hash-images"),
        ("build_manifest",),
        ("convert_detection", "--dry-run", "--limit", "1"),
        ("convert_detection", "--image-mode", "copy"),
        ("validate_detection",),
    ]
    for command in commands:
        result = invoke(*command)
        assert result.returncode == 0, result.stdout + result.stderr
    assert read_json(tmp_path / "outputs/detection_validation/summary.json")["passed"]
    manifest = read_json(tmp_path / "data/processed/manifest_summary.json")
    assert manifest["path_base"] == "repository_root"
    assert all(row["annotation_path"].startswith("data/extracted/") for row in read_csv(tmp_path / "data/processed/manifest.csv"))
    protected = invoke("validate_dataset", "--out", str(dataset / "reports"))
    assert protected.returncode == 2
    assert not (dataset / "reports").exists()
    existing = invoke("build_manifest")
    assert existing.returncode == 2 and "--overwrite" in existing.stderr
    assert invoke("build_manifest", "--overwrite").returncode == 0
    assert fingerprint_path(dataset) == before


def test_opt_in_real_data_regression(tmp_path: Path) -> None:
    data_path = os.environ.get("ICT_REAL_DATA_ROOT")
    if not data_path:
        pytest.skip("Set ICT_REAL_DATA_ROOT to opt in to the existing AI Hub 71900 dataset regression.")
    path = Path(data_path).resolve()
    assert path.is_dir(), "ICT_REAL_DATA_ROOT must point to the extracted dataset directory."
    state = runner(tmp_path, path).run("real-data", STAGES)
    audit = read_json(artifact(tmp_path, state, "audit") / "summary.json")
    manifest = read_json(artifact(tmp_path, state, "manifest") / "manifest_summary.json")
    conversion = read_json(artifact(tmp_path, state, "convert") / "detection/conversion_summary.json")
    validation = read_json(artifact(tmp_path, state, "validate_detection") / "summary.json")
    assert audit["total_samples"] == 15_372
    assert manifest["by_decision"] == {"candidate": 14_718, "hold_review": 645, "exclude": 9}
    assert manifest["by_split_decision"]["train/candidate"] == 13_062
    assert manifest["by_split_decision"]["validation/candidate"] == 1_656
    assert conversion["selected_manifest_rows"] == 14_718
    assert sum(count for key, count in conversion["counts"].items() if key.startswith("objects_")) == 16_877
    assert validation["issue_count"] == 0 and validation["passed"]
