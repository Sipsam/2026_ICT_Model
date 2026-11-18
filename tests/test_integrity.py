"""Fixture and opt-in source regressions for split integrity and detector provenance."""

import json
import os
import sqlite3
from collections import Counter
from contextlib import closing
from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image, ImageDraw
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.data.preprocessing import leakage
from ict_pipeline.data.preprocessing.common import read_csv, write_csv
from ict_pipeline.data.preprocessing.leakage import AnalysisConfig, analyze_leakage, observe
from ict_pipeline.data.preprocessing.policy import CLASS_MAP, SENSOR_FIELDS, label_policy
from ict_pipeline.data.preprocessing.revision import SEAL, read_dataset_seal
from ict_pipeline.data.preprocessing.split import SplitConfig, plan_split
from ict_pipeline.evaluation.contracts import EvaluationConfig, EvaluationError, EvaluationSplit
from ict_pipeline.evaluation.runner import evaluate_runs, read_report
from ict_pipeline.models import training
from ict_pipeline.models.contracts import Checkpoints, ModelName, TrainingError, TrainingRequest
from ict_pipeline.models.dataset import inspect_data, verify_shared_labels
from ict_pipeline.models.registry import REGISTRY
from ict_pipeline.pipeline.store import RunStore, fingerprint_path
from test_evaluation import FakePredictor, write_run


@pytest.fixture
def source(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    root.mkdir()
    rows = []
    for index in range(12):
        split = "train" if index % 2 == 0 else "validation"
        timestamp = f"20250101-1200{index:02}"
        stem = f"SD_DEF_CC_A_{timestamp}_{index:05}"
        image = root / f"{stem}.jpg"
        annotation = root / f"{stem}.json"
        visual = Image.new("RGB", (32, 32), ((index * 21) % 256, index * 5, 48))
        ImageDraw.Draw(visual).rectangle((3, 5, 15, 20), fill="#613d85")
        visual.save(image, quality=80 if index != 3 else 92)
        if index == 1:
            previous = root / rows[0]["image_path"]
            image.write_bytes(previous.read_bytes())
        if index == 3:
            with Image.open(root / rows[2]["image_path"]) as previous:
                previous.save(image, quality=90)
        sensor = [{
            "timestamp": f"2025-01-01T12:00:{index + elapsed:02}",
            **dict(zip(SENSOR_FIELDS, (30 + index // 2, 50, .2, .03, 66), strict=True)),
            "defect_detected": index % 2,
        } for elapsed in (0, 1)]
        document = {
            "images": [{"id": 1, "file_name": image.name, "width": 32, "height": 32}],
            "annotations": [{"image_id": 1, "category_id": category, "bbox": [1, 2, 3, 4]}
                            for category in CLASS_MAP] if index != 11 else [],
            "sensor_data": [{"sensor_sequence": sensor}],
        }
        if index in (4, 5):
            document["production_session_id"] = "session-one"
        if index in (6, 7):
            document["capture_group_id"] = "capture-one"
        annotation.write_text(json.dumps(document), encoding="utf-8")
        decision = "hold_review" if index == 9 else "candidate"
        rows.append({"split": split, "sample_id": stem, "image_path": image.name, "annotation_path": annotation.name,
                     "sensor_path": "", "decision": decision, "use_for_detection": decision,
                     "manufacturer": "A", "target_box_count": "0" if index == 11 else "5"})
    manifest = tmp_path / "manifest.csv"
    write_csv(manifest, rows, list(rows[0]))
    return root, manifest


def test_evidence_and_group_constraints(source: tuple[Path, Path]) -> None:
    root, manifest = source
    report = analyze_leakage(root, manifest, root, AnalysisConfig())
    cross = [group for group in report["groups"] if len(group["by_original_split"]) > 1]
    statuses = {group["method"]: group["status"] for group in cross}
    assert statuses["image_sha256"] == statuses["production_session"] == "confirmed"
    assert statuses["image_similarity"] == statuses["sensor_similarity"] == statuses["manufacturer"] == "potential"
    assert statuses["capture_group"] == "potential"
    assert report["unknown_provenance"]["production_session"] == 10
    plan = plan_split(report, SplitConfig(seed=17, group_near_images=True))
    assert plan == plan_split(report, SplitConfig(seed=17, group_near_images=True))
    assert sum(plan["counts"].values()) == 11
    assert all(not missing for missing in plan["missing_classes"].values())
    assert all(method not in plan["remaining_cross_split_groups"] for method in plan["grouping_methods"])
    assert plan["test_origin"] == "internal_same_source"
    constrained = plan_split(report, SplitConfig(grouping="manufacturer"))
    assert constrained["candidate_components"] == 1
    assert len(constrained["insufficient_class_group_support"]) == 5
    assert 0 in constrained["counts"].values()


def test_sensor_evidence_ignores_target_and_checks_workbook(source: tuple[Path, Path]) -> None:
    root, manifest = source
    row = {**read_csv(manifest)[0], "sensor_path": "sensor.xlsx"}
    annotation = root / row["annotation_path"]
    document = json.loads(annotation.read_text())
    embedded = document["sensor_data"][0]["sensor_sequence"]
    original = observe(root, row, AnalysisConfig()).groups["sensor_exact"]
    book = Workbook()
    book.active.append(["timestamp", *SENSOR_FIELDS, "defect_detected"])
    for reading in embedded:
        book.active.append([reading["timestamp"], *(reading[key] for key in SENSOR_FIELDS), 1 - reading["defect_detected"]])
    book.save(root / row["sensor_path"])
    verified = observe(root, row, AnalysisConfig(verify_xlsx=True))
    assert verified.groups["sensor_exact"] == original
    book.active.cell(row=2, column=2).value += 1
    book.save(root / row["sensor_path"])
    conflicting = observe(root, row, AnalysisConfig(verify_xlsx=True))
    assert "sensor_exact" not in conflicting.groups
    assert "sensor_value_mismatch" in conflicting.issues


def test_analysis_checkpoint_reuses_verified_observations(
    source: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, manifest = source
    config = AnalysisConfig()
    checkpoint = tmp_path / "observations.sqlite"

    def interrupt(done: int, total: int | None) -> None:
        if done == 3:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        analyze_leakage(root, manifest, root, config, interrupt, checkpoint=checkpoint)
    with closing(sqlite3.connect(checkpoint)) as db:
        assert db.execute("SELECT status FROM analysis").fetchone() == ("interrupted",)
        assert db.execute("SELECT COUNT(*) FROM observations").fetchone() == (3,)
    changed = root / leakage.source_rows(manifest, root, root)[0]["image_path"]
    changed.write_bytes(changed.read_bytes() + b"changed")
    expected = analyze_leakage(root, manifest, root, config)
    calls = []

    def track(path: Path, row: dict[str, str], config: AnalysisConfig) -> leakage.Observation:
        calls.append(row["image_path"])
        return observe(path, row, config)

    monkeypatch.setattr(leakage, "observe", track)
    assert analyze_leakage(root, manifest, root, config, checkpoint=checkpoint) == expected
    assert len(calls) == 10 and changed.name in calls
    with pytest.raises(ValueError, match="does not match"):
        analyze_leakage(root, manifest, root, AnalysisConfig(image_hamming=0), checkpoint=checkpoint)


def command_args(source: tuple[Path, Path]) -> list[str]:
    root, manifest = source
    return ["--root", str(root), "--manifest", str(manifest)]


@pytest.fixture
def published(source: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    root, manifest = source
    before = fingerprint_path(root), fingerprint_path(manifest)
    cli = CliRunner()
    args = command_args(source)
    analysis = tmp_path / "analysis"
    report = cli.invoke(app, ["integrity", "analyze", *args, "--out", str(analysis)])
    assert report.exit_code == 0, report.output
    args += ["--analysis", str(analysis / "leakage.json")]
    output = tmp_path / "revision"
    preview = cli.invoke(app, ["integrity", "split", *args, "--dry-run", "--out", str(output)])
    assert preview.exit_code == 0 and not output.exists(), preview.output
    baseline = tmp_path / "baseline"
    result = cli.invoke(app, ["integrity", "split", *args, "--strategy", "aihub", "--publish", "--out", str(baseline)])
    assert result.exit_code == 0, result.output
    assert all(row["split"] == row["original_split"] for row in read_csv(baseline / "manifest.csv"))
    command = ["integrity", "split", *args, "--publish", "--export-detection", "--out", str(output)]
    result = cli.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert cli.invoke(app, command).exit_code == 2
    assert (fingerprint_path(root), fingerprint_path(manifest)) == before
    assert json.loads((output / "validation" / "summary.json").read_text())["passed"]
    return output


def test_revision_export_and_identical_evaluation(published: Path) -> None:
    dataset = published / "detection"
    seal = read_dataset_seal(dataset)
    assert seal is not None
    revision = json.loads((published / "split.json").read_text())
    assert seal["split"]["revision_id"] == revision["revision_id"]
    assert all(row["split"] == row["original_split"] for row in read_csv(published / "manifest.csv")
               if row["decision"] != "candidate")
    credentials = Credentials()
    coco = inspect_data(dataset, "coco", credentials)
    yolo = inspect_data(dataset, "yolo", credentials)
    assert yolo.split_fingerprints == coco.split_fingerprints
    assert set(coco.splits) == {"train", "val"}
    assert "test:" not in (dataset / "yolo" / "data.yaml").read_text()
    runs = []
    for name in ModelName:
        run = write_run(published.parent, name, coco.split_fingerprints)
        path = run / "run.json"
        record = json.loads(path.read_text())
        record["dataset_revision"] = {"selected_split": seal}
        path.write_text(json.dumps(record), encoding="utf-8")
        runs.append(run)

    def evaluate(path: Path) -> list[Path]:
        return list(evaluate_runs(
            published.parent, tuple(runs), path, EvaluationSplit.TEST, EvaluationConfig(warmup=0),
            published.parent / "reports", "best", True, None, None, lambda *_: None,
            lambda name, *_: FakePredictor(name),
        ))

    results = [read_report(path) for path in evaluate(dataset)]
    assert len(results) == 3
    assert all(result["dataset"] == results[0]["dataset"] for result in results)
    assert results[0]["dataset"]["role"] == "internal_group_holdout"
    assert results[0]["dataset"]["external_independence"] is False
    path = runs[0] / "run.json"
    record = json.loads(path.read_text())
    record.pop("dataset_revision")
    path.write_text(json.dumps(record))
    with pytest.raises(EvaluationError, match="must match every checkpoint"):
        evaluate(dataset)
    labels = dataset / "coco" / "annotations" / "instances_test.json"
    labels.write_bytes(labels.read_bytes() + b" ")
    with pytest.raises(ValueError, match="have changed"):
        read_dataset_seal(dataset)
    with pytest.raises(EvaluationError, match="split revision"):
        evaluate(dataset)


@pytest.fixture
def mocked_training(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, str]]:
    seen = []

    class Adapter:
        architecture = "fake"

        def train(self, request: TrainingRequest) -> Checkpoints:
            seen.append(request.data.split_fingerprints)
            paths = training.checkpoint_paths(ModelName.YOLO, request.output)
            paths.best.parent.mkdir()
            paths.best.write_bytes(b"best")
            paths.last.write_bytes(b"last")
            return paths

    backend = type(REGISTRY[ModelName.YOLO])
    monkeypatch.setattr(backend, "versions", lambda *_: {})
    monkeypatch.setattr(backend, "load", lambda *_: Adapter())
    monkeypatch.setattr(training, "resolve_device", lambda *_: "cpu")
    monkeypatch.setattr(training, "import_module", lambda *_: object())
    return seen


def test_training_imbalance_joins_sealed_groups(published: Path) -> None:
    dataset = published / "detection"
    runner = CliRunner()
    args = ["training-data", "imbalance", "--dataset", str(dataset)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["manifest_verified_by_seal"] and report["test_available"]
    assert "test" not in report["splits"]
    train, val = (report["splits"][split] for split in ("train", "val"))
    assert set(train["by_group"]).isdisjoint(val["by_group"])
    assert "A" in train["by_manufacturer"]
    audit = runner.invoke(app, [*args, "--include-test"])
    assert audit.exit_code == 0, audit.output
    test = json.loads(audit.stdout)["splits"]["test"]
    assert set(test["by_group"]).isdisjoint(set(train["by_group"]) | set(val["by_group"]))
    manifest = published / "manifest.csv"
    with manifest.open("a", encoding="utf-8") as stream:
        stream.write("wrong,data\n")
    rejected = runner.invoke(app, args)
    assert rejected.exit_code == 2 and "differs from the selected split" in rejected.stderr


def test_training_preflight_binds_revision_and_rejects_tampering(
    published: Path, mocked_training: list[dict[str, str]],
) -> None:
    dataset = published / "detection"
    plan = training.plan_training(published.parent, (ModelName.YOLO,), dataset, published.parent / "runs", None, None)
    list(training.execute_training(plan, lambda *_: None))
    metadata = json.loads((plan.destination / "yolo" / "run.json").read_text())
    assert metadata["dataset_revision"]["selected_split"] == read_dataset_seal(dataset)
    assert metadata["split_fingerprints"] == mocked_training[0]
    assert metadata["label_policy"] == json.loads(json.dumps(label_policy()))
    (dataset / SEAL).write_text("{}")
    with pytest.raises(TrainingError, match="selected split revision"):
        list(training._execute_training(plan, lambda *_: None))


def test_run_reuses_only_the_verified_bundle(published: Path, mocked_training: list[dict[str, str]]) -> None:
    cli = CliRunner()
    args = ["--run-id", "selected"]
    result = cli.invoke(app, ["run", *args, "--split-bundle", str(published), "--model", "yolo", "--until", "train",
                              "--approve-detection-train"])
    assert result.exit_code == 0, result.output
    store = RunStore(published.parent / "runs", "selected")
    state = store.read()
    assert state is not None and state.selection == ["validate_detection", "train"]
    artifact = store.artifact_dir("train", state.stages["train"].attempt)
    data = json.loads((artifact / "checkpoints" / "yolo" / "run.json").read_text())
    assert data["dataset_revision"]["selected_split"] == read_dataset_seal(published / "detection")
    reused = cli.invoke(app, ["resume", *args])
    assert reused.exit_code == 0 and "train reused" in reused.output, reused.output
    assert len(mocked_training) == 1
    labels = published / "detection" / "coco" / "annotations" / "instances_val.json"
    labels.write_bytes(labels.read_bytes() + b" ")
    denied = cli.invoke(app, ["resume", *args])
    assert denied.exit_code == 1 and "verified split bundle" in denied.output, denied.output
    assert len(mocked_training) == 1


def test_publish_rechecks_inputs_and_refuses_empty_group_plan(
    source: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    root, _ = source
    cli = CliRunner()
    args = command_args(source)
    assert cli.invoke(app, ["integrity", "analyze", *args, "--out", "analysis"]).exit_code == 0
    args += ["--analysis", "analysis/leakage.json", "--publish", "--out", "revision"]
    constrained = cli.invoke(app, ["integrity", "split", *args, "--grouping", "manufacturer"])
    assert constrained.exit_code == 2 and "empty" in constrained.output
    image = next(root.glob("*.jpg"))
    image.write_bytes(image.read_bytes() + b"changed")
    stale = cli.invoke(app, ["integrity", "split", *args])
    assert stale.exit_code == 2 and "Source files changed" in stale.output
    assert not (tmp_path / "revision").exists()


def test_legacy_formats_reject_annotation_mismatch(published: Path) -> None:
    dataset = published / "detection"
    label = next(path for path in (dataset / "yolo" / "labels" / "val").glob("*.txt") if path.stat().st_size)
    label.write_text(label.read_text().replace("0 ", "1 ", 1))
    credentials = Credentials()
    yolo = inspect_data(dataset, "yolo", credentials)
    coco = inspect_data(dataset, "coco", credentials)
    with pytest.raises(TrainingError, match="annotations differ"):
        verify_shared_labels(yolo, coco)


def test_real_integrity_regression() -> None:
    root = os.environ.get("ICT_REAL_DATA_ROOT")
    manifest = os.environ.get("ICT_REAL_DATA_MANIFEST")
    if not root or not manifest:
        pytest.skip("Set ICT_REAL_DATA_ROOT and ICT_REAL_DATA_MANIFEST to opt in.")
    path_base = Path(os.environ.get("ICT_REAL_DATA_PATH_BASE", root))
    report = analyze_leakage(Path(root), Path(manifest), path_base, AnalysisConfig())
    assert report["original_split_counts"] == {"train": 13664, "validation": 1708}
    assert report["candidate_split_counts"] == {"train": 13062, "validation": 1656}
    assert report["unknown_provenance"]["production_session"] == 15372
    cross_methods = Counter(group["method"] for group in report["groups"] if len(group["by_original_split"]) > 1)
    assert cross_methods["manufacturer"] == 4
    assert cross_methods["source_capture"] == 11
    assert cross_methods["image_sha256"] == 0
