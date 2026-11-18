"""Exercise the local CLI journey with real files and mocked transport and ML."""

import json
import os
import shutil
import tarfile
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest
from typer.testing import CliRunner

from ict_pipeline import cli
from ict_pipeline.config.settings import CREDENTIAL_LABELS
from ict_pipeline.data import ingestion, providers
from ict_pipeline.data.model import Selection, Split
from ict_pipeline.data.preprocessing import stages
from ict_pipeline.data.providers import MockProvider, RetryableDownload
from ict_pipeline.environment import doctor
from ict_pipeline.environment.model import Graphics, Resources, Torch
from ict_pipeline.evaluation import runner as evaluation
from ict_pipeline.evaluation.contracts import EvaluationConfig, Prediction, Preprocessing
from ict_pipeline.models import training
from ict_pipeline.models.contracts import Checkpoints, ModelName, TrainConfig, TrainingRequest
from ict_pipeline.models.dataset import Sample
from ict_pipeline.models.registry import Backend, REGISTRY, resolve_device
from ict_pipeline.pipeline.model import StageContext, StageFailure, StageResult
from ict_pipeline.pipeline.store import RunState, RunStore, fingerprint_path

from test_preprocessing import sample
from test_training import config_directory, fake_torch, optional_runtime

pytestmark = pytest.mark.e2e
SECRET = "fixture-key-78e2c47c"
SHADOWED = "fixture-dotenv-key-aa72a"
TRAIN = "SD_DEF_CC_ACME_20240101_001"
VAL = "SD_DEF_CC_ACME_20240102_002"
NORMAL = "SD_NOR_CC_ACME_20240102_003"


def command(*args: str, code: int = 0, contains: str = "") -> str:
    result = CliRunner().invoke(cli.app, list(args), input="")
    assert result.exit_code == code, result.output
    assert contains in result.output
    assert all(secret not in result.output for secret in (SECRET, SHADOWED))
    return result.stdout


@dataclass(frozen=True)
class Fixture:
    root: Path
    source: Path
    selection: Selection
    archives: Path
    provider: MockProvider

    def state(self, run_id: str) -> RunState:
        state = RunStore(self.root / "runs", run_id).read()
        assert state is not None
        return state

    def artifact(self, run_id: str, stage: str) -> Path:
        store = RunStore(self.root / "runs", run_id)
        return store.artifact_dir(stage, self.state(run_id).stages[stage].attempt)

    def assert_redacted(self) -> None:
        for folder in ("configs", "runs", "outputs", "data/raw"):
            for path in (self.root / folder).rglob("*"):
                data = path.read_bytes() if path.is_file() else b""
                for secret in (SECRET, SHADOWED):
                    assert secret not in path.as_posix()
                    assert secret.encode() not in data, path


@pytest.fixture
def journey(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    monkeypatch.chdir(tmp_path)
    for key in tuple(os.environ):
        if key.startswith(("ICT_", "MLFLOW_TRACKING_")) or key in CREDENTIAL_LABELS:
            monkeypatch.delenv(key)
    monkeypatch.setenv("AIHUB_API_KEY", SECRET)
    (tmp_path / ".env").write_text(f"AIHUB_API_KEY={SHADOWED}\n", encoding="utf-8")
    source = tmp_path / "fixture-samples"
    sample(source, "train", TRAIN, "cold_solder", (23,), color=16)
    sample(source, "validation", VAL, "cold_solder", (23,), color=32)
    sample(source, "validation", NORMAL, "cold_solder", (7,), color=48)
    archives = tmp_path / "fixture-zips"
    archives.mkdir()
    folders = {}
    for folder in sorted(source.iterdir()):
        with ZipFile(archives / f"{folder.name}.zip", "w") as archive:
            for item in sorted(folder.iterdir()):
                archive.write(item, item.name)
                folders[folder.name] = (item.suffix.lower(),)
    splits = (Split("train", (123,), 65536), Split("validation", (456,), 65536))
    selection = Selection(71900, splits, folders, "a" * 64, Path("data/raw"), Path("data/extracted"))
    bundles = {}
    for split in splits:
        bundles[split.name] = tmp_path / f"{split.name}.tar"
        with tarfile.open(bundles[split.name], "w") as bundle:
            for archive in sorted(archives.glob(f"{split.name}_*.zip")):
                bundle.add(archive, arcname=archive.name)
    monkeypatch.setattr(cli, "load_selection", lambda: selection)
    return Fixture(tmp_path, source, selection, archives, MockProvider(bundles))


class FixtureAdapter:
    architecture = "fixture"

    def train(self, request: TrainingRequest) -> Checkpoints:
        assert request.device == "cpu"
        assert (request.config.epochs, request.config.batch_size, request.config.image_size) == (1, 1, 64)
        request.progress(f"fixture training {SECRET}", 1, 1)
        checkpoints = training.checkpoint_paths(ModelName.FASTER_RCNN, request.output)
        for path in (checkpoints.best, checkpoints.last):
            path.write_bytes(b"fixture-only model weights")
        return checkpoints


class FixturePredictor:
    device = "cpu"
    hardware: dict[str, object] = {"device": "cpu", "cpu": "fixture CPU"}
    postprocessing: dict[str, object] = {}
    libraries = {"torch": "fixture"}

    def synchronize(self) -> None:
        pass

    def predict(self, sample: Sample) -> Prediction:
        preprocessing = Preprocessing((sample.width, sample.height), (64, 64), "aspect_ratio")
        if sample.image.stem == VAL:
            return Prediction(VAL, ((2, 3, 10, 9),), (1,), (0.9,), preprocessing)
        assert sample.image.stem == NORMAL
        return Prediction(NORMAL, (), (), (), preprocessing)


def predictor(
    model: ModelName, checkpoint: Path, training_config: TrainConfig, config: EvaluationConfig, scratch: Path,
) -> FixturePredictor:
    assert resolve_device(fake_torch(), training_config.model_copy(update={"device": config.device}),
                          REGISTRY[model]) == "cpu"
    assert checkpoint.read_bytes() == b"fixture-only model weights"
    return FixturePredictor()


def test_fresh_cli_journey(journey: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "provider_for", lambda *_: journey.provider)
    monkeypatch.setattr(training, "import_module", lambda _: fake_torch())
    monkeypatch.setattr(Backend, "versions", lambda _: {"torch": "fixture"})
    monkeypatch.setattr(Backend, "load", lambda _: FixtureAdapter())
    monkeypatch.setattr(evaluation, "evaluate_runs", partial(evaluation.evaluate_runs, factory=predictor))
    monkeypatch.setattr(doctor, "probe_torch", lambda: Torch("unavailable", "not_installed"))
    monkeypatch.setattr(doctor, "probe_hardware", lambda *_: (
        Resources("Fixture CPU", 2, None, None), Graphics("unknown", (), "unavailable", None),
    ))
    command("--version", contains="ict ")
    diagnosed = json.loads(command("doctor", "--json"))
    assert diagnosed["execution_modes"]["local_cpu"] == "available"
    assert diagnosed["torch"]["detail"] == "not_installed"
    command("setup", code=2, contains="--non-interactive")
    setup = ("setup", "--non-interactive", "--dataset-source", "aihub", "--data-path", "data/extracted")
    command(*setup, "--dry-run", contains="Dry run complete")
    assert not (journey.root / "configs").exists()
    command(*setup, contains="Saved configs/local.yaml")
    command(*setup, contains="Kept existing configs/local.yaml")

    command("run", "--dry-run", contains="validate_detection   run")
    assert not (journey.root / "runs").exists()
    command("run", "--run-id", "prep", code=1, contains="approve transfer")
    assert not journey.provider.requests
    command("resume", "--run-id", "prep", "--approve-download", contains="Detection dataset")
    assert journey.provider.requests == [(71900, "train"), (71900, "validation")]
    command("status", "--run-id", "prep", contains="validate_detection   completed")
    state = journey.state("prep")
    raw_hash = fingerprint_path(journey.root / "data/raw")
    resumed = command("resume", "--run-id", "prep", contains="convert reused")
    assert journey.state("prep").stages == state.stages
    dataset = Path(next(line.removeprefix("Detection dataset   ") for line in resumed.splitlines()
                        if line.startswith("Detection dataset   ")))
    dataset_hash = fingerprint_path(dataset)
    for split, expected in (("train", {TRAIN}), ("val", {VAL, NORMAL})):
        assert {path.stem for path in (dataset / "coco/images" / split).iterdir()} == expected
    assert (dataset / "yolo/labels/val" / f"{NORMAL}.txt").read_bytes() == b""

    profiles = journey.root / "profiles"
    profiles.mkdir()
    profile = profiles / "faster_rcnn.yaml"
    train = ("train", "--model", "faster_rcnn", "--dataset", str(dataset), "--config-dir", str(profiles))
    profile.write_text("epochs: 1\nbatch_size: 1\nimage_size: 64\ndevice: cuda\n", encoding="utf-8")
    command(*train, code=1, contains="requires explicit approval")
    command(*train, "--approve-train", code=1, contains="CUDA is unavailable")
    profile.write_text("epochs: 1\nbatch_size: 1\nimage_size: 64\ndevice: mps\n", encoding="utf-8")
    command(*train, "--approve-train", code=1, contains="does not support the requested device")
    profile.write_text("epochs: 1\nbatch_size: 1\nimage_size: 64\ndevice: cpu\n", encoding="utf-8")
    trained = command(*train, "--approve-train", contains="fixture training [REDACTED]")
    model_dir = Path(trained.splitlines()[-1].removeprefix("Model run   "))

    evaluate = ("evaluate", "--run", str(model_dir), "--dataset", str(dataset),
                "--device", "cpu", "--image-size", "64", "--warmup", "0")
    command(*evaluate, "--image-size", "33", code=2, contains="Invalid evaluation options")
    (model_dir / "last.pt").rename(model_dir / "retained-last.pt")
    command(*evaluate, "--checkpoint", "last", code=1, contains="last checkpoint is missing")
    one = Path(command(*evaluate).strip())
    two = Path(command(*evaluate).strip())
    first, second = (evaluation.read_report(path) for path in (one, two))
    assert first["metrics"] == second["metrics"]
    assert first["metrics"]["map50_95"] == 1.0
    assert first["dataset"]["role"] == "validation_for_tuning"
    assert first["dataset"]["image_count"] == 2
    assert command(*evaluate, "--resume-from", str(one.parent)).strip() == str(one)
    catalog = Path(command("report", "--format", "json").strip())
    report = json.loads(catalog.read_text(encoding="utf-8"))
    assert {row["status"] for row in report["runs"]} == {"failed", "completed"}
    assert len(report["comparison_groups"]) == 1
    for fmt in ("html", "csv"):
        assert Path(command("report", "--format", fmt).strip()).is_file()
    assert fingerprint_path(journey.root / "data/raw") == raw_hash
    assert fingerprint_path(dataset) == dataset_hash
    assert (journey.root / ".env").read_text(encoding="utf-8") == f"AIHUB_API_KEY={SHADOWED}\n"
    journey.assert_redacted()


def test_reuse_raw_interrupt_resume_and_invalidate(journey: Fixture, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = shutil.copytree(journey.archives, journey.root / journey.selection.raw_root)
    raw_hash = fingerprint_path(raw)
    monkeypatch.delenv("AIHUB_API_KEY")
    (journey.root / ".env").write_text("", encoding="utf-8")
    # An explicitly selected profile can reuse archives without a key.
    profile = journey.root / "offline.yaml"
    profile.write_text("dataset_source: aihub\ndata_path: data/extracted\n", encoding="utf-8")
    convert = stages.convert

    def cancel(context: StageContext) -> StageResult:
        (context.output_dir / "partial.txt").write_text("interrupted fixture", encoding="utf-8")
        raise KeyboardInterrupt

    monkeypatch.setattr(stages, "convert", cancel)
    run = ("--run-id", "offline", "--config", str(profile))
    command("run", *run, code=130, contains="Run cancelled")
    command("status", "--run-id", "offline", contains="convert   interrupted")
    interrupted = journey.state("offline")
    store = RunStore(journey.root / "runs", "offline")
    attempt = interrupted.stages["convert"].attempt
    assert not store.artifact_dir("convert", attempt).exists()
    assert (store.working_dir("convert", attempt) / "partial.txt").is_file()
    monkeypatch.setattr(stages, "convert", convert)
    command("resume", *run, contains="manifest reused")
    first = journey.state("offline")
    for name in ("download", "extract", "audit", "manifest"):
        assert interrupted.stages[name] == first.stages[name]
    assert first.stages["convert"].attempt != attempt

    profile.write_text("dataset_source: aihub\ndata_path: new-extraction-path\n", encoding="utf-8")
    command("resume", *run, contains="Detection dataset")
    second = journey.state("offline")
    assert all(row.attempt != first.stages[name].attempt for name, row in second.stages.items())
    assert second.stages["download"].config_fingerprint != first.stages["download"].config_fingerprint
    command("resume", *run, contains="validate_detection reused")
    assert journey.state("offline").stages == second.stages
    assert fingerprint_path(raw) == raw_hash
    journey.assert_redacted()


@pytest.mark.parametrize(("failure", "message"), [
    ("invalid-key", "rejected the credential"),
    ("network", "Check network connectivity"),
    ("storage", "Insufficient disk space"),
    ("archive", "missing or incomplete"),
    ("adapter", "[REDACTED]"),
])
def test_failures_are_actionable_and_unpublished(
    journey: Fixture, monkeypatch: pytest.MonkeyPatch, failure: str, message: str,
) -> None:
    requests: list[bool] = []

    @contextmanager
    def response(
        command: tuple[str, ...], url: str, header: bytes | None, timeout: int,
    ) -> Iterator[tuple[None, None, int, dict[str, str]]]:
        requests.append(header is not None)
        if failure == "network":
            raise RetryableDownload("AI Hub returned no usable HTTP response. Check network connectivity.")
        if header is not None:
            assert header == f"apikey: {SECRET}\n".encode()
        yield None, None, 200 if header is None else 403, {}

    monkeypatch.setattr(providers, "_curl_command", lambda *_: ("fixture-curl",))
    monkeypatch.setattr(providers, "_response", response)
    if failure == "storage":
        monkeypatch.setattr(ingestion.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    if failure == "archive":
        raw = shutil.copytree(journey.archives, journey.root / journey.selection.raw_root)
        next(raw.glob("train_*.zip")).write_bytes(b"corrupted fixture ZIP")
    if failure == "adapter":
        monkeypatch.setattr(cli, "provider_for", lambda *_: journey.provider)

        def fail(context: StageContext) -> StageResult:
            raise StageFailure(f"Fixture adapter failed with {SECRET} and {SHADOWED}.")

        monkeypatch.setattr(stages, "audit", fail)

    command("setup", "--non-interactive", "--dataset-source", "aihub", "--data-path", "data/extracted")
    approval = () if failure == "archive" else ("--approve-download",)
    command("run", "--run-id", "failure", *approval, code=1, contains=message)
    stage = "audit" if failure == "adapter" else "download"
    command("status", "--run-id", "failure", contains=f"{stage}   failed")
    assert journey.state("failure").status == "failed"
    assert not (journey.root / "runs/failure/artifacts" / stage).exists()
    assert requests == {"network": [False], "invalid-key": [False, True]}.get(failure, [])
    command("report", "--format", "json")
    journey.assert_redacted()


def test_corrupt_annotations_preserve_audit_and_can_be_retried(journey: Fixture) -> None:
    extracted = shutil.copytree(journey.source, journey.root / journey.selection.extracted_root)
    for path in extracted.rglob("*.json"):
        path.write_text("{", encoding="utf-8")
    bad_hash = fingerprint_path(extracted)
    command("setup", "--non-interactive", "--dataset-source", "skip")
    command("run", "--run-id", "broken", code=1, contains="Stage convert")
    command("status", "--run-id", "broken", contains="convert   failed")
    state = journey.state("broken")
    store = RunStore(journey.root / "runs", "broken")
    audit = store.artifact_dir("audit", state.stages["audit"].attempt)
    assert "json_parse_error" in (audit / "issues.csv").read_text(encoding="utf-8")
    assert fingerprint_path(extracted) == bad_hash

    command("setup", "--non-interactive", "--dataset-source", "skip", "--data-path", str(journey.source), "--replace")
    command("resume", "--run-id", "broken", contains="Detection dataset")
    assert journey.state("broken").stages["audit"].attempt != state.stages["audit"].attempt
    assert fingerprint_path(extracted) == bad_hash
    journey.assert_redacted()


@pytest.fixture
def staged_detectors(monkeypatch: pytest.MonkeyPatch) -> tuple[Counter[ModelName], set[ModelName]]:
    calls: Counter[ModelName] = Counter()
    interrupt: set[ModelName] = set()

    @dataclass
    class Adapter:
        name: ModelName
        architecture = "fixture"

        def train(self, request: TrainingRequest) -> Checkpoints:
            calls[self.name] += 1
            assert request.device == "cpu"
            paths = training.checkpoint_paths(self.name, request.output)
            paths.last.parent.mkdir(exist_ok=True)
            start = 1
            if request.resume:
                start = int(request.resume.last.read_text()) + 1
            for epoch in range(start, request.config.epochs + 1):
                for path in (paths.best, paths.last):
                    path.write_text(str(epoch))
                request.record_metrics({"epoch": epoch})
                if self.name in interrupt:
                    interrupt.remove(self.name)
                    raise KeyboardInterrupt
            return paths

    monkeypatch.setattr(training, "import_module", lambda _: fake_torch())
    monkeypatch.setattr(Backend, "versions", lambda _: {"torch": "fixture"})
    monkeypatch.setattr(Backend, "load", lambda backend: Adapter(next(
        name for name, entry in REGISTRY.items() if entry == backend
    )))
    monkeypatch.setattr(evaluation, "load_predictor", lambda *_: FixturePredictor())
    return calls, interrupt


def test_detection_dag_all_models_resume_and_invalidate(
    journey: Fixture, staged_detectors: tuple[Counter[ModelName], set[ModelName]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, interrupt = staged_detectors
    monkeypatch.setattr(cli, "provider_for", lambda *_: journey.provider)
    command("setup", "--non-interactive", "--dataset-source", "aihub", "--data-path", "data/extracted")
    config = config_directory(journey.root, epochs=2)
    run = ("run", "--run-id", "full", "--all", "--config-dir", str(config), "--evaluation-image-size", "64")
    resume = ("resume", "--run-id", "full")
    approve = "--approve-detection-train"
    source_hash = fingerprint_path(journey.source)
    preview = command(*run, "--dry-run", contains="report   run")
    assert all(f"Detector {name}" in preview for name in REGISTRY)
    assert not (journey.root / "runs").exists()

    command(*run, "--approve-download", "--approve-train", code=1, contains="requires explicit approval")
    assert not calls
    assert journey.state("full").selection == [
        "download", "extract", "audit", "manifest", "convert", "validate_detection",
        "train", "validate_models", "evaluate", "report",
    ]
    initial = journey.state("full")
    dataset = journey.artifact("full", "convert") / "detection"
    conversion_hash = fingerprint_path(dataset)
    raw_hash = fingerprint_path(journey.root / "data/raw")
    interrupt.add(ModelName.FASTER_RCNN)
    command(*resume, approve, code=130, contains="Run cancelled")
    assert calls == {ModelName.YOLO: 1, ModelName.FASTER_RCNN: 1}
    stopped = journey.state("full")
    assert stopped.stages["train"].status == "interrupted"
    assert not (journey.root / "runs/full/artifacts/train").exists()
    command(*resume, code=1, contains="Training faster_rcnn requires explicit approval")
    command(*resume, approve, contains="Detection report")
    assert calls == {ModelName.YOLO: 1, ModelName.FASTER_RCNN: 2, ModelName.RT_DETR: 1}
    first = journey.state("full")
    for stage in initial.selection[:6]:
        assert initial.stages[stage] == first.stages[stage]
    assert all(record.status == "completed" for record in first.stages.values())
    assert journey.provider.requests == [(71900, "train"), (71900, "validation")]

    checkpoints = journey.artifact("full", "train") / "checkpoints"
    predictions = journey.artifact("full", "validate_models") / "predictions"
    metrics_dir = journey.artifact("full", "evaluate") / "model_reports"
    reports = sorted(metrics_dir.glob("*.json"))
    comparison = evaluation.compare_reports(journey.root, tuple(reports))
    for name, report in zip(REGISTRY, comparison["models"], strict=True):
        metadata = json.loads((checkpoints / name / "run.json").read_text(encoding="utf-8"))
        if name == ModelName.FASTER_RCNN:
            assert metadata["initialization"] == "resume"
            assert metadata["metrics"]["epoch"] == 2
        assert metadata["dataset_revision"]["detection"] == conversion_hash
        assert metadata["dataset_revision"]["annotations"]["coco"]["val"]
        for record in metadata["checkpoints"].values():
            assert fingerprint_path(checkpoints / name / record["path"]) == record["fingerprint"]
        prediction_file = predictions / f"{name}.json"
        predicted = evaluation.read_result(prediction_file, "Prediction artifact")
        assert set(predicted["predictions"]) == {VAL, NORMAL}
        assert predicted["predictions"][NORMAL]["boxes"] == []
        assert predicted["checkpoint"]["fingerprint"] == metadata["checkpoints"]["best"]["fingerprint"]
        assert predicted["dataset"] == report["dataset"]
        assert report["prediction_fingerprint"] == fingerprint_path(prediction_file)
        assert report["metrics"]["map50_95"] == 1.0
    report_dir = journey.artifact("full", "report")
    document = json.loads((report_dir / "report.json").read_text())
    assert len(document["comparison_groups"]) == 1
    assert [row["model"] for row in document["comparison_groups"][0]["models"]] == list(REGISTRY)
    assert "ICT run comparison" in (report_dir / "report.html").read_text(encoding="utf-8")
    command("compare", *(str(path) for path in reports), contains="matching_speed_environment")
    command(*resume, contains="report reused")
    assert journey.state("full").stages == first.stages

    # Only the modified model profile needs training, even for a byte-only YAML edit.
    old_artifacts = fingerprint_path(journey.root / "runs/full/artifacts")
    with (config / "rt_detr.yaml").open("a", encoding="utf-8") as stream:
        stream.write("# Profile revision\n")
    command(*resume, code=1, contains="Training rt_detr requires explicit approval")
    assert fingerprint_path(journey.root / "runs/full/artifacts") == old_artifacts
    command(*resume, approve, contains="report completed")
    assert calls == {ModelName.YOLO: 1, ModelName.FASTER_RCNN: 2, ModelName.RT_DETR: 2}
    second = journey.state("full")
    for stage in initial.selection[:6]:
        assert second.stages[stage] == first.stages[stage]
    assert second.stages["train"].attempt != first.stages["train"].attempt

    # A deliberately altered fixture artifact triggers cache repair.
    path = journey.artifact("full", "validate_models") / "predictions/yolo.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError, match="incomplete, corrupt"):
        evaluation.read_result(path, "Prediction artifact")
    command(*resume, contains="train reused")
    third = journey.state("full")
    assert second.stages["train"] == third.stages["train"]
    assert second.stages["validate_models"].attempt != third.stages["validate_models"].attempt
    assert calls == {ModelName.YOLO: 1, ModelName.FASTER_RCNN: 2, ModelName.RT_DETR: 2}
    assert fingerprint_path(journey.source) == source_hash
    assert fingerprint_path(dataset) == conversion_hash
    assert fingerprint_path(journey.root / "data/raw") == raw_hash
    journey.assert_redacted()


def test_detection_dag_fresh_full_run_and_force_recovery(
    journey: Fixture, staged_detectors: tuple[Counter[ModelName], set[ModelName]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, interrupt = staged_detectors
    monkeypatch.setattr(cli, "provider_for", lambda *_: journey.provider)
    command("setup", "--non-interactive", "--dataset-source", "aihub", "--data-path", "data/extracted")
    config = config_directory(journey.root)
    options = ("--run-id", "fresh", "--model", "yolo", "--config-dir", str(config), "--evaluation-image-size", "64")
    approve = "--approve-detection-train"
    command("run", *options, "--approve-download", approve, contains="Detection report")
    first = journey.state("fresh")
    assert len(first.stages) == 10 and first.status == "completed"
    assert calls == {ModelName.YOLO: 1}

    def unexpected_inference(*args: object) -> None:
        raise AssertionError("Scoring should read the published predictions.")

    monkeypatch.setattr(evaluation, "load_predictor", unexpected_inference)
    command("run", *options, "--only", "evaluate", "--force", contains="Detection metrics")
    forced = journey.state("fresh")
    assert forced.stages["train"] == first.stages["train"]
    assert forced.stages["validate_models"] == first.stages["validate_models"]
    before = RunStore(journey.root / "runs", "fresh").artifact_dir("evaluate", first.stages["evaluate"].attempt)
    after = journey.artifact("fresh", "evaluate")
    assert evaluation.read_report(before / "model_reports/01-yolo.json")["metrics"] == evaluation.read_report(
        after / "model_reports/01-yolo.json"
    )["metrics"]

    # Force opens a fresh generation. A fully checkpointed final epoch can be
    # finalized on the next approved attempt without running another epoch.
    command("run", *options, "--only", "train", "--force", code=1, contains="requires explicit approval")
    assert calls == {ModelName.YOLO: 1}
    interrupt.add(ModelName.YOLO)
    command("resume", "--run-id", "fresh", approve, code=130, contains="Run cancelled")
    command("resume", "--run-id", "fresh", approve, contains="Detector checkpoints")
    assert calls == {ModelName.YOLO: 2}
    assert journey.state("fresh").stages["train"].attempt != first.stages["train"].attempt


def test_detection_dag_rejects_mismatched_labels_and_bad_predictions(
    journey: Fixture, staged_detectors: tuple[Counter[ModelName], set[ModelName]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls, _ = staged_detectors
    command("setup", "--non-interactive", "--dataset-source", "skip", "--data-path", str(journey.source))
    configuration = config_directory(journey.root)
    args = ("run", "--run-id", "single", "--model", "faster_rcnn", "--config-dir", str(configuration),
            "--evaluation-image-size", "64", "--approve-detection-train")
    convert = stages.convert

    def inconsistent(context: StageContext) -> StageResult:
        result = convert(context)
        label = context.output_dir / f"detection/yolo/labels/val/{VAL}.txt"
        label.write_text(label.read_text().replace("0 ", "1 ", 1))
        return result

    monkeypatch.setattr(stages, "convert", inconsistent)
    command(*args, code=1, contains="Detection validation found")
    assert not calls
    assert "train" not in journey.state("single").stages
    monkeypatch.setattr(stages, "convert", convert)
    monkeypatch.setattr(evaluation, "load_predictor", lambda *_: SimpleNamespace(
        hardware={}, postprocessing={}, libraries={}, synchronize=lambda: None,
        predict=lambda sample: Prediction("wrong", (), (), (), Preprocessing((sample.width, sample.height),
                                                                              (64, 64), "aspect_ratio")),
    ))
    command(*args, "--force", code=1, contains="Prediction rows or preprocessing metadata")
    assert calls == {ModelName.FASTER_RCNN: 1}
    assert not (journey.root / "runs/single/artifacts/validate_models").exists()
    monkeypatch.setattr(evaluation, "load_predictor", lambda *_: FixturePredictor())
    command("resume", "--run-id", "single", contains="Detection report")
    assert calls == {ModelName.FASTER_RCNN: 1}
    report = journey.artifact("single", "evaluate") / "model_reports/01-faster_rcnn.json"
    assert evaluation.read_report(report)["metrics"]["map50_95"] == 1.0
    dataset = journey.artifact("single", "convert") / "detection"
    model = journey.artifact("single", "train") / "checkpoints/faster_rcnn"
    changed = shutil.copytree(dataset, journey.root / "changed-labels")
    annotation = changed / "coco/annotations/instances_val.json"
    data = json.loads(annotation.read_text())
    data["annotations"][0]["category_id"] = 2
    annotation.write_text(json.dumps(data), encoding="utf-8")
    command("evaluate", "--run", str(model), "--dataset", str(changed), code=1, contains="verified detection revision")


@pytest.mark.parametrize("name", tuple(REGISTRY))
def test_detection_dag_actual_cpu_backend(name: ModelName, journey: Fixture) -> None:
    torch = optional_runtime(name)
    sample(journey.source, "train", "SD_NOR_CC_ACME_20240101_004", "cold_solder", (7,), color=64)
    config = config_directory(journey.root)
    command("setup", "--non-interactive", "--dataset-source", "skip", "--data-path", str(journey.source))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        command(
            "run", "--run-id", "cpu", "--model", name, "--config-dir", str(config),
            "--evaluation-image-size", "64", "--approve-detection-train", contains="Detection report",
        )
        model_report, = (journey.artifact("cpu", "evaluate") / "model_reports").glob("*.json")
        report = evaluation.read_report(model_report)
        assert report["model"] == name
        assert report["speed"]["hardware"]["device"] == "cpu"
        assert report["dataset"]["image_count"] == report["speed"]["timed_images"] == 2
        assert 0 <= report["metrics"]["map50_95"] <= 1
        command("resume", "--run-id", "cpu", contains="train reused")
    finally:
        torch.set_num_threads(threads)
