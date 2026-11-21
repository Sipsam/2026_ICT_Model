import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.data import detector_finalization as finalization
from ict_pipeline.data.group_retraining import read_group_verification
from ict_pipeline.data.preprocessing.common import read_csv, write_csv, write_json
from ict_pipeline.data.preprocessing import leakage
from ict_pipeline.data.preprocessing.leakage import AnalysisConfig, analyze_leakage
from ict_pipeline.data.preprocessing.policy import label_policy
from ict_pipeline.data.preprocessing.split import SplitConfig, plan_split, publish_split
from ict_pipeline.evaluation.product import write_artifact
from ict_pipeline.evaluation.product_protocol import FrozenValidation
from ict_pipeline.models import training as detector_training
from ict_pipeline.models.contracts import Checkpoints, ModelName, TrainConfig
from ict_pipeline.models.dataset import DetectionData, inspect_data
from ict_pipeline.models.registry import Backend
from ict_pipeline.multimodal.audit import audit_cohort
from ict_pipeline.multimodal.preprocessing import PreprocessingConfig, fit_preprocessor
from ict_pipeline.multimodal.training_plan import ProductTrainingConfig, plan_training as plan_product_training
from ict_pipeline.pipeline.store import digest_json, fingerprint_path

from multimodal_fixtures import make_sample


@pytest.mark.parametrize("finalize", [False, True], ids=["group-verification", "detector-finalization"])
def test_group_commands_keep_test_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, finalize: bool,
) -> None:
    if finalize:
        pytest.importorskip("torch")
        pytest.importorskip("torchvision")
    source = tmp_path / "source"
    manifest = tmp_path / "provider.csv"
    rows = [make_sample(source, index, values=(index * 5, index * 5 + 1, index * 5 + 2))
            for index in range(1, 7)]
    for index, row in enumerate(rows, 1):
        color = tuple(255 if index & bit else 0 for bit in (1, 2, 4))
        Image.new("RGB", (24, 18), color).save(source / row["image_path"])
    write_csv(manifest, rows, list(rows[0]))
    report = analyze_leakage(source, manifest, source, AnalysisConfig(verify_xlsx=True))
    plan = plan_split(report, SplitConfig(val_fraction=0.25, test_fraction=0.25,
                                        group_near_images=True, group_near_sensors=True))
    bundle = tmp_path / "bundle"
    publish_split(source, manifest, source, report, plan, bundle, export_detection=True, allow_missing_classes=True,
                  progress=lambda _: lambda *_: None)
    revision = json.loads((bundle / "split.json").read_bytes())
    audit = audit_cohort(source, bundle / "manifest.csv", revision_path=bundle / "split.json")
    cohort = audit.for_product_training()
    index = tmp_path / "training-cohort.json"
    write_json(index, {"cohort": cohort.model_dump(mode="json"), "fingerprint": cohort.fingerprint})
    frozen = tmp_path / "frozen-validation.json"
    write_artifact(frozen, FrozenValidation.from_audit(cohort, tuple(
        record.sample_id for record in cohort.records if record.split == "validation"
    )))
    state = tmp_path / "sensor-state.json"
    fitted = fit_preprocessor(cohort, PreprocessingConfig())
    write_json(state, fitted.payload())

    reserved = {source / getattr(record, name) for record in cohort.records if record.split == "test"
                for name in ("image_path", "annotation_path", "sensor_path")}
    original = Path.open

    def guard(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
        assert path not in reserved
        if path.is_relative_to(bundle / "detection"):
            assert "test" not in path.parts and path.name != "instances_test.json"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)
    out = tmp_path / "verification"
    args = ["integrity", "verify-group", "--root", str(source), "--bundle", str(bundle),
            "--revision-id", revision["revision_id"], "--cohort", str(index),
            "--cohort-fingerprint", cohort.fingerprint, "--frozen-validation", str(frozen),
            "--preprocessing", str(state)]
    cache = ["--checkpoint", str(tmp_path / "observations.sqlite")]
    result = CliRunner().invoke(app, [*args, *cache, "--out", str(out)])
    assert result.exit_code == 0, (result.output, result.exception)
    receipt = out / "verification.json"
    saved = json.loads(receipt.read_bytes())["verification"]
    assert saved["counts"] == revision["counts"]
    assert saved["test"]["payloads_opened"] is False
    assert saved["test"]["payload_seal_reverified"] is False
    assert saved["training_fingerprint"] == fitted.training_fingerprint

    with monkeypatch.context() as mocks:
        mocks.setattr(leakage, "observe", lambda *_: pytest.fail("Verified observations should be reused"))
        retry = tmp_path / "verification-retry"
        result = CliRunner().invoke(app, [*args, *cache, "--out", str(retry)])
        assert result.exit_code == 0, (result.output, result.exception)
        assert receipt.read_bytes() == (retry / "verification.json").read_bytes()

    dataset = bundle / "detection"
    training = detector_training.plan_training(tmp_path, (ModelName.FASTER_RCNN,), dataset, tmp_path / "runs", None, None,
                             group_verification=receipt)
    assert training.dataset_revision["group_split_revision"] == revision["revision_id"]

    def check_detector_data(_name: ModelName, _config: object, _plan: object,
                            data: DetectionData, *_: object) -> Checkpoints:
        assert set(data.splits) == {"train", "val"}
        for split, role in (("train", "train"), ("val", "validation")):
            assert {sample.image.stem for sample in data.splits[split].samples} == {
                record.sample_id for record in cohort.records if record.split == role
            }
        return Checkpoints(tmp_path / "best.pt", tmp_path / "last.pt")

    with monkeypatch.context() as mocks:
        mocks.setattr(Backend, "versions", lambda _: {})
        mocks.setattr(Backend, "load", lambda _: SimpleNamespace())
        mocks.setattr(detector_training, "import_module", lambda _: SimpleNamespace())
        mocks.setattr(detector_training, "resolve_device", lambda *_: "cpu")
        mocks.setattr(detector_training, "_train_model", check_detector_data)
        assert [model for model, _ in detector_training._execute_training(training, lambda *_: None)] == [
            ModelName.FASTER_RCNN,
        ]

    forged = fitted.model_copy(update={"statistics": (
        fitted.statistics[0].model_copy(update={"center": 2026.0}), *fitted.statistics[1:],
    )})
    forged_path = tmp_path / "forged-state.json"
    write_json(forged_path, forged.payload())
    result = CliRunner().invoke(app, [*args[:-1], str(forged_path), "--out", str(tmp_path / "bad-state")])
    assert result.exit_code != 0 and "fresh fit on group Train" in result.output

    prior = tmp_path / "prior.csv"
    previous = next(row for row in read_csv(bundle / "manifest.csv") if row["split"] == "train")
    write_csv(prior, [{**previous, "split": "test"}], list(previous))
    result = CliRunner().invoke(app, [*args, "--out", str(tmp_path / "custody-conflict"),
                                     "--prior-test-manifest", str(prior)])
    assert result.exit_code != 0 and "Prior Test custody conflict" in result.output

    detector = tmp_path / "old-detector"
    detector.mkdir()
    checkpoint = detector / "best.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    write_json(detector / "run.json", {"status": "completed", "model": "faster_rcnn"})
    templates = Path(__file__).resolve().parents[1] / "docs" / "profiles" / "group-retraining"
    profiles = []
    for name in ("fpn-image-only", "fpn-sensor-concat"):
        raw = yaml.safe_load((templates / f"{name}.yaml").read_text(encoding="utf-8"))
        raw.update(source=str(source), bundle=str(bundle), revision_id=revision["revision_id"],
                   cohort=str(index), cohort_fingerprint=cohort.fingerprint, preprocessing=str(state),
                   group_verification=str(receipt))
        raw["architecture"]["preprocessing_fingerprint"] = fitted.fingerprint
        raw["validation"].update(cohort=str(frozen), fingerprint=saved["frozen_validation_fingerprint"])
        raw["detector_initialization"].update(path=str(checkpoint), sha256="a" * 64)
        profile = ProductTrainingConfig.model_validate_json(json.dumps(raw))
        profiles.append(profile)
        assert profile.epochs == 30 and profile.architecture.image.freeze_backbone
        with pytest.raises(ValueError, match="completed best checkpoint from the verified group Detector"):
            plan_product_training(profile, tmp_path / name, workspace=tmp_path, progress=lambda *_: None)

    if finalize:
        import torch

        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            _check_finalization(tmp_path, bundle, receipt, index, tuple(profiles), monkeypatch)
        finally:
            torch.set_num_threads(threads)

    path = dataset / "coco" / "annotations" / "instances_train.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="inputs changed"):
        read_group_verification(dataset, receipt)


def _check_finalization(
    tmp_path: Path, bundle: Path, receipt: Path, index: Path, profiles: tuple[ProductTrainingConfig, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from ict_pipeline.models.coco import FasterRCNNAdapter
    from ict_pipeline.multimodal.fpn import build_detector
    from ict_pipeline.multimodal.model_config import FPNDetectorConfig

    source_run = tmp_path / "detector" / "original-run" / "faster_rcnn"
    source_run.mkdir(parents=True)
    (source_run / "events.jsonl").write_text('{"stage":"epoch 47/50 validation batch"}\n', encoding="utf-8")
    weights = build_detector(FPNDetectorConfig(image_size=64)).state_dict()
    training = TrainConfig(image_size=64, optimizer="SGD", lr=0.005)
    best_loss, last_loss = 0.3986949825265438, 0.4377073348879239
    state = {"format": finalization.BACKEND.checkpoint_format, "architecture": finalization.ARCHITECTURE,
             "best_epoch": 16, "best_loss": best_loss, "model": weights, "optimizer": {}, "scaler": {}}
    data = inspect_data(bundle / "detection", "coco", Credentials())
    run = {
        "schema_version": 1, "status": "failed", "error": finalization.FAILURE, "model": "faster_rcnn",
        "architecture": finalization.ARCHITECTURE, "checkpoint_format": finalization.BACKEND.checkpoint_format,
        "dataset_format": "coco", "label_policy": label_policy(), "best_criterion": finalization.BACKEND.best_criterion,
        "dataset_revision": read_group_verification(bundle / "detection", receipt),
        "split_fingerprints": data.split_fingerprints, "annotation_fingerprint": data.annotation_fingerprint,
        "split_annotation_fingerprints": data.split_annotation_fingerprints,
        "initialization_mode": "scratch", "configuration": training.model_dump(mode="json"),
        "run_fingerprint": "a" * 64, "checkpoint_epoch": 46,
        "metrics": {"epoch": 46, "val_loss": last_loss, "best_val_loss": best_loss},
    }
    best_state = {**state, "epoch": 16, "val_loss": best_loss}
    last_state = {**state, "epoch": 46, "val_loss": last_loss}

    def write_source(best: dict, last: dict, **changes: object) -> None:
        checkpoints = {}
        for role, payload in (("best", best), ("last", last)):
            path = source_run / f"{role}.pt"
            torch.save(payload, path)
            checkpoints[role] = {"path": path.name, "fingerprint": fingerprint_path(path)}
        (source_run / "run.json").write_text(json.dumps({**run, "checkpoints": checkpoints, **changes}), encoding="utf-8")

    runner = CliRunner()
    args = ["integrity", "finalize-detector", "--source-run", str(source_run), "--bundle", str(bundle),
            "--group-verification", str(receipt), "--cohort", str(index), "--best-epoch", "16", "--completed-epoch", "46"]
    out = tmp_path / "finalized"

    def invoke(*extra: str, command: list[str] = args) -> Any:
        return runner.invoke(app, [*command, "--out", str(out), *extra])

    def reject(message: str, command: list[str] = args) -> None:
        result = invoke(command=command)
        assert result.exit_code == 2 and message in result.output, (result.output, result.exception)
        assert not out.exists()

    for value in (float("nan"), float("inf")):
        key = next(key for key, tensor in weights.items() if tensor.is_floating_point())
        corrupt = {**weights, key: torch.full_like(weights[key], value)}
        write_source({**best_state, "model": corrupt}, last_state)
        reject("must be finite")
    for best, last, message in (
        ({**best_state, "epoch": 15}, last_state, "epoch or loss is invalid"),
        (best_state, {**last_state, "epoch": 45}, "epoch or loss is invalid"),
        (best_state, {**last_state, "best_epoch": 15}, "best epoch or loss disagrees"),
        (best_state, {**last_state, "best_loss": 0.4}, "best epoch or loss disagrees"),
    ):
        write_source(best, last)
        reject(message)
    write_source(best_state, last_state, dataset_revision={**run["dataset_revision"], "group_split_revision": "b" * 64})
    reject("group revision, verification receipt")
    write_source(best_state, last_state, split_fingerprints={"val": "b" * 64})
    reject("Detector dataset fingerprints")
    write_source(best_state, last_state)
    group = json.loads(receipt.read_bytes())["verification"]
    wrong_receipt = tmp_path / "wrong-receipt.json"
    mismatched = {**group, "preprocessing_fingerprint": "b" * 64}
    write_json(wrong_receipt, {"verification": mismatched, "fingerprint": digest_json(mismatched)})
    reject("group revision, verification receipt", [str(wrong_receipt) if arg == str(receipt) else arg for arg in args])
    wrong_cohort = {**group, "cohort_fingerprint": "b" * 64}
    wrong_receipt.write_text(json.dumps({"verification": wrong_cohort, "fingerprint": digest_json(wrong_cohort)}))
    reject("sanitized Cohort", [str(wrong_receipt) if arg == str(receipt) else arg for arg in args])
    for role in ("best", "last"):
        write_source(best_state, last_state)
        with (source_run / f"{role}.pt").open("ab") as stream:
            stream.write(b"tampered")
        reject(f"{role}.pt fingerprint differs")

    write_source(best_state, last_state)
    snapshot = {p.name: (fingerprint_path(p), p.stat().st_mtime_ns) for p in source_run.iterdir()}
    build_model = FasterRCNNAdapter.build_model

    def nonfinite_model(adapter: FasterRCNNAdapter, config: TrainConfig) -> Any:
        model = build_model(adapter, config)
        model.forward = lambda _: [{
            "boxes": torch.tensor([[0., 0., 1., 1.]]), "labels": torch.tensor([1]), "scores": torch.tensor([float("nan")]),
        }]
        return model

    with monkeypatch.context() as mocks:
        mocks.setattr(FasterRCNNAdapter, "build_model", nonfinite_model)
        failed_inference = invoke("--validation-device", "cpu")
        assert failed_inference.exit_code == 2 and "non-finite predictions" in failed_inference.output
        assert not out.exists()
    result = invoke("--validation-device", "cpu")
    assert result.exit_code == 0, (result.output, result.exception)
    assert {p.name: (fingerprint_path(p), p.stat().st_mtime_ns) for p in source_run.iterdir()} == snapshot
    assert {p.name for p in out.iterdir()} == {"best.pt", "finalization.json"}
    assert (out / "best.pt").read_bytes() == (source_run / "best.pt").read_bytes()
    record = json.loads((out / "finalization.json").read_bytes())["finalization"]
    assert record["status"] == "finalized_early"
    assert record["source"]["status"] == "failed" and record["source"]["run_id"] == "original-run"
    assert record["completed_epoch"] == 46 and record["best_epoch"] == 16 and record["best_val_loss"] == best_loss
    inference = record["checks"]["validation_inference"]
    assert inference["status"] == "passed" and inference["split"] == "val"
    assert inference["finite_predictions"] and 0 <= inference["map50_95"] <= 1
    assert inference["samples"] == len(data.splits["val"].samples)
    assert invoke().exit_code == 2
    sha = hashlib.sha256((out / "best.pt").read_bytes()).hexdigest()
    assert record["checkpoints"]["best"]["sha256"] == sha

    without_inference = tmp_path / "finalized-unscored"
    result = runner.invoke(app, [*args, "--out", str(without_inference)])
    assert result.exit_code == 0, (result.output, result.exception)
    unscored = json.loads((without_inference / "finalization.json").read_bytes())["finalization"]
    assert unscored["checks"]["validation_inference"] == {"status": "not_run", "split": "val"}

    completed = tmp_path / "completed"
    completed.mkdir()
    (completed / "best.pt").write_bytes((out / "best.pt").read_bytes())
    completed_run = json.loads((source_run / "run.json").read_bytes())
    completed_run.update(status="completed")
    completed_run.pop("error")
    write_json(completed / "run.json", completed_run)

    for profile in profiles:
        raw = profile.model_dump(mode="json")
        raw["architecture"]["image"]["detector"]["image_size"] = 64
        raw["detector_initialization"] = {"path": str(source_run / "best.pt"), "sha256": sha}
        failed = ProductTrainingConfig.model_validate_json(json.dumps(raw))
        with pytest.raises(ValueError, match="completed best checkpoint"):
            plan_product_training(failed, tmp_path / "fpn", workspace=tmp_path, progress=lambda *_: None)
        raw["detector_initialization"]["path"] = str(out / "best.pt")
        finalized = ProductTrainingConfig.model_validate_json(json.dumps(raw))
        plan = plan_product_training(finalized, tmp_path / "fpn", workspace=tmp_path, progress=lambda *_: None)
        assert plan.manifest["detector_initialization"]["finalization_fingerprint"] == digest_json(record)

    raw["detector_initialization"]["path"] = str(completed / "best.pt")
    completed_profile = ProductTrainingConfig.model_validate_json(json.dumps(raw))
    plan = plan_product_training(completed_profile, tmp_path / "fpn", workspace=tmp_path, progress=lambda *_: None)
    assert "finalization_fingerprint" not in plan.manifest["detector_initialization"]

    with (out / "best.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="Invalid finalized Detector"):
        plan_product_training(finalized, tmp_path / "fpn", workspace=tmp_path, progress=lambda *_: None)
