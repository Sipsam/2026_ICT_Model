import json
from pathlib import Path
from types import SimpleNamespace
from typing import IO, Any

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.data.group_retraining import read_group_verification
from ict_pipeline.data.preprocessing.common import read_csv, write_csv, write_json
from ict_pipeline.data.preprocessing import leakage
from ict_pipeline.data.preprocessing.leakage import AnalysisConfig, analyze_leakage
from ict_pipeline.data.preprocessing.split import SplitConfig, plan_split, publish_split
from ict_pipeline.evaluation.product import write_artifact
from ict_pipeline.evaluation.product_protocol import FrozenValidation
from ict_pipeline.models import training as detector_training
from ict_pipeline.models.contracts import Checkpoints, ModelName
from ict_pipeline.models.dataset import DetectionData
from ict_pipeline.models.registry import Backend
from ict_pipeline.multimodal.audit import audit_cohort
from ict_pipeline.multimodal.preprocessing import PreprocessingConfig, fit_preprocessor
from ict_pipeline.multimodal.training_plan import ProductTrainingConfig, plan_training as plan_product_training

from test_multimodal import _sample


def test_group_command_and_detector_scope_keep_test_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    manifest = tmp_path / "provider.csv"
    rows = [_sample(source, index, values=(index * 5, index * 5 + 1, index * 5 + 2))
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
    for name in ("fpn-image-only", "fpn-sensor-concat"):
        raw = yaml.safe_load((templates / f"{name}.yaml").read_text(encoding="utf-8"))
        raw.update(source=str(source), bundle=str(bundle), revision_id=revision["revision_id"],
                   cohort=str(index), cohort_fingerprint=cohort.fingerprint, preprocessing=str(state),
                   group_verification=str(receipt))
        raw["architecture"]["preprocessing_fingerprint"] = fitted.fingerprint
        raw["validation"].update(cohort=str(frozen), fingerprint=saved["frozen_validation_fingerprint"])
        raw["detector_initialization"].update(path=str(checkpoint), sha256="a" * 64)
        profile = ProductTrainingConfig.model_validate_json(json.dumps(raw))
        assert profile.epochs == 30 and profile.architecture.image.freeze_backbone
        with pytest.raises(ValueError, match="completed best checkpoint from the verified group Detector"):
            plan_product_training(profile, tmp_path / name, workspace=tmp_path, progress=lambda *_: None)

    path = dataset / "coco" / "annotations" / "instances_train.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="inputs changed"):
        read_group_verification(dataset, receipt)
