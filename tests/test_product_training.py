import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.data.preprocessing.common import read_csv, write_csv, write_json
from ict_pipeline.data.preprocessing.leakage import source_hashes
from ict_pipeline.data.preprocessing.policy import label_policy
from ict_pipeline.data.preprocessing.revision import SEAL, seal_dataset
from ict_pipeline.environment.model import Backend, EnvironmentReport, Graphics, Host, Python, Resources, Torch
from ict_pipeline.evaluation.product import classification_metrics, write_artifact
from ict_pipeline.evaluation.product_protocol import ExperimentReport, FrozenValidation
from ict_pipeline.inspection.contracts import ProductModelIdentity
from ict_pipeline.models.contracts import TrainingError
from ict_pipeline.multimodal import training, training_plan
from ict_pipeline.multimodal.audit import Cohort, audit_cohort
from ict_pipeline.multimodal.model_config import ArchitectureConfig, SensorEncoderConfig
from ict_pipeline.multimodal.preprocessing import PreprocessingConfig, fit_preprocessor
from ict_pipeline.multimodal.training_plan import ProductTrainingConfig, SchedulerConfig, ValidationConfig
from ict_pipeline.pipeline.store import digest_json

from test_multimodal import _sample


def _noop(*_: object) -> None:
    pass


@pytest.fixture
def profile(tmp_path: Path, request: pytest.FixtureRequest) -> ProductTrainingConfig:
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rows = [_sample(source, index, source=group, split=provider, normal=index % 2 == 0)
            for index, group, provider in ((1, "A", "train"), (2, "A", "train"),
                                           (3, "B", "validation"), (4, "B", "validation"), (5, "C", "validation"))]
    for index, row in enumerate(rows):
        row["original_split"] = row["split"]
        row["group_id"] = f"group-{index}"
    rows[-1]["split"] = "test"
    if getattr(request, "param", False):
        extra = _sample(source, 6, source="D", split="validation", normal=True)
        rows.append({**extra, "original_split": "validation", "group_id": "group-5"})
    manifest = bundle / "manifest.csv"
    write_csv(manifest, rows, list(rows[0]))
    revision = {
        "schema_version": 1, "label_policy": label_policy(),
        "selected_manifest_fingerprint": digest_json(read_csv(manifest)),
        "dataset_fingerprint": digest_json([
            [{key: row[key] for key in ("image_path", "sensor_path", "annotation_path")}, source_hashes(source, row)]
            for row in sorted(rows, key=lambda row: (row["original_split"], row["sample_id"], row["annotation_path"]))
        ]),
        "assignments": [],
    }
    revision["revision_id"] = digest_json(revision)
    revision_path = bundle / "split.json"
    write_json(revision_path, revision)
    detection = bundle / "detection"
    for fmt in ("yolo", "coco"):
        (detection / fmt).mkdir(parents=True)
    write_json(detection / SEAL, seal_dataset(detection, revision))

    audited = audit_cohort(source, manifest, revision_path=revision_path)
    write_json(tmp_path / "custody-index.json", {
        "cohort": audited.model_dump(mode="json"), "fingerprint": audited.fingerprint,
    })
    cohort = audited.for_product_training()
    cohort_path = tmp_path / "cohort.json"
    write_json(cohort_path, {"cohort": cohort.model_dump(mode="json"), "fingerprint": cohort.fingerprint})
    state = fit_preprocessor(cohort, PreprocessingConfig())
    state_path = tmp_path / "sensor-state.json"
    write_json(state_path, state.payload())
    frozen = FrozenValidation.from_audit(cohort, tuple(row["sample_id"] for row in rows[2:4]))
    frozen_path = tmp_path / "frozen.json"
    write_artifact(frozen_path, frozen)
    return ProductTrainingConfig(
        source=source, bundle=bundle, revision_id=revision["revision_id"], cohort=cohort_path,
        cohort_fingerprint=cohort.fingerprint, preprocessing=state_path,
        architecture=ArchitectureConfig(preprocessing_fingerprint=state.fingerprint),
        branch="sensor", epochs=3, validation=ValidationConfig(
            cohort=frozen_path, fingerprint=frozen.identity.cohort_fingerprint,
        ),
    )


def _plan(config: ProductTrainingConfig, tmp_path: Path) -> training_plan.ProductTrainingPlan:
    return training_plan.plan_training(config, tmp_path / "product-run", workspace=tmp_path, progress=_noop)


@pytest.mark.parametrize("branch", ("image", "sensor", "paired"))
def test_plan_enforces_split_scope(
    tmp_path: Path, profile: ProductTrainingConfig, monkeypatch: pytest.MonkeyPatch, branch: str,
) -> None:
    rows = Cohort.load(profile.cohort).records
    assert all(row.sequence is None and row.targets is None for row in rows if row.split == "test")
    test_names = [path for row in rows if row.split == "test"
                  for path in (row.annotation_path, row.image_path, row.sensor_path)]
    original_hash = training_plan.file_sha

    def check(path: Path) -> str:
        assert path.relative_to(profile.source).as_posix() not in test_names
        return original_hash(path)

    monkeypatch.setattr(training_plan, "file_sha", check)
    plan = _plan(profile.model_copy(update={"branch": branch}), tmp_path)
    assert [row.split for row in plan.train.records] == ["train", "train"]
    assert [row.split for row in plan.validation.records] == ["validation", "validation"]
    assert plan.manifest["samples"]["train"]["used"] == 2
    assert plan.validation_protocol().cohort == plan.frozen.identity
    assert not plan.output.exists()
    no_val = _plan(profile.model_copy(update={"validation": None}), tmp_path)
    assert no_val.validation is None and no_val.frozen is None


def test_revision_source_and_preprocessing_drift(tmp_path: Path, profile: ProductTrainingConfig) -> None:
    custody_index = tmp_path / "custody-index.json"
    with pytest.raises(ValueError, match="Test evidence removed"):
        _plan(profile.model_copy(update={
            "cohort": custody_index, "cohort_fingerprint": Cohort.load(custody_index).fingerprint,
        }), tmp_path)
    with pytest.raises(ValueError, match="sealed split bundle"):
        _plan(profile.model_copy(update={"revision_id": "a" * 64}), tmp_path)
    with pytest.raises(ValueError, match="pinned cohort"):
        _plan(profile.model_copy(update={"cohort_fingerprint": "b" * 64}), tmp_path)
    architecture = profile.architecture.model_copy(update={"preprocessing_fingerprint": "a" * 64})
    with pytest.raises(ValueError, match="frozen Train preprocessing fingerprint"):
        _plan(profile.model_copy(update={"architecture": architecture}), tmp_path)
    record = next(row for row in Cohort.load(profile.cohort).records if row.split == "train")
    with (profile.source / record.sensor_path).open("ab") as stream:
        stream.write(b"altered")
    with pytest.raises(ValueError, match="Source bytes changed"):
        _plan(profile, tmp_path)


def test_split_fingerprint_and_ambiguous_groups(tmp_path: Path, profile: ProductTrainingConfig) -> None:
    cohort = Cohort.load(profile.cohort)
    payload = cohort.model_dump(mode="json")
    payload["split_fingerprint"] = "a" * 64
    fingerprint = digest_json(payload)
    corrupt = tmp_path / "corrupt-index.json"
    write_json(corrupt, {"cohort": payload, "fingerprint": fingerprint})
    with pytest.raises(ValueError, match="split fingerprint"):
        _plan(profile.model_copy(update={"cohort": corrupt, "cohort_fingerprint": fingerprint}), tmp_path)

    train = next(row for row in cohort.records if row.split == "train").model_dump(mode="json")
    heldout = next(row for row in cohort.records if row.split == "test").model_dump(mode="json")
    heldout["groups"].append(["production_session", "same-session"])
    train["groups"].append(["production_session", "same-session"])
    with pytest.raises(ValueError, match="Cross-split"):
        training_plan._check_isolation([train, heldout], {0})
    with pytest.raises(ValueError, match="Ambiguous duplicate"):
        training_plan._check_isolation([train, train], {0})
    invalid = tuple(row.model_copy(update={"alignment": "ambiguous"}) if row.sample_id == train["sample_id"]
                    else row for row in cohort.records)
    state = fit_preprocessor(cohort, PreprocessingConfig())
    with pytest.raises(ValueError, match="Ambiguous sensor"):
        training_plan._views(profile, cohort.model_copy(update={"records": invalid}), state, None)


def test_missing_modalities_and_temporal_guard(tmp_path: Path, profile: ProductTrainingConfig) -> None:
    cohort = Cohort.load(profile.cohort)
    architecture = profile.architecture.model_copy(update={"sensor": SensorEncoderConfig(kind="gru")})
    with pytest.raises(ValueError, match="nonzero frozen sensor window"):
        _plan(profile.model_copy(update={"architecture": architecture}), tmp_path)
    state = fit_preprocessor(cohort, PreprocessingConfig(window_seconds=2))
    state_path = tmp_path / "temporal-state.json"
    write_json(state_path, state.payload())
    temporal = profile.model_copy(update={"preprocessing": state_path, "architecture": architecture.model_copy(update={
        "preprocessing_fingerprint": state.fingerprint,
    })})
    assert _plan(temporal, tmp_path).validation_protocol().alignment_policy == "audited_pair_decision_window-v1"
    architecture = temporal.architecture.model_copy(update={"sensor": SensorEncoderConfig(kind="gru", min_steps=4)})
    with pytest.raises(ValueError, match="enough observed steps"):
        _plan(temporal.model_copy(update={"architecture": architecture}), tmp_path)

    records = list(cohort.records)
    records[0] = records[0].model_copy(update={"image_available": False})
    missing = cohort.model_copy(update={"records": tuple(records)})
    paired = profile.model_copy(update={"branch": "paired"})
    with pytest.raises(ValueError, match="requested modality"):
        training_plan._views(paired, missing, state, None)
    train, _, counts = training_plan._views(paired.model_copy(update={"missing_train": "drop"}), missing, state, None)
    assert len(train) == counts["train"]["used"] == 1


@pytest.fixture
def ready(monkeypatch: pytest.MonkeyPatch) -> None:
    def environment(_path: Path) -> EnvironmentReport:
        return EnvironmentReport(
            Host("fixture", None, None, None, None), Python("3.11", "CPython", True),
            Resources("cpu", 1, None, None, 4 * 1024**3), Graphics("unavailable", (), "unavailable", None),
            Torch("available", cpu=Backend("available", "available", precisions_tested=("fp32",))), {}, {}, (),
        )
    monkeypatch.setattr(training_plan, "collect_environment", environment)


def test_preflight_and_cli_gate(tmp_path: Path, profile: ProductTrainingConfig, ready: None,
                                monkeypatch: pytest.MonkeyPatch) -> None:
    raw = profile.model_dump(mode="json")
    with pytest.raises(ValueError, match="requires frozen Validation"):
        ProductTrainingConfig.model_validate_json(json.dumps({**raw, "checkpoint_policy": "best_validation"}))
    with pytest.raises(ValueError):
        SchedulerConfig(gamma=1.1)
    with pytest.raises(ValueError):
        ProductTrainingConfig.model_validate_json(json.dumps({**raw, "optimizer": {"learning_rate": -1}}))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    out = tmp_path / "product-run"
    args = ["multimodal", "train", "--config", str(path), "--out", str(out)]
    monkeypatch.setattr(training, "execute_training", lambda *args, **kwargs: pytest.fail("Execution is gated"))
    runner = CliRunner()
    preview = runner.invoke(app, [*args, "--dry-run", "--approve-train"])
    assert preview.exit_code == 0, preview.output
    payload = json.loads(preview.stdout)
    assert payload["experiment_status"] == "pending" and payload["preflight"]["ready"]
    gated = runner.invoke(app, args)
    assert gated.exit_code == 2 and "--approve-train" in gated.output
    raw["device"] = "rocm"
    path.write_text(json.dumps(raw), encoding="utf-8")
    gpu = runner.invoke(app, [*args, "--dry-run"])
    assert gpu.exit_code == 1, gpu.output
    assert not json.loads(gpu.stdout)["preflight"]["ready"]
    assert not out.exists()


class MockEngine:
    def __init__(self, plan: training_plan.ProductTrainingPlan) -> None:
        self.plan = plan
        self.epochs: list[int] = []
        self.restored: list[int] = []
        self.interrupt_epoch: int | None = None

    def restore(self, _directory: Path, receipt: training.EpochReceipt) -> None:
        self.restored.append(receipt.epoch)

    def train_epoch(self, epoch: int, _progress: object) -> float:
        self.epochs.append(epoch)
        if epoch == self.interrupt_epoch:
            raise KeyboardInterrupt
        return 0.75

    def save(self, directory: Path, epoch: int) -> None:
        for name in ("model.pt", "resume.pt"):
            (directory / name).write_bytes(f"mock {name} {epoch}".encode())

    def evaluate(self, checkpoint_hash: str, _progress: object) -> ExperimentReport:
        config = self.plan.config
        protocol = self.plan.validation_protocol()
        # Hand-authored scores exercise report/selection contracts only.
        scores = {sample.sample_id: ((0.9 if sample.targets else 0.1) if self.epochs[-1] < 3 else 0.1,)
                  for sample in self.plan.frozen.samples}
        identity = ProductModelIdentity(
            checkpoint_format="ict-multimodal-state-v1", checkpoint_sha256=checkpoint_hash,
            architecture_version=1, architecture_fingerprint=self.plan.manifest["architecture_fingerprint"],
            image_encoder="rgb_grid", image_encoder_version=1, sensor_encoder="mlp", sensor_encoder_version=1,
            preprocessing_version=1, preprocessing_fingerprint=self.plan.state.fingerprint,
            image_preprocessing_fingerprint=protocol.image_preprocessing_fingerprint,
            window_policy=self.plan.state.config.window_policy, fusion="late", fusion_version=1,
            fusion_fingerprint=digest_json(config.architecture.fusion.model_dump(mode="json")),
            device="cpu", seed=42,
        )
        return ExperimentReport(
            experiment_id="product-run", status="measured", branch=config.branch, protocol=protocol,
            product_model=identity, fusion=config.architecture.fusion, prediction_fingerprint=digest_json(scores),
            code_fingerprint="a" * 64, runtime={}, classification_metrics=classification_metrics(
                self.plan.frozen, protocol, scores,
            ),
        )


@pytest.mark.parametrize(("policy", "chosen"), (("final_epoch", 3), ("best_validation", 1)))
def test_mock_execution_and_recovery(
    tmp_path: Path, profile: ProductTrainingConfig, ready: None, monkeypatch: pytest.MonkeyPatch,
    policy: str, chosen: int,
) -> None:
    pytest.importorskip("torch")
    from ict_pipeline.multimodal import training_torch

    config = profile.model_copy(update={
        "checkpoint_policy": policy,
        "validation": profile.validation.model_copy(update={"frequency": "each_epoch" if chosen == 1 else "final"}),
    })
    plan = _plan(config, tmp_path)
    engine = MockEngine(plan)

    @contextmanager
    def factory(_plan: training_plan.ProductTrainingPlan) -> Iterator[MockEngine]:
        yield engine

    monkeypatch.setattr(training_torch, "training_engine", factory)
    options = {"credentials": Credentials(), "progress": lambda _: _noop}
    with pytest.raises(TrainingError, match="approval|approve-train"):
        training.execute_training(plan, approved=False, resume=False, **options)
    assert not plan.output.exists()
    engine.interrupt_epoch = 2
    with pytest.raises(KeyboardInterrupt):
        training.execute_training(plan, approved=True, resume=False, **options)
    assert json.loads((plan.output / "run.json").read_bytes())["status"] == "interrupted"
    assert len(training.inspect_recovery(plan, resume=True)) == 1
    incompatible = _plan(config.model_copy(update={"gradient_accumulation": 2}), tmp_path)
    with pytest.raises(ValueError, match="configuration is incompatible"):
        training.inspect_recovery(incompatible, resume=True)
    epoch = training.epoch_directory(plan.output, 1)
    contents = (epoch / "resume.pt").read_bytes()
    (epoch / "resume.pt").write_bytes(b"truncated")
    with pytest.raises(ValueError, match="checkpoint hash"):
        training.inspect_recovery(plan, resume=True)
    (epoch / "resume.pt").write_bytes(contents)
    engine.interrupt_epoch = None
    result = training.execute_training(plan, approved=True, resume=True, **options)
    saved = json.loads(result.read_bytes())
    assert saved["status"] == "completed" and saved["selected_checkpoint"]["epoch"] == chosen
    assert saved["profile"]["checkpoint_policy"] == policy
    assert engine.epochs == [1, 2, 2, 3] and engine.restored == [1]
    assert (training.epoch_directory(plan.output, 1) / "validation.json").exists() == (chosen == 1)
    with pytest.raises(ValueError, match="already completed"):
        training.inspect_recovery(plan, resume=True)


def test_optimizer_scheduler_and_invalid_mask(
    tmp_path: Path, profile: ProductTrainingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from ict_pipeline.multimodal.model import MultimodalModel, ProductOutput
    from ict_pipeline.multimodal.training_torch import _branch_logits, configure_optimizer

    def fail_step(self: object, *args: object, **kwargs: object) -> None:
        pytest.fail("Optimizer steps are prohibited")

    monkeypatch.setattr(torch.optim.AdamW, "step", fail_step)
    plan = _plan(profile.model_copy(update={"scheduler": SchedulerConfig(kind="step", step_size=2, gamma=0.5)}),
                 tmp_path)
    model = MultimodalModel(plan.config.architecture, plan.state)
    optimizer, scheduler = configure_optimizer(model, plan)
    assert optimizer.param_groups[0]["lr"] == profile.optimizer.learning_rate
    assert scheduler.step_size == 2 and scheduler.gamma == 0.5
    logits = torch.zeros((1, 1))
    for mask in (torch.zeros(1, dtype=torch.bool), torch.ones(1, dtype=torch.float32), torch.ones((1, 1), dtype=torch.bool)):
        output = ProductOutput("train", ("one",), ("smt-target-defect-presence-v1"), ("any_target_defect",),
                               logits, logits, logits, mask, mask)
        with pytest.raises(ValueError, match="usable modality mask"):
            _branch_logits(output, "sensor", 1)

def test_tensor_resume_contract_without_optimizer_steps(
    tmp_path: Path, profile: ProductTrainingConfig, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from ict_pipeline.multimodal.model_config import WeightStatus
    from ict_pipeline.multimodal.training_torch import TorchEpochEngine

    def fail_step(self: object, *args: object, **kwargs: object) -> None:
        pytest.fail("Optimizer steps are prohibited")

    monkeypatch.setattr(torch.optim.AdamW, "step", fail_step)
    plan = _plan(profile, tmp_path)
    engine = TorchEpochEngine(plan, "cpu")
    epoch = tmp_path / "checkpoint-fixture"
    epoch.mkdir()
    engine.model.weight_status = WeightStatus(sensor="trained")
    state = engine.optimizer.state[next(engine.model.parameters())]
    # Construct a tensor-layout fixture without a forward, backward, or step.
    state["step"] = torch.tensor(1.0)
    for name in ("exp_avg", "exp_avg_sq"):
        state[name] = torch.zeros_like(next(engine.model.parameters()))
    engine.scheduler.last_epoch = 1
    engine.scheduler._step_count = 2
    engine.save(epoch, 1)
    receipt = training.EpochReceipt(
        compatibility=plan.compatibility, execution="a" * 64, epoch=1, previous=None,
        training_loss=0.0, elapsed_seconds=0.0, model_sha256="a" * 64, resume_sha256="a" * 64,
        validation_sha256=None, validation_metric=None,
    )
    restored = TorchEpochEngine(plan, "cpu")
    restored.restore(epoch, receipt)
    assert restored.scheduler.last_epoch == 1
    assert restored.model.branch_status("sensor") == "trained"
    assert len(restored.optimizer.state) == 1
    payload = torch.load(epoch / "resume.pt", map_location="cpu", weights_only=True)
    payload["preprocessing"] = "b" * 64
    torch.save(payload, epoch / "resume.pt")
    with pytest.raises(ValueError, match="architecture, or preprocessing"):
        restored.restore(epoch, receipt)


@pytest.mark.parametrize("branch", ("image", "paired"))
def test_fpn_cpu_training_and_reload(
    tmp_path: Path, profile: ProductTrainingConfig, branch: str, ready: None,
) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ict_pipeline.data.preprocessing.leakage import file_sha
    from ict_pipeline.evaluation.product import compare_experiments
    from ict_pipeline.multimodal.checkpoint import load_checkpoint
    from ict_pipeline.multimodal.dataset import collate
    from ict_pipeline.multimodal.fpn import ARCHITECTURE, FORMAT, build_detector, load_detector
    from ict_pipeline.multimodal.model_config import FPNDetectorConfig, FusionConfig, ImageEncoderConfig
    from ict_pipeline.multimodal.training_plan import DetectorInitialization
    from ict_pipeline.multimodal.training_torch import training_engine

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        spec = FPNDetectorConfig(image_size=64)
        detector = build_detector(spec)
        path = tmp_path / "best.pt"
        torch.save({
            "format": FORMAT, "architecture": ARCHITECTURE, "epoch": 1, "best_epoch": 1,
            "best_loss": 0.1, "val_loss": 0.1, "model": detector.state_dict(), "optimizer": {}, "scaler": {},
        }, path)
        sha256 = file_sha(path)
        image = ImageEncoderConfig(kind="faster_rcnn_fpn", freeze_backbone=True, detector=spec)
        common = profile.architecture.model_copy(update={"target_policy": "smt-view-class-presence-v1"})
        architecture = common.model_copy(update={
            "image": image, "fusion": FusionConfig(kind="concat" if branch == "paired" else "late"),
        })
        validation = profile.validation.model_copy(update={"frequency": "each_epoch"})
        settings = {
            "branch": branch, "validation": validation, "checkpoint_policy": "best_validation",
            "selection_metric": "macro_auroc",
        }
        rgb_plan = _plan(profile.model_copy(update={**settings, "architecture": common}), tmp_path)
        config = ProductTrainingConfig.model_validate({
            **dict(profile), **settings, "architecture": architecture,
            "detector_initialization": DetectorInitialization(path=path, sha256=sha256),
        })
        plan = _plan(config, tmp_path)
        assert plan.frozen.identity == rgb_plan.frozen.identity
        assert plan.state.fingerprint == rgb_plan.state.fingerprint
        assert plan.manifest["samples"] == rgb_plan.manifest["samples"]
        assert plan.checkpoint_bytes > rgb_plan.checkpoint_bytes
        assert plan.estimated_device_bytes > rgb_plan.estimated_device_bytes

        epoch = tmp_path / "fpn-epoch"
        epoch.mkdir()
        with training_engine(plan) as engine:
            model = engine.model
            backbone = model.image_encoder.backbone
            initial = {name: value.clone() for name, value in backbone.state_dict().items()}
            for name, value in detector.backbone.state_dict().items():
                torch.testing.assert_close(initial[name], value, rtol=0, atol=0)
            model.train()
            assert all(not module.training for module in backbone.modules())
            assert all(not param.requires_grad for param in backbone.parameters())
            params = dict(model.named_parameters())
            trainable = ["image_encoder.projection.weight"]
            if branch == "paired":
                trainable.extend(name for name in params if name.startswith(("fusion.", "sensor_encoder.")))
            else:
                trainable.append("image_head.weight")
            before = {name: params[name].detach().clone() for name in trainable}
            assert engine.train_epoch(1, _noop) > 0
            assert any(not torch.equal(before[name], params[name]) for name in trainable if name.startswith("image_encoder."))
            for prefix in (("fusion.", "sensor_encoder.") if branch == "paired" else ("image_head.",)):
                assert any(not torch.equal(before[name], params[name]) for name in trainable if name.startswith(prefix))
            for name, value in backbone.state_dict().items():
                torch.testing.assert_close(initial[name], value, rtol=0, atol=0)
            engine.save(epoch, 1)
            checkpoint_sha = file_sha(epoch / "model.pt")
            report = engine.evaluate(checkpoint_sha, _noop)
            training._check_report(plan, report, checkpoint_sha)
            assert len(report.classification_metrics) == 5
            assert training.selection_metric(plan, report) is not None
            batch = collate([plan.validation[0]])
            with torch.inference_mode():
                scores = model(batch).for_branch(branch)[0].sigmoid()
                reloaded = load_checkpoint(epoch / "model.pt", plan.state)
                assert torch.equal(scores, reloaded(batch).for_branch(branch)[0].sigmoid())
            receipt = training.EpochReceipt(
                compatibility=plan.compatibility, execution="a" * 64, epoch=1, previous=None,
                training_loss=0.1, elapsed_seconds=0.0, model_sha256=checkpoint_sha,
                resume_sha256=file_sha(epoch / "resume.pt"), validation_sha256=None, validation_metric=None,
            )
        with training_engine(plan) as engine:
            engine.restore(epoch, receipt)
            assert engine.scheduler.last_epoch == 1
            assert engine.evaluate(checkpoint_sha, _noop).prediction_fingerprint == report.prediction_fingerprint
            assert engine.train_epoch(2, _noop) > 0
        with training_engine(rgb_plan) as engine:
            engine.train_epoch(1, _noop)
            rgb_report = engine.evaluate("a" * 64, _noop).model_copy(update={"experiment_id": "rgb"})
        assert len(compare_experiments((report, rgb_report), compare_image_encoders=True)) == 2
        with pytest.raises(ValueError, match="identical frozen"):
            compare_experiments((report, rgb_report.model_copy(update={
                "protocol": rgb_report.protocol.model_copy(update={"threshold": 0.25}),
            })), compare_image_encoders=True)

        if branch == "paired":
            config_path = tmp_path / "fpn.json"
            config_path.write_text(config.model_copy(update={"epochs": 1}).model_dump_json(), encoding="utf-8")
            args = ["multimodal", "train", "--config", str(config_path), "--out", str(plan.output)]
            preview = CliRunner().invoke(app, [*args, "--dry-run"])
            assert preview.exit_code == 0, preview.output
            payload = json.loads(preview.stdout)
            assert payload["preflight"]["ready"] and payload["detector_initialization"]["sha256"] == sha256
            assert not plan.output.exists()
            executed = CliRunner().invoke(app, [*args, "--approve-train"])
            assert executed.exit_code == 0, executed.output
            metadata = json.loads((plan.output / "run.json").read_bytes())
            assert metadata["selected_checkpoint"]["epoch"] == 1
            assert (plan.output / metadata["selected_checkpoint"]["path"]).is_file()
            assert metadata["validation_report"]["macro_auroc"] is not None
            assert metadata["validation_report"]["macro_f1"] is not None

        with pytest.raises(ValueError, match="SHA-256"):
            load_detector(path, "0" * 64, spec)
        with pytest.raises(ValueError, match="detector layout"):
            load_detector(path, sha256, FPNDetectorConfig(image_size=64, layout="pretrained"))
        assert file_sha(path) == sha256
    finally:
        torch.set_num_threads(threads)
