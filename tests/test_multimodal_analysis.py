import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.data.preprocessing.common import read_csv, write_csv, write_json
from ict_pipeline.data.preprocessing.leakage import AnalysisConfig, analyze_leakage, file_sha
from ict_pipeline.evaluation.product import classification_metrics, read_artifact, write_artifact
from ict_pipeline.evaluation.product_protocol import ExperimentReport, FrozenValidation
from ict_pipeline.inspection.contracts import ProductModelIdentity
from ict_pipeline.multimodal import analysis
from ict_pipeline.multimodal.audit import Cohort
from ict_pipeline.multimodal.model_config import FusionConfig, ImageEncoderConfig
from ict_pipeline.multimodal.preprocessing import PreprocessingState
from ict_pipeline.multimodal.training_plan import (
    DetectorInitialization, ProductTrainingConfig, ProductTrainingPlan, _views,
)
from ict_pipeline.pipeline.store import digest_json

from test_product_training import profile  # noqa: F401


def _run(profile: ProductTrainingConfig, folder: Path, role: analysis.ModelRole, epoch: int) -> Path:
    cohort = Cohort.load(profile.cohort)
    frozen = read_artifact(profile.validation.cohort, FrozenValidation)
    state = PreprocessingState.load(profile.preprocessing)
    architecture = profile.architecture.model_copy(update={
        "image": ImageEncoderConfig.model_validate({"kind": "faster_rcnn_fpn", "detector": {"image_size": 640}}),
        "fusion": FusionConfig(kind="concat" if role == "concat" else "late"),
        "target_policy": "smt-view-class-presence-v1",
    })
    config = profile.model_copy(update={
        "architecture": architecture, "branch": "paired" if role == "concat" else "image",
        "checkpoint_policy": "best_validation", "selection_metric": "macro_auroc",
        "validation": profile.validation.model_copy(update={"frequency": "each_epoch"}),
        "detector_initialization": DetectorInitialization(path=folder / "unused.pt", sha256="a" * 64),
    })
    train, validation, counts = _views(config, cohort, state, frozen)
    plan = ProductTrainingPlan(config, folder, state, train, validation, frozen, {}, "", 0, 0, 0)
    protocol = plan.validation_protocol()
    epoch_dir = folder / f"epoch-{epoch:06d}"
    epoch_dir.mkdir(parents=True)
    model = epoch_dir / "model.pt"
    model.write_bytes(b"mock selected checkpoint")
    (folder / "sensor-state.json").write_bytes(profile.preprocessing.read_bytes())
    scores = {sid: (0.7, 0.1, 0.1, 0.1, 0.1) for sid in frozen.identity.sample_ids}
    identity = ProductModelIdentity(
        checkpoint_format="ict-multimodal-state-v1", checkpoint_sha256=file_sha(model),
        architecture_version=1, architecture_fingerprint="b" * 64,
        image_encoder="faster_rcnn_fpn", image_encoder_version=1, sensor_encoder="mlp", sensor_encoder_version=1,
        preprocessing_version=1, preprocessing_fingerprint=state.fingerprint,
        image_preprocessing_fingerprint=protocol.image_preprocessing_fingerprint,
        window_policy=state.config.window_policy, fusion=architecture.fusion.kind, fusion_version=1,
        fusion_fingerprint=digest_json(architecture.fusion.model_dump(mode="json")),
        device="cpu", seed=42,
    )
    report = epoch_dir / "validation.json"
    write_artifact(report, ExperimentReport(
        experiment_id=role, status="measured", branch=config.branch, protocol=protocol,
        product_model=identity, fusion=architecture.fusion, prediction_fingerprint=digest_json(scores),
        code_fingerprint="a" * 64, runtime={}, classification_metrics=classification_metrics(frozen, protocol, scores),
    ))
    saved = config.model_dump(mode="json")
    for key in ("source", "bundle", "cohort", "preprocessing"):
        del saved[key]
    del saved["detector_initialization"]["path"]
    del saved["validation"]["cohort"]
    write_json(folder / "run.json", {
        "schema_version": 1, "task": "product_classification", "status": "completed",
        "profile": saved, "validation_identity": frozen.identity.model_dump(mode="json"),
        "train_state_fingerprint": state.training_fingerprint, "preprocessing_fingerprint": state.fingerprint,
        "samples": counts, "architecture_fingerprint": identity.architecture_fingerprint, "completed_epochs": 30,
        "selected_checkpoint": {"epoch": epoch, "path": model.relative_to(folder).as_posix(), "sha256": file_sha(model)},
        "validation_report": {"path": report.relative_to(folder).as_posix(), "sha256": file_sha(report)},
    })
    return folder


def test_analysis_cli(tmp_path: Path, profile: ProductTrainingConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    image = _run(profile, tmp_path / "image", "image", 30)
    concat = _run(profile, tmp_path / "concat", "concat", 14)
    original = read_csv(profile.bundle / "manifest.csv")
    for row in original:
        row["split"] = row["original_split"]
    manifest = tmp_path / "original.csv"
    write_csv(manifest, original, list(original[0]))
    audit = analyze_leakage(profile.source, manifest, profile.source, AnalysisConfig())
    indexed = {row["sample_id"]: i for i, row in enumerate(audit["samples"])}
    members = [indexed[original[i]["sample_id"]] for i in (0, 2, 4)]
    audit["groups"] = [group for group in audit["groups"] if group["method"] not in (
        "image_similarity", "source_day", "capture_group",
    )]
    audit["groups"].extend([
        {"method": "image_similarity", "members": members, "group_id": "similar"},
        {"method": "source_day", "members": members, "group_id": "overlap"},
        {"method": "capture_group", "members": [members[0], members[2]], "group_id": "held-out-only"},
    ])
    audit_path = tmp_path / "leakage.json"
    write_json(audit_path, audit)

    def scores(
        run: analysis.AnalysisRun, device: str, batch_size: int, donors: dict[str, dict[str, str]],
        progress: analysis.ProgressFactory, checkpoint_dir: Path | None,
    ) -> tuple[dict[str, analysis.Scores], dict[str, str]]:
        assert device == "cpu" and batch_size == 2
        values = {run.validation[i].sample_id: (0.8, 0.1, 0.1, 0.1, 0.1) for i in range(len(run.validation))}
        return {key: values for key in ("original", *donors)}, (
            {"sensor_removed": "untrained"} if donors else {}
        )

    monkeypatch.setattr(analysis, "score_run", scores)
    out = tmp_path / "reports"
    cli = CliRunner()
    assert cli.invoke(app, ["multimodal", "analyze", "--help"]).exit_code == 0
    args = ["multimodal", "analyze", "--root", str(profile.source), "--audit", str(audit_path),
            "--cohort", str(profile.cohort), "--frozen-validation", str(profile.validation.cohort),
            "--image-run", str(image), "--concat-run", str(concat), "--out", str(out), "--batch-size", "2"]
    result = cli.invoke(app, args)
    assert result.exit_code == 0, (result.output, result.exception)
    leak = json.loads((out / "leakage.json").read_bytes())["models"]["concat"]
    assert leak["methods"]["image_similarity"]["validation_ids"] == [original[2]["sample_id"]]
    assert leak["methods"]["capture_group"]["group_count"] == 0
    assert all(original[4]["sample_id"] not in item["validation_ids"] for item in leak["groups"])
    assert leak["methods"]["source_day"]["validation_ids"] == [original[2]["sample_id"]]
    assert len(leak["any_method_ids"]["validation"]) == len(set().union(
        *(set(group["validation_ids"]) for group in leak["groups"]),
    ))
    groups = json.loads((out / "groups.json").read_bytes())["results"]
    similar = next(row for row in groups if row["group"] == "cross_group")
    assert similar["sample_count"] == 1 and similar["macro_auroc"] is None
    assert len(similar["classes"]) == 5
    assert sum(row["positive"] + row["negative"] for row in similar["classes"]) == 5
    ablations = json.loads((out / "ablation.json").read_bytes())["results"]
    assert {tuple(row["sample_ids"]) for row in ablations} == {tuple(leak["validation_ids"])}
    assert {row["status"] for row in ablations} == {"measured", "skipped"}
    assert (out / "manufacturers.csv").is_file()
    assert cli.invoke(app, args).exit_code == 2
    (concat / "epoch-000014" / "model.pt").write_bytes(b"changed")
    failed = cli.invoke(app, [*args, "--out", str(tmp_path / "fresh")])
    assert failed.exit_code == 2 and "fingerprint mismatch" in failed.output
    assert not (tmp_path / "fresh").exists()


def test_sensor_donors() -> None:
    ids = ("a", "b", "c", "d", "e", "f")
    groups = dict(zip(ids, ("X", "X", "X", "Y", "Y", ""), strict=True))
    within = analysis.sensor_donors(ids, groups, 7, within_manufacturer=True)
    overall = analysis.sensor_donors(ids, groups, 7, within_manufacturer=False)
    assert within == analysis.sensor_donors(ids[::-1], groups, 7, within_manufacturer=True)
    assert all(groups[donor] == groups[sid] and (donor != sid or sid == "f") for sid, donor in within.items())
    assert set(overall.values()) == set(ids) and all(sid != donor for sid, donor in overall.items())
    assert overall != within


@pytest.mark.parametrize("trained_image", [False, True])
def test_scoring_interventions(
    tmp_path: Path, profile: ProductTrainingConfig, monkeypatch: pytest.MonkeyPatch, trained_image: bool,
) -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from ict_pipeline.multimodal import checkpoint, training_torch
    from ict_pipeline.multimodal.dataset import Batch
    from ict_pipeline.multimodal.model import ProductOutput

    path = _run(profile, tmp_path / "concat", "concat", 14)
    cohort = Cohort.load(profile.cohort)
    frozen = read_artifact(profile.validation.cohort, FrozenValidation)
    run = analysis._inspect_run("concat", path, profile.source, profile.cohort, profile.validation.cohort, cohort, frozen)
    batches: list[Batch] = []

    class FixtureModel:
        config = run.profile.architecture
        architecture_fingerprint = run.architecture_fingerprint
        image_encoder = SimpleNamespace(input_contract={})

        def to(self, device: str) -> "FixtureModel":
            assert device == "cpu"
            return self

        def eval(self) -> "FixtureModel":
            return self

        def branch_status(self, branch: str) -> str:
            return "trained" if branch == "paired" or (branch == "image" and trained_image) else "untrained"

        def __call__(self, batch: Batch) -> ProductOutput:
            batches.append(batch)
            logits = torch.zeros((len(batch.samples), 5))
            return ProductOutput("validation", tuple(row.sample_id for row in batch.samples),
                                 self.config.target_policy, self.config.output_names, logits, logits, logits,
                                 torch.tensor([image for image, _ in batch.modalities]),
                                 torch.tensor([sensor for _, sensor in batch.modalities]))

    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *args, **kwargs: FixtureModel())
    # Import the trainer before patching so its module-level loader binding stays real.
    assert training_torch.load_checkpoint is not checkpoint.load_checkpoint
    monkeypatch.setattr(analysis, "resolve_device", lambda *args: "cpu")
    a, b = frozen.identity.sample_ids
    cache = tmp_path / "cache"
    options = (run, "cpu", 2, {"global_shuffle": {a: b, b: a}}, lambda _: lambda *_: None, cache)

    def interrupt_after_commit(done: int, total: int | None) -> None:
        if done == total:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        analysis.score_run(*options[:4], lambda _: interrupt_after_commit, cache)
    initial_calls = len(batches)
    scores, skipped = analysis.score_run(*options)
    assert len(batches) == initial_calls
    assert set(scores["original"]) == set(scores["global_shuffle"]) == {a, b}
    baseline, shuffled = batches[:2]
    for original, changed, donor in zip(baseline.samples, shuffled.samples, baseline.samples[::-1], strict=True):
        assert changed.image is original.image and changed.original_targets == original.original_targets
        assert changed.sensor == donor.sensor
    if trained_image:
        assert not skipped and set(scores["sensor_removed"]) == {a, b}
        assert all(row.sensor is None for row in batches[2].samples)
    else:
        assert "sensor_removed" in skipped and len(batches) == 2
    next(cache.glob("*/*.json")).write_text('{"scores": {}, "fingerprint": ""}', encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint batch is corrupt"):
        analysis.score_run(*options)
