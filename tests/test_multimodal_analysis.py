import json
from pathlib import Path
from typing import IO, Any

import pytest
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.data import detector_finalization
from ict_pipeline.data.group_retraining import GroupInputs, verify_group_retraining
from ict_pipeline.data.preprocessing.common import read_csv, write_csv, write_json
from ict_pipeline.data.preprocessing.leakage import AnalysisConfig, analyze_leakage, file_sha
from ict_pipeline.data.preprocessing.split import SplitConfig, plan_split, publish_split
from ict_pipeline.evaluation.product import classification_metrics, read_artifact, write_artifact
from ict_pipeline.evaluation.product_protocol import ExperimentReport, FrozenValidation
from ict_pipeline.inspection.contracts import ProductModelIdentity
from ict_pipeline.multimodal import analysis
from ict_pipeline.models.weights import LAYOUTS
from ict_pipeline.multimodal.audit import Cohort, audit_cohort
from ict_pipeline.multimodal.model_config import FusionConfig, ImageEncoderConfig
from ict_pipeline.multimodal.preprocessing import PreprocessingConfig, PreprocessingState, fit_preprocessor
from ict_pipeline.multimodal.training_plan import (
    DetectorInitialization, ProductTrainingConfig, ProductTrainingPlan, _load_index, _views,
)
from ict_pipeline.pipeline.store import digest_json, fingerprint_path

from multimodal_fixtures import make_sample
from product_training_fixtures import product_training_profile as product_training_profile


def _run(profile: ProductTrainingConfig, folder: Path, role: analysis.ModelRole, epoch: int) -> Path:
    frozen = read_artifact(profile.validation.cohort, FrozenValidation)
    cohort, seal, _ = _load_index(profile, frozen, lambda *_: None)
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
        "detector_initialization": profile.detector_initialization or
        DetectorInitialization(path=folder / "unused.pt", sha256="a" * 64),
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
    group_metadata = {}
    if profile.group_verification:
        saved["group_verification"] = json.loads(profile.group_verification.read_bytes())["fingerprint"]
        group_metadata = {
            "dataset_fingerprint": cohort.input_fingerprint, "split_fingerprint": cohort.split_fingerprint,
            "seal_fingerprint": seal,
            "detector_initialization": {
                "format": "ict-torch-state-v1", "architecture": LAYOUTS["faster_rcnn"].architecture,
                "sha256": config.detector_initialization.sha256,
            },
        }
    write_json(folder / "run.json", {
        "schema_version": 1, "task": "product_classification", "status": "completed",
        "profile": saved, "validation_identity": frozen.identity.model_dump(mode="json"),
        "train_state_fingerprint": state.training_fingerprint, "preprocessing_fingerprint": state.fingerprint,
        "samples": counts, "architecture_fingerprint": identity.architecture_fingerprint, "completed_epochs": 30,
        "selected_checkpoint": {"epoch": epoch, "path": model.relative_to(folder).as_posix(), "sha256": file_sha(model)},
        "validation_report": {"path": report.relative_to(folder).as_posix(), "sha256": file_sha(report)},
        **group_metadata,
    })
    return folder


@pytest.mark.parametrize("profile", [False, True], indirect=True, ids=["exact", "extra-validation"])
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
    audit["cross_split_groups"] = [
        group["group_id"] for group in audit["groups"]
        if len({audit["samples"][i]["original_split"] for i in group["members"]}) > 1
    ]
    audit_path = tmp_path / "leakage.json"
    write_json(audit_path, {"fingerprint": digest_json(audit), "report": audit})
    frozen = read_artifact(profile.validation.cohort, FrozenValidation)
    candidates = {row.sample_id for row in Cohort.load(profile.cohort).records
                  if row.split == "validation" and row.detection_candidate}
    extra_ids = candidates.difference(frozen.identity.sample_ids)
    assert extra_ids == {row["sample_id"] for row in original[5:]}
    assert len(candidates) == len(frozen.identity.sample_ids) + len(original[5:])

    def scores(
        run: analysis.AnalysisRun, device: str, batch_size: int, donors: dict[str, dict[str, str]],
        progress: analysis.ProgressFactory, checkpoint_dir: Path | None,
    ) -> tuple[dict[str, analysis.Scores], dict[str, str]]:
        assert device == "cpu" and batch_size == 2
        assert tuple(sorted(row.sample_id for row in run.validation.records)) == frozen.identity.sample_ids
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
    config = json.loads((out / "analysis.json").read_bytes())
    assert config["audit_configuration"] == json.loads(audit_path.read_bytes())["report"]["configuration"]
    assert config["audit_sha256"] == file_sha(audit_path)
    leak = json.loads((out / "leakage.json").read_bytes())["models"]["concat"]
    assert extra_ids.isdisjoint(leak["validation_ids"])
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
    tampered = tmp_path / "tampered.json"
    write_json(tampered, {"fingerprint": digest_json(audit), "report": {**audit, "samples": []}})
    failed = cli.invoke(app, [*args, "--out", str(tmp_path / "fresh"), "--audit", str(tampered)])
    assert failed.exit_code == 2
    assert "Verify cohort, audit and selected checkpoints" in failed.output
    assert "leakage.json fingerprint does not match report contents" in failed.output
    assert not (tmp_path / "fresh").exists()

    receipt = concat / "run.json"
    original_receipt = receipt.read_bytes()
    saved = json.loads(original_receipt)
    del saved["selected_checkpoint"]
    receipt.write_text(json.dumps(saved), encoding="utf-8")
    failed = cli.invoke(app, [*args, "--out", str(tmp_path / "fresh")])
    assert failed.exit_code == 2
    assert "Verify concat run and project leakage" in failed.output and "(KeyError)" in failed.output
    assert "selected_checkpoint" not in failed.output and "Traceback" not in failed.output
    assert not (tmp_path / "fresh").exists()
    receipt.write_bytes(original_receipt)
    (concat / "epoch-000014" / "model.pt").write_bytes(b"changed")
    failed = cli.invoke(app, [*args, "--out", str(tmp_path / "fresh")])
    assert failed.exit_code == 2 and "fingerprint mismatch" in failed.output
    assert not (tmp_path / "fresh").exists()


@pytest.mark.parametrize(("field", "value", "message"), [
    ("schema_version", True, "report.schema_version"),
    ("dataset", "other", "report.dataset"),
    ("samples", {}, "report.samples must be an array"),
    ("samples", [{"original_split": "train", "sample_id": "sample", "paths": {}}], "report.samples[0].hashes"),
    ("groups", None, "report.groups must be an array"),
    ("groups", [{"group_id": "g", "method": "source_day", "members": [{}]}], "report.groups[0].members"),
    ("cross_split_groups", {}, "report.cross_split_groups must be an array"),
    ("cross_split_groups", ["missing"], "report.cross_split_groups must list"),
])
def test_leakage_report_rejects_invalid_fields(field: str, value: object, message: str) -> None:
    report = {"schema_version": 1, "dataset": "aihub-smt-71900", "samples": [], "groups": [],
              "cross_split_groups": [], field: value}
    raw = json.dumps({"fingerprint": digest_json(report), "report": report}).encode()
    with pytest.raises(ValueError) as exc_info:
        analysis._load_leakage_report(raw)
    assert message in str(exc_info.value)


@pytest.mark.parametrize(("raw", "message"), [
    (b"{", "valid JSON"),
    (b"[]", "JSON object"),
    (b'{"report": {}}', "fingerprint"),
    (json.dumps({"fingerprint": "a" * 64, "report": []}).encode(), "report must be an object"),
])
def test_leakage_report_rejects_invalid_wrapper(raw: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        analysis._load_leakage_report(raw)


@pytest.mark.parametrize("failure", ["count", "fingerprint", "validation_ids"])
def test_analysis_reports_split_mismatch(
    tmp_path: Path, profile: ProductTrainingConfig, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    path = _run(profile, tmp_path / "image", "image", 30)
    cohort = Cohort.load(profile.cohort)
    frozen_path = profile.validation.cohort
    frozen = read_artifact(frozen_path, FrozenValidation)

    if failure == "validation_ids":
        def mismatched_views(
            config: ProductTrainingConfig, scoped: Cohort, state: PreprocessingState, selection: FrozenValidation,
        ) -> tuple[analysis.MultimodalDataset, analysis.MultimodalDataset, dict[str, object]]:
            train, validation, counts = _views(config, scoped, state, selection)
            assert validation is not None
            validation.records = validation.records[:-1]
            return train, validation, counts

        monkeypatch.setattr(analysis, "_views", mismatched_views)
        message = "image: reconstructed Validation sample IDs differ from FrozenValidation"
    else:
        metadata = path / "run.json"
        saved = json.loads(metadata.read_bytes())
        if failure == "count":
            saved["samples"]["validation"]["candidates"] += 1
            split = "validation"
        else:
            saved["samples"]["train"]["sample_fingerprint"] = "0" * 64
            split = "train"
        metadata.write_text(json.dumps(saved), encoding="utf-8")
        message = f"image: reconstructed sample counts or sample fingerprints differ from run.json samples for {split}"

    with pytest.raises(ValueError, match=message):
        analysis._inspect_run("image", path, profile.source, profile.cohort, frozen_path, cohort, frozen)


@pytest.fixture
def group_analysis(tmp_path: Path) -> tuple[list[str], Path, Path, set[Path]]:
    source, bundle = tmp_path / "source", tmp_path / "bundle"
    manifest = tmp_path / "provider.csv"
    rows = [make_sample(source, i, normal=i % 2 == 0, values=(5 * i, 5 * i + 1, 5 * i + 2))
            for i in range(1, 7)]
    for i, row in enumerate(rows, 1):
        color = tuple(255 if i & bit else 0 for bit in (1, 2, 4))
        Image.new("RGB", (24, 18), color).save(source / row["image_path"])
    write_csv(manifest, rows, list(rows[0]))
    audit = analyze_leakage(source, manifest, source, AnalysisConfig(verify_xlsx=True))
    audit_path = tmp_path / "sensor-integrity-analysis" / "leakage.json"
    audit_path.parent.mkdir()
    write_json(audit_path, {"fingerprint": digest_json(audit), "report": audit})
    plan = plan_split(audit, SplitConfig(val_fraction=0.25, test_fraction=0.25,
                                       group_near_images=True, group_near_sensors=True))
    publish_split(source, manifest, source, audit, plan, bundle, export_detection=True,
                  allow_missing_classes=True, progress=lambda _: lambda *_: None)
    revision_path = bundle / "split.json"
    cohort = audit_cohort(source, bundle / "manifest.csv", revision_path=revision_path).for_product_training()
    index = tmp_path / "training-cohort.json"
    write_json(index, {"cohort": cohort.model_dump(mode="json"), "fingerprint": cohort.fingerprint})
    assert any(row.split == "validation" and row.provider_split == "train" for row in cohort.records)
    frozen = FrozenValidation.from_audit(cohort, tuple(
        row.sample_id for row in cohort.records if row.split == "validation"
    ))
    frozen_path = tmp_path / "frozen.json"
    write_artifact(frozen_path, frozen)
    state = fit_preprocessor(cohort, PreprocessingConfig())
    state_path = tmp_path / "state.json"
    write_json(state_path, state.payload())
    prior = tmp_path / "prior.csv"
    heldout = [row for row in read_csv(bundle / "manifest.csv") if row["split"] == "test"]
    write_csv(prior, heldout, list(heldout[0]))
    verified = verify_group_retraining(
        GroupInputs(source, bundle, cohort.revision_id, index, cohort.fingerprint),
        frozen, state, Credentials(), lambda _: lambda *_: None, prior_manifests=(prior,),
    )
    assert verified["passed"], verified["blockers"]
    verification = tmp_path / "verification.json"
    fingerprint = digest_json(verified)
    write_json(verification, {"verification": verified, "fingerprint": fingerprint})
    detector = tmp_path / "detector" / "best.pt"
    detector.parent.mkdir()
    detector.write_bytes(b"synthetic original Detector")
    write_json(detector.parent / "run.json", {
        "status": "completed", "model": "faster_rcnn", "architecture": LAYOUTS["faster_rcnn"].architecture,
        "dataset_revision": {"group_split_revision": cohort.revision_id, "group_verification": fingerprint,
                             "detector_scope": verified["detector_scope"]},
        "initialization_mode": "scratch", "configuration": {"image_size": 640},
        "checkpoints": {"best": {"path": detector.name, "fingerprint": fingerprint_path(detector)}},
    })
    config = ProductTrainingConfig.model_validate({
        "source": source, "bundle": bundle, "cohort": index, "cohort_fingerprint": cohort.fingerprint,
        "revision_id": cohort.revision_id, "preprocessing": state_path, "group_verification": verification,
        "detector_initialization": {"path": detector, "sha256": file_sha(detector)},
        "branch": "paired", "epochs": 30,
        "architecture": {"preprocessing_fingerprint": state.fingerprint,
                         "image": {"kind": "faster_rcnn_fpn", "detector": {"image_size": 640}}},
        "validation": {"cohort": frozen_path, "fingerprint": frozen.identity.cohort_fingerprint},
    })
    concat = _run(config, tmp_path / "concat", "concat", 14)
    image = _run(config, tmp_path / "image", "image", 30)
    out = tmp_path / "analysis"
    args = ["multimodal", "analyze", "--root", str(source), "--audit", str(audit_path), "--cohort", str(index),
            "--frozen-validation", str(frozen_path), "--concat-run", str(concat), "--image-run", str(image),
            "--out", str(out), "--group-verification", str(verification), "--group-bundle", str(bundle),
            "--detector-initialization", str(detector), "--prior-test-manifest", str(prior)]
    reserved = {source / row[key] for row in heldout for key in ("image_path", "sensor_path", "annotation_path")}
    return args, out, concat / "run.json", reserved


@pytest.mark.parametrize(("mode", "finalized"), [("ablation", False), ("groups", False), ("ablation", True)])
def test_group_analysis_keeps_test_payloads_closed(
    group_analysis: tuple[list[str], Path, Path, set[Path]], monkeypatch: pytest.MonkeyPatch,
    mode: str, finalized: bool,
) -> None:
    args, out, concat_metadata, reserved = group_analysis
    if finalized:
        original_best = Path(args[args.index("--detector-initialization") + 1])
        best = out.parent / "finalized" / "best.pt"
        best.parent.mkdir()
        best.write_bytes(original_best.read_bytes())
        group = json.loads(Path(args[args.index("--group-verification") + 1]).read_bytes())["verification"]
        revision = json.loads((original_best.parent / "run.json").read_bytes())["dataset_revision"]
        record = {
            "schema_version": 1, "status": detector_finalization.STATUS,
            "reason": detector_finalization.REASON, "model": "faster_rcnn",
            "architecture": detector_finalization.ARCHITECTURE,
            "source": {"status": "failed", "error": detector_finalization.FAILURE,
                       "run_fingerprint": "a" * 64, "run_json_sha256": "b" * 64},
            "dataset_revision": revision,
            **{key: group[key] for key in ("publication_fingerprint", "cohort_fingerprint", "split_fingerprint")},
            "detector_layout": "scratch", "image_size": 640, "planned_epochs": 3,
            "best_epoch": 1, "completed_epoch": 2, "best_val_loss": 0.1,
            "checkpoints": {
                "best": {"path": best.name, "sha256": file_sha(best), "fingerprint": fingerprint_path(best),
                         "epoch": 1, "best_epoch": 1, "val_loss": 0.1, "best_loss": 0.1},
                "last": {"epoch": 2, "best_epoch": 1, "best_loss": 0.1},
            },
            "checks": dict.fromkeys(("group_inputs", "checkpoint_fingerprints", "checkpoint_epochs_and_losses",
                                     "best_model_layout_and_finiteness"), "passed") | {
                "test_payloads_opened": False, "validation_inference": {"status": "not_run", "split": "val"},
            },
        }
        fingerprint = digest_json(record)
        write_json(best.parent / detector_finalization.RECEIPT, {"finalization": record, "fingerprint": fingerprint})
        args[args.index("--detector-initialization") + 1] = str(best)
        saved = json.loads(concat_metadata.read_bytes())
        saved["detector_initialization"]["finalization_fingerprint"] = fingerprint
        concat_metadata.write_text(json.dumps(saved), encoding="utf-8")
    original = Path.open

    def guard(path: Path, *args: Any, **kwargs: Any) -> IO[Any]:
        assert path.resolve() not in reserved
        assert not (path.is_relative_to(out.parent / "bundle" / "detection")
                    and ("test" in path.parts or path.name == "instances_test.json"))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guard)

    def scores(
        run: analysis.AnalysisRun, device: str, batch_size: int, donors: dict[str, dict[str, str]],
        progress: analysis.ProgressFactory, checkpoint_dir: Path | None,
    ) -> tuple[dict[str, analysis.Scores], dict[str, str]]:
        assert all(row.split == "validation" and row.provider_split == "train" for row in run.validation.records)
        values = {run.validation[i].sample_id: (0.8, 0.1, 0.1, 0.1, 0.1) for i in range(len(run.validation))}
        return {key: values for key in ("original", *donors)}, {"sensor_removed": "untrained"} if donors else {}

    monkeypatch.setattr(analysis, "score_run", scores)
    result = CliRunner().invoke(app, [*args, "--mode", mode])
    assert result.exit_code == 0, (result.output, result.exception)
    config = json.loads((out / "analysis.json").read_bytes())
    assert config["audit_source_identity_split"] == "provider_split"
    assert config["models"]["concat"]["role_split"] == "verified_group_revision"
    projected = json.loads((out / "leakage.json").read_bytes())["models"]["concat"]
    assert projected["methods"]["source_day"]["validation_ids"] == projected["validation_ids"]
    if finalized:
        assert config["models"]["concat"]["detector_initialization"]["finalization_fingerprint"] == fingerprint
    results = json.loads((out / f"{mode}.json").read_bytes())["results"]
    if mode == "ablation":
        assert {row["group"]: row["status"] for row in results} == {
            "original": "measured", "within_manufacturer_shuffle": "measured",
            "global_shuffle": "measured", "sensor_removed": "skipped",
        }
        failure = CliRunner().invoke(app, [*args, "--mode", mode, "--out", str(out.parent / "new"),
                                          "--audit", str(out / "leakage.json")])
        assert failure.exit_code == 2 and "leakage.json fingerprint" in failure.output
    else:
        assert {row["model"] for row in results} == {"image", "concat"}


@pytest.mark.parametrize(("failure", "message"), [
    ("receipt", "Group Verification Receipt fingerprint"),
    ("revision", "Group Split Revision"),
    ("cohort", "run profile, cohort or FrozenValidation"),
    ("train", "sample fingerprints differ from run.json samples for train"),
    ("validation", "sample fingerprints differ from run.json samples for validation"),
    ("frozen", "FrozenValidation differs"),
    ("prior", "Prior Test custody conflict"),
    ("missing_prior", "exact prior Test reservation manifests"),
    ("audit_path", "Leakage audit source differs"),
    ("audit_hash", "Leakage audit source differs"),
    ("detector", "Detector initialization provenance differs"),
    ("unverified", "group analysis requires matching run provenance"),
    ("missing_receipt", "group analysis requires matching run provenance"),
    ("provider_guard", "split differs from the original provider split"),
])
def test_group_analysis_rejects_inconsistent_inputs(
    group_analysis: tuple[list[str], Path, Path, set[Path]], failure: str, message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, out, run, _ = group_analysis
    saved = json.loads(run.read_bytes())
    if failure == "receipt":
        saved["profile"]["group_verification"] = "0" * 64
    elif failure == "revision":
        saved["profile"]["revision_id"] = "0" * 64
    elif failure == "cohort":
        saved["profile"]["cohort_fingerprint"] = "0" * 64
    elif failure in ("train", "validation"):
        saved["samples"][failure]["sample_fingerprint"] = "0" * 64
    elif failure == "frozen":
        path = Path(args[args.index("--frozen-validation") + 1])
        cohort = Cohort.load(Path(args[args.index("--cohort") + 1]))
        frozen = read_artifact(path, FrozenValidation)
        changed = FrozenValidation.from_audit(cohort, frozen.identity.sample_ids[:1])
        path = path.with_name("changed-frozen.json")
        write_artifact(path, changed)
        args[args.index("--frozen-validation") + 1] = str(path)
        saved["profile"]["validation"]["fingerprint"] = changed.identity.cohort_fingerprint
        saved["validation_identity"] = changed.identity.model_dump(mode="json")
    elif failure == "prior":
        path = Path(args[args.index("--prior-test-manifest") + 1])
        bundle = Path(args[args.index("--group-bundle") + 1])
        row = next(row for row in read_csv(bundle / "manifest.csv") if row["split"] == "train")
        path = path.with_name("changed-prior.csv")
        write_csv(path, [{**row, "split": "test"}], list(row))
        args[args.index("--prior-test-manifest") + 1] = str(path)
    elif failure == "missing_prior":
        args = args[:args.index("--prior-test-manifest")]
    elif failure in ("audit_path", "audit_hash"):
        path = Path(args[args.index("--audit") + 1])
        report = json.loads(path.read_bytes())["report"]
        ids = set(saved["validation_identity"]["sample_ids"])
        row = next(row for row in report["samples"] if row["sample_id"] in ids)
        row["paths" if failure == "audit_path" else "hashes"]["image_path"] = "changed"
        path = path.with_name("changed-audit.json")
        write_json(path, {"report": report, "fingerprint": digest_json(report)})
        args[args.index("--audit") + 1] = str(path)
    elif failure == "detector":
        saved["detector_initialization"]["sha256"] = "0" * 64
    elif failure == "missing_receipt":
        args = args[:args.index("--group-verification")]
    else:
        del saved["profile"]["group_verification"]
        if failure == "provider_guard":
            args = args[:args.index("--group-verification")]
    run.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(analysis, "score_run", lambda *_: pytest.fail("Invalid inputs must fail before scoring"))
    result = CliRunner().invoke(app, [*args, "--mode", "ablation"])
    assert result.exit_code == 2 and message in result.output, (result.output, result.exception)
    assert not out.exists()


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
