import json
from dataclasses import asdict
from pathlib import Path

import pytest
from pydantic import ValidationError

from ict_pipeline.evaluation import runner
from ict_pipeline.evaluation.contracts import CLASSES, EvaluationError, EvaluationSplit, Prediction, Preprocessing
from ict_pipeline.evaluation.product import (
    compare_experiments, evaluate_detector, evaluate_detector_artifact, evaluate_products, pending_experiment,
    read_artifact, write_artifact,
)
from ict_pipeline.evaluation.product_protocol import (
    DetectionProtocol, DetectorExperimentModel, FrozenValidation, ProductProtocol,
)
from ict_pipeline.inspection.contracts import (
    ImageSource, InspectionResult, InspectionThresholds, Modality, ModelProvenance, ProductInspection,
    ProductModelIdentity, ProductScoreRecord,
)
from ict_pipeline.inspection.service import InspectionService
from ict_pipeline.config.settings import Credentials
from ict_pipeline.models.contracts import ModelName
from ict_pipeline.multimodal.audit import Cohort
from ict_pipeline.multimodal.contracts import Target
from ict_pipeline.multimodal.model_config import FusionConfig
from ict_pipeline.pipeline.store import digest_json, fingerprint_path

from test_evaluation import write_run, write_split
from test_product_inspection import FINGERPRINT, POLICY, InputFixture, fixture_inputs, fixture_model  # noqa: F401


@pytest.fixture
def cohort(inputs: InputFixture) -> FrozenValidation:
    records = tuple(inputs.record.model_copy(update={
        "sample_id": f"part-{index}", "image_sha256": str(index) * 64, "targets": truth,
    }) for index, truth in enumerate(((Target(category=1, box=(1., 2., 4., 5.)),), (), None, ()), 1))
    audit = Cohort(
        manifest_fingerprint=FINGERPRINT, input_fingerprint="b" * 64, split_fingerprint="c" * 64,
        implementation_fingerprint="d" * 64, revision_id=None, records=records,
    )
    with pytest.raises(ValueError, match="Validation"):
        FrozenValidation.from_audit(audit.model_copy(update={
            "records": (records[0].model_copy(update={"split": "test"}), *records[1:]),
        }), ("part-1",))
    return FrozenValidation.from_audit(audit, ("part-4", "part-2", "part-1", "part-3"))


def synthetic_scoring_records(
    cohort: FrozenValidation, branch: Modality, *, weight_status: str = "trusted_trained",
) -> tuple[InspectionResult, ...]:
    """Synthetic metric inputs with the same shape as a trusted branch's exported scores."""
    model = ProductModelIdentity(
        checkpoint_format="ict-multimodal-state-v1", checkpoint_sha256=FINGERPRINT,
        architecture_version=1, architecture_fingerprint=FINGERPRINT,
        image_encoder="rgb_grid", image_encoder_version=1, sensor_encoder="mlp", sensor_encoder_version=1,
        image_preprocessing_fingerprint=FINGERPRINT,
        preprocessing_version=1, preprocessing_fingerprint=FINGERPRINT, window_policy="at-or-before-capture-v1",
        fusion="late", fusion_version=1, fusion_fingerprint=digest_json(FusionConfig().model_dump(mode="json")),
        device="cpu", seed=42,
    )
    return tuple(InspectionResult(
        request_id=sample.sample_id, source_id="fixture",
        image=(ImageSource(width=sample.image_size[0], height=sample.image_size[1], format="PNG",
                           sha256=sample.image_sha256) if branch != "sensor" else None),
        sensor_source_id="sensor-fixture" if branch != "image" else None,
        provenance=ModelProvenance(mode="real", product_only=True), thresholds=InspectionThresholds(),
        outcome="review_required", reason="product_review",
        multimodal=ProductInspection(
            sample_id=sample.sample_id, input_mode=branch, branch=branch, target_policy=POLICY,
            score_kind="uncalibrated_sigmoid", weight_status=weight_status, threshold=0.5,
            scores=(ProductScoreRecord(name="any_target_defect", class_id=None, score=score,
                                       threshold_crossed=score >= 0.5),),
            sensor_status="available" if branch != "image" else "absent",
            sensor_fingerprint=sample.sensor_fingerprint if branch != "image" else None,
            capture_timestamp=sample.capture_timestamp,
            alignment="verified_receipt" if branch == "paired" else "not_applicable",
            pairing_fingerprint=FINGERPRINT if branch == "paired" else None,
            model=model,
        ),
    ) for sample, score in zip(cohort.samples, (0.8, 0.8, 0.99, 0.2), strict=True))


def test_frozen_comparison_metrics_and_serialization(tmp_path: Path, cohort: FrozenValidation) -> None:
    protocol = ProductProtocol(cohort=cohort.identity, target_policy=POLICY, preprocessing_fingerprint=FINGERPRINT,
                               image_preprocessing_fingerprint=FINGERPRINT)
    reports = [evaluate_products(branch, cohort, protocol, branch, reversed(synthetic_scoring_records(cohort, branch)),
                                 fusion=FusionConfig()) for branch in ("paired", "sensor", "image")]
    assert all(report.status == "measured" for report in reports)
    assert [row.experiment_id for row in compare_experiments(reports)] == ["image", "paired", "sensor"]
    for report in reports:
        metric, = report.classification_metrics
        assert metric.confusion_matrix == ((1, 1), (0, 1))
        assert metric.missing_labels == 1
        assert (metric.precision, metric.recall, metric.f1, metric.auroc) == (0.5, 1, 2 / 3, 0.75)
        assert report.detection_metrics is None

    pending = pending_experiment("pending", protocol, branch="paired")
    assert [item.status for item in compare_experiments((pending, reports[0]))] == ["measured", "pending"]
    assert pending.classification_metrics is None
    untrained = evaluate_products("untrained", cohort, protocol, "image",
                                  synthetic_scoring_records(cohort, "image", weight_status="untrained"),
                                  fusion=FusionConfig())
    assert untrained.status == "untrained" and untrained.classification_metrics is None
    for name, artifact in (("cohort", cohort), ("measured", reports[0]), ("pending", pending)):
        path = tmp_path / f"{name}.json"
        copy = tmp_path / f"{name}-copy.json"
        write_artifact(path, artifact)
        write_artifact(copy, artifact)
        assert read_artifact(path, type(artifact)) == artifact
        assert path.read_bytes() == copy.read_bytes()
        with pytest.raises(FileExistsError):
            write_artifact(path, artifact)
        with path.open("ab") as stream:
            stream.write(b"corrupt")
        with pytest.raises(ValueError, match="corrupt"):
            read_artifact(path, type(artifact))


def test_protocol_mismatches_are_rejected(cohort: FrozenValidation) -> None:
    protocol = ProductProtocol(cohort=cohort.identity, target_policy=POLICY, preprocessing_fingerprint=FINGERPRINT,
                               image_preprocessing_fingerprint=FINGERPRINT)
    records = synthetic_scoring_records(cohort, "paired")
    for override in (
        {"preprocessing_fingerprint": "d" * 64}, {"target_policy": "smt-view-class-presence-v1"},
        {"image_preprocessing_fingerprint": "d" * 64},
        {"threshold": 0.6}, {"cohort": cohort.identity.model_copy(update={"split_fingerprint": "d" * 64})},
        {"cohort": cohort.identity.model_copy(update={"annotation_fingerprint": "d" * 64})},
    ):
        with pytest.raises(ValueError):
            evaluate_products("rejected", cohort, protocol.model_copy(update=override), "paired", records,
                              fusion=FusionConfig())
        different = pending_experiment("different", protocol.model_copy(update=override), branch="paired")
        with pytest.raises(ValueError, match="matching"):
            compare_experiments((pending_experiment("reference", protocol, branch="image"), different))
    for invalid in (records[:-1], (*records, records[0]), synthetic_scoring_records(cohort, "image")):
        with pytest.raises(ValueError):
            evaluate_products("rejected", cohort, protocol, "paired", invalid, fusion=FusionConfig())
    unverifiable = records[0].model_copy(update={
        "image": records[0].image.model_copy(update={"sha256": "e" * 64}),
    })
    with pytest.raises(ValueError, match="evidence"):
        evaluate_products("rejected", cohort, protocol, "paired", (unverifiable, *records[1:]), fusion=FusionConfig())
    with pytest.raises(ValidationError, match="validation"):
        FrozenValidation.model_validate(cohort.model_dump() | {"split": "test"})


def test_missing_classes_labels_and_raw_detector_metrics(cohort: FrozenValidation) -> None:
    # Drop the unknown row for the detector and use a separate product cohort with only negatives.
    complete = cohort.model_copy(update={"samples": (cohort.samples[0], cohort.samples[1])})
    detector_protocol = DetectionProtocol(cohort=complete.identity)
    model = DetectorExperimentModel(
        model="faster_rcnn", checkpoint_fingerprint=FINGERPRINT, run_metadata_fingerprint=FINGERPRINT,
        checkpoint_format="ict-torch-state-v1", adapter_version="fixture-v1", training_seed=42,
        weight_status="trusted_trained",
    )
    preprocessing = Preprocessing((40, 24), (40, 24), "stretch")
    predictions = {
        "part-1": Prediction("part-1", ((1., 2., 5., 7.),), (1,), (0.9,), preprocessing),
        "part-2": Prediction("part-2", (), (), (), preprocessing),
    }
    detector = evaluate_detector("detector", complete, detector_protocol, model=model,
                                 input_cohort=complete.identity, predictions=predictions)
    assert detector.detection_metrics["map50_95"] == 1
    assert detector.detection_metrics["per_class"][1]["ap50"] is None
    assert detector.classification_metrics is None
    with pytest.raises(ValueError, match="provenance"):
        evaluate_detector("wrong", complete, detector_protocol, model=model,
                          input_cohort=cohort.identity, predictions=predictions)
    with pytest.raises(ValueError, match="complete"):
        evaluate_detector("missing", cohort, DetectionProtocol(cohort=cohort.identity), model=model,
                          input_cohort=cohort.identity, predictions=predictions)
    negatives = cohort.model_copy(update={"samples": (cohort.samples[1], cohort.samples[3])})
    protocol = ProductProtocol(cohort=negatives.identity, target_policy=POLICY, preprocessing_fingerprint=FINGERPRINT,
                               image_preprocessing_fingerprint=FINGERPRINT)
    negative_rows = tuple(row for row in synthetic_scoring_records(cohort, "sensor")
                          if row.multimodal.sample_id in negatives.identity.sample_ids)
    report = evaluate_products("negative", negatives, protocol, "sensor", negative_rows, fusion=FusionConfig())
    metric, = report.classification_metrics
    assert (metric.precision, metric.recall, metric.f1, metric.auroc) == (0.0, None, 0.0, None)
    with pytest.raises(ValueError, match="matching"):
        compare_experiments((report, detector))
    assert cohort.samples[2].labels("smt-view-class-presence-v1") == (None,) * 5
    partial = cohort.samples[1].model_copy(update={"missing_classes": (2,)})
    assert partial.labels(POLICY) == (None,)
    assert partial.labels("smt-view-class-presence-v1") == (False, None, False, False, False)


def test_mock_predictions_never_produce_metrics(tmp_path: Path, inputs: InputFixture, cohort: FrozenValidation) -> None:
    service = InspectionService.from_product(fixture_model(tmp_path, inputs), tmp_path, credentials=Credentials())
    result = service.inspect_product(inputs.request)
    frozen = cohort.model_copy(update={"samples": (cohort.samples[0],)})
    protocol = ProductProtocol(cohort=frozen.identity, target_policy=POLICY, preprocessing_fingerprint=FINGERPRINT,
                               image_preprocessing_fingerprint=FINGERPRINT)
    report = evaluate_products("fixture", frozen, protocol, "sensor", (result,))
    assert report.status == "simulated" and report.classification_metrics is None


def test_per_class_presence_and_unscorable_labels(cohort: FrozenValidation) -> None:
    policy = "smt-view-class-presence-v1"
    records = []
    for sample, source in zip(cohort.samples, synthetic_scoring_records(cohort, "image"), strict=True):
        prediction = source.multimodal.model_dump(mode="json")
        prediction.update(target_policy=policy, scores=[
            {"class_id": category, "name": name, "score": 0.9 if truth else 0.1, "threshold_crossed": bool(truth)}
            for (category, name), truth in zip(CLASSES.items(), sample.labels(policy), strict=True)
        ])
        records.append(source.model_copy(update={"multimodal": ProductInspection.model_validate_json(
            json.dumps(prediction),
        )}))
    protocol = ProductProtocol(cohort=cohort.identity, target_policy=policy, preprocessing_fingerprint=FINGERPRINT,
                               image_preprocessing_fingerprint=FINGERPRINT)
    report = evaluate_products("classes", cohort, protocol, "image", records, fusion=FusionConfig())
    assert [metric.name for metric in report.classification_metrics] == list(CLASSES.values())
    assert report.classification_metrics[0].auroc == 1
    assert report.classification_metrics[1].auroc is None
    assert all(metric.missing_labels == 1 for metric in report.classification_metrics)
    unknown = cohort.model_copy(update={"samples": (cohort.samples[2],)})
    result = evaluate_products("unknown", unknown, protocol.model_copy(update={"cohort": unknown.identity}),
                               "image", (records[2],), fusion=FusionConfig())
    assert result.status == "unscorable" and result.classification_metrics is None


def test_native_detector_validation_artifact_binds_bytes_labels_and_checkpoint(tmp_path: Path, cohort: FrozenValidation) -> None:
    root = tmp_path / "detection"
    write_split(root, "val")
    credentials = Credentials()

    def progress(stage: str, done: int, total: int) -> None:
        pass

    labels, hashes, annotation, _ = runner.inspect_split(root, "val", credentials, progress, labels_required=True)
    run = write_run(tmp_path, ModelName.FASTER_RCNN, {"val": digest_json(hashes)})
    checkpoint = runner.inspect_checkpoint(run, "best", credentials)
    samples = tuple(cohort.samples[0].model_copy(update={
        "sample_id": sample.image.stem, "image_sha256": hashes[sample.image.name],
        "image_size": (sample.width, sample.height),
        "targets": tuple(Target(category=obj.category, box=obj.box) for obj in sample.objects),
    }) for sample in labels.samples)
    frozen = cohort.model_copy(update={"samples": tuple(sorted(samples, key=lambda sample: sample.sample_id))})
    protocol = DetectionProtocol(cohort=frozen.identity)
    raw = tmp_path / "predictions.json"
    predictions = {
        sample.image.stem: Prediction(
            sample.image.stem, ((10., 20., 40., 60.),) if sample.objects else (),
            (1,) if sample.objects else (), (0.9,) if sample.objects else (),
            Preprocessing((sample.width, sample.height), (64, 64), "stretch"),
        ) for sample in labels.samples
    }
    artifact = {
        "schema_version": 1, "status": "completed", **runner.checkpoint_record(checkpoint, "best"),
        "dataset": runner.dataset_record(EvaluationSplit.VALIDATION, len(hashes), digest_json(hashes), annotation, {}),
        "run_metadata_fingerprint": fingerprint_path(run / "run.json"),
        "protocol": runner.evaluation_protocol(protocol.evaluation), "code_fingerprints": runner.evaluation_code(),
        "libraries": {"torch": "fixture"}, "postprocessing": {},
        "predictions": {sample_id: asdict(row) for sample_id, row in predictions.items()},
    }
    runner.write_result(raw, artifact)
    kwargs = {"artifact_path": raw, "detection_root": root, "run": run, "trust_run": True,
              "credentials": credentials, "progress": progress}
    report = evaluate_detector_artifact("detector", frozen, protocol, **kwargs)
    assert report.detection_metrics["map50"] == 1 and report.detector_model.checkpoint_fingerprint == checkpoint.fingerprint

    held_out = tmp_path / "held-out.json"
    runner.write_result(held_out, artifact | {"dataset": artifact["dataset"] | {"split": "test"}})
    with pytest.raises(EvaluationError, match="incompatible"):
        evaluate_detector_artifact("held-out", frozen, protocol, **(kwargs | {"artifact_path": held_out}))
    labels.samples[0].image.write_bytes(b"other bytes")
    with pytest.raises(EvaluationError, match="incompatible"):
        evaluate_detector_artifact("wrong-image", frozen, protocol, **kwargs)
