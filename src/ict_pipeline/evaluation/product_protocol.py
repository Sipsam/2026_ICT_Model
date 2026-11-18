"""Frozen Validation inputs and typed experiment reports for SMT product inference."""

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, model_validator

from ..inspection.contracts import Contract, Modality, Probability, ProductModelIdentity, ProductTargetPolicy, SchemaVersion
from ..multimodal.audit import Cohort
from ..multimodal.contracts import Target
from ..multimodal.inference_contracts import SourceID, capture_instant, causal_sequence, sequence_fingerprint
from ..multimodal.model_config import FusionConfig
from ..pipeline.store import Fingerprint, digest_json
from .contracts import CLASSES, EvaluationConfig


class ValidationSample(Contract):
    sample_id: SourceID
    image_sha256: Fingerprint
    image_size: tuple[int, int]
    sensor_fingerprint: Fingerprint
    capture_timestamp: str
    annotation_sha256: Fingerprint | None
    targets: tuple[Target, ...] | None
    missing_classes: tuple[int, ...] = ()

    @model_validator(mode="after")
    def check_annotations(self) -> Self:
        width, height = self.image_size
        if (min(width, height) <= 0
                or capture_instant(self.capture_timestamp).isoformat() != self.capture_timestamp
                or tuple(sorted(set(self.missing_classes))) != self.missing_classes
                or set(self.missing_classes) - CLASSES.keys()
                or (self.targets is not None and self.annotation_sha256 is None)):
            raise ValueError("Invalid Validation sample metadata.")
        for target in self.targets or ():
            x, y, w, h = target.box
            if x + w > width + 1e-6 or y + h > height + 1e-6 or target.category in self.missing_classes:
                raise ValueError("Validation targets must be known classes in original-image coordinates.")
        return self

    def labels(self, policy: ProductTargetPolicy) -> tuple[bool | None, ...]:
        present = {target.category for target in self.targets or ()}
        by_class = tuple(
            None if self.targets is None or category in self.missing_classes else category in present
            for category in CLASSES
        )
        if policy == "smt-view-class-presence-v1":
            return by_class
        return (True if any(by_class) else None if None in by_class else False,)


class ValidationIdentity(Contract):
    split: Literal["validation"] = "validation"
    dataset_fingerprint: Fingerprint
    split_fingerprint: Fingerprint
    annotation_fingerprint: Fingerprint
    cohort_fingerprint: Fingerprint
    sample_ids: tuple[SourceID, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ids(self) -> Self:
        if tuple(sorted(set(self.sample_ids))) != self.sample_ids:
            raise ValueError("The frozen cohort requires unique sample IDs in canonical order.")
        return self


class FrozenValidation(Contract):
    schema_version: SchemaVersion = 1
    split: Literal["validation"] = "validation"
    dataset_fingerprint: Fingerprint
    split_fingerprint: Fingerprint
    samples: tuple[ValidationSample, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def check_ids(self) -> Self:
        _ = self.identity
        return self

    @property
    def identity(self) -> ValidationIdentity:
        annotations = [
            [row.sample_id, row.annotation_sha256, row.missing_classes,
             None if row.targets is None else [target.model_dump(mode="json") for target in row.targets]]
            for row in self.samples
        ]
        return ValidationIdentity(
            dataset_fingerprint=self.dataset_fingerprint, split_fingerprint=self.split_fingerprint,
            annotation_fingerprint=digest_json(annotations), cohort_fingerprint=digest_json(self.model_dump(mode="json")),
            sample_ids=tuple(sample.sample_id for sample in self.samples),
        )

    @classmethod
    def from_audit(cls, cohort: Cohort, sample_ids: tuple[str, ...]) -> Self:
        """Freeze explicitly selected, eligible and verified paired Validation observations."""
        selected = set(sample_ids)
        if not selected or len(selected) != len(sample_ids):
            raise ValueError("Choose distinct Validation sample IDs before examining experiment results.")
        samples = []
        for row in cohort.records:
            if row.sample_id not in selected:
                continue
            if (row.split != "validation" or row.quarantined or not row.detection_candidate
                    or not row.image_available or row.image_sha256 is None or row.image_size is None
                    or row.alignment != "matched" or row.sequence is None or row.capture_timestamp is None):
                raise ValueError("The common cohort requires eligible, verified paired Validation samples.")
            capture = capture_instant(row.capture_timestamp)
            sequence = causal_sequence(row.sequence, capture)
            if sequence is None or not sequence.available:
                raise ValueError("Common cohort samples require observed sensor evidence at capture.")
            samples.append(ValidationSample(
                sample_id=row.sample_id, image_sha256=row.image_sha256, image_size=row.image_size,
                sensor_fingerprint=sequence_fingerprint(sequence), capture_timestamp=capture.isoformat(),
                annotation_sha256=row.annotation_sha256, targets=row.targets,
            ))
        if {sample.sample_id for sample in samples} != selected:
            raise ValueError("Every selected Validation sample ID must exist in the audited cohort.")
        return cls(dataset_fingerprint=cohort.input_fingerprint, split_fingerprint=cohort.split_fingerprint,
                   samples=tuple(sorted(samples, key=lambda sample: sample.sample_id)))


class ProductProtocol(Contract):
    task: Literal["product"] = "product"
    cohort: ValidationIdentity
    target_policy: ProductTargetPolicy
    preprocessing_fingerprint: Fingerprint
    image_preprocessing_fingerprint: Fingerprint
    threshold: Probability = 0.5
    missing_modality_policy: Literal["require_requested_branch-v1"] = "require_requested_branch-v1"
    alignment_policy: Literal["verified_receipt_at_capture-v1"] = "verified_receipt_at_capture-v1"


class DetectionProtocol(Contract):
    task: Literal["detection"] = "detection"
    cohort: ValidationIdentity
    evaluation: EvaluationConfig = EvaluationConfig()


Protocol = Annotated[ProductProtocol | DetectionProtocol, Field(discriminator="task")]


class DetectorExperimentModel(Contract):
    model: Literal["yolo", "faster_rcnn", "rt_detr"]
    checkpoint_fingerprint: Fingerprint
    checkpoint_format: SourceID
    run_metadata_fingerprint: Fingerprint
    # The existing detector adapter owns the model-specific resize and postprocessing settings.
    adapter_version: SourceID
    training_seed: int = Field(ge=0)
    weight_status: Literal["trusted_trained"]
    libraries: dict[str, str] = Field(default_factory=dict)


class ClassificationMetric(Contract):
    name: str
    missing_labels: int = Field(ge=0)
    # Rows are true [negative, positive]; columns are predicted [negative, positive].
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    precision: Probability | None
    recall: Probability | None
    f1: Probability | None
    auroc: Probability | None


class ExperimentReport(Contract):
    schema_version: SchemaVersion = 1
    experiment_id: SourceID
    protocol: Protocol
    status: Literal["pending", "simulated", "untrained", "measured", "unscorable"]
    branch: Modality | None = None
    product_model: ProductModelIdentity | None = None
    fusion: FusionConfig | None = None
    detector_model: DetectorExperimentModel | None = None
    prediction_fingerprint: Fingerprint | None = None
    classification_metrics: tuple[ClassificationMetric, ...] | None = None
    detection_metrics: dict[str, JsonValue] | None = None
    code_fingerprint: Fingerprint
    runtime: dict[str, str]

    @model_validator(mode="after")
    def check_kind(self) -> Self:
        product = isinstance(self.protocol, ProductProtocol)
        measured = self.status == "measured"
        if (product != (self.branch is not None)
                or (product and self.detector_model is not None)
                or (not product and (self.product_model is not None or self.fusion is not None))
                or (measured and product) != (self.classification_metrics is not None)
                or (measured and not product) != (self.detection_metrics is not None)
                or (self.status == "pending") != (self.prediction_fingerprint is None)
                or (measured and not product and self.detector_model is None)
                or (self.status in ("measured", "untrained", "unscorable") and product and self.product_model is None)
                or (self.status == "simulated" and (not product or self.product_model is not None))
                or (self.status in ("untrained", "unscorable") and not product)):
            raise ValueError("Experiment status, model kind and metrics are inconsistent.")
        if product and self.product_model is not None:
            if (self.fusion is None or self.fusion.threshold != self.protocol.threshold
                    or self.fusion.kind != self.product_model.fusion
                    or digest_json(self.fusion.model_dump(mode="json")) != self.product_model.fusion_fingerprint
                    or self.product_model.image_preprocessing_fingerprint != self.protocol.image_preprocessing_fingerprint
                    or self.product_model.preprocessing_fingerprint != self.protocol.preprocessing_fingerprint):
                raise ValueError("Product experiments require matching frozen preprocessing, fusion and threshold.")
        elif self.fusion is not None:
            raise ValueError("Fusion metadata requires a product checkpoint.")
        return self
