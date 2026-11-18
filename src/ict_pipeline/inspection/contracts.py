"""Public version 1 contracts for inspection requests and records."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal, Self, get_args

from pydantic import (
    AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, JsonValue, SerializerFunctionWrapHandler,
    model_serializer, model_validator,
)

from ..data.preprocessing.policy import POLICY_VERSION, SENSOR_FIELDS, TARGET_IDS
from ..evaluation.contracts import CLASSES
from ..models.contracts import ModelName
from ..pipeline.store import Fingerprint

MAX_IMAGE_BYTES = 64 * 1024**2
MAX_IMAGE_PIXELS = 25_000_000

def _check_identifier(value: str) -> str:
    if not value.isprintable() or value != value.strip():
        raise ValueError("Use a printable identifier without leading or trailing whitespace.")
    return value


Identifier = Annotated[str, Field(min_length=1, max_length=256), AfterValidator(_check_identifier)]
Probability = Annotated[float, Field(ge=0, le=1)]
Outcome = Literal["pass", "defect_detected", "review_required", "inference_failed"]
Failure = Literal["invalid_request", "missing_image", "invalid_image", "dimension_mismatch", "prediction_failed"]
Reason = Literal[Failure, "verified_good", "detections_retained", "no_detections", "simulated_prediction",
                 "product_review", "product_unavailable"]
FAILURES = frozenset(get_args(Failure))
Device = Literal["cpu", "cuda", "rocm", "mps"]
CheckpointKind = Literal["best", "last"]
MockScenario = Literal["empty", "defect"]


def _require_integer(value: object) -> int:
    if type(value) is not int:
        raise ValueError("Schema version must be an integer.")
    return value


SchemaVersion = Annotated[Literal[1], BeforeValidator(_require_integer)]


class InspectionError(Exception):
    """An actionable inspection error safe to display."""


class TransientInspectionError(InspectionError):
    """A predictor may raise this when it can safely retry an operation."""


class InspectionInterrupted(InspectionError):
    """The caller cancelled before entering the predictor."""


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, hide_input_in_errors=True, allow_inf_nan=False)


class ImageSize(Contract):
    width: int = Field(gt=0)
    height: int = Field(gt=0)


class ImageSource(ImageSize):
    format: Literal["JPEG", "PNG"]
    sha256: Fingerprint


class SensorInput(Contract):
    source_id: Identifier
    values: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def check_fields(self) -> Self:
        if self.values.keys() - set(SENSOR_FIELDS):
            raise ValueError("Use supported SMT measurement names.")
        return self


class InspectionInput(Contract):
    schema_version: SchemaVersion = 1
    request_id: Identifier
    source_id: Identifier
    expected_size: ImageSize | None = None
    sensor: SensorInput | None = None


class InspectionMetadataInput(InspectionInput):
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_metadata(self) -> Self:
        json.dumps(self.metadata, allow_nan=False)
        return self


class InspectionRequest(InspectionMetadataInput):
    image_path: Path


class Detection(Contract):
    class_id: int
    class_name: str
    source_category_id: int
    bbox_xyxy: tuple[float, float, float, float]
    confidence: Probability

    @model_validator(mode="after")
    def check_detection(self) -> Self:
        if CLASSES.get(self.class_id) != self.class_name or TARGET_IDS.get(self.source_category_id) != self.class_name:
            raise ValueError("Detection class identifiers must match the SMT label policy.")
        x1, y1, x2, y2 = self.bbox_xyxy
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("Detection boxes must be ordered with positive area.")
        return self


class InspectionThresholds(Contract):
    confidence: Probability = 0.25
    nms_iou: float = Field(default=0.7, gt=0, le=1)


class ModelProvenance(Contract):
    model_config = ConfigDict(json_schema_mode_override="validation")

    mode: Literal["real", "mock"]
    label_policy: Literal[POLICY_VERSION] = POLICY_VERSION
    model: ModelName | None = None
    checkpoint_kind: CheckpointKind | None = None
    checkpoint_fingerprint: Fingerprint | None = None
    run_metadata_fingerprint: Fingerprint | None = None
    device: str | None = None
    image_size: int | None = Field(default=None, ge=64, multiple_of=32)
    libraries: dict[str, str] = Field(default_factory=dict)
    mock_scenario: MockScenario | None = None
    product_only: Literal[True] | None = None

    @model_serializer(mode="wrap")
    def serialize_provenance(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.product_only is None:
            data.pop("product_only", None)
        return data

    @model_validator(mode="after")
    def check_mode(self) -> Self:
        fields = (self.model, self.checkpoint_kind, self.checkpoint_fingerprint,
                  self.run_metadata_fingerprint, self.device, self.image_size)
        if self.product_only is True:
            if (self.mode != "real" or self.mock_scenario is not None or self.libraries
                    or any(value is not None for value in fields)):
                raise ValueError("Product-only provenance carries no spatial detector identity.")
        elif self.mode == "real" and (None in fields or self.mock_scenario is not None):
            raise ValueError("Real inference requires checkpoint and runtime provenance.")
        if self.mode == "mock" and (any(value is not None for value in fields)
                                    or self.libraries or self.mock_scenario is None):
            raise ValueError("Mock inference requires a scenario and carries no real model provenance.")
        return self


Modality = Literal["image", "sensor", "paired"]
ProductTargetPolicy = Literal["smt-target-defect-presence-v1", "smt-view-class-presence-v1"]
SensorStatus = Literal["absent", "no_observations", "insufficient_steps", "available"]


class ProductScoreRecord(Contract):
    name: str
    class_id: int | None
    score: Probability
    threshold_crossed: bool


class ProductModelIdentity(Contract):
    checkpoint_format: Literal["ict-multimodal-state-v1"]
    checkpoint_sha256: Fingerprint
    architecture_version: SchemaVersion
    architecture_fingerprint: Fingerprint
    image_encoder: Literal["rgb_grid", "faster_rcnn_fpn"]
    image_encoder_version: SchemaVersion
    image_preprocessing_fingerprint: Fingerprint
    sensor_encoder: Literal["mlp", "gru"]
    sensor_encoder_version: SchemaVersion
    preprocessing_version: SchemaVersion
    preprocessing_fingerprint: Fingerprint
    window_policy: Literal["capture-through-decision-v1", "at-or-before-capture-v1"]
    fusion: Literal["late", "concat", "gated"]
    fusion_version: SchemaVersion
    fusion_fingerprint: Fingerprint
    device: Device
    seed: int = Field(ge=0, lt=2**63)
    libraries: dict[str, str] = Field(default_factory=dict)


class ProductInspection(Contract):
    schema_version: SchemaVersion = 1
    sample_id: Identifier
    input_mode: Modality
    branch: Modality | None
    target_policy: ProductTargetPolicy
    score_kind: Literal["uncalibrated_sigmoid", "simulated_fixture"]
    weight_status: Literal["simulated", "untrained", "trusted_trained", "unavailable"]
    threshold: Probability
    scores: tuple[ProductScoreRecord, ...] = ()
    sensor_status: SensorStatus
    capture_timestamp: str | None = None
    sensor_fingerprint: Fingerprint | None = None
    alignment: Literal["not_applicable", "verified_receipt"] = "not_applicable"
    pairing_fingerprint: Fingerprint | None = None
    missing_modality_policy: Literal["available_branch_only-v1"] = "available_branch_only-v1"
    model: ProductModelIdentity | None = None

    @model_validator(mode="after")
    def check_evidence(self) -> Self:
        image = self.input_mode != "sensor"
        sensor = self.sensor_status == "available"
        expected = "paired" if image and sensor else "image" if image else "sensor" if sensor else None
        paired = self.input_mode == "paired"
        has_sensor = self.input_mode != "image"
        simulated = self.score_kind == "simulated_fixture"
        labels = ((None, "any_target_defect"),) if self.target_policy == "smt-target-defect-presence-v1" else tuple(CLASSES.items())
        if self.capture_timestamp is not None:
            capture = datetime.fromisoformat(self.capture_timestamp)
            if capture.isoformat() != self.capture_timestamp or capture.utcoffset() not in (None, timedelta(0)):
                raise ValueError("Product evidence requires a canonical local or UTC capture timestamp.")
        if (self.branch != expected
                or has_sensor != (self.sensor_status != "absent")
                or has_sensor != (self.sensor_fingerprint is not None)
                or (has_sensor and self.capture_timestamp is None)
                or paired != (self.alignment == "verified_receipt")
                or paired != (self.pairing_fingerprint is not None)
                or simulated != (self.model is None)
                or (expected is None) != (self.weight_status == "unavailable")
                or (expected is not None and simulated != (self.weight_status == "simulated"))
                or tuple((score.class_id, score.name) for score in self.scores) != (labels if expected else ())
                or any(score.threshold_crossed != (score.score >= self.threshold) for score in self.scores)):
            raise ValueError("Product prediction, modality and provenance are inconsistent.")
        return self


class InspectionResult(Contract):
    model_config = ConfigDict(json_schema_mode_override="validation")

    schema_version: SchemaVersion = 1
    request_id: Identifier | None = None
    source_id: Identifier | None = None
    input_line: int | None = Field(default=None, gt=0)
    image: ImageSource | None = None
    sensor_source_id: Identifier | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: ModelProvenance
    thresholds: InspectionThresholds
    outcome: Outcome
    reason: Reason
    detections: tuple[Detection, ...] = ()
    verification_reference: Identifier | None = None
    multimodal: ProductInspection | None = None

    @model_serializer(mode="wrap")
    def serialize_result(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        if self.multimodal is None:
            data.pop("multimodal", None)
        return data

    @property
    def modality(self) -> Modality | None:
        return self.multimodal.input_mode if self.multimodal else "image" if self.image else None

    @model_validator(mode="after")
    def check_decision(self) -> Self:
        if self.outcome == "inference_failed":
            valid = self.reason in FAILURES and not self.detections and self.multimodal is None
        elif self.multimodal is not None:
            product = self.multimodal
            simulated = product.score_kind == "simulated_fixture"
            reason = ("simulated_prediction" if simulated else
                      "product_review" if product.branch else "product_unavailable")
            valid = (self.outcome == "review_required" and self.reason == reason
                     and simulated == (self.provenance.mode == "mock")
                     and (self.image is not None) == (product.input_mode != "sensor")
                     and (self.sensor_source_id is not None) == (product.input_mode != "image")
                     and (not self.detections or (self.image is not None and self.provenance.model is not None)))
        elif self.provenance.mode == "mock":
            valid = self.outcome == "review_required" and self.reason == "simulated_prediction"
        elif self.outcome == "pass":
            valid = self.reason == "verified_good" and not self.detections and bool(self.verification_reference)
        elif self.outcome == "defect_detected":
            valid = self.reason == "detections_retained" and bool(self.detections)
        else:
            valid = self.reason == "no_detections" and not self.detections
        if not valid or (self.outcome != "pass" and self.verification_reference is not None):
            raise ValueError("Outcome, reason and evidence are inconsistent.")
        if self.outcome != "inference_failed" and (
            not self.request_id or not self.source_id or (self.image is None and self.multimodal is None)
            or (self.provenance.mode == "real" and self.provenance.model is None and self.multimodal is None)
        ):
            raise ValueError("Successful inspection requires input evidence and source identifiers.")
        if self.image and any(
            item.bbox_xyxy[2] > self.image.width or item.bbox_xyxy[3] > self.image.height
            or item.confidence < self.thresholds.confidence for item in self.detections
        ):
            raise ValueError("Detections must be within the source image and meet the confidence threshold.")
        json.dumps(self.metadata, allow_nan=False)
        return self
