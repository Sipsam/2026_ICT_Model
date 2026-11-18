"""Public version 1 contracts for inspection requests and records."""

import json
from pathlib import Path
from typing import Annotated, Literal, Self, get_args

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, JsonValue, model_validator

from ..data.preprocessing.policy import POLICY_VERSION, SENSOR_FIELDS, TARGET_IDS
from ..evaluation.contracts import CLASSES
from ..models.contracts import ModelName
from ..pipeline.store import Fingerprint

def _check_identifier(value: str) -> str:
    if not value.isprintable() or value != value.strip():
        raise ValueError("Use a printable identifier without leading or trailing whitespace.")
    return value


Identifier = Annotated[str, Field(min_length=1, max_length=256), AfterValidator(_check_identifier)]
Probability = Annotated[float, Field(ge=0, le=1)]
Outcome = Literal["pass", "defect_detected", "review_required", "inference_failed"]
Failure = Literal["invalid_request", "missing_image", "invalid_image", "dimension_mismatch", "prediction_failed"]
Reason = Literal[Failure, "verified_good", "detections_retained", "no_detections", "simulated_prediction"]
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

    @model_validator(mode="after")
    def check_mode(self) -> Self:
        fields = (self.model, self.checkpoint_kind, self.checkpoint_fingerprint,
                  self.run_metadata_fingerprint, self.device, self.image_size)
        if self.mode == "real" and (None in fields or self.mock_scenario is not None):
            raise ValueError("Real inference requires checkpoint and runtime provenance.")
        if self.mode == "mock" and (any(value is not None for value in fields)
                                    or self.libraries or self.mock_scenario is None):
            raise ValueError("Mock inference requires a scenario and carries no real model provenance.")
        return self


class InspectionResult(Contract):
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

    @model_validator(mode="after")
    def check_decision(self) -> Self:
        if self.outcome == "inference_failed":
            valid = self.reason in FAILURES and not self.detections
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
        if self.outcome != "inference_failed" and (self.image is None or not self.request_id or not self.source_id):
            raise ValueError("Successful inspection requires a decoded image and source identifiers.")
        if self.image and any(
            item.bbox_xyxy[2] > self.image.width or item.bbox_xyxy[3] > self.image.height
            or item.confidence < self.thresholds.confidence for item in self.detections
        ):
            raise ValueError("Detections must be within the source image and meet the confidence threshold.")
        json.dumps(self.metadata, allow_nan=False)
        return self
