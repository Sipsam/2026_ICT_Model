"""Versioned architecture and training settings without a tensor dependency."""

from typing import Literal, Self

from pydantic import Field, model_validator

from ..data.preprocessing.policy import NAMES
from ..inspection.contracts import Contract, Modality, ProductTargetPolicy, SchemaVersion
from ..pipeline.store import Fingerprint, digest_json
from .contracts import CHANNELS, Channels

TargetPolicy = ProductTargetPolicy
Branch = Modality
TrainingStatus = Literal["untrained", "trained"]
CLASS_NAMES = tuple(NAMES)


class SensorEncoderConfig(Contract):
    schema_version: SchemaVersion = 1
    kind: Literal["mlp", "gru"] = "mlp"
    features: Literal["normalized_mean", "statistics"] = "normalized_mean"
    channels: Channels = CHANNELS
    hidden_dim: int = Field(default=32, gt=0)
    embedding_dim: int = Field(default=16, gt=0)
    min_steps: int = Field(default=2, ge=2)

    @model_validator(mode="after")
    def check_features(self) -> Self:
        if self.channels != CHANNELS:
            raise ValueError("Sensor encoder requires canonical measured channel order.")
        if self.kind == "gru" and self.features != "normalized_mean":
            raise ValueError("The temporal encoder requires per-step values instead of statistical features.")
        return self


class ImageEncoderConfig(Contract):
    schema_version: SchemaVersion = 1
    kind: Literal["rgb_grid", "faster_rcnn_fpn"] = "rgb_grid"
    embedding_dim: int = Field(default=16, gt=0)
    grid_size: int = Field(default=8, ge=1, le=64)
    fpn_level: str = Field(default="0", pattern=r"^(?:[0-9]+|pool)$")


class FusionConfig(Contract):
    schema_version: SchemaVersion = 1
    kind: Literal["late", "concat", "gated"] = "late"
    image_weight: float = Field(default=1.0, gt=0, le=1e6)
    sensor_weight: float = Field(default=1.0, gt=0, le=1e6)
    hidden_dim: int = Field(default=32, gt=0)
    threshold: float = Field(default=0.5, ge=0, le=1)


class ArchitectureConfig(Contract):
    schema_version: SchemaVersion = 1
    sensor: SensorEncoderConfig = SensorEncoderConfig()
    image: ImageEncoderConfig = ImageEncoderConfig()
    fusion: FusionConfig = FusionConfig()
    target_policy: TargetPolicy = "smt-target-defect-presence-v1"
    class_names: tuple[str, ...] = CLASS_NAMES
    preprocessing_fingerprint: Fingerprint
    seed: int = Field(default=42, ge=0, lt=2**63)

    @model_validator(mode="after")
    def check_classes(self) -> Self:
        if self.class_names != CLASS_NAMES:
            raise ValueError("Product models require the canonical SMT class mapping.")
        return self

    @property
    def output_names(self) -> tuple[str, ...]:
        if self.target_policy == "smt-target-defect-presence-v1":
            return ("any_target_defect",)
        return self.class_names

    @property
    def fingerprint(self) -> str:
        return digest_json(self.model_dump(mode="json"))


class WeightStatus(Contract):
    schema_version: SchemaVersion = 1
    image: TrainingStatus = "untrained"
    sensor: TrainingStatus = "untrained"
    paired: TrainingStatus = "untrained"


class OptimizerConfig(Contract):
    """Declarative settings for future callers that own the training loop."""

    schema_version: SchemaVersion = 1
    algorithm: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=1e-3, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
