"""Validated SMT-specific training data configuration."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ...data.preprocessing.policy import NAMES


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, hide_input_in_errors=True)


class Photometric(Policy):
    brightness: float = Field(default=0.0, ge=0, le=0.12)
    contrast: float = Field(default=0.0, ge=0, le=0.12)
    noise: int = Field(default=0, ge=0, le=6)
    saturation: float = Field(default=0.0, ge=0, le=0.10)


class Geometric(Policy):
    rotation_degrees: float = Field(default=0.0, ge=0, le=7)
    translation_fraction: float = Field(default=0.0, ge=0, le=0.05)
    scale_delta: float = Field(default=0.0, ge=0, le=0.08)
    crop_fraction: float = Field(default=0.0, ge=0, le=0.05)
    min_visibility: float = Field(default=1.0, ge=0.95, le=1)
    horizontal_flip_probability: float = Field(default=0.0, ge=0, le=1)
    flip_policy: Literal["disabled", "verified_mirror_invariant"] = "disabled"

    @model_validator(mode="after")
    def check_orientation(self) -> Self:
        if self.horizontal_flip_probability and self.flip_policy != "verified_mirror_invariant":
            raise ValueError("Horizontal flips require verified_mirror_invariant approval for the selected cohort.")
        return self


class Augmentation(Policy):
    photometric: Photometric | None = None
    geometric: Geometric | None = None


class Balancing(Policy):
    method: Literal["weighted", "class_aware"]
    strength: float = Field(default=0.5, gt=0, le=1)
    max_weight_ratio: float = Field(default=4.0, ge=1, le=20)


class Optimization(Policy):
    augmentation: Augmentation | None = None
    balancing: Balancing | None = None


BACKGROUND = "background"
BUCKETS = (*NAMES, BACKGROUND)
