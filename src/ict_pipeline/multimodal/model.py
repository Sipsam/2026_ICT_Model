"""Product prediction, dataset integration, and differentiable loss interfaces."""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F

from ..evaluation.contracts import CLASSES, Prediction
from ..pipeline.store import digest_json
from .contracts import Split
from .dataset import Batch, DatasetSample, collate
from .encoders import (
    FasterRCNNImageEncoder, ImageEncoder, MLPSensorEncoder, RGBGridImageEncoder, SensorEncoder,
    TemporalSensorEncoder,
)
from .fusion import ConcatenationFusion, GatedFusion, LateFusion
from .model_config import ArchitectureConfig, Branch, TargetPolicy, TrainingStatus, WeightStatus
from .preprocessing import PreprocessingState, SensorWindow

SensorStatus = Literal["absent", "no_observations", "insufficient_steps", "available"]


@dataclass(frozen=True)
class ProductTargets:
    split: Split
    sample_ids: tuple[str, ...]
    policy: TargetPolicy
    output_names: tuple[str, ...]
    values: Tensor


@dataclass(frozen=True)
class ProductOutput:
    split: Split | None
    sample_ids: tuple[str, ...]
    policy: TargetPolicy
    output_names: tuple[str, ...]
    image_logits: Tensor
    sensor_logits: Tensor
    paired_logits: Tensor
    image_available: Tensor
    sensor_usable: Tensor

    @property
    def paired(self) -> Tensor:
        return self.image_available & self.sensor_usable

    def for_branch(self, branch: Branch) -> tuple[Tensor, Tensor]:
        if branch == "image":
            return self.image_logits, self.image_available
        if branch == "sensor":
            return self.sensor_logits, self.sensor_usable
        if branch == "paired":
            return self.paired_logits, self.paired
        raise ValueError("Use a supported product loss branch.")


@dataclass(frozen=True)
class DetectorEvidence:
    source: str  # caller's detector run or checkpoint identifier
    prediction: Prediction


@dataclass(frozen=True)
class ProductInput:
    """Label-free input. Use transform_sensor for raw, aligned sensor sequences."""

    sample_id: str
    image: Image.Image | None = None
    sensor: SensorWindow | None = None
    original_size: tuple[int, int] | None = None

    @classmethod
    def from_sample(cls, sample: DatasetSample) -> "ProductInput":
        return cls(sample.sample_id, sample.image, sample.sensor, sample.original_size)


@dataclass(frozen=True)
class ProductScore:
    name: str
    class_id: int | None
    score: float
    threshold_crossed: bool


@dataclass(frozen=True)
class ProductPrediction:
    sample_id: str
    target_policy: TargetPolicy
    architecture_fingerprint: str
    preprocessing_fingerprint: str | None
    fusion: str
    branch: Branch | None
    image_available: bool
    sensor: SensorStatus
    weight_status: TrainingStatus | None
    threshold: float
    scores: tuple[ProductScore, ...]
    detection: DetectorEvidence | None
    schema_version: Literal[1] = 1
    score_kind: Literal["uncalibrated_sigmoid"] = "uncalibrated_sigmoid"
    outcome: Literal["review_required"] = "review_required"


def product_targets(batch: Batch, config: ArchitectureConfig) -> ProductTargets:
    """Project original canonical annotations to ordered presence targets."""
    if not batch.samples or len({sample.split for sample in batch.samples}) != 1:
        raise ValueError("Product targets require a nonempty single-split batch.")
    rows = []
    for sample in batch.samples:
        present = {item.category for item in sample.original_targets}
        if any(type(item.category) is not int or item.category not in CLASSES for item in sample.original_targets):
            raise ValueError("Product targets require canonical one-based detection classes.")
        rows.append([float(bool(present))] if len(config.output_names) == 1 else
                    [float(category in present) for category in CLASSES])
    return ProductTargets(batch.samples[0].split, tuple(sample.sample_id for sample in batch.samples),
                          config.target_policy, config.output_names, torch.tensor(rows, dtype=torch.float32, device="cpu"))


def product_loss(output: ProductOutput, targets: ProductTargets, *, branch: Branch) -> Tensor:
    """Binary cross-entropy over available rows for the selected product head."""
    logits, usable = output.for_branch(branch)
    values = targets.values
    if (output.split != targets.split or output.sample_ids != targets.sample_ids
            or output.policy != targets.policy or output.output_names != targets.output_names
            or logits.shape != values.shape or not ((values == 0) | (values == 1)).all()):
        raise ValueError("Product targets must match the output batch and ordered binary label policy.")
    if not usable.any():
        raise ValueError("The selected product branch has no usable rows.")
    return F.binary_cross_entropy_with_logits(logits[usable], values.to(logits)[usable])


def _check_evidence(sample: ProductInput, evidence: DetectorEvidence) -> None:
    prediction = evidence.prediction
    size = sample.original_size or (sample.image.size if sample.image is not None else None)
    if (sample.image is None or size is None or not evidence.source.strip() or prediction.sample_id != sample.sample_id
            or prediction.preprocessing.original_size != size
            or len(prediction.boxes) != len(prediction.class_ids)
            or len(prediction.boxes) != len(prediction.confidence_scores)):
        raise ValueError("Detector evidence must match the original image and sample ID.")
    width, height = size
    for box, category, score in zip(prediction.boxes, prediction.class_ids, prediction.confidence_scores, strict=True):
        x1, y1, x2, y2 = box
        if (type(category) is not int or category not in CLASSES or not 0 <= score <= 1
                or not all(map(math.isfinite, box)) or not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height)):
            raise ValueError("Detector evidence requires canonical boxes and confidences.")


class MultimodalModel(nn.Module):
    """Product heads over independently reusable image and sensor encoders."""

    def __init__(
        self, config: ArchitectureConfig, preprocessing: PreprocessingState, *, detector: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if config.preprocessing_fingerprint != preprocessing.fingerprint:
            raise ValueError("Architecture preprocessing fingerprint mismatch.")
        self.config = config
        self.weight_status = WeightStatus()
        self.sensor_encoder: SensorEncoder
        self.image_encoder: ImageEncoder
        self.fusion: nn.Module
        # Initialize on CPU without altering the caller's CPU or accelerator RNG.
        with torch.random.fork_rng(devices=[]), torch.device("cpu"):
            torch.random.default_generator.manual_seed(config.seed)
            sensor_type = MLPSensorEncoder if config.sensor.kind == "mlp" else TemporalSensorEncoder
            self.sensor_encoder = sensor_type(config.sensor, preprocessing)
            if config.image.kind == "rgb_grid" and detector is None:
                self.image_encoder = RGBGridImageEncoder(config.image)
            elif config.image.kind == "faster_rcnn_fpn" and detector is not None:
                self.image_encoder = FasterRCNNImageEncoder(config.image, detector)
            else:
                raise ValueError("Supply an initialized detector only for the Faster R-CNN image encoder.")
            image_dim = self.image_encoder.embedding_dim
            sensor_dim = self.sensor_encoder.embedding_dim
            output_dim = len(config.output_names)
            self.image_head = nn.Linear(image_dim, output_dim, dtype=torch.float32)
            self.sensor_head = nn.Linear(sensor_dim, output_dim, dtype=torch.float32)
            if config.fusion.kind == "late":
                self.fusion = LateFusion(config.fusion, output_dim)
            else:
                fusion_type = ConcatenationFusion if config.fusion.kind == "concat" else GatedFusion
                self.fusion = fusion_type(config.fusion, image_dim, sensor_dim, output_dim)

    @property
    def architecture_fingerprint(self) -> str:
        return digest_json([self.config.fingerprint, self.image_encoder.input_contract])

    def _batch_inputs(self, batch: Batch) -> tuple[ProductInput, ...]:
        if batch != collate(batch.samples):
            raise ValueError("Batch sensor values, masks, lengths, or modality flags disagree with its samples.")
        fingerprint = batch.samples[0].preprocessing_fingerprint
        if fingerprint not in (None, self.config.preprocessing_fingerprint):
            raise ValueError("Batch preprocessing fingerprint mismatch.")
        return tuple(ProductInput.from_sample(sample) for sample in batch.samples)

    def forward(self, batch: Batch) -> ProductOutput:
        inputs = self._batch_inputs(batch)
        return self._forward_inputs(inputs, batch.samples[0].split)

    def _forward_inputs(self, inputs: Sequence[ProductInput], split: Split | None) -> ProductOutput:
        if not inputs or any(not sample.sample_id.strip() for sample in inputs):
            raise ValueError("Product inputs require a nonempty batch and nonblank sample IDs.")
        sensors = self.sensor_encoder([sample.sensor for sample in inputs])
        reference = self.image_head.weight
        image_available = torch.tensor([sample.image is not None for sample in inputs],
                                       device=reference.device, dtype=torch.bool)
        image = reference.new_zeros((len(inputs), self.image_encoder.embedding_dim))
        images = [sample.image for sample in inputs if sample.image is not None]
        if images:
            image[image_available] = self.image_encoder(images).to(image)
        image_logits = self.image_head(image).masked_fill(~image_available[:, None], 0)
        sensor_logits = self.sensor_head(sensors.embeddings).masked_fill(~sensors.usable[:, None], 0)
        paired = image_available & sensors.usable
        paired_logits = torch.zeros_like(image_logits)
        if paired.any():
            if self.config.fusion.kind == "late":
                paired_logits[paired] = self.fusion(image_logits[paired], sensor_logits[paired])
            else:
                paired_logits[paired] = self.fusion(image[paired], sensors.embeddings[paired].to(image)).to(paired_logits)
        if not all(torch.isfinite(logits).all() for logits in (image_logits, sensor_logits, paired_logits)):
            raise ValueError("Product model produced non-finite logits.")
        return ProductOutput(
            split, tuple(sample.sample_id for sample in inputs), self.config.target_policy,
            self.config.output_names, image_logits, sensor_logits, paired_logits, image_available, sensors.usable,
        )

    def branch_status(self, branch: Branch) -> TrainingStatus:
        if branch == "paired" and self.config.fusion.kind == "late":
            return "trained" if self.weight_status.image == self.weight_status.sensor == "trained" else "untrained"
        return getattr(self.weight_status, branch)

    @torch.inference_mode()
    def predict(
        self, batch: Batch | Sequence[ProductInput], *, detections: Sequence[DetectorEvidence | None] | None = None,
    ) -> tuple[ProductPrediction, ...]:
        if self.training:
            raise ValueError("Call eval() before product inference.")
        inputs = self._batch_inputs(batch) if isinstance(batch, Batch) else tuple(batch)
        if detections is None:
            detections = [None] * len(inputs)
        if len(detections) != len(inputs):
            raise ValueError("Supply one optional detector prediction per batch row.")
        output = self._forward_inputs(inputs, None)
        rows = []
        fingerprint = self.architecture_fingerprint
        for index, (sample, evidence) in enumerate(zip(inputs, detections, strict=True)):
            if evidence is not None:
                _check_evidence(sample, evidence)
            branch: Branch | None = None
            scores: tuple[ProductScore, ...] = ()
            image, sensor = bool(output.image_available[index]), bool(output.sensor_usable[index])
            if image or sensor:
                branch = "paired" if image and sensor else "image" if image else "sensor"
                logits, _ = output.for_branch(branch)
                scores = tuple(
                    ProductScore(name, column if len(output.output_names) > 1 else None, score,
                                 score >= self.config.fusion.threshold)
                    for column, (name, score) in enumerate(
                        zip(output.output_names, logits[index].sigmoid().tolist(), strict=True), 1,
                    )
                )
            sensor_status: SensorStatus = "available" if sensor else "absent"
            if not sensor and sample.sensor is not None:
                sensor_status = "insufficient_steps" if sample.sensor.available else "no_observations"
            rows.append(ProductPrediction(
                sample.sample_id, output.policy, fingerprint,
                sample.sensor.preprocessing_fingerprint if sample.sensor is not None else None,
                self.config.fusion.kind, branch, image, sensor_status,
                self.branch_status(branch) if branch is not None else None,
                self.config.fusion.threshold, scores, evidence,
            ))
        return tuple(rows)
