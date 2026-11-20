"""Masked sensor and image embeddings for product-level models."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Self

import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_sequence

from ..classification.sensors import parse_timestamp
from .contracts import CHANNELS
from .features import FEATURE_NAMES, statistical_features
from .model_config import ImageEncoderConfig, SensorEncoderConfig
from .preprocessing import PreprocessingState, SensorWindow


@dataclass(frozen=True)
class SensorEncoding:
    embeddings: Tensor
    usable: Tensor  # [B] bool. Unusable rows have zero embeddings.


class SensorEncoder(nn.Module):
    """Encode frozen, timestamped windows. A None window represents absent sensor data."""

    def __init__(self, config: SensorEncoderConfig, preprocessing: PreprocessingState) -> None:
        super().__init__()
        if config.channels != CHANNELS or config.channels != preprocessing.channels:
            raise ValueError("Sensor encoder and preprocessing must use the same canonical channel order.")
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.preprocessing_fingerprint = preprocessing.fingerprint
        self.window_seconds = preprocessing.config.window_seconds

    def _validate_windows(self, windows: Sequence[SensorWindow | None]) -> None:
        if not windows:
            raise ValueError("Sensor encoding requires at least one row.")
        for window in windows:
            if window is None:
                continue
            if window.preprocessing_fingerprint != self.preprocessing_fingerprint:
                raise ValueError("Sensor window preprocessing fingerprint mismatch.")
            if len(window.timestamps) > 1:
                elapsed = parse_timestamp(window.timestamps[-1]) - parse_timestamp(window.timestamps[0])
                if elapsed.total_seconds() > self.window_seconds:
                    raise ValueError("Sensor timestamps exceed the frozen decision window.")

    def forward(self, windows: Sequence[SensorWindow | None]) -> SensorEncoding:
        raise NotImplementedError


class MLPSensorEncoder(SensorEncoder):
    def __init__(self, config: SensorEncoderConfig, preprocessing: PreprocessingState) -> None:
        super().__init__(config, preprocessing)
        if config.kind != "mlp":
            raise ValueError("MLP sensor encoder requires the mlp configuration.")
        width = len(FEATURE_NAMES) if config.features == "statistics" else len(CHANNELS)
        self.network = nn.Sequential(
            nn.Linear(2 * width, config.hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, self.embedding_dim, dtype=torch.float32),
        )

    def forward(self, windows: Sequence[SensorWindow | None]) -> SensorEncoding:
        self._validate_windows(windows)
        reference = self.network[0].weight
        rows = reference.new_zeros((len(windows), self.network[0].in_features))
        usable = torch.tensor([window is not None and window.available for window in windows],
                              device=rows.device, dtype=torch.bool)
        for index, window in enumerate(windows):
            if window is None or not window.available:
                continue
            if self.config.features == "statistics":
                features = statistical_features(window).values
                values = rows.new_tensor([value if value is not None else 0.0 for value in features])
                observed = torch.tensor([value is not None for value in features], device=rows.device)
            else:
                # Exclude imputed or missing values before converting to the model dtype.
                measured = [[value if valid else 0.0 for value, valid in zip(row, mask, strict=True)]
                            for row, mask in zip(window.values, window.observed, strict=True)]
                counts = rows.new_tensor(window.observed).sum(dim=0)
                values = (rows.new_tensor(measured) / counts.clamp_min(1)).sum(dim=0)
                observed = counts > 0
            rows[index] = torch.cat((values, observed.to(rows.dtype)))
        if not torch.isfinite(rows).all():
            raise ValueError("Observed sensor features exceed the encoder's numeric range.")
        embeddings = self.network(rows).masked_fill(~usable[:, None], 0.0)
        return SensorEncoding(embeddings, usable)


class TemporalSensorEncoder(SensorEncoder):
    def __init__(self, config: SensorEncoderConfig, preprocessing: PreprocessingState) -> None:
        super().__init__(config, preprocessing)
        if config.kind != "gru" or self.window_seconds == 0:
            raise ValueError("The GRU requires a temporal configuration and nonzero frozen decision window.")
        self.gru = nn.GRU(len(CHANNELS) * 2 + 1, config.hidden_dim, batch_first=True, dtype=torch.float32)
        self.projection = nn.Linear(config.hidden_dim, self.embedding_dim, dtype=torch.float32)

    def forward(self, windows: Sequence[SensorWindow | None]) -> SensorEncoding:
        self._validate_windows(windows)
        reference = self.projection.weight
        embeddings = reference.new_zeros((len(windows), self.embedding_dim))
        usable = torch.zeros(len(windows), device=reference.device, dtype=torch.bool)
        sequences = []
        for index, window in enumerate(windows):
            if window is None:
                continue
            steps = [(stamp, row, mask) for stamp, row, mask in
                     zip(window.timestamps, window.values, window.observed, strict=True) if any(mask)]
            if len(steps) < self.config.min_steps:
                continue
            start = parse_timestamp(steps[0][0])
            values = reference.new_tensor([
                [*(value if valid else 0.0 for value, valid in zip(row, mask, strict=True)), *mask,
                 (parse_timestamp(stamp) - start).total_seconds()]
                for stamp, row, mask in steps
            ])
            if not torch.isfinite(values).all():
                raise ValueError("Observed temporal features exceed the encoder's numeric range.")
            sequences.append(values)
            usable[index] = True
        if sequences:
            _, hidden = self.gru(pack_sequence(sequences, enforce_sorted=False))
            embeddings[usable] = self.projection(hidden[-1]).to(embeddings)
        return SensorEncoding(embeddings, usable)


class ImageEncoder(nn.Module):
    """Map a nonempty list of present RGB images to [B, embedding_dim]."""

    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim

    @property
    def input_contract(self) -> dict[str, object]:
        return {}

    def forward(self, images: Sequence[Image.Image]) -> Tensor:
        raise NotImplementedError


class RGBGridImageEncoder(ImageEncoder):
    def __init__(self, config: ImageEncoderConfig) -> None:
        super().__init__(config)
        if config.kind != "rgb_grid":
            raise ValueError("RGB image encoder requires the rgb_grid configuration.")
        self.projection = nn.Linear(3 * config.grid_size**2, self.embedding_dim, dtype=torch.float32)

    def forward(self, images: Sequence[Image.Image]) -> Tensor:
        if not images or any(image.mode != "RGB" for image in images):
            raise ValueError("Image encoding requires at least one RGB image.")
        size = (self.config.grid_size, self.config.grid_size)
        rows = [list(image.resize(size, Image.Resampling.BILINEAR).tobytes()) for image in images]
        return self.projection(self.projection.weight.new_tensor(rows) / 255)


class FasterRCNNImageEncoder(ImageEncoder):
    """Pool a public FPN output from an independently configured detector backbone."""

    def __init__(self, config: ImageEncoderConfig, detector: nn.Module) -> None:
        super().__init__(config)
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.transform import GeneralizedRCNNTransform

        if (config.kind != "faster_rcnn_fpn" or not isinstance(detector, FasterRCNN)
                or type(detector.transform) is not GeneralizedRCNNTransform
                or not isinstance(getattr(detector.backbone, "out_channels", None), int)):
            raise ValueError("FPN image encoding requires an initialized TorchVision Faster R-CNN detector.")
        transform = detector.transform
        # The existing predictor may attach hooks to track transformed box coordinates.
        self.transform = GeneralizedRCNNTransform(
            transform.min_size[-1], transform.max_size, list(transform.image_mean), list(transform.image_std),
            size_divisible=transform.size_divisible, fixed_size=transform.fixed_size,
        ).eval()
        self.backbone = deepcopy(detector.backbone).cpu().float().eval()
        if config.freeze_backbone:
            self.backbone.requires_grad_(False)
        self.projection = nn.Linear(self.backbone.out_channels, self.embedding_dim, dtype=torch.float32)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.transform.eval()
        if self.config.freeze_backbone:
            self.backbone.eval()
        return self

    @property
    def input_contract(self) -> dict[str, object]:
        from torchvision import __version__

        return {
            "torchvision": __version__,
            "backbone_type": type(self.backbone).__qualname__,
            "channels": self.backbone.out_channels,
            "min_size": list(self.transform.min_size),
            "max_size": self.transform.max_size,
            "image_mean": self.transform.image_mean,
            "image_std": self.transform.image_std,
            "size_divisible": self.transform.size_divisible,
            "fixed_size": list(self.transform.fixed_size) if self.transform.fixed_size is not None else None,
        }

    def forward(self, images: Sequence[Image.Image]) -> Tensor:
        from torchvision.transforms.functional import pil_to_tensor

        if not images or any(image.mode != "RGB" for image in images):
            raise ValueError("Image encoding requires at least one RGB image.")
        reference = self.projection.weight
        embeddings = []
        for image in images:
            pixels = pil_to_tensor(image).to(reference) / 255
            transformed, _ = self.transform([pixels])
            maps = self.backbone(transformed.tensors)
            feature = maps.get(self.config.fpn_level) if isinstance(maps, Mapping) else None
            if feature is None or feature.ndim != 4 or feature.shape[1] != self.projection.in_features:
                raise ValueError("Requested FPN level is unavailable or has incompatible channels.")
            embeddings.append(F.adaptive_avg_pool2d(feature, 1).flatten(1))
        return self.projection(torch.cat(embeddings))
