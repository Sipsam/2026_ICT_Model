"""Product-level fusion operators. Detector confidence and boxes stay outside fusion."""

import torch
from torch import Tensor, nn

from .model_config import FusionConfig


def _check_inputs(image: Tensor, sensor: Tensor, image_dim: int, sensor_dim: int) -> None:
    if (image.ndim != 2 or sensor.ndim != 2 or image.shape != (sensor.shape[0], image_dim)
            or sensor.shape[1] != sensor_dim or image.device != sensor.device or image.dtype != sensor.dtype
            or not image.is_floating_point()):
        raise ValueError("Fusion requires compatible rank-two image and sensor tensors.")


class LateFusion(nn.Module):
    """Weighted average of logits from heads with the same product target mapping."""

    def __init__(self, config: FusionConfig, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim
        total = config.image_weight + config.sensor_weight
        self.image_fraction = config.image_weight / total
        self.sensor_fraction = config.sensor_weight / total

    def forward(self, image_logits: Tensor, sensor_logits: Tensor) -> Tensor:
        _check_inputs(image_logits, sensor_logits, self.output_dim, self.output_dim)
        return self.image_fraction * image_logits + self.sensor_fraction * sensor_logits


class ConcatenationFusion(nn.Module):
    def __init__(self, config: FusionConfig, image_dim: int, sensor_dim: int, output_dim: int) -> None:
        super().__init__()
        self.image_dim, self.sensor_dim = image_dim, sensor_dim
        self.head = nn.Sequential(
            nn.Linear(image_dim + sensor_dim, config.hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, output_dim, dtype=torch.float32),
        )

    def forward(self, image: Tensor, sensor: Tensor) -> Tensor:
        _check_inputs(image, sensor, self.image_dim, self.sensor_dim)
        return self.head(torch.cat((image, sensor), dim=1))


class GatedFusion(nn.Module):
    def __init__(self, config: FusionConfig, image_dim: int, sensor_dim: int, output_dim: int) -> None:
        super().__init__()
        self.image_dim, self.sensor_dim = image_dim, sensor_dim
        self.image_projection = nn.Linear(image_dim, config.hidden_dim, dtype=torch.float32)
        self.sensor_projection = nn.Linear(sensor_dim, config.hidden_dim, dtype=torch.float32)
        self.gate = nn.Linear(image_dim + sensor_dim, config.hidden_dim, dtype=torch.float32)
        self.head = nn.Linear(config.hidden_dim, output_dim, dtype=torch.float32)

    def forward(self, image: Tensor, sensor: Tensor) -> Tensor:
        _check_inputs(image, sensor, self.image_dim, self.sensor_dim)
        gate = self.gate(torch.cat((image, sensor), dim=1)).sigmoid()
        return self.head(gate * self.image_projection(image) + (1 - gate) * self.sensor_projection(sensor))
