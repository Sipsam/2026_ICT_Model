"""Local initialization of the product FPN from a pinned detector checkpoint."""

import hashlib
from io import BytesIO
from pathlib import Path
from pickle import UnpicklingError

import torch
from torch import nn

from ..models.coco import build_faster_rcnn
from ..models.weights import LAYOUTS
from .model_config import FPNDetectorConfig

FORMAT = "ict-torch-state-v1"
ARCHITECTURE = LAYOUTS["faster_rcnn"].architecture


def build_detector(config: FPNDetectorConfig) -> nn.Module:
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        return build_faster_rcnn(config.image_size, pretrained_layout=config.layout == "pretrained")


def load_detector(path: Path, sha256: str, config: FPNDetectorConfig) -> nn.Module:
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("FPN initialization checkpoint SHA-256 mismatch.")
    try:
        payload = torch.load(BytesIO(content), weights_only=True, map_location="cpu")
    except (UnpicklingError, RuntimeError, EOFError, ValueError):
        raise ValueError("Cannot read the tensor-only FPN initialization checkpoint.") from None
    return detector_from_checkpoint(payload, config)


def detector_from_checkpoint(payload: object, config: FPNDetectorConfig) -> nn.Module:
    if (not isinstance(payload, dict) or payload.keys() != {
        "format", "architecture", "epoch", "best_epoch", "best_loss", "val_loss", "model", "optimizer", "scaler",
    } or payload["format"] != FORMAT or payload["architecture"] != ARCHITECTURE):
        raise ValueError("FPN initialization requires an ICT MobileNetV3 Faster R-CNN detector checkpoint.")
    detector = build_detector(config)
    expected = detector.state_dict()
    state = payload["model"]
    # Explicit key checks also reject missing BatchNorm counters that PyTorch
    # may otherwise synthesize for legacy state dictionaries.
    if not isinstance(state, dict) or state.keys() != expected.keys() or any(
        type(value) is not torch.Tensor or value.layout != torch.strided
        or value.shape != expected[name].shape or value.dtype != expected[name].dtype
        or not torch.isfinite(value).all().item()
        for name, value in state.items()
    ):
        raise ValueError("FPN initialization tensors must be finite and match the configured detector layout.")
    detector.load_state_dict(state, strict=True)
    return detector
