"""Exclusive local publication and tensor-only reload for product models."""

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from torch import Tensor, nn

from ..data.preprocessing.console import protect_source_data
from .model import MultimodalModel
from .model_config import ArchitectureConfig, WeightStatus
from .preprocessing import PreprocessingState

FORMAT = "ict-multimodal-state-v1"


def save_checkpoint(path: Path, model: MultimodalModel) -> None:
    """Publish a new checkpoint in an existing output directory."""
    path = path.absolute()
    protect_source_data(path.resolve(), Path.cwd())
    if path.exists() or path.is_symlink():
        raise FileExistsError("Choose a fresh multimodal checkpoint path.")
    tensors = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(value.is_floating_point() and (
        value.dtype != torch.float32 or not torch.isfinite(value).all()
    ) for value in tensors.values()):
        raise ValueError("Multimodal checkpoints require finite float32 parameters and buffers.")
    payload = {
        "format": FORMAT,
        "config": model.config.model_dump_json(),
        "architecture_fingerprint": model.architecture_fingerprint,
        "weight_status": model.weight_status.model_dump_json(),
        "state_dict": tensors,
    }
    with NamedTemporaryFile(dir=path.parent, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            os.link(temporary, path)
        finally:
            stream.close()
            temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: Path, preprocessing: PreprocessingState, *, expected_config: ArchitectureConfig | None = None,
    detector: nn.Module | None = None,
) -> MultimodalModel:
    """Reconstruct and validate an architecture, then load exact CPU tensor state."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.keys() != {
        "format", "config", "architecture_fingerprint", "weight_status", "state_dict",
    } or payload["format"] != FORMAT:
        raise ValueError("Unsupported multimodal checkpoint format.")
    config = ArchitectureConfig.model_validate_json(payload["config"])
    status = WeightStatus.model_validate_json(payload["weight_status"])
    if expected_config is not None and config != expected_config:
        raise ValueError("Multimodal checkpoint configuration mismatch.")
    if config.image.detector is not None:
        from .fpn import build_detector

        detector = build_detector(config.image.detector)
    model = MultimodalModel(config, preprocessing, detector=detector if config.image.kind == "faster_rcnn_fpn" else None)
    if payload["architecture_fingerprint"] != model.architecture_fingerprint:
        raise ValueError("Multimodal checkpoint architecture fingerprint mismatch.")
    expected = model.state_dict()
    tensors = payload["state_dict"]
    if not isinstance(tensors, dict) or tensors.keys() != expected.keys():
        raise ValueError("Multimodal checkpoint tensor keys mismatch.")
    for name, reference in expected.items():
        value = tensors[name]
        if (type(value) is not Tensor or value.layout != torch.strided or value.shape != reference.shape
                or value.dtype != reference.dtype or not torch.isfinite(value).all()):
            raise ValueError(f"Multimodal checkpoint tensor is incompatible: {name}")
    model.load_state_dict(tensors, strict=True)
    model.weight_status = status
    return model.eval()
