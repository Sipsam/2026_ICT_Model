"""Resolve, verify and transfer offline detector checkpoints."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from urllib.parse import urlsplit

from ..config.settings import Credentials, is_filesystem_path
from .contracts import CheckpointConfig, ModelName, TrainConfig, TrainingError, TrainingRequest


@dataclass(frozen=True)
class WeightLayout:
    architecture: str
    library: str
    source_classes: int


LAYOUTS = {
    ModelName.YOLO: WeightLayout("yolo11n.yaml", "ultralytics", 80),
    ModelName.FASTER_RCNN: WeightLayout("fasterrcnn_mobilenet_v3_large_320_fpn", "torchvision", 91),
    ModelName.RT_DETR: WeightLayout("rt-detr-resnet18", "transformers", 80),
}


@dataclass(frozen=True)
class VerifiedWeights:
    path: Path
    spec: CheckpointConfig
    source_classes: int


def checkpoint_source(
    name: ModelName, config: TrainConfig, directory: Path | None, credentials: Credentials,
) -> Path | None:
    spec = config.checkpoint
    if spec is None:
        return None
    layout = LAYOUTS[name]
    if spec.architecture != layout.architecture:
        raise TrainingError(f"{name} requires checkpoint architecture {layout.architecture}.")
    if spec.format == "ultralytics_pickle" and name != ModelName.YOLO:
        raise TrainingError("Native Ultralytics pickle is supported only for YOLO.")
    if name == ModelName.YOLO and config.optimizer_name == "SGD" and (
        config.fine_tuning is not None and config.fine_tuning.momentum == 0
    ):
        raise TrainingError("Ultralytics SGD uses Nesterov momentum. Set fine_tuning.momentum above zero.")
    origin = urlsplit(spec.origin)
    if "://" in spec.origin and (origin.scheme != "https" or not origin.hostname or origin.username):
        raise TrainingError("Use a public HTTPS URL or a provider identifier for checkpoint origin.")
    for field in (spec.source, spec.origin, spec.revision, spec.license, spec.library_version):
        credentials.protect_path(field, "Checkpoint configuration")
    if not is_filesystem_path(spec.source) or Path(spec.source).drive and not Path(spec.source).is_absolute():
        raise TrainingError("Use a local checkpoint file path. Stage approved downloads in the local cache first.")
    path = Path(spec.source)
    if path.is_absolute():
        path = path.resolve()
    elif directory is not None:
        path = (directory / path).resolve()
    else:
        raise TrainingError("A relative checkpoint source requires a model profile directory.")
    credentials.protect_path(str(path), "Resolved checkpoint path")
    return path


def verify_weights(
    name: ModelName, config: TrainConfig, source: Path | None, libraries: Mapping[str, str],
) -> VerifiedWeights | None:
    spec = config.checkpoint
    if spec is None:
        return None
    layout = LAYOUTS[name]
    if libraries.get(layout.library) != spec.library_version:
        raise TrainingError(f"{name} checkpoint requires {layout.library} {spec.library_version} on this target.")
    weight_libraries(config)
    if source is None or not source.is_file():
        raise TrainingError(f"{name} checkpoint is missing from the local cache. Stage the pinned file and retry.")
    _verify_hash(source, spec.sha256)
    return VerifiedWeights(source, spec, spec.source_classes or layout.source_classes)


def weight_packages(config: TrainConfig) -> tuple[str, ...]:
    return ("safetensors",) if config.checkpoint and config.checkpoint.format == "safetensors" else ()


def weight_libraries(config: TrainConfig) -> dict[str, str]:
    try:
        return {package: version(package) for package in weight_packages(config)}
    except PackageNotFoundError:
        raise TrainingError("Safetensors loading requires ict-pipeline[pretrained-safetensors] on this target.") from None


def _verify_hash(path: Path, expected: str) -> None:
    try:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError:
        raise TrainingError("Cannot read the local pretrained checkpoint.") from None
    if actual != expected:
        raise TrainingError("Pretrained checkpoint SHA-256 mismatch. Restore the pinned file from a trusted source.")


@contextmanager
def _verified_copy(weights: VerifiedWeights, output: Path) -> Iterator[Path]:
    # Deserialize the same private bytes that were hashed, even if the shared cache
    # is replaced while training is starting.
    from shutil import copyfile

    with TemporaryDirectory(prefix="init-", dir=output) as directory:
        path = Path(directory) / "weights"
        try:
            copyfile(weights.path, path)
        except OSError:
            raise TrainingError("Cannot stage the local pretrained checkpoint. Check storage and permissions.") from None
        _verify_hash(path, weights.spec.sha256)
        yield path


def load_weights(request: TrainingRequest, *, native_type: type | None = None) -> Mapping[str, Any]:
    import torch

    weights = request.pretrained
    if weights is None or weights.spec != request.config.checkpoint:
        raise TrainingError("Pretrained training requires the verified checkpoint for this profile.")
    with _verified_copy(weights, request.output) as path:
        try:
            if weights.spec.format == "safetensors":
                from safetensors.torch import load_file

                state = load_file(path, device="cpu")
            elif weights.spec.format == "ultralytics_pickle":
                if native_type is None or not weights.spec.trusted_pickle:
                    raise TrainingError("Native pickle requires a trusted YOLO checkpoint.")
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                model = checkpoint.get("ema")
                if model is None:
                    model = checkpoint.get("model")
                if type(model) is not native_type or model.yaml.get("nc") != weights.source_classes:
                    raise TrainingError("Expected a native YOLO detection model with the declared source classes.")
                state = model.state_dict()
            else:
                state = torch.load(path, map_location="cpu", weights_only=True)
        except TrainingError:
            raise
        except Exception:
            raise TrainingError("Cannot decode the pinned checkpoint using its declared format.") from None
    if not isinstance(state, Mapping) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise TrainingError("Expected a flat checkpoint mapping of tensor names to tensors.")
    return state


def transfer_state(
    model: Any, source: Mapping[str, Any], source_shapes: Mapping[str, tuple[int, ...]],
    reset_keys: set[str], request: TrainingRequest, *, aliases: Mapping[str, str] | None = None,
) -> None:
    """Allow omissions only for declared tied aliases and reset only approved head keys."""
    import torch

    target = model.state_dict()
    if target.keys() != source_shapes.keys() or not reset_keys <= target.keys():
        raise TrainingError("The configured transfer layout differs from the detector architecture.")
    state = dict(source)
    missing = sorted(target.keys() - source.keys())
    unexpected = sorted(source.keys() - target.keys())
    restored = {}
    incompatible = set()
    for key, alias in (aliases or {}).items():
        if key in state and alias in state:
            if not torch.equal(state[key], state[alias]):
                incompatible.update((key, alias))
        elif key in state or alias in state:
            absent, present = (alias, key) if key in state else (key, alias)
            state[absent] = state[present]
            restored[absent] = present
    for key in target.keys() & state.keys():
        value, expected = state[key], target[key]
        if (
            tuple(value.shape) != source_shapes[key] or value.layout != torch.strided
            or (expected.is_floating_point() and not value.is_floating_point())
            or (not expected.is_floating_point() and value.dtype != expected.dtype)
        ):
            incompatible.add(key)
    request.record_transfer({
        "missing_keys": missing, "unexpected_keys": unexpected, "incompatible_keys": sorted(incompatible),
        "reset_keys": sorted(reset_keys), "restored_aliases": restored,
    })
    if set(missing) - restored.keys() or unexpected or incompatible:
        raise TrainingError("Checkpoint tensors are incompatible with the declared architecture. See transfer diagnostics in run.json.")
    retained = target.keys() - reset_keys
    if any(state[key].shape != target[key].shape for key in retained):
        raise TrainingError("The configured transfer would change a non-head tensor shape.")
    model.load_state_dict({key: target[key] if key in reset_keys else state[key] for key in target}, strict=True)


def training_scope(name: ModelName, config: TrainConfig, source: Path | None) -> dict[str, object]:
    size = None
    if source is not None:
        try:
            if source.is_file():
                size = source.stat().st_size
        except OSError:
            pass
    return {
        "architecture": LAYOUTS[name].architecture, "initialization": config.initialization,
        "checkpoint_source": str(source) if source is not None else None,
        "device": config.device, "precision": config.precision, "epochs": config.epochs,
        "image_size": config.image_size, "batch_size": config.batch_size,
        "fp32_rgb_batch_bytes": config.batch_size * 3 * config.image_size**2 * 4,
        "checkpoint_bytes": size,
        "storage": "Run checkpoints, a temporary checkpoint copy if pretrained"
                   + (", and a private Train/Validation dataset copy" if name == ModelName.YOLO else ""),
        "memory": "Model, optimizer and activation memory are additional and depend on the backend and device.",
    }
