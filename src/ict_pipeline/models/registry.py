"""Backend discovery and device selection without eager ML imports."""

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType

from .contracts import DataFormat, ModelAdapter, ModelName, TrainConfig, TrainingError


@dataclass(frozen=True)
class Backend:
    module: str
    adapter: str
    data_format: DataFormat
    extra: str
    libraries: tuple[str, ...]
    license: str
    checkpoint_format: str
    best_criterion: str
    devices: tuple[str, ...]
    supports_resume: bool = True

    def versions(self) -> dict[str, str]:
        try:
            versions = {name: version(name) for name in self.libraries}
        except PackageNotFoundError:
            raise TrainingError(f"Missing training dependencies. Install ict-pipeline[{self.extra}] in this environment.") from None
        if self.data_format == "yolo" and versions["ultralytics"] != "8.3.228":
            raise TrainingError("Offline YOLO training requires Ultralytics 8.3.228. Use ict-pipeline[train-yolo].")
        if "transformers" in versions and versions["transformers"] != "4.57.6":
            raise TrainingError("RT-DETR requires Transformers 4.57.6. Use ict-pipeline[train-rt-detr].")
        return versions

    def load(self) -> ModelAdapter:
        return getattr(import_module(f".{self.module}", __package__), self.adapter)()


REGISTRY: dict[ModelName, Backend] = {
    ModelName.YOLO: Backend(
        "yolo", "YoloAdapter", "yolo", "train-yolo",
        ("torch", "torchvision", "ultralytics", "numpy"), "AGPL-3.0 / Ultralytics Enterprise",
        "ultralytics-pt", "Ultralytics validation fitness", ("cpu", "cuda", "rocm", "mps"),
    ),
    ModelName.FASTER_RCNN: Backend(
        "coco", "FasterRCNNAdapter", "coco", "train-faster-rcnn",
        ("torch", "torchvision", "Pillow", "numpy"), "BSD-3-Clause",
        "ict-torch-state-v1", "minimum validation loss", ("cpu", "cuda", "rocm"),
    ),
    ModelName.RT_DETR: Backend(
        "coco", "RTDETRAdapter", "coco", "train-rt-detr",
        ("torch", "torchvision", "transformers", "scipy", "Pillow", "numpy"), "Apache-2.0",
        "ict-torch-state-v1", "minimum validation loss", ("cpu", "cuda", "rocm"),
    ),
}


def resolve_device(torch: ModuleType, config: TrainConfig, backend: Backend) -> str:
    if config.device != "auto" and config.device not in backend.devices:
        raise TrainingError("This adapter does not support the requested device. Use CPU or a supported GPU backend.")
    cuda = torch.cuda.is_available() and torch.cuda.device_count() > 0
    gpu = "rocm" if torch.version.hip else "cuda"
    mps = torch.backends.mps.is_available()
    requested = config.device
    if requested == "auto":
        requested = gpu if cuda and gpu in backend.devices else "cpu"
        if requested == "cpu" and mps and "mps" in backend.devices and config.precision == "fp32":
            requested = "mps"
    if requested in ("cuda", "rocm") and (not cuda or requested != gpu):
        raise TrainingError(f"{requested.upper()} is unavailable in this PyTorch build. Run ict doctor or use device: cpu.")
    if requested == "mps" and not mps:
        raise TrainingError("MPS is unavailable in this PyTorch build. Run ict doctor or use device: cpu.")
    if config.precision != "fp32" and requested not in ("cuda", "rocm"):
        raise TrainingError("Mixed precision requires CUDA or ROCm. Set precision: fp32 for CPU or MPS.")
    if config.precision == "bf16":
        if backend.data_format == "yolo":
            raise TrainingError("This YOLO backend supports fp32 and fp16. Select one of those precisions.")
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise TrainingError("This GPU lacks native BF16 support. Use fp32 or fp16.")
    return "cuda:0" if requested in ("cuda", "rocm") else requested


def failure_message(exc: Exception, backend: Backend) -> str:
    # Inspect backend errors for classification without exposing arbitrary library text.
    message = str(exc).lower()
    if isinstance(exc, MemoryError) or "outofmemory" in type(exc).__name__.lower() or any(
        hint in message for hint in ("out of memory", "can't allocate memory", "cannot allocate memory", "defaultcpuallocator")
    ):
        return "Training ran out of memory. Reduce batch_size or image_size and start a new run."
    if isinstance(exc, (ImportError, ModuleNotFoundError)) or "torchvision::" in message or "custom c++ ops" in message:
        return f"Cannot load the training runtime. Check compatible PyTorch and TorchVision builds and ict-pipeline[{backend.extra}]."
    if any(hint in message for hint in (
        "not implemented", "not supported on", "no kernel image", "invalid device ordinal",
        "nvidia driver", "cuda driver", "hip error",
    )):
        return "A required operator is unavailable on this device. Try device: cpu and precision: fp32."
    if isinstance(exc, OSError):
        return "Training could not read or write a file. Check dataset integrity, storage, and permissions."
    return f"Training backend failed ({type(exc).__name__}). Check the dataset, configuration, and runtime with ict doctor."
