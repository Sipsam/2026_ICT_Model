"""Conservative detector settings from observations made on the execution host."""

from dataclasses import dataclass
from typing import Literal

from .disk import MIB, RESERVE_BYTES
from .environment.model import EnvironmentReport
from .models.contracts import ModelName, TrainingError
from .models.registry import REGISTRY
from .training_storage import CHECKPOINT_ALLOWANCE

Device = Literal["cpu", "cuda", "rocm", "mps"]
Precision = Literal["fp32", "fp16", "bf16"]


@dataclass(frozen=True)
class Overrides:
    device: Device | None = None
    precision: Precision | None = None
    image_size: int | None = None
    batch_size: int | None = None
    workers: int | None = None
    gradient_accumulation: int | None = None


@dataclass(frozen=True)
class Recommendation:
    model: ModelName
    device: Device
    precision: Precision
    image_size: int
    batch_size: int
    workers: int
    gradient_accumulation: int | None
    tested_precisions: tuple[str, ...]
    measured_available_ram_bytes: int | None
    measured_free_device_bytes: int | None
    measured_free_disk_bytes: int | None
    estimated_host_bytes: int
    estimated_device_bytes: int | None
    estimated_disk_floor_bytes: int
    checks_passed: bool
    issues: tuple[str, ...]
    reasoning: tuple[str, ...]


# Planning allowances in MiB, with variable memory expressed per image at 640 px.
# These are deliberately conservative and must be calibrated by a representative pilot.
_PROFILES = {
    ModelName.YOLO: (768, 480, 8),
    ModelName.FASTER_RCNN: (1024, 800, 4),
    ModelName.RT_DETR: (1408, 1024, 4),
}
_OPS = {
    ModelName.YOLO: {"nms"},
    ModelName.FASTER_RCNN: {"nms", "roi_align_backward"},
    ModelName.RT_DETR: set(),
}


def recommend(
    name: ModelName, environment: EnvironmentReport, overrides: Overrides = Overrides(), *,
    free_disk_bytes: int | None = None,
) -> Recommendation:
    backend = REGISTRY[name]
    issues: list[str] = []
    reasons = ["FP32 is the baseline for the first pilot. Mixed precision requires a passing autocast probe."]
    supported = tuple(device for device in ("cuda", "rocm", "mps", "cpu") if device in backend.devices)
    device = overrides.device or next((
        candidate for candidate in supported
        if getattr(environment.torch, candidate).readiness == "available"
        and "fp32" in getattr(environment.torch, candidate).precisions_tested
        and _OPS[name].issubset(getattr(environment.torch, candidate).operators_tested)
    ), "cpu")
    observed = getattr(environment.torch, device)
    precision = overrides.precision or "fp32"
    tested = tuple(mode for mode in observed.precisions_tested
                   if (mode == "fp32" or device in ("cuda", "rocm"))
                   and (mode != "bf16" or name != ModelName.YOLO))
    if device not in supported:
        issues.append(f"{name} does not expose the {device} training adapter.")
    if observed.readiness != "available":
        issues.append(f"PyTorch has no passing tensor execution probe on {device}.")
    if precision not in tested:
        issues.append(f"{precision} is unsupported or has no passing precision probe for {name} on {device}.")
    if missing := _OPS[name].difference(observed.operators_tested):
        issues.append(f"Required operator probes missing or failed on {device}: {', '.join(sorted(missing))}.")
    try:
        backend.versions()
    except TrainingError as exc:
        issues.append(str(exc))

    size = overrides.image_size if overrides.image_size is not None else 320
    if size < 64 or size % 32:
        raise TrainingError("Image size must be a multiple of 32 and at least 64.")
    accumulation = overrides.gradient_accumulation
    if accumulation is not None and (name == ModelName.YOLO or accumulation < 1):
        raise TrainingError("Set a positive gradient accumulation interval for Faster R-CNN or RT-DETR. "
                            "YOLO uses its native accumulation schedule.")
    fixed, per_image, cap = _PROFILES[name]
    ram = environment.resources.available_ram_bytes
    vram = observed.free_device_bytes
    budget = vram if device in ("cuda", "rocm") else ram
    if device == "mps" and budget is not None and observed.shared_memory_limit_bytes is not None:
        budget = min(budget, observed.shared_memory_limit_bytes)
    multiplier = 2 if name == ModelName.YOLO else 1  # native YOLO Validation uses double batch size
    variable = max(32, round(per_image * (size / 640)**2 * multiplier)) * MIB
    estimated_base = fixed * MIB
    if budget is None:
        recommended_batch = 1
        reasons.append("Available execution memory is unknown. Start with one image per batch.")
    else:
        recommended_batch = max(1, min(cap, (budget * 65 // 100 - estimated_base) // variable))
        reasons.append("Batch estimate uses 65% of observed free execution memory and a model-specific cap.")
    batch = recommended_batch if overrides.batch_size is None else overrides.batch_size
    if batch < 1:
        raise TrainingError("Batch size must be positive.")
    cpus = environment.resources.logical_cpus or 1
    workers = overrides.workers
    if workers is None:
        # Spawn overhead and duplicate Python/image state can dwarf tiny pilots.
        workers = min(2, cpus // 4) if device in ("cuda", "rocm") and (ram or 0) >= 8 * 1024**3 else 0
    if not 0 <= workers <= cpus:
        raise TrainingError("Set DataLoader workers between zero and the visible logical CPU count.")
    if device in ("cpu", "mps") and name == ModelName.YOLO and workers:
        issues.append("Ultralytics forces zero DataLoader workers on CPU and MPS. Specify zero workers.")
    host_overhead = (512 + 256 * workers * multiplier + 32 * batch * multiplier) * MIB
    execution = estimated_base + variable * batch
    host = host_overhead + (execution if device in ("cpu", "mps") else 0)
    accelerator = execution if device in ("cuda", "rocm") else None
    if (ram is not None and host > ram * 85 // 100
            or accelerator is not None and vram is not None and accelerator > vram * 85 // 100
            or device == "mps" and budget is not None and execution > budget * 85 // 100):
        issues.append("Estimated memory exceeds the available budget. Reduce batch size, image size, or workers.")
    disk = (free_disk_bytes if free_disk_bytes is not None
            else environment.resources.working_directory_free_bytes)
    disk_floor = CHECKPOINT_ALLOWANCE[name] * (4 if name == ModelName.YOLO else 3) + RESERVE_BYTES
    if disk is not None and disk < disk_floor:
        issues.append("Low disk space for checkpoints. Choose a different output volume.")
    reasons.append("Worker count accounts for host RAM and CPU count. Explicit settings take precedence.")
    reasons.append("Memory and checkpoint allowances are estimates. Native detector execution requires an approved pilot.")
    return Recommendation(
        name, device, precision, size, batch, workers, accumulation, tested, ram, vram, disk, host, accelerator,
        disk_floor, not issues, tuple(issues), tuple(reasons),
    )
