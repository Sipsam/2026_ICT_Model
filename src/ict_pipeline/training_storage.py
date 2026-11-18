"""Estimate training storage and publish immutable checkpoint snapshots."""

import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING

from .data.preprocessing.common import require_separate_output
from .data.preprocessing.console import protect_source_data
from .disk import MIB, RESERVE_BYTES, StorageError, require_space, volume_at
from .models.contracts import ModelName, TrainingError

if TYPE_CHECKING:
    from .models.training import TrainingPlan


# Planning allowances, not benchmark results. The estimate assumes filesystem
# copies even when hard links are available and leaves room for atomic writes.
CHECKPOINT_ALLOWANCE = {
    ModelName.YOLO: 64 * MIB,
    ModelName.FASTER_RCNN: 320 * MIB,
    ModelName.RT_DETR: 512 * MIB,
}


@dataclass(frozen=True)
class StorageEstimate:
    path: str
    measured_free_bytes: int
    measured_dataset_copy_bytes: int
    measured_staging_bytes: int
    estimated_checkpoint_bytes: int
    estimated_cache_bytes: int
    required_bytes: int


def cache_root(plan: "TrainingPlan") -> Path | None:
    configured = os.environ.get("ICT_TRAIN_CACHE_ROOT")
    if not configured or ModelName.YOLO not in plan.models:
        return None
    path = (plan.workspace / configured).resolve()
    plan.credentials.protect_path(str(path), "training cache root")
    try:
        for source in (plan.dataset, plan.resume_from, *plan.weight_sources.values()):
            if source is not None:
                require_separate_output(source, path)
        protect_source_data(path, plan.workspace)
        protect_source_data(path, Path.cwd())
    except ValueError:
        raise TrainingError("Choose a training cache root outside protected or input directories.") from None
    return path


def _file_bytes(path: Path | None) -> int:
    if path is None:
        return 0
    try:
        return path.stat().st_size
    except FileNotFoundError:
        # A preview can still display its estimate before approved weights are staged.
        return 0


def storage_estimates(plan: "TrainingPlan") -> tuple[StorageEstimate, ...]:
    dataset_bytes = cache_bytes = staged_bytes = checkpoints = 0
    scratch = cache_root(plan)
    for name, config in plan.models.items():
        staged_bytes += _file_bytes(plan.weight_sources.get(name))
        unit = CHECKPOINT_ALLOWANCE[name]
        if plan.resume_from is not None:
            directory = plan.resume_from / "weights" if name == ModelName.YOLO else plan.resume_from
            last, best = (_file_bytes(directory / f"{kind}.pt") for kind in ("last", "best"))
            staged_bytes += last + best
            unit = max(unit, last, best)
        periodic = config.epochs // config.save_period if config.save_period else 0
        copies = 2 + (2 if name == ModelName.YOLO else 1) + periodic + 2 * plan.publish_checkpoints
        checkpoints += copies * unit
        if name == ModelName.YOLO:
            for kind in ("images", "labels"):
                dataset_bytes += sum(path.stat().st_size for path in (plan.dataset / "yolo" / kind).rglob("*")
                                     if path.is_file() and path.parent.name in ("train", "val"))
            # Ultralytics caches labels and metadata beside the private copy.
            cache_bytes = max(16 * MIB, dataset_bytes // 10)
    try:
        output = volume_at(plan.destination)
        cache = volume_at(scratch) if scratch is not None else output
    except StorageError as exc:
        raise TrainingError(str(exc)) from None
    on_output = output.device == cache.device
    writes = checkpoints + staged_bytes
    return (
        StorageEstimate(
            str(output.path), output.free_bytes, dataset_bytes if on_output else 0, staged_bytes, checkpoints,
            cache_bytes if on_output else 0, RESERVE_BYTES + writes + (dataset_bytes + cache_bytes if on_output else 0),
        ),
        *(() if on_output else (
            StorageEstimate(str(cache.path), cache.free_bytes, dataset_bytes, 0, 0, cache_bytes,
                            RESERVE_BYTES + dataset_bytes + cache_bytes),
        )),
    )


def check_training_storage(estimates: tuple[StorageEstimate, ...]) -> None:
    for estimate in estimates:
        if estimate.measured_free_bytes < estimate.required_bytes:
            raise TrainingError(
                f"Low disk space for training at {estimate.path}. "
                f"Measured free bytes: {estimate.measured_free_bytes}; estimated required bytes: "
                f"{estimate.required_bytes}. Choose another runs or cache volume, or reduce periodic checkpoints."
            )


def guard_training_write(path: Path, write_bytes: int = 0) -> None:
    try:
        require_space(path, write_bytes, label="training")
    except StorageError as exc:
        raise TrainingError(str(exc)) from None


def _stage_snapshot(source: Path, target: Path) -> Path:
    temporary = target.with_suffix(".snapshot-partial")
    try:
        try:
            temporary.hardlink_to(source)
        except OSError:
            guard_training_write(target.parent, source.stat().st_size)
            copyfile(source, temporary)
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def commit_checkpoint(temporary: Path, last: Path, snapshots: tuple[Path, ...]) -> None:
    """Stage aliases before replacing any prior checkpoint, with a copy fallback."""
    with ExitStack() as cleanup:
        cleanup.callback(temporary.unlink, missing_ok=True)
        staged = []
        for target in snapshots:
            alias = _stage_snapshot(temporary, target)
            cleanup.callback(alias.unlink, missing_ok=True)
            staged.append((alias, target))
        for alias, target in staged:
            alias.replace(target)
        temporary.replace(last)
