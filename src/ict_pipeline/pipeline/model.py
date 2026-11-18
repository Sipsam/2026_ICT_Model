"""Stage declarations and portable artifact contracts."""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PipelineError(Exception):
    """An operator-facing diagnostic produced by the runner."""


class StageFailure(PipelineError):
    """An adapter failure with a diagnostic safe to persist in run state."""


_WINDOWS_DEVICE = re.compile(r"(?:con|prn|aux|nul|com[0-9]|lpt[0-9])", re.IGNORECASE)


def valid_id(value: str) -> bool:
    return (
        re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", value) is not None
        and _WINDOWS_DEVICE.fullmatch(value) is None
    )


def portable_artifact(path: Path) -> bool:
    return bool(path.parts) and all(
        part != ".."
        and re.fullmatch(r"[a-zA-Z0-9_-][a-zA-Z0-9_.-]*", part) is not None
        and not part.endswith(".")
        and _WINDOWS_DEVICE.fullmatch(part.partition(".")[0]) is None
        for part in path.parts
    )


@dataclass(frozen=True)
class StageInput:
    name: str
    path: Path
    # Relative to the workspace for external files, or the producing stage's
    # published artifact directory when a producer is declared.
    producer: str | None = None


class StageResult(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    MISSING_REQUIREMENT = "missing_requirement"
    INVALID_DATA = "invalid_data"


@dataclass(frozen=True)
class StageContext:
    inputs: Mapping[str, Path]
    settings: Mapping[str, object]
    output_dir: Path
    artifact_dir: Path
    progress: Callable[[int, int | None], None]
    input_fingerprints: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Stage:
    name: str
    dependencies: tuple[str, ...]
    inputs: tuple[StageInput, ...]
    outputs: tuple[Path, ...]
    config_keys: tuple[str, ...]
    version: str
    execute: Callable[[StageContext], StageResult] | None = None


class StageGraph:
    def __init__(self, stages: Iterable[Stage]) -> None:
        self.stages: dict[str, Stage] = {}
        for stage in stages:
            if not valid_id(stage.name) or stage.name in self.stages:
                raise PipelineError("Stage names must be unique portable identifiers.")
            self.stages[stage.name] = stage
        if not self.stages:
            raise PipelineError("Declare at least one stage.")

        ordered: dict[str, None] = {}
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in ordered:
                return
            if name in visiting:
                raise PipelineError(f"Dependency cycle involving stage {name}.")
            visiting.add(name)
            stage = self.stages[name]
            if not stage.version or not stage.outputs or any(
                not portable_artifact(path) for path in stage.outputs
            ):
                raise PipelineError(f"Declare a version and portable output paths for {name}.")
            if len({item.name for item in stage.inputs}) != len(stage.inputs) or any(
                not valid_id(item.name) for item in stage.inputs
            ):
                raise PipelineError(f"Declare unique portable input names for {name}.")
            for dependency in stage.dependencies:
                if dependency not in self.stages:
                    raise PipelineError(f"Stage {name} has an unknown dependency.")
                visit(dependency)
            for item in stage.inputs:
                if item.producer is not None and (
                    item.producer not in stage.dependencies
                    or item.path not in self.stages[item.producer].outputs
                ):
                    raise PipelineError(f"Input {item.name} for {name} requires a declared dependency output.")
            visiting.remove(name)
            ordered[name] = None

        for name in self.stages:
            visit(name)
        self.order = tuple(ordered)

    def select(
        self, start: str | None, until: str | None, only: Iterable[str]
    ) -> tuple[str, ...]:
        only = tuple(only)
        if only and (start is not None or until is not None):
            raise PipelineError("Use --only by itself or select a range with --from and --until.")
        if any(name not in self.stages for name in (*only, start, until) if name is not None):
            raise PipelineError("Unknown stage. Choose from " + ", ".join(self.order) + ".")
        if only:
            return tuple(name for name in self.order if name in only)
        first = self.order.index(start) if start is not None else 0
        last = self.order.index(until) if until is not None else len(self.order) - 1
        if first > last:
            raise PipelineError("--from must precede or equal --until.")
        return self.order[first:last + 1]
