"""Versioned data contract for local environment diagnostics."""

from dataclasses import asdict, dataclass
from typing import Literal

Status = Literal["available", "unavailable", "unknown"]


@dataclass(frozen=True)
class Host:
    os: str
    release: str | None
    architecture: str | None
    wsl: bool | None
    container: bool | None


@dataclass(frozen=True)
class Python:
    version: str
    implementation: str
    venv: bool


@dataclass(frozen=True)
class Resources:
    cpu_model: str | None
    logical_cpus: int | None
    ram_bytes: int | None
    working_directory_free_bytes: int | None


@dataclass(frozen=True)
class GPU:
    vendor: str | None
    model: str | None
    vram_bytes: int | None = None


@dataclass(frozen=True)
class Graphics:
    inventory: Status
    devices: tuple[GPU, ...]
    nvidia_smi: Status
    nvidia_driver: str | None


@dataclass(frozen=True)
class Backend:
    availability: Status = "unknown"
    readiness: Status = "unknown"


@dataclass(frozen=True)
class Torch:
    status: Status
    detail: str | None = None
    version: str | None = None
    cuda_build: str | None = None
    rocm_build: str | None = None
    cpu: Backend = Backend()
    cuda: Backend = Backend()
    rocm: Backend = Backend()
    mps: Backend = Backend()


@dataclass(frozen=True)
class EnvironmentReport:
    host: Host
    python: Python
    resources: Resources
    graphics: Graphics
    torch: Torch
    tools: dict[str, Status]
    execution_modes: dict[str, Status]
    recommendations: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"schema_version": 1, **asdict(self)}
