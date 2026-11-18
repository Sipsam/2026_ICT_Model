"""Isolate PyTorch import and device initialization in a timed child process."""

import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict
from typing import Protocol

from .model import Backend, Status, Torch
from .probes import run_probe

_RESULT_PREFIX = "ICT_TORCH_PROBE="
_UNAVAILABLE = Backend("unavailable", "unavailable")


def _missing_torch() -> Torch:
    return Torch(
        "unavailable", "not_installed",
        cpu=_UNAVAILABLE, cuda=_UNAVAILABLE, rocm=_UNAVAILABLE, mps=_UNAVAILABLE,
    )


class _Tensor(Protocol):
    def __add__(self, other: int) -> "_Tensor": ...

    def item(self) -> float: ...


def _check_backend(
    is_available: Callable[[], bool],
    ones: Callable[..., _Tensor],
    device: str,
) -> Backend:
    try:
        if not is_available():
            return _UNAVAILABLE
    except Exception:
        return Backend()
    try:
        # item() waits for the result, surfacing asynchronous device errors.
        ready = (ones(1, device=device) + 1).item() == 2
    except Exception:
        ready = False
    return Backend("available", "available" if ready else "unavailable")


def _inspect_torch() -> Torch:
    try:
        import torch
    except ModuleNotFoundError as error:
        if error.name == "torch":
            return _missing_torch()
        return Torch("unknown", "import_failed")
    except Exception:
        return Torch("unknown", "import_failed")

    cuda = getattr(torch.version, "cuda", None)
    rocm = getattr(torch.version, "hip", None)
    mps = getattr(torch.backends, "mps", None)
    return Torch(
        status="available",
        version=str(torch.__version__),
        cuda_build=cuda,
        rocm_build=rocm,
        cpu=_check_backend(lambda: True, torch.ones, "cpu"),
        cuda=_check_backend(torch.cuda.is_available, torch.ones, "cuda:0") if cuda else _UNAVAILABLE,
        rocm=_check_backend(torch.cuda.is_available, torch.ones, "cuda:0") if rocm else _UNAVAILABLE,
        mps=_check_backend(mps.is_available, torch.ones, "mps") if mps else _UNAVAILABLE,
    )


def _status(value: object) -> Status:
    if value == "available":
        return "available"
    if value == "unavailable":
        return "unavailable"
    return "unknown"


def _backend(value: object) -> Backend:
    if isinstance(value, dict):
        return Backend(_status(value.get("availability")), _status(value.get("readiness")))
    return Backend()


def _version(value: object) -> str | None:
    if isinstance(value, str) and re.fullmatch(r"\d[0-9a-zA-Z.+_-]{0,79}", value):
        return value
    return None


def probe_torch() -> Torch:
    # -B avoids writing bytecode and -P avoids importing torch from the working directory.
    # Use the current interpreter so the probe sees its installed packages.
    status, output = run_probe(
        [sys.executable, "-B", "-P", "-m", "ict_pipeline.environment.torch_probe"],
        timeout=20,
    )
    if status == "available":
        for line in reversed(output.splitlines()):
            if not line.startswith(_RESULT_PREFIX):
                continue
            try:
                result = json.loads(line.removeprefix(_RESULT_PREFIX))
            except ValueError:
                break
            if not isinstance(result, dict):
                break
            if result.get("status") == "available":
                return Torch(
                    status="available",
                    version=_version(result.get("version")),
                    cuda_build=_version(result.get("cuda_build")),
                    rocm_build=_version(result.get("rocm_build")),
                    cpu=_backend(result.get("cpu")),
                    cuda=_backend(result.get("cuda")),
                    rocm=_backend(result.get("rocm")),
                    mps=_backend(result.get("mps")),
                )
            if result.get("detail") == "not_installed":
                return _missing_torch()
            if result.get("detail") == "import_failed":
                return Torch("unknown", "import_failed")
            break
    return Torch("unknown", "probe_failed_or_timed_out")


if __name__ == "__main__":
    print(_RESULT_PREFIX + json.dumps(asdict(_inspect_torch())))
