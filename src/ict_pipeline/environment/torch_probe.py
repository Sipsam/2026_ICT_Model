"""Isolate PyTorch import and device initialization in a timed child process."""

import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict, replace
from typing import Any, Protocol

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


def _measure(torch: Any, base: Backend, device: str) -> Backend:
    if base.readiness != "available":
        return base
    details: dict[str, Any] = {}
    if device.startswith("cuda:"):
        try:
            details["free_device_bytes"], details["total_device_bytes"] = torch.cuda.mem_get_info(device)
            details["device_name"] = torch.cuda.get_device_name(device)
        except Exception:
            pass
    if device == "mps":
        try:
            details["shared_memory_limit_bytes"] = torch.mps.recommended_max_memory()
        except Exception:
            pass
    precisions = []
    for mode in (("fp32", "fp16", "bf16") if device.startswith("cuda:") else ("fp32",)):
        try:
            if mode == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
                continue
            x = torch.ones((8, 8), device=device, requires_grad=True)
            with torch.autocast("cuda", enabled=mode != "fp32",
                                dtype=torch.bfloat16 if mode == "bf16" else torch.float16):
                y = x @ x
            if mode != "fp32" and y.dtype != (torch.bfloat16 if mode == "bf16" else torch.float16):
                continue
            y.float().mean().backward()
            if torch.isfinite(x.grad).all().item():
                precisions.append(mode)
        except Exception:
            continue
    operators = []
    try:
        from torchvision.ops import nms, roi_align

        box = torch.tensor([[0., 0., 2., 2.]], device=device)
        if nms(box, torch.ones(1, device=device), 0.5).item() == 0:
            operators.append("nms")
        if device != "mps":
            feature = torch.ones((1, 1, 4, 4), device=device, requires_grad=True)
            roi_align(feature, [box], (2, 2)).sum().backward()
            if torch.isfinite(feature.grad).all().item():
                operators.append("roi_align_backward")
    except Exception:
        pass
    return replace(base, precisions_tested=tuple(precisions), operators_tested=tuple(operators), **details)


def _inspect_torch(*, detailed: bool = False) -> Torch:
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
    report = Torch(
        status="available",
        version=str(torch.__version__),
        cuda_build=cuda,
        rocm_build=rocm,
        cpu=_check_backend(lambda: True, torch.ones, "cpu"),
        cuda=_check_backend(torch.cuda.is_available, torch.ones, "cuda:0") if cuda else _UNAVAILABLE,
        rocm=_check_backend(torch.cuda.is_available, torch.ones, "cuda:0") if rocm else _UNAVAILABLE,
        mps=_check_backend(mps.is_available, torch.ones, "mps") if mps else _UNAVAILABLE,
    )
    if not detailed:
        return report
    torch.set_num_threads(1)
    return replace(report, **{
        name: _measure(torch, getattr(report, name), device)
        for name, device in (("cpu", "cpu"), ("cuda", "cuda:0"), ("rocm", "cuda:0"), ("mps", "mps"))
    })


def _status(value: object) -> Status:
    if value == "available":
        return "available"
    if value == "unavailable":
        return "unavailable"
    return "unknown"


def _backend(value: object) -> Backend:
    if isinstance(value, dict):
        def count(key: str) -> int | None:
            item = value.get(key)
            return item if type(item) is int and item >= 0 else None

        def choices(key: str, allowed: tuple[str, ...]) -> tuple[str, ...]:
            items = value.get(key)
            return tuple(item for item in allowed if item in items) if isinstance(items, list) else ()

        name = value.get("device_name")
        return Backend(
            _status(value.get("availability")), _status(value.get("readiness")),
            name[:200] if isinstance(name, str) and name.isprintable() else None,
            count("free_device_bytes"), count("total_device_bytes"), count("shared_memory_limit_bytes"),
            choices("precisions_tested", ("fp32", "fp16", "bf16")),
            choices("operators_tested", ("nms", "roi_align_backward")),
        )
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
    print(_RESULT_PREFIX + json.dumps(asdict(_inspect_torch(detailed=True))))
