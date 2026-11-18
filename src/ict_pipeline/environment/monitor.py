"""Best-effort resource observations for one foreground model attempt."""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Any


def _rss() -> int | None:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
        except (OSError, ValueError):
            pass
    try:
        from psutil import Process

        return Process().memory_info().rss
    except Exception:
        return None


@contextmanager
def observe_training(device: str, result: dict[str, object]) -> Iterator[None]:
    stopped = Event()
    rss = None
    mps_peak = None
    accelerator: Any = None
    torch = sys.modules.get("torch")
    if torch is not None and device.startswith("cuda"):
        try:
            accelerator = torch.cuda
            accelerator.reset_peak_memory_stats(device)
        except Exception:
            accelerator = None

    def sample() -> None:
        nonlocal rss, mps_peak
        if (current := _rss()) is not None:
            rss = max(rss or 0, current)
        if torch is not None and device == "mps":
            try:
                mps_peak = max(mps_peak or 0, torch.mps.current_allocated_memory())
            except Exception:
                pass

    def poll() -> None:
        while not stopped.wait(0.1):
            sample()

    sample()
    sampler = Thread(target=poll, daemon=True)
    sampler.start()
    started = perf_counter()
    try:
        yield
    finally:
        result["adapter_wall_seconds"] = perf_counter() - started
        stopped.set()
        sampler.join()
        sample()
        result["sampled_peak_process_rss_bytes"] = rss
        result["sample_interval_seconds"] = 0.1
        result["peak_torch_device_allocated_bytes"] = None
        result["peak_torch_device_reserved_bytes"] = None
        result["sampled_peak_mps_allocated_bytes"] = mps_peak
        if accelerator is not None:
            try:
                result["peak_torch_device_allocated_bytes"] = accelerator.max_memory_allocated(device)
                result["peak_torch_device_reserved_bytes"] = accelerator.max_memory_reserved(device)
            except Exception:
                pass


def logical_file_bytes(paths: Iterator[Path]) -> int:
    return sum(path.stat().st_size for path in paths if path.is_file())
