"""Capture the execution environment without loading ML frameworks."""

import re
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from ..environment.probes import probe_hardware, probe_host, probe_python, run_probe

_PACKAGES = ("ict-pipeline", "pydantic", "PyYAML", "typer", "rich", "python-dotenv", "Pillow", "openpyxl")


def runtime_snapshot(workspace: Path, packages: tuple[str, ...] = ()) -> dict[str, object]:
    _, commit = run_probe(["git", "-C", str(workspace), "rev-parse", "--verify", "HEAD"])
    host = probe_host()
    resources, graphics = probe_hardware(host, workspace)
    versions: dict[str, str | None] = {}
    for package in dict.fromkeys((*_PACKAGES, *packages)):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return {
        "git_commit": commit if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit) else None,
        "python": asdict(probe_python()),
        "packages": versions,
        "os": asdict(host),
        "hardware": {"resources": asdict(resources), "graphics": asdict(graphics)},
    }
