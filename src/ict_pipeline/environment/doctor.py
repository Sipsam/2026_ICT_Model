"""Collect an environment report for the CLI and future setup workflows."""

from pathlib import Path

from .model import EnvironmentReport, Status
from .probes import probe_hardware, probe_host, probe_python, probe_tools
from .torch_probe import probe_torch


def collect_environment(directory: Path) -> EnvironmentReport:
    host = probe_host()
    python = probe_python()
    resources, graphics = probe_hardware(host, directory)
    torch = probe_torch()
    tools = probe_tools(host, directory)

    modes: dict[str, Status] = {
        "local_cpu": "available",
        "torch_cpu": torch.cpu.readiness,
        "torch_cuda": torch.cuda.readiness,
        "torch_rocm": torch.rocm.readiness,
        "torch_mps": torch.mps.readiness,
        "wsl": "available" if host.wsl else "unknown" if tools["wsl"] == "available" else "unavailable",
        "container": "available" if host.container else "unknown",
    }
    recommendations = []
    if "available" not in (torch.cuda.readiness, torch.rocm.readiness, torch.mps.readiness):
        recommendations.append("Use local CPU for CLI diagnostics and preprocessing.")
        if torch.cpu.readiness == "available":
            recommendations.append("Use PyTorch CPU for small device smoke tests.")
        recommendations.append(
            "For GPU workloads, check the driver and PyTorch build for your backend or use a verified GPU host."
        )
    if torch.detail == "not_installed":
        recommendations.append("Check the intended Python environment for PyTorch before running model workloads.")
    elif torch.status == "unknown":
        recommendations.append("Check whether PyTorch imports successfully in this Python environment.")
    if tools["docker"] == "available" and not host.container:
        recommendations.append("Verify the Docker daemon and device access before relying on Docker.")
    if tools["wsl"] == "available" and not host.wsl:
        recommendations.append("Verify a WSL distribution and run doctor inside it before relying on WSL.")
    if tools["aihubshell"] == "unavailable":
        recommendations.append("Use an existing dataset or locate the AI Hub tool before acquisition.")
    return EnvironmentReport(host, python, resources, graphics, torch, tools, modes, tuple(recommendations))
