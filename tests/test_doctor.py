import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.environment import doctor, probes, torch_probe
from ict_pipeline.environment.model import Backend, GPU, Status, Torch


@pytest.fixture
def mock_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probes, "_PROC", tmp_path / "proc")
    monkeypatch.setattr(probes, "_PCI", tmp_path / "pci")
    monkeypatch.setattr(probes, "_NVIDIA_VERSION", tmp_path / "nvidia-version")
    monkeypatch.setattr(probes, "_exists", lambda path: False)
    monkeypatch.setattr(probes, "_is_file", lambda path: False)
    monkeypatch.setattr(probes.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(probes.shutil, "disk_usage", lambda path: SimpleNamespace(free=10_000))
    monkeypatch.setattr(doctor, "probe_torch", torch_probe._missing_torch)
    for name in ("WSL_INTEROP", "WSL_DISTRO_NAME", "container", "CONTAINER_SANDBOX_MOUNT_POINT"):
        monkeypatch.delenv(name, raising=False)


def mock_commands(
    monkeypatch: pytest.MonkeyPatch,
    commands: Mapping[str, str],
    outputs: Mapping[tuple[str, ...], str],
) -> None:
    def run(command: list[str], timeout: float = 4) -> tuple[Status, str]:
        assert 0 < timeout <= 8
        return "available", outputs[tuple(command[:3])]

    monkeypatch.setattr(probes.shutil, "which", commands.get)
    monkeypatch.setattr(probes, "run_probe", run)


@pytest.mark.usefixtures("mock_host")
def test_windows_cim_and_nvidia_inventory_without_torch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probes, "_platform_identity", lambda: ("Windows", "10.0.26100", "AMD64"))
    hardware = {
        "cpu": "AMD Ryzen",
        "ram": 16_000_000_000,
        "gpu": [
            {"Name": "Radeon Graphics", "PNPDeviceID": "PCI\\VEN_1002"},
            {"Name": "GeForce RTX", "PNPDeviceID": "PCI\\VEN_10DE"},
        ],
    }
    mock_commands(
        monkeypatch,
        {name: name for name in ("powershell.exe", "nvidia-smi.exe", "wsl.exe", "git", "docker")},
        {
            ("powershell.exe", "-NoProfile", "-NonInteractive"): json.dumps(hardware),
            ("nvidia-smi.exe", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"):
                '"NVIDIA GeForce RTX", 8192, 552.12',
        },
    )
    monkeypatch.setattr(probes, "_is_file", lambda path: path == tmp_path / "tools" / "aihubshell")
    report = doctor.collect_environment(tmp_path)
    assert report.host.os == "Windows"
    assert report.host.architecture == "AMD64"
    assert report.host.wsl is False
    assert report.resources.cpu_model == "AMD Ryzen"
    assert report.resources.ram_bytes == 16_000_000_000
    assert report.resources.logical_cpus == 8
    assert report.resources.working_directory_free_bytes == 10_000
    assert report.graphics.devices == (GPU("AMD", "Radeon Graphics"), GPU("NVIDIA", "NVIDIA GeForce RTX", 8 * 1024**3))
    assert report.graphics.nvidia_driver == "552.12"
    assert report.torch.detail == "not_installed"
    assert report.execution_modes["torch_cuda"] == "unavailable"
    assert report.execution_modes["wsl"] == "unknown"
    assert report.execution_modes["container"] == "unknown"
    assert report.tools == {
        "git": "available", "curl": "unavailable", "unzip": "unavailable",
        "docker": "available", "wsl": "available", "aihubshell": "available",
    }


@pytest.mark.usefixtures("mock_host")
def test_linux_pci_and_container_with_no_nvidia_smi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probes, "_platform_identity", lambda: ("Linux", "6.8.0", "x86_64"))
    proc = tmp_path / "proc"
    (proc / "1").mkdir(parents=True)
    (proc / "1" / "cgroup").write_text("0::/docker/123\n", encoding="utf-8")
    (proc / "cpuinfo").write_text("model name : AMD Ryzen\n", encoding="utf-8")
    (proc / "meminfo").write_text("MemTotal: 8192000 kB\n", encoding="utf-8")
    slot = tmp_path / "pci" / "0000:01:00.0"
    attributes = {"class": "0x030000", "vendor": "0x1002", "mem_info_vram_total": "8589934592"}
    read_file = probes._read

    def read(path: Path) -> str | None:
        return attributes.get(path.name) if path.parent == slot else read_file(path)

    monkeypatch.setattr(probes, "_PCI", SimpleNamespace(iterdir=lambda: iter([slot])))
    monkeypatch.setattr(probes, "_read", read)
    mock_commands(
        monkeypatch, {"lspci": "lspci"},
        {("lspci", "-D", "-mm"): '0000:01:00.0 "VGA compatible controller [0300]" "AMD [1002]" "RX 6600 [73ff]"'},
    )
    report = doctor.collect_environment(tmp_path)
    assert report.host.wsl is False
    assert report.host.container is True
    assert report.resources.cpu_model == "AMD Ryzen"
    assert report.resources.ram_bytes == 8192000 * 1024
    assert report.graphics.devices == (GPU("AMD", "RX 6600 [73ff]", 8 * 1024**3),)
    assert report.graphics.nvidia_smi == "unavailable"
    assert report.execution_modes["container"] == "available"
    assert report.execution_modes["torch_rocm"] == "unavailable"


@pytest.mark.usefixtures("mock_host")
def test_macos_apple_gpu_and_shared_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probes, "_platform_identity", lambda: ("Darwin", "24.0.0", "arm64"))
    mock_commands(
        monkeypatch, {"sysctl": "sysctl", "system_profiler": "system_profiler"},
        {
            ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M3",
            ("sysctl", "-n", "hw.memsize"): "17179869184",
            ("system_profiler", "-json", "-timeout"): json.dumps({
                "SPDisplaysDataType": [{"_name": "Apple M3", "spdisplays_vram_shared": "16 GB"}],
            }),
        },
    )
    report = doctor.collect_environment(tmp_path)
    assert report.host.architecture == "arm64"
    assert report.resources.cpu_model == "Apple M3"
    assert report.resources.ram_bytes == 16 * 1024**3
    assert report.graphics.devices == (GPU("Apple", "Apple M3"),)
    assert report.execution_modes["torch_mps"] == "unavailable"


@pytest.mark.usefixtures("mock_host")
def test_wsl_nvidia_smi_fallback_without_vram(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probes, "_platform_identity", lambda: ("Linux", "6.6.0-microsoft-standard-WSL2", "x86_64"))
    executable = "/usr/lib/wsl/lib/nvidia-smi"
    monkeypatch.setattr(probes, "_exists", lambda path: path == Path(executable))
    mock_commands(
        monkeypatch, {},
        {(executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"):
            "NVIDIA RTX, [N/A], 550.30"},
    )
    report = doctor.collect_environment(tmp_path)
    assert report.host.wsl is True
    assert report.execution_modes["wsl"] == "available"
    assert report.graphics.devices == (GPU("NVIDIA", "NVIDIA RTX"),)
    assert report.graphics.nvidia_driver == "550.30"


@pytest.mark.usefixtures("mock_host")
def test_linux_reports_loaded_nvidia_driver_when_tool_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(probes, "_platform_identity", lambda: ("Linux", "6.8.0", "x86_64"))
    (tmp_path / "nvidia-version").write_text("550.54.14\n", encoding="utf-8")
    mock_commands(monkeypatch, {}, {})
    report = doctor.collect_environment(tmp_path)
    assert report.graphics.nvidia_smi == "unavailable"
    assert report.graphics.nvidia_driver == "550.54.14"
    assert report.graphics.inventory == "unknown"


@pytest.mark.parametrize(
    ("cuda", "hip", "failing_device", "expected"),
    [
        ("12.8", None, "cuda:0", (Backend("available", "unavailable"), Backend("unavailable", "unavailable"))),
        (None, "6.4", "", (Backend("unavailable", "unavailable"), Backend("available", "available"))),
    ],
)
def test_torch_executes_tensor_operation(
    cuda: str | None, hip: str | None, failing_device: str,
    expected: tuple[Backend, Backend], monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        def __init__(self, device: str, value: int) -> None:
            self.device = device
            self.value = value

        def __add__(self, value: int) -> "Tensor":
            return Tensor(self.device, self.value + value)

        def item(self) -> int:
            if self.device == failing_device:
                raise RuntimeError("device error at synchronization")
            return self.value

    checked = []

    def ones(size: int, *, device: str) -> Tensor:
        assert size == 1
        checked.append(device)
        return Tensor(device, 1)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        __version__="2.8.0",
        version=SimpleNamespace(cuda=cuda, hip=hip),
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
        ones=ones,
    ))
    result = torch_probe._inspect_torch()
    assert result.cpu == result.mps == Backend("available", "available")
    assert (result.cuda, result.rocm) == expected
    assert checked == ["cpu", "cuda:0", "mps"]


def test_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    result = torch_probe._inspect_torch()
    assert result.status == "unavailable"
    assert result.detail == "not_installed"
    assert result.cuda.readiness == "unavailable"


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired(["probe"], 20), PermissionError("SECRET")])
def test_probe_failure_returns_unknown_without_leaking_diagnostics(
    failure: Exception, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(command: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        assert command[0] == sys.executable
        assert options["timeout"] == 20
        assert options["stdin"] == options["stderr"] == subprocess.DEVNULL
        raise failure

    monkeypatch.setattr(probes.subprocess, "run", fail)
    result = torch_probe.probe_torch()
    assert result == Torch("unknown", "probe_failed_or_timed_out")


def test_torch_probe_reads_only_framed_output(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "status": "available", "version": "2.8.0+cpu",
        "cpu": {"availability": "available", "readiness": "available"},
    }
    monkeypatch.setattr(torch_probe, "run_probe", lambda command, timeout: (
        "available", f"SECRET\nICT_TORCH_PROBE={json.dumps(payload)}\nother output",
    ))
    result = torch_probe.probe_torch()
    assert result.version == "2.8.0+cpu"
    assert result.cpu.readiness == "available"
    assert "SECRET" not in repr(result)


@pytest.mark.usefixtures("mock_host")
def test_doctor_cli_and_json_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probes, "_platform_identity", lambda: ("Linux", "6.8.0", "x86_64"))
    mock_commands(monkeypatch, {}, {})
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    human = runner.invoke(app, ["doctor"])
    assert human.exit_code == 0
    assert "ICT environment" in human.stdout
    assert "Execution modes" in human.stdout
    assert "Recommendations" in human.stdout

    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0
    assert result.stderr == ""
    payload = json.loads(result.stdout)
    schema_path = Path(__file__).resolve().parents[1] / "docs" / "doctor.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == schema["properties"]["schema_version"]["const"] == 1
    assert set(payload) == set(schema["required"])
    for name in ("host", "python", "resources", "graphics", "torch", "tools", "execution_modes"):
        assert set(payload[name]) == set(schema["$defs"][name]["required"])
    assert payload["graphics"]["inventory"] == "unknown"
    assert payload["graphics"]["devices"] == []
    assert payload["graphics"]["nvidia_driver"] is None
    assert payload["torch"]["detail"] == "not_installed"
    assert payload["execution_modes"]["local_cpu"] == "available"
    assert not list(tmp_path.iterdir())
