"""Read-only adapters for host resources, GPU inventory, and tool discovery."""

import csv
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from .model import GPU, Graphics, Host, Python, Resources, Status

_PROC = Path("/proc")
_PCI = Path("/sys/bus/pci/devices")
_NVIDIA_VERSION = Path("/sys/module/nvidia/version")
_VENDORS = {"10de": "NVIDIA", "1002": "AMD", "8086": "Intel", "106b": "Apple"}
_WINDOWS_HARDWARE = """
[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name
$ram = Get-CimInstance Win32_ComputerSystem | Select-Object -ExpandProperty TotalPhysicalMemory
$gpu = Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID,AdapterCompatibility
@{cpu = $cpu; ram = $ram; gpu = @($gpu)} | ConvertTo-Json -Compress -Depth 3
"""


def run_probe(command: list[str], timeout: float = 4) -> tuple[Status, str]:
    """Return successful stdout for parsing. Discard subprocess diagnostics."""
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown", ""
    if result.returncode != 0:
        return "unknown", ""
    return "available", result.stdout.strip()


def _read(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8", errors="replace") as stream:
            return stream.read(262_144).strip()
    except OSError:
        return None


def _exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _platform_identity() -> tuple[str, str | None, str | None]:
    if sys.platform == "win32":
        version = sys.getwindowsversion().platform_version
        arch = os.getenv("PROCESSOR_ARCHITEW6432") or os.getenv("PROCESSOR_ARCHITECTURE")
        return "Windows", ".".join(map(str, version)), arch
    kernel = os.uname()
    return kernel.sysname, kernel.release or None, kernel.machine or None


def probe_host() -> Host:
    system, release, arch = _platform_identity()
    wsl: bool | None = False
    container: bool | None = None
    if system == "Linux":
        kernel = _read(_PROC / "version")
        wsl = bool(
            "microsoft" in f"{release} {kernel}".lower()
            or os.getenv("WSL_INTEROP")
            or os.getenv("WSL_DISTRO_NAME")
        )
        cgroup = _read(_PROC / "1" / "cgroup")
        if (
            _exists(Path("/.dockerenv"))
            or _exists(Path("/run/.containerenv"))
            or bool(_read(Path("/run/systemd/container")))
            or os.getenv("container")
            or any(marker in (cgroup or "") for marker in ("docker", "kubepods", "containerd", "libpod", "lxc"))
        ):
            container = True
        # A private cgroup namespace can hide container evidence.
    elif system == "Windows" and os.getenv("CONTAINER_SANDBOX_MOUNT_POINT"):
        container = True
    elif system == "Darwin":
        container = False
    return Host(system, release, arch, wsl, container)


def probe_python() -> Python:
    return Python(platform.python_version(), platform.python_implementation(), sys.prefix != sys.base_prefix)


def _object(output: str) -> dict[str, object]:
    try:
        value = json.loads(output)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}


def _string(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    return None


def _positive_int(value: object) -> int | None:
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError:
            return None
    return value if type(value) is int and value > 0 else None


def _vendor(name: str) -> str | None:
    normalized = name.lower()
    for vendor, pattern in (
        ("NVIDIA", r"nvidia"),
        ("AMD", r"\bamd\b|radeon|advanced micro devices|\bati\b"),
        ("Intel", r"intel"),
        ("Apple", r"apple"),
    ):
        if re.search(pattern, normalized):
            return vendor
    return None


def _windows_hardware() -> tuple[str | None, int | None, list[GPU]]:
    powershell = shutil.which("powershell.exe") or shutil.which("pwsh.exe")
    if not powershell:
        return None, None, []
    _, output = run_probe(
        [powershell, "-NoProfile", "-NonInteractive", "-Command", _WINDOWS_HARDWARE],
        timeout=8,
    )
    hardware = _object(output)
    devices = []
    controllers = hardware.get("gpu")
    if isinstance(controllers, list):
        for item in controllers:
            if not isinstance(item, dict) or not (name := _string(item.get("Name"))):
                continue
            pci = re.search(r"VEN_([0-9a-f]{4})", _string(item.get("PNPDeviceID")) or "", re.I)
            vendor = _VENDORS.get(pci[1].lower()) if pci else None
            vendor = vendor or _vendor(f"{item.get('AdapterCompatibility')} {name}")
            # Win32_VideoController.AdapterRAM is only 32 bits.
            devices.append(GPU(vendor, name))
    return _string(hardware.get("cpu")), _positive_int(hardware.get("ram")), devices


def _pci_display_names() -> dict[str, tuple[str, str]]:
    lspci = shutil.which("lspci")
    if not lspci:
        return {}
    _, output = run_probe([lspci, "-D", "-mm", "-nn"])
    devices = {}
    for line in output.splitlines():
        try:
            fields = shlex.split(line)
        except ValueError:
            continue
        if len(fields) >= 4 and re.search(r"\[03[0-9a-f]{2}\]$", fields[1], re.I):
            slot, _, vendor, model, *_ = fields
            vendor_id = re.search(r"\[([0-9a-f]{4})\]$", vendor, re.I)
            if vendor_id:
                devices[slot.lower()] = (vendor_id[1].lower(), model)
    return devices


def _linux_graphics() -> list[GPU]:
    names = _pci_display_names()
    try:
        slots = sorted(_PCI.iterdir())
    except OSError:
        slots = []
    slots_by_name = {
        slot.name.lower(): slot
        for slot in slots
        if (_read(slot / "class") or "").startswith("0x03")
    }
    devices = []
    for slot_name in sorted(names.keys() | slots_by_name.keys()):
        vendor_id, model = names.get(slot_name, ("", None))
        slot = slots_by_name.get(slot_name)
        vram = None
        if slot is not None:
            vendor_id = (_read(slot / "vendor") or "").removeprefix("0x").lower() or vendor_id
            # amdgpu exposes dedicated VRAM in bytes where the driver can report it.
            vram = _positive_int(_read(slot / "mem_info_vram_total"))
        devices.append(GPU(_VENDORS.get(vendor_id), model, vram))
    return devices


def _mac_hardware() -> tuple[str | None, int | None, list[GPU]]:
    cpu = ram = None
    if sysctl := shutil.which("sysctl"):
        _, cpu = run_probe([sysctl, "-n", "machdep.cpu.brand_string"])
        _, ram = run_probe([sysctl, "-n", "hw.memsize"])
    devices = []
    if profiler := shutil.which("system_profiler"):
        _, output = run_probe([profiler, "-json", "-timeout", "5", "SPDisplaysDataType"], timeout=8)
        displays = _object(output).get("SPDisplaysDataType")
        if isinstance(displays, list):
            for item in displays:
                if not isinstance(item, dict):
                    continue
                name = _string(item.get("sppci_model")) or _string(item.get("_name"))
                vendor = _vendor(f"{item.get('spdisplays_vendor')} {name}")
                dedicated = _string(item.get("spdisplays_vram"))
                match = re.fullmatch(r"(\d+)\s*(MB|GB)", dedicated or "", re.I)
                vram = None
                if match:
                    multiplier = 1024**2 if match[2].upper() == "MB" else 1024**3
                    vram = int(match[1]) * multiplier
                if name or vendor or vram:
                    devices.append(GPU(vendor, name, vram))
    return cpu or None, _positive_int(ram), devices


def _nvidia_graphics(host: Host) -> tuple[Status, list[GPU], str | None]:
    command = "nvidia-smi.exe" if host.os == "Windows" else "nvidia-smi"
    executable = shutil.which(command)
    if not executable and host.wsl and _exists(Path("/usr/lib/wsl/lib/nvidia-smi")):
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    if not executable:
        return "unavailable", [], None
    status, output = run_probe([
        executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits",
    ])
    devices = []
    driver = None
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != 3 or not (name := _string(row[0])) or name.startswith("["):
            continue
        mib = _positive_int(row[1])
        devices.append(GPU("NVIDIA", name, mib * 1024 ** 2 if mib is not None else None))
        version = row[2].strip()
        if re.fullmatch(r"\d+(?:\.\d+)+", version):
            driver = version
    if not devices:
        status = "unknown"
    return status, devices, driver


def probe_hardware(host: Host, directory: Path) -> tuple[Resources, Graphics]:
    cpu, ram, devices = None, None, []
    if host.os == "Windows":
        cpu, ram, devices = _windows_hardware()
    elif host.os == "Linux":
        cpuinfo = _read(_PROC / "cpuinfo") or ""
        model = re.search(r"^(?:model name|Hardware)\s*:[ \t]*(.+)$", cpuinfo, re.M)
        cpu = _string(model[1]) if model else None
        memory = re.search(r"^MemTotal:\s+(\d+) kB$", _read(_PROC / "meminfo") or "", re.M)
        ram = int(memory[1]) * 1024 if memory else None
        devices = _linux_graphics()
    elif host.os == "Darwin":
        cpu, ram, devices = _mac_hardware()

    try:
        free = shutil.disk_usage(directory).free
    except OSError:
        free = None
    nvidia_smi, nvidia, driver = _nvidia_graphics(host)
    if nvidia:
        # Prefer the driver's view of NVIDIA devices to avoid duplicate entries.
        devices = [device for device in devices if device.vendor != "NVIDIA"] + nvidia
    if driver is None and host.os == "Linux":
        version = _read(_NVIDIA_VERSION)
        if version and re.fullmatch(r"\d+(?:\.\d+)+", version):
            driver = version
    return (
        Resources(cpu, os.cpu_count(), ram, free, available_ram(host.os)),
        Graphics("available" if devices else "unknown", tuple(devices), nvidia_smi, driver),
    )


def available_ram(system: str) -> int | None:
    if system == "Linux":
        match = re.search(r"^MemAvailable:\s+(\d+) kB$", _read(_PROC / "meminfo") or "", re.M)
        available = int(match[1]) * 1024 if match else None
        # In a container, host MemAvailable can exceed its cgroup budget.
        group = next((line[3:] for line in (_read(_PROC / "self/cgroup") or "").splitlines()
                      if line.startswith("0::/")), "/")
        mount = Path("/sys/fs/cgroup")
        leaf = mount / group.lstrip("/")
        for directory in (leaf, *leaf.parents):
            if not directory.is_relative_to(mount):
                break
            limit = _positive_int(_read(directory / "memory.max"))
            used = _positive_int(_read(directory / "memory.current"))
            if limit is not None and used is not None:
                remaining = max(0, limit - used)
                available = remaining if available is None else min(remaining, available)
        return available
    if system == "Windows" and sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total", "virtual_avail", "extended")
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.available)
    if system == "Darwin" and (vm_stat := shutil.which("vm_stat")):
        _, output = run_probe([vm_stat])
        page = re.search(r"page size of (\d+) bytes", output)
        free = re.search(r"^Pages free:\s+(\d+)\.", output, re.M)
        inactive = re.search(r"^Pages inactive:\s+(\d+)\.", output, re.M)
        if page and free and inactive:
            return int(page[1]) * (int(free[1]) + int(inactive[1]))
    return None


def probe_tools(host: Host, directory: Path) -> dict[str, Status]:
    tools: dict[str, Status] = {
        tool: "available" if shutil.which(tool) else "unavailable"
        for tool in ("git", "curl", "unzip", "docker")
    }
    tools["wsl"] = "available" if (host.os == "Windows" or host.wsl) and shutil.which("wsl.exe") else "unavailable"
    source_root = Path(__file__).resolve().parents[3]
    aihub_present = shutil.which("aihubshell") or any(
        _is_file(root / "tools" / "aihubshell") for root in (directory, source_root)
    )
    tools["aihubshell"] = "available" if aihub_present else "unavailable"
    return tools
