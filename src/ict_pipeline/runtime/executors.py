"""Adapters for native processes, local Docker, and OpenSSH."""

import csv
import io
import json
import os
import re
import shlex
import subprocess
from pathlib import Path, PurePosixPath
from shutil import disk_usage, which
from typing import Protocol

from ..config.settings import Credentials
from .jobs import Progress, absolute_path, dispatch, job_store, require_disjoint_paths
from .model import ExecutorError, WorkerRequest

_WORKER = ("-m", "ict_pipeline.runtime.worker")
_CONTAINER_ROOT = PurePosixPath("/ict/jobs")
_SSH_ENVIRONMENT = {
    "PATH", "HOME", "USER", "LOGNAME", "USERPROFILE", "USERNAME", "HOMEDRIVE", "HOMEPATH",
    "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "SSH_AUTH_SOCK",
}


class Executor(Protocol):
    def execute(self, request: WorkerRequest, on_event: Progress | None = None) -> dict[str, object]: ...


def _command(
    argv: list[str], *, label: str, payload: str | None = None, timeout: float | None = 45,
    env: dict[str, str] | None = None,
) -> str:
    executable = which(argv[0])
    if executable is None or Path(executable).suffix.lower() in (".cmd", ".bat"):
        raise ExecutorError(f"{label} requires an installed native executable on PATH.")
    try:
        result = subprocess.run(
            [executable, *argv[1:]], input=payload, stdin=subprocess.DEVNULL if payload is None else None,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, encoding="utf-8", errors="replace",
            env=env, timeout=timeout, check=False, shell=False,
        )
    except subprocess.TimeoutExpired:
        raise ExecutorError(f"{label} timed out. For SSH jobs, check job status before retrying.") from None
    except (OSError, ValueError):
        raise ExecutorError(f"Could not start {label}. Check executable access and settings.") from None
    if result.returncode:
        raise ExecutorError(f"{label} failed (exit {result.returncode}). Check the target runtime and access.")
    return result.stdout


def _worker_result(value: str, credentials: Credentials) -> dict[str, object]:
    # SSH login text can precede the single protocol response.
    try:
        response = json.loads(value.strip().splitlines()[-1])
        if not isinstance(response, dict) or response.get("schema_version") != 1:
            raise ValueError
        error = response.get("error")
        if isinstance(error, str):
            raise ExecutorError(credentials.redact_text(error))
        result = credentials.redact(response.get("result"))
        if not isinstance(result, dict):
            raise ValueError
    except (ValueError, IndexError):
        raise ExecutorError("The target did not return a compatible runtime response. Check its ICT installation.") from None
    return result


class LocalExecutor:
    def __init__(self, credentials: Credentials | None = None) -> None:
        self.credentials = credentials or Credentials.load({}, os.environ)

    def execute(self, request: WorkerRequest, on_event: Progress | None = None) -> dict[str, object]:
        if request.action not in {"preflight", "run", "status", "logs", "snapshot"}:
            raise ExecutorError("Use the local runner for foreground jobs. Detached jobs require SSH.")
        result = self.credentials.redact(dispatch(request, self.credentials, on_event))
        assert isinstance(result, dict)
        return result


def _bind(host: Path, guest: PurePosixPath, *, readonly: bool) -> str:
    # Docker's --mount parser accepts CSV quoting for paths with commas.
    stream = io.StringIO(newline="")
    fields = ["type=bind", f"source={host}", f"target={guest}"]
    if readonly:
        fields.extend(("readonly", "bind-recursive=disabled"))
    csv.writer(stream).writerow(fields)
    return stream.getvalue().removesuffix("\r\n")


class ContainerExecutor:
    def __init__(self, image: str, credentials: Credentials | None = None) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/@:-]{0,254}", image):
            raise ExecutorError("Provide a trusted, locally built Docker image name or digest.")
        self.image = image
        self.credentials = credentials or Credentials.load({}, os.environ)

    def execute(self, request: WorkerRequest, on_event: Progress | None = None) -> dict[str, object]:
        if request.action not in {"preflight", "run", "status", "logs", "snapshot"}:
            raise ExecutorError("Docker runs in the foreground. Detached jobs require SSH.")
        task = request.task
        if task and task.device not in ("cpu", "cuda"):
            raise ExecutorError("This Docker adapter supports CPU and NVIDIA CUDA on Linux containers.")
        if task and request.action == "run":
            task.require_approval()

        store = job_store(request, self.credentials)
        paths = []
        model_mount = f"/ict/model-run/{task.model}" if task and task.model else "/ict/model-run"
        inputs = (
            (task.dataset, "/ict/dataset"), (task.model_run, model_mount), (task.config_dir, "/ict/train-config"),
        ) if task else ()
        for value, guest in inputs:
            if value is None:
                continue
            self.credentials.protect_path(value, "Docker input")
            path = absolute_path(value)
            self.credentials.protect_path(str(path), "resolved Docker input")
            require_disjoint_paths(path, store.directory)
            if not path.is_dir():
                raise ExecutorError("A Docker input directory is missing on the local host.")
            if task and task.model and guest == model_mount and path.name != task.model:
                raise ExecutorError("Select the model directory of an earlier ICT training run.")
            paths.extend(("--mount", _bind(path, PurePosixPath(guest), readonly=True)))
        if not os.access(store.directory.parent, os.W_OK | os.X_OK):
            raise ExecutorError("The Docker job workspace is not writable.")
        if request.action == "run" and (store.directory / "job.json").exists():
            raise ExecutorError("This job ID already exists. Use a new ID.")

        env = dict(os.environ)
        # Pin the verified local endpoint. A network daemon would interpret
        # bind-mount paths on another machine.
        host_override = env.get("DOCKER_HOST") if not env.get("DOCKER_CONTEXT") else None
        endpoint = host_override or _command(
            ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"], label="Docker context check",
            timeout=10,
        ).strip()
        if not re.fullmatch(r"(?:unix:///|npipe:////\./pipe/)[^\r\n]+", endpoint):
            raise ExecutorError("Select a local Docker daemon using a Unix socket or Windows named pipe.")
        for name in ("DOCKER_HOST", "DOCKER_CONTEXT"):
            env.pop(name, None)
        docker = ["docker", "--host", endpoint]
        kind = _command([*docker, "info", "--format", "{{.OSType}}"], label="Docker daemon check", timeout=15, env=env)
        if kind.strip() != "linux":
            raise ExecutorError("Switch Docker to Linux containers.")
        _command([*docker, "image", "inspect", "--format", "{{.Id}}", self.image],
                 label="Local Docker image check", timeout=15, env=env)
        argv = [
            *docker, "run", "--rm", "--interactive", "--pull=never", "--network=none",
            "--log-driver=none", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--env", "HOME=/tmp",
            "--entrypoint", "python", *paths,
        ]
        for scheme in ("HTTP", "HTTPS", "FTP", "ALL", "NO"):
            proxy = f"{scheme}_PROXY"
            # The Docker client otherwise injects proxies from its config file.
            argv.extend(("--env", f"{proxy}=", "--env", f"{proxy.lower()}="))
        if os.name == "posix":
            argv.extend(("--user", f"{os.getuid()}:{os.getgid()}"))
        if task and task.device == "cuda":
            argv.extend(("--gpus", "all"))
        if request.action == "run":
            store.directory.mkdir(exist_ok=True)
        if request.action != "preflight":
            if not store.directory.is_dir():
                raise ExecutorError("No job directory found. Check the workspace and job ID.")
            # Recheck links after making the directory and just before binding.
            job_store(request, self.credentials)
            argv.extend(("--mount", _bind(store.directory, _CONTAINER_ROOT / request.job, readonly=False)))
        mapped = request.model_copy(update={
            "workspace": str(_CONTAINER_ROOT),
            "task": task.model_copy(update={
                "dataset": "/ict/dataset", "model_run": model_mount if task.model_run else None,
                "config_dir": "/ict/train-config" if task.config_dir else None,
            }) if task else None,
        })
        result = _worker_result(_command(
            [*argv, self.image, *_WORKER], label="Docker worker", payload=mapped.model_dump_json(),
            timeout=None if request.action == "run" else 90, env=env,
        ), self.credentials)
        if request.action == "preflight" and task:
            free_bytes = disk_usage(store.directory.parent).free
            result.update(job_directory=str(store.directory), dataset=task.dataset, free_bytes=free_bytes)
            storage = result.get("storage")
            if storage is not None:
                if not isinstance(storage, list) or not storage or any(
                    not isinstance(row, dict) or type(row.get("measured_free_bytes")) is not int
                    or type(row.get("required_bytes")) is not int or not isinstance(row.get("path"), str)
                    for row in storage
                ):
                    raise ExecutorError("The Docker worker returned an incompatible storage estimate.")
                storage[0].update(path=str(store.directory.parent), measured_free_bytes=free_bytes)
                result.pop("storage_reason", None)
                result["storage_ready"] = True
                for row in storage:
                    if row["measured_free_bytes"] < row["required_bytes"]:
                        result.update(
                            storage_ready=False,
                            storage_reason=f"Low disk space for training at {row['path']}. "
                            f"Measured free bytes: {row['measured_free_bytes']}; "
                            f"estimated required bytes: {row['required_bytes']}.",
                        )
                        break
        return result


class RemoteExecutor:
    def __init__(
        self, host: str, *, user: str | None = None, python: str = "python3", credentials: Credentials | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,252}", host) or (
            user is not None and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}", user)
        ):
            raise ExecutorError("Provide a valid SSH host alias and, if needed, a separate username.")
        interpreter = PurePosixPath(python)
        if not python.isprintable() or not python or (not interpreter.is_absolute() and
                not re.fullmatch(r"python3(?:\.\d+)?", python)):
            raise ExecutorError("Use python3 or the absolute path to a provisioned remote interpreter.")
        self._command = [
            "ssh", "-T", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "UpdateHostKeys=no",
            "-o", "ForwardAgent=no", "-o", "ForwardX11=no", "-o", "ClearAllForwardings=yes",
            "-o", "PermitLocalCommand=no", "-o", "ControlPath=none",
            "-o", "ConnectTimeout=8", "-o", "ServerAliveInterval=10", "-o", "ServerAliveCountMax=2",
        ]
        if user is not None:
            self._command.extend(("-l", user))
        self._command.extend(("--", host, shlex.join((python, *_WORKER, "--ssh"))))
        self.credentials = credentials or Credentials.load({}, os.environ)
        self._path_credentials = Credentials()
        for name, secret in self.credentials.values.items():
            if name not in {"REMOTE_HOST", "REMOTE_USER"}:
                self._path_credentials.add(name, secret.get_secret_value())

    def execute(self, request: WorkerRequest, on_event: Progress | None = None) -> dict[str, object]:
        if request.action == "run":
            request = request.model_copy(update={"action": "submit"})
        if request.action not in {"preflight", "submit", "resume", "status", "logs", "snapshot"}:
            raise ExecutorError("SSH supports preflight, detached jobs, and inspection.")
        paths = (request.workspace,)
        if request.task:
            paths += (request.task.dataset, request.task.model_run, request.task.config_dir)
        for path in paths:
            if path is None:
                continue
            if not PurePosixPath(path).is_absolute():
                raise ExecutorError("Use absolute Linux paths for the remote workspace and inputs.")
            self._path_credentials.protect_path(path, "remote filesystem path")
        if request.action == "submit" and request.task is not None:
            request.task.require_approval()
        return _worker_result(_command(
            self._command, label="SSH worker (check known_hosts, agent, and remote Python)", payload=request.model_dump_json(),
            timeout=90, env={name: value for name, value in os.environ.items() if name.upper() in _SSH_ENVIRONMENT},
        ), self.credentials)
