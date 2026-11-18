"""CLI for executor selection and job lifecycle operations."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import sleep
from typing import Annotated, Literal, NoReturn

import typer
from pydantic import ValidationError

from ..config.settings import Credentials
from ..config.storage import SetupFiles
from ..data.preprocessing.console import protect_source_data
from ..models.contracts import ModelName
from ..pipeline.model import valid_id
from .executors import ContainerExecutor, Executor, LocalExecutor, RemoteExecutor
from .jobs import _SAFE_ERRORS
from .model import Action, Device, ExecutorError, JobRequest, WorkerRequest, Workflow

runtime_app = typer.Typer(help="Run workflows locally, in Docker, or on a Linux SSH host.", no_args_is_help=True)


@dataclass(frozen=True)
class Session:
    executor: Executor
    workspace: str
    credentials: Credentials
    remote: bool


def _failure(exc: Exception) -> NoReturn:
    if isinstance(exc, _SAFE_ERRORS):
        message = str(exc)
    elif isinstance(exc, ValidationError):
        details = ", ".join(str(error["msg"]) for error in exc.errors(include_input=False, include_context=False))
        message = f"Invalid runtime options. {details}"
    else:
        message = "Cannot access runtime files. Check paths, permissions, and available storage."
    typer.echo(message, err=True)
    raise typer.Exit(1) from None


@runtime_app.callback()
def configure(
    ctx: typer.Context,
    workspace: Annotated[str, typer.Option(help="Existing parent directory for jobs on the target.")],
    executor: Annotated[Literal["local", "docker", "ssh"], typer.Option(help="Choose the execution target.")] = "local",
    image: Annotated[str, typer.Option(help="Locally built Docker image.")] = "ict-pipeline:cpu",
    ssh_host: Annotated[str | None, typer.Option(help="SSH config alias. Defaults to REMOTE_HOST and REMOTE_USER.")] = None,
    remote_python: Annotated[str, typer.Option(help="Python executable on the Linux SSH target.")] = "python3",
) -> None:
    try:
        credentials = SetupFiles.read(Path.cwd(), None).credentials(os.environ)
        target: Executor
        if executor == "ssh":
            host = credentials.values.get("REMOTE_HOST")
            user = credentials.values.get("REMOTE_USER")
            if ssh_host:
                credentials.add("REMOTE_HOST", ssh_host)
            if not ssh_host and host is None:
                raise ExecutorError("Configure an SSH host alias with --ssh-host or REMOTE_HOST.")
            target = RemoteExecutor(
                ssh_host or host.get_secret_value(), user=user.get_secret_value() if user and not ssh_host else None,
                python=remote_python, credentials=credentials,
            )
        else:
            workspace = str(Path(workspace).resolve())
            target = LocalExecutor(credentials) if executor == "local" else ContainerExecutor(image, credentials)
        ctx.obj = Session(target, workspace, credentials, executor == "ssh")
    except (*_SAFE_ERRORS, OSError, ValueError, RuntimeError) as exc:
        _failure(exc)


def _task(
    session: Session, workflow: Workflow, dataset: str, device: Device, model: ModelName | None,
    model_run: str | None, approve_train: bool, config_dir: str | None,
) -> JobRequest:
    if not session.remote:
        dataset = str(Path(dataset).resolve())
        model_run = str(Path(model_run).resolve()) if model_run else None
        config_dir = str(Path(config_dir).resolve()) if config_dir else None
    return JobRequest(
        workflow=workflow, dataset=dataset, device=device, model=model, model_run=model_run,
        config_dir=config_dir, approve_train=approve_train,
    )


def _print(value: object) -> None:
    typer.echo(json.dumps(value, ensure_ascii=True))


def _invoke(
    session: Session, action: Action, job: str, *, task: JobRequest | None = None, after: int = 0,
) -> dict[str, object]:
    session.credentials.protect_path(job, "job ID")
    request = WorkerRequest(action=action, workspace=session.workspace, job=job, task=task, after=after)
    return session.executor.execute(request, _print if action == "run" else None)


Job = Annotated[str, typer.Argument(help="Stable job identifier on the target.")]
Dataset = Annotated[str, typer.Option(help="Existing extracted data or generated detection directory on the target.")]
ModelRun = Annotated[str | None, typer.Option(help="Existing model attempt for evaluation or explicit training resumption.")]
TrainConfigDir = Annotated[str | None, typer.Option(help="Directory containing the model's YAML configuration on the target.")]


@runtime_app.command()
def preflight(
    ctx: typer.Context, job: Job, dataset: Dataset, workflow: Workflow = "preprocess",
    device: Device = "cpu", model: ModelName | None = None, model_run: ModelRun = None, config_dir: TrainConfigDir = None,
) -> None:
    """Check input paths, storage, and device readiness without starting a job."""
    try:
        task = _task(ctx.obj, workflow, dataset, device, model, model_run, False, config_dir)
        result = _invoke(ctx.obj, "preflight", job, task=task)
        _print(result)
    except (*_SAFE_ERRORS, ValidationError, OSError, ValueError, RuntimeError) as exc:
        _failure(exc)
    if not result["ready"]:
        raise typer.Exit(1)


@runtime_app.command()
def run(
    ctx: typer.Context, job: Job, dataset: Dataset, workflow: Workflow = "preprocess",
    device: Device = "cpu", model: ModelName | None = None, model_run: ModelRun = None, config_dir: TrainConfigDir = None,
    approve_train: Annotated[bool, typer.Option("--approve-train", help="Authorize this training attempt.")] = False,
) -> None:
    """Run a local or Docker workflow, or submit a detached SSH job."""
    session: Session = ctx.obj
    try:
        if not valid_id(job):
            raise ExecutorError("Use a portable job ID of up to 64 lowercase letters, digits, hyphens, or underscores.")
        task = _task(session, workflow, dataset, device, model, model_run, approve_train, config_dir)
        if workflow == "train" and not approve_train:
            _print(_invoke(session, "preflight", job, task=task))
            task.require_approval()
        result = _invoke(session, "run", job, task=task)
        _print(result)
    except (*_SAFE_ERRORS, ValidationError, OSError, ValueError, RuntimeError) as exc:
        _failure(exc)
    except KeyboardInterrupt:
        typer.echo("Stopped waiting. Inspect the job status before retrying.", err=True)
        raise typer.Exit(130) from None
    if result["status"] in {"failed", "interrupted"}:
        raise typer.Exit(1)


@runtime_app.command()
def resume(ctx: typer.Context, job: Job) -> None:
    """Retry an interrupted SSH preprocessing job with its saved request."""
    try:
        _print(_invoke(ctx.obj, "resume", job))
    except (*_SAFE_ERRORS, ValidationError, OSError, ValueError, RuntimeError) as exc:
        _failure(exc)


@runtime_app.command()
def status(ctx: typer.Context, job: Job) -> None:
    """Read the job's persisted state."""
    try:
        _print(_invoke(ctx.obj, "status", job))
    except (*_SAFE_ERRORS, ValidationError, OSError, ValueError, RuntimeError) as exc:
        _failure(exc)


@runtime_app.command()
def logs(
    ctx: typer.Context, job: Job, after: Annotated[int, typer.Option(min=0)] = 0,
    watch: Annotated[bool, typer.Option(help="Poll events until the job stops.")] = False,
) -> None:
    """Read structured progress with a resumable byte cursor."""
    try:
        while True:
            result = _invoke(ctx.obj, "logs", job, after=after)
            _print(result)
            if not watch:
                return
            next_cursor = int(str(result["next"]))
            if result["status"] != "running" and next_cursor == after:
                if result["status"] != "completed":
                    raise typer.Exit(1)
                return
            if next_cursor == after:
                sleep(2)
            after = next_cursor
    except (*_SAFE_ERRORS, ValidationError, OSError, ValueError, RuntimeError) as exc:
        _failure(exc)
    except KeyboardInterrupt:
        typer.echo("Stopped polling. The job can still be inspected by ID.", err=True)
        raise typer.Exit(130) from None


@runtime_app.command()
def sync(ctx: typer.Context, job: Job, output: Annotated[Path, typer.Option(help="New local JSON file.")]) -> None:
    """Copy a finished job's state and result metadata into a local JSON file."""
    try:
        session: Session = ctx.obj
        session.credentials.protect_path(str(output), "sync output")
        output = output.absolute()
        resolved = output.resolve()
        session.credentials.protect_path(str(resolved), "resolved sync output")
        protect_source_data(resolved, Path.cwd().resolve())
        if output.exists() or output.is_symlink():
            raise ExecutorError("The sync destination already exists. Choose a new file.")
        result = _invoke(session, "snapshot", job)
        with NamedTemporaryFile(dir=output.parent, suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(json.dumps(session.credentials.redact(result), indent=2).encode() + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                # Publish exclusively and atomically in the same directory.
                os.link(temporary, output)
            finally:
                stream.close()
                temporary.unlink(missing_ok=True)
        typer.echo(str(output))
    except (*_SAFE_ERRORS, ValidationError, OSError, ValueError, RuntimeError) as exc:
        _failure(exc)
