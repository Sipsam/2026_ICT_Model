"""Wire contract shared by the runtime adapters and their target workers."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..config.settings import is_filesystem_path
from ..models.contracts import ModelName
from ..pipeline.model import PipelineError, valid_id

Device = Literal["cpu", "cuda", "rocm", "mps"]
Workflow = Literal["preprocess", "train", "evaluate"]
Action = Literal["preflight", "run", "submit", "resume", "status", "logs", "snapshot", "work"]


class ExecutorError(PipelineError):
    """A runtime failure safe to display."""


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    workflow: Workflow = "preprocess"
    dataset: str
    device: Device = "cpu"
    model: ModelName | None = None
    model_run: str | None = None
    config_dir: str | None = None
    approve_train: bool = False

    @field_validator("dataset", "model_run", "config_dir")
    @classmethod
    def check_path(cls, value: str | None) -> str | None:
        if value is not None and not is_filesystem_path(value):
            raise ValueError("Use a filesystem path.")
        return value

    @model_validator(mode="after")
    def check_workflow(self) -> Self:
        if self.workflow == "preprocess" and (self.device != "cpu" or self.model or self.model_run):
            raise ValueError("Preprocessing uses CPU and requires only a dataset.")
        if self.workflow == "train" and self.model is None:
            raise ValueError("Training requires a model.")
        if self.workflow == "evaluate" and (self.model_run is None or self.model is not None):
            raise ValueError("Evaluation requires a completed model run directory.")
        if self.workflow != "train" and (self.config_dir or self.approve_train):
            raise ValueError("Training config and approval apply only to training.")
        return self

    def require_approval(self) -> None:
        if self.workflow == "train" and not self.approve_train:
            raise ExecutorError("Review preflight, then pass --approve-train to authorize training.")


class WorkerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    action: Action
    workspace: str
    job: str
    task: JobRequest | None = None
    after: int = Field(default=0, ge=0)

    @field_validator("job")
    @classmethod
    def check_job(cls, value: str) -> str:
        if not valid_id(value):
            raise ValueError("Use a portable job ID.")
        return value

    @field_validator("workspace")
    @classmethod
    def check_workspace(cls, value: str) -> str:
        if not is_filesystem_path(value):
            raise ValueError("Use a workspace filesystem path.")
        return value

    @model_validator(mode="after")
    def require_task(self) -> Self:
        if (self.task is not None) != (self.action in {"preflight", "run", "submit"}):
            raise ValueError("This action requires a task only for preflight or a new run.")
        return self
