"""Resolve and validate setup settings without importing runtime backends."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.resources import files
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    ValidationInfo,
    field_validator,
)


class SetupError(Exception):
    """A setup error safe to show in the terminal."""


def is_filesystem_path(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and "://" not in value
        and value.isprintable()
    )


class Settings(BaseModel):
    """The public YAML and config snapshot contract."""

    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True, frozen=True)

    execution_mode: Literal["local", "wsl", "docker", "remote"]
    dataset_source: Literal["aihub", "existing", "skip"]
    data_path: str | None = Field(default=None, validate_default=True)
    output_path: str
    model: Literal["yolo", "faster_rcnn", "rt_detr"]
    device: Literal["auto", "cpu", "cuda", "rocm", "mps"]
    training_profile: Literal["smoke", "standard", "full"]
    tracking: Literal["off", "local", "mlflow"]

    @field_validator("data_path", "output_path")
    @classmethod
    def validate_path(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            if info.data.get("dataset_source") in {"existing", "aihub"}:
                raise ValueError("required for the selected dataset source")
            return None
        if not is_filesystem_path(value):
            raise ValueError("enter a non-empty filesystem path")
        return value.strip()

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(exclude_none=True), sort_keys=False)


CREDENTIAL_LABELS = {
    "AIHUB_API_KEY": "AI Hub API key",
    "MLFLOW_TRACKING_URI": "MLflow tracking URI",
    "REMOTE_HOST": "Remote SSH host",
    "REMOTE_USER": "Remote SSH user",
    "REMOTE_SSH_KEY_PATH": "Remote SSH key path",
}


def required_credentials(settings: Settings) -> list[str]:
    required: list[str] = []
    if settings.dataset_source == "aihub":
        required.append("AIHUB_API_KEY")
    if settings.tracking == "mlflow":
        required.append("MLFLOW_TRACKING_URI")
    if settings.execution_mode == "remote":
        required.extend(("REMOTE_HOST", "REMOTE_USER"))
    return required


@dataclass
class Credentials:
    """Credential values stay out of the public Settings model."""

    values: dict[str, SecretStr] = field(default_factory=dict, repr=False)
    _known_values: set[str] = field(default_factory=set, repr=False)

    @classmethod
    def load(cls, dotenv: Mapping[str, str], environ: Mapping[str, str]) -> "Credentials":
        credentials = cls()
        for source in (dotenv, environ):
            for name in CREDENTIAL_LABELS:
                if name in source and source[name].strip():
                    credentials.add(name, source[name])
        return credentials

    def add(self, name: str, value: str) -> None:
        self.values[name] = SecretStr(value)
        self._known_values.add(value)

    def protect_path(self, value: object, name: str) -> None:
        if isinstance(value, str) and any(
            secret == value or (len(secret) >= 4 and secret in value)
            for secret in self._known_values
        ):
            raise SetupError(f"{name} contains a credential. Use a filesystem path.")

    def require(self, settings: Settings) -> None:
        missing = [name for name in required_credentials(settings) if name not in self.values]
        if missing:
            raise SetupError("Set these missing credentials in the environment or .env. " + ", ".join(missing))


def merge_settings(
    local_yaml: str, environ: Mapping[str, str], overrides: Mapping[str, str]
) -> dict[str, object]:
    """Apply CLI > process environment > local YAML > packaged defaults."""
    try:
        local = yaml.safe_load(local_yaml)
    except (yaml.YAMLError, ValueError, RecursionError):
        raise SetupError("Cannot parse the setup YAML.") from None
    if local is None:
        local = {}
    if not isinstance(local, dict):
        raise SetupError("Setup YAML must contain a mapping of settings.")
    if any(key not in Settings.model_fields for key in local):
        raise SetupError("Setup YAML contains unsupported settings. Put credentials in .env.")

    defaults = files(__package__).joinpath("default.yaml").read_text(encoding="utf-8")
    merged = yaml.safe_load(defaults)
    merged.update(local)
    for name in Settings.model_fields:
        env_name = f"ICT_{name.upper()}"
        if env_name in environ:
            merged[name] = environ[env_name]
    merged.update(overrides)
    return merged


def validate_settings(values: Mapping[str, object], credentials: Credentials) -> Settings:
    """Validate the merged inputs and report field names without raw input."""
    try:
        settings = Settings.model_validate(values)
    except ValidationError as exc:
        issues = []
        for error in exc.errors(include_input=False, include_context=False, include_url=False):
            name = error["loc"][0]
            if name not in Settings.model_fields:
                issues.append("unsupported setting")
            elif error["type"] == "missing":
                issues.append(f"{name} is required")
            elif name == "data_path" and values.get(name) is None:
                issues.append("data_path is required for the selected dataset source")
            else:
                issues.append(f"{name} is invalid")
        raise SetupError("Invalid setup configuration. " + ". ".join(issues) + ".") from None
    credentials.protect_path(settings.data_path, "data_path")
    credentials.protect_path(settings.output_path, "output_path")
    return settings


