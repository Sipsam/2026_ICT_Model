"""Resolve and validate setup settings without importing runtime backends."""

import json
import re
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

from ..classification.contracts import ClassificationConfig


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
    classification: ClassificationConfig | None = None

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

_SECRET_NAMES = (
    r"api[_-]?key|access[_-]?key|password|passwd|token|secret|authorization|credentials?"
    r"|private[_-]?key|ssh[_-]?key|tracking[_-]?uri|remote[_-]?(?:host|user)|signature"
)
_SECRET_FIELD = re.compile(rf"(?:^|[_-])(?:{_SECRET_NAMES})(?:$|[_-])", re.I)
_SECRET_ASSIGNMENT = re.compile(
    rf"""(?i)([\w-]*(?:{_SECRET_NAMES})[\w-]*["']?\s*(?:=|:)\s*)"""
    r"""(?:"[^"\r\n]*"|'[^'\r\n]*'|[^\s&,;]+)"""
)


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
            for name in (*CREDENTIAL_LABELS, "MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_USERNAME",
                         "MLFLOW_TRACKING_PASSWORD"):
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

    def redact_text(self, value: str) -> str:
        for secret in sorted(self._known_values, key=len, reverse=True):
            if len(secret) >= 4:
                value = value.replace(secret, "[REDACTED]")
            elif secret == value:
                return "[REDACTED]"
        value = re.sub(r"(?i)([a-z][a-z0-9+.-]*://)[^/\s@]+@", r"\1[REDACTED]@", value)
        value = re.sub(r"(?s)-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
                       "[REDACTED]", value)
        value = re.sub(r"(?i)\b(Bearer|Basic)\s+[\w./+=~-]+", r"\1 [REDACTED]", value)
        return _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", value)

    def redact(self, value: object) -> object:
        if isinstance(value, Mapping):
            return {
                self.redact_text(str(key)): "[REDACTED]"
                if _SECRET_FIELD.search(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(key))) else self.redact(item)
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, (str, SecretStr)):
            return self.redact_text(str(value))
        return value

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
            if name == "classification":
                try:
                    merged[name] = json.loads(environ[env_name])
                except ValueError:
                    raise SetupError("ICT_CLASSIFICATION must be a JSON object or null.") from None
            else:
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


