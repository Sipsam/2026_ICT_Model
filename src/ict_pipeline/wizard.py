"""Prompt for a setup configuration and confirm publication."""

import getpass
import sys
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import get_args

import typer
from pydantic import SecretStr

from .config.settings import (
    CREDENTIAL_LABELS,
    Credentials,
    Settings,
    SetupError,
    is_filesystem_path,
    merge_settings,
    required_credentials,
    validate_settings,
)
from .config.storage import SetupFiles


def _has_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_settings(
    values: dict[str, object], credentials: Credentials, overrides: Mapping[str, str]
) -> None:
    for name, label in (
        ("execution_mode", "Execution mode"),
        ("dataset_source", "Dataset source"),
        ("data_path", "Dataset path"),
        ("output_path", "Output path"),
        ("model", "Detection model"),
        ("device", "Device"),
        ("training_profile", "Training profile"),
        ("tracking", "Experiment tracking"),
    ):
        if name in overrides or (name == "data_path" and values["dataset_source"] == "skip"):
            continue
        choices = get_args(Settings.model_fields[name].annotation)
        default = values.get(name)
        if name in {"data_path", "output_path"}:
            if name == "data_path" and default is None:
                default = "data/extracted"
            credentials.protect_path(default, name)
            values[name] = typer.prompt(label, default=default if is_filesystem_path(default) else None)
            continue

        if name == "dataset_source" and default is None:
            default = "skip"
        options = "/".join(choices)
        while True:
            answer = typer.prompt(
                f"{label} ({options})", default=default if default in choices else None
            )
            if answer in choices:
                values[name] = answer
                break
            typer.echo(f"Choose one of {options}.", err=True)


def _prompt_credentials(settings: Settings, credentials: Credentials) -> dict[str, SecretStr]:
    additions: dict[str, SecretStr] = {}
    for name in required_credentials(settings):
        if name in credentials.values:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                value = getpass.getpass(f"{CREDENTIAL_LABELS[name]} (hidden): ")
        except (getpass.GetPassWarning, OSError):
            raise SetupError(
                "Hidden input is unavailable. Set credentials in the environment or .env."
            ) from None
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise SetupError(f"{name} requires a non-empty, single-line value.")
        credentials.add(name, value)
        additions[name] = credentials.values[name]
    return additions


def run_setup(
    root: Path,
    config_path: Path | None,
    environ: Mapping[str, str],
    overrides: Mapping[str, str],
    *,
    non_interactive: bool,
    dry_run: bool,
    replace: bool,
) -> None:
    if not non_interactive and not _has_terminal():
        raise SetupError("Interactive setup requires a terminal. Use --non-interactive for headless setup.")

    setup_files = SetupFiles.read(root, config_path)
    credentials = setup_files.credentials(environ)
    values = merge_settings(setup_files.config_text, environ, overrides)
    if not non_interactive:
        typer.echo("Press Ctrl+C to cancel before saving.")
        _prompt_settings(values, credentials, overrides)
    settings = validate_settings(values, credentials)
    additions = {} if non_interactive else _prompt_credentials(settings, credentials)
    credentials.require(settings)
    if additions:
        # A credential entered during the wizard must also stay out of the preview and YAML.
        credentials.protect_path(settings.data_path, "data_path")
        credentials.protect_path(settings.output_path, "output_path")

    typer.echo("Resolved setup for configs/local.yaml")
    typer.echo(settings.to_yaml(), nl=False)
    if additions:
        action = "update" if setup_files.env_contents is not None else "create"
        typer.echo(f"Setup would {action} .env with the credentials entered.")
    typer.echo(
        "Setup saves configuration. Downloads, installs, training, and remote jobs require separate approval."
    )
    if dry_run:
        typer.echo("Dry run complete.")
        return
    if not non_interactive:
        action = "Replace" if setup_files.local_contents is not None else "Save"
        destination = "configs/local.yaml and save credentials to .env" if additions else "configs/local.yaml"
        if not typer.confirm(f"{action} {destination}?", default=False):
            typer.echo("Setup cancelled.")
            return
        replace = True
    written = setup_files.save(settings, additions, replace=replace)
    typer.echo("Saved configs/local.yaml." if written else "Kept existing configs/local.yaml.")
