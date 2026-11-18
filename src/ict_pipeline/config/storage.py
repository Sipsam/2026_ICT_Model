"""Read and publish setup files within a workspace."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from dotenv.parser import parse_stream
from pydantic import SecretStr

from .settings import Credentials, Settings, SetupError, merge_settings, validate_settings


@dataclass(frozen=True)
class SetupFiles:
    root: Path
    config_text: str = field(repr=False)
    local_contents: bytes | None = field(repr=False)
    env_contents: bytes | None = field(repr=False)

    @classmethod
    def read(cls, root: Path, config_path: Path | None) -> "SetupFiles":
        root = root.resolve()
        local = _read(root / "configs/local.yaml", "configs/local.yaml")
        source = local if config_path is None else _read(root / config_path, "the selected YAML file")
        if config_path is not None and source is None:
            raise SetupError("The selected YAML file does not exist.")
        return cls(root, _decode(source, "setup YAML"), local, _read(root / ".env", ".env"))

    def credentials(self, environ: Mapping[str, str]) -> Credentials:
        bindings: dict[str, str] = {}
        for binding in parse_stream(StringIO(_decode(self.env_contents, ".env"))):
            if binding.error:
                raise SetupError(f"Cannot parse .env at line {binding.original.line}.")
            if binding.key is not None:
                bindings[binding.key] = binding.value or ""
        return Credentials.load(bindings, environ)

    def save(self, settings: Settings, additions: Mapping[str, SecretStr], *, replace: bool) -> bool:
        """Return whether YAML was written, preserving identical outputs."""
        target = self.root / "configs/local.yaml"
        dotenv = self.root / ".env"
        desired = settings.to_yaml().encode("utf-8")
        write_config = desired != self.local_contents
        if self.local_contents is not None and write_config and not replace:
            raise SetupError("configs/local.yaml already exists. Use --replace to replace it.")

        # Check both files before writing either. Credentials are published first
        # so a retry can reuse them if the YAML write is interrupted.
        if write_config and _read(target, "configs/local.yaml") != self.local_contents:
            raise SetupError("configs/local.yaml changed during setup. Run setup again.")
        if additions and _read(dotenv, ".env") != self.env_contents:
            raise SetupError(".env changed during setup. Run setup again.")
        if additions:
            lines = []
            for name, secret in additions.items():
                escaped = secret.get_secret_value().replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{name}="{escaped}"\n')
            contents = self.env_contents or b""
            if contents and not contents.endswith(b"\n"):
                contents += b"\n"
            try:
                contents += "".join(lines).encode("utf-8")
            except UnicodeError:
                raise SetupError("Cannot store a credential in .env. Use UTF-8 text.") from None
            _publish(dotenv, contents, self.env_contents, ".env")
        if write_config:
            _publish(target, desired, self.local_contents, "configs/local.yaml")
        return write_config


def load_config(
    root: Path,
    *,
    config_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> tuple[Settings, Credentials]:
    """Load validated settings and required credentials for downstream commands."""
    environ = os.environ if environ is None else environ
    setup_files = SetupFiles.read(root, config_path)
    credentials = setup_files.credentials(environ)
    settings = validate_settings(
        merge_settings(setup_files.config_text, environ, overrides or {}), credentials
    )
    credentials.require(settings)
    return settings, credentials


def _read(path: Path, label: str) -> bytes | None:
    try:
        if path.is_symlink() and not path.exists():
            raise SetupError(f"Cannot read {label}. The symbolic link has no target.")
        if path.exists() and not path.is_file():
            raise SetupError(f"Cannot read {label}. A regular file is required.")
        return path.read_bytes()
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise SetupError(f"Cannot read {label}. Check the path and permissions.") from None


def _decode(contents: bytes | None, label: str) -> str:
    try:
        return (contents or b"").decode("utf-8-sig")
    except UnicodeError:
        raise SetupError(f"Cannot read {label}. Use UTF-8 encoding.") from None


def _publish(path: Path, contents: bytes, original: bytes | None, label: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, prefix=f"{path.name}.tmp-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(contents)
                stream.close()
                if original is None:
                    # Publish a complete file without overwriting a concurrent creation.
                    path.hardlink_to(temporary)
                else:
                    if _read(path, label) != original:
                        raise SetupError(f"{label} changed during setup. Run setup again.")
                    temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError):
        raise SetupError(f"Cannot write {label}. Check permissions or retry setup.") from None
