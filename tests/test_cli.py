import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from textwrap import dedent

import pytest
from typer.testing import CliRunner

from ict_pipeline.cli import app

COMMANDS = (
    "doctor",
    "setup",
    "run",
    "status",
    "resume",
    "data",
    "train",
    "evaluate",
    "compare",
    "report",
    "clean",
)

runner = CliRunner()


def test_help_lists_all_commands() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--version" in result.stdout
    assert set(COMMANDS).issubset(result.stdout.split())


def test_version_matches_installed_package() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout.strip() == f"ict {version('ict-pipeline')}"
    assert result.stderr == ""


@pytest.mark.parametrize("command", [command for command in COMMANDS if command != "doctor"])
def test_planned_command_reports_unavailability(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, [command])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert result.stderr.strip() == f"ict {command} is not yet implemented."
    assert list(tmp_path.iterdir()) == []


def test_cli_load_and_help_do_not_import_ml_libraries(tmp_path: Path) -> None:
    script = dedent("""\
        import sys
        from collections.abc import Sequence
        from importlib.abc import MetaPathFinder
        from types import ModuleType

        blocked = {"torch", "torchvision", "ultralytics", "cupy", "cuda", "pynvml"}
        attempted: list[str] = []

        class BlockMLImports(MetaPathFinder):
            def find_spec(
                self,
                fullname: str,
                path: Sequence[str] | None = None,
                target: ModuleType | None = None,
            ) -> None:
                if fullname.split(".", 1)[0] in blocked:
                    attempted.append(fullname)
                    raise ModuleNotFoundError(fullname)

        sys.meta_path.insert(0, BlockMLImports())
        from ict_pipeline.cli import app

        app(["--help"], standalone_mode=False)
        assert not attempted, attempted
        """)

    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", script],
        cwd=tmp_path,
        capture_output=True,
        encoding="utf-8",
        timeout=20,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "--version" in result.stdout
