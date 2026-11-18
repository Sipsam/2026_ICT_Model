import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from textwrap import dedent

from typer.testing import CliRunner

from ict_pipeline.cli import app

COMMANDS = (
    "doctor",
    "setup",
    "run",
    "status",
    "runs",
    "resume",
    "data",
    "train",
    "evaluate",
    "inspect",
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


def test_cli_load_and_help_do_not_import_ml_libraries(tmp_path: Path) -> None:
    script = dedent("""\
        import sys
        from collections.abc import Sequence
        from importlib.abc import MetaPathFinder
        from types import ModuleType

        blocked = {"torch", "torchvision", "ultralytics", "transformers", "cupy", "cuda", "pynvml", "mlflow"}
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
        app(["train", "--help"], standalone_mode=False)
        app(["evaluate", "--help"], standalone_mode=False)
        app(["inspect", "--help"], standalone_mode=False)
        app(["compare", "--help"], standalone_mode=False)
        app(["runs", "--help"], standalone_mode=False)
        app(["report", "--help"], standalone_mode=False)
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
