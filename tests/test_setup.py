import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from ict_pipeline import wizard
from ict_pipeline.cli import app
from ict_pipeline.config.settings import CREDENTIAL_LABELS, Settings, SetupError
from ict_pipeline.config.storage import SetupFiles, load_config

runner = CliRunner()
headless = ["setup", "--non-interactive"]


@pytest.fixture(autouse=True)
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    for name in CREDENTIAL_LABELS:
        monkeypatch.delenv(name, raising=False)
    for name in Settings.model_fields:
        monkeypatch.delenv(f"ICT_{name.upper()}", raising=False)


def test_precedence(tmp_path: Path) -> None:
    config = tmp_path / "input.yaml"
    config.write_text(
        "dataset_source: aihub\ndata_path: dataset\nmodel: faster_rcnn\n"
        "training_profile: standard\noutput_path: from-yaml\n",
        encoding="utf-8",
    )
    (tmp_path / ".env").write_text(
        "AIHUB_API_KEY=from-dotenv\nICT_DEVICE=cuda\n", encoding="utf-8"
    )
    settings, credentials = load_config(
        tmp_path,
        config_path=Path("input.yaml"),
        environ={
            "AIHUB_API_KEY": "from-process",
            "ICT_MODEL": "rt_detr",
            "ICT_OUTPUT_PATH": "from-env",
        },
        overrides={"output_path": "from-cli"},
    )
    assert settings.model == "rt_detr"
    assert settings.output_path == "from-cli"
    assert settings.training_profile == "standard"
    assert settings.device == "cpu"
    assert credentials.values["AIHUB_API_KEY"].get_secret_value() == "from-process"
    assert "from-process" not in repr(credentials) + settings.to_yaml()
    assert "from-dotenv" not in repr(credentials) + settings.to_yaml()


def test_headless_precedence_and_preview(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "input.yaml"
    config.write_text("dataset_source: skip\nmodel: faster_rcnn\n", encoding="utf-8")
    monkeypatch.setenv("ICT_MODEL", "rt_detr")
    args = [*headless, "--config", "input.yaml", "--model", "yolo"]

    preview = runner.invoke(app, [*args, "--dry-run"])

    assert preview.exit_code == 0, preview.output
    assert "model: yolo" in preview.stdout
    assert list(tmp_path.iterdir()) == [config]
    saved = runner.invoke(app, args)
    assert saved.exit_code == 0, saved.output
    assert yaml.safe_load(Path("configs/local.yaml").read_text(encoding="utf-8"))["model"] == "yolo"
    assert config.read_text(encoding="utf-8") == "dataset_source: skip\nmodel: faster_rcnn\n"


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], "dataset_source is required"),
        (["--dataset-source", "existing"], "data_path is required"),
        (["--dataset-source", "skip", "--device", "broken"], "device is invalid"),
        (["--dataset-source", "skip", "--output-path", " "], "output_path is invalid"),
        (["--config", "missing.yaml"], "selected YAML file does not exist"),
    ],
)
def test_missing_or_invalid_settings(tmp_path: Path, args: list[str], expected: str) -> None:
    result = runner.invoke(app, [*headless, *args])

    assert result.exit_code == 2
    assert expected in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_reports_all_required_credentials() -> None:
    result = runner.invoke(
        app, [*headless, "--dataset-source", "aihub", "--data-path", "dataset",
              "--tracking", "mlflow", "--execution-mode", "remote"]
    )

    assert result.exit_code == 2
    assert all(key in result.stderr for key in CREDENTIAL_LABELS if key != "REMOTE_SSH_KEY_PATH")
    assert "REMOTE_SSH_KEY_PATH" not in result.stderr


@pytest.mark.parametrize(
    "content",
    [
        "dataset_source: skip\nmodel: secret-marker\n",
        "dataset_source: skip\nsecret-marker: secret-marker\n",
        "dataset_source: skip\nmodel: [secret-marker\n",
    ],
)
def test_yaml_errors_do_not_echo_input(content: str) -> None:
    Path("input.yaml").write_text(content, encoding="utf-8")
    result = runner.invoke(app, [*headless, "--config", "input.yaml"])

    assert result.exit_code == 2
    assert "secret-marker" not in result.output
    assert not Path("configs").exists()


def test_malformed_dotenv_does_not_echo_input(caplog: pytest.LogCaptureFixture) -> None:
    Path(".env").write_text('AIHUB_API_KEY="secret-marker\n', encoding="utf-8")
    result = runner.invoke(app, [*headless, "--dataset-source", "skip"])

    assert result.exit_code == 2
    assert "Cannot parse .env" in result.stderr
    assert "secret-marker" not in result.output + caplog.text


def test_shadowed_credential_rejected_in_public_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    Path(".env").write_text("AIHUB_API_KEY=shadowed-marker\n", encoding="utf-8")
    monkeypatch.setenv("AIHUB_API_KEY", "active-marker")
    result = runner.invoke(
        app, [*headless, "--dataset-source", "skip", "--output-path", "test/shadowed-marker"]
    )

    assert result.exit_code == 2
    assert "output_path contains a credential" in result.stderr
    assert "shadowed-marker" not in result.output
    assert "active-marker" not in result.output
    assert not (tmp_path / "configs").exists()
    with pytest.raises(SetupError, match="output_path contains a credential") as exc:
        load_config(tmp_path, overrides={"dataset_source": "skip", "output_path": "active-marker"})
    assert "active-marker" not in str(exc.value)


def test_headless_refuses_replacement_without_flag() -> None:
    config = Path("configs/local.yaml")
    config.parent.mkdir()
    original = b"# Keep this comment\r\ndataset_source: skip\r\n"
    config.write_bytes(original)
    args = [*headless, "--model", "rt_detr"]

    result = runner.invoke(app, args)

    assert result.exit_code == 2
    assert "--replace" in result.stderr
    assert config.read_bytes() == original
    assert runner.invoke(app, [*args, "--dry-run"]).exit_code == 0
    assert config.read_bytes() == original
    assert runner.invoke(app, [*args, "--replace"]).exit_code == 0
    assert b"model: rt_detr" in config.read_bytes()
    assert "Kept existing" in runner.invoke(app, args).stdout


def test_concurrent_config_change_preserved(tmp_path: Path) -> None:
    setup_files = SetupFiles.read(tmp_path, None)
    settings, _ = load_config(tmp_path, overrides={"dataset_source": "skip"})
    target = Path("configs/local.yaml")
    target.parent.mkdir()
    target.write_text("dataset_source: existing\n", encoding="utf-8")

    with pytest.raises(SetupError, match="changed during setup"):
        setup_files.save(settings, {}, replace=True)
    assert target.read_text(encoding="utf-8") == "dataset_source: existing\n"


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wizard, "_has_terminal", lambda: True)


@pytest.mark.parametrize("answer", ["n\n", ""])
def test_interactive_cancellation_preserves_existing_files(answer: str, terminal: None) -> None:
    config = Path("configs/local.yaml")
    config.parent.mkdir()
    original = b"dataset_source: skip\nmodel: faster_rcnn\n"
    config.write_bytes(original)
    dotenv = Path(".env")
    dotenv.write_bytes(b"# keep\r\n")

    result = runner.invoke(app, ["setup"], input="\n" * 7 + answer)

    assert result.exit_code == (0 if answer else 130), result.output
    assert "Setup cancelled." in result.output
    assert config.read_bytes() == original
    assert dotenv.read_bytes() == b"# keep\r\n"


def test_interactive_hidden_credentials_and_save(
    terminal: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = b"# keep\r\nUNRELATED='literal'\r\n"
    Path(".env").write_bytes(original)
    secret = 'secret-marker-"\\$${OTHER}'
    prompts: list[str] = []

    def hidden_prompt(prompt: str) -> str:
        prompts.append(prompt)
        return secret

    monkeypatch.setattr(wizard.getpass, "getpass", hidden_prompt)
    inputs = "\n" * 7 + "y\n"
    args = ["setup", "--dataset-source", "aihub"]
    dry_run = runner.invoke(app, [*args, "--dry-run"], input=inputs)
    assert dry_run.exit_code == 0, dry_run.output
    assert Path(".env").read_bytes() == original
    assert not Path("configs").exists()

    result = runner.invoke(app, args, input=inputs)

    assert result.exit_code == 0, result.output
    assert prompts == ["AI Hub API key (hidden): "] * 2
    assert "secret-marker" not in result.output + dry_run.output
    assert Path(".env").read_bytes().startswith(original)
    settings, credentials = load_config(tmp_path, environ={})
    assert credentials.values["AIHUB_API_KEY"].get_secret_value() == secret
    assert "secret-marker" not in settings.to_yaml() + Path("configs/local.yaml").read_text(encoding="utf-8")


def test_interactive_refuses_echoing_credential_fallback(
    terminal: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(_prompt: str) -> str:
        warnings.warn("missing tty", wizard.getpass.GetPassWarning)
        pytest.fail("the echoing fallback was reached")

    monkeypatch.setattr(wizard.getpass, "getpass", unavailable)
    result = runner.invoke(app, ["setup", "--dataset-source", "aihub"], input="\n" * 7)

    assert result.exit_code == 2, result.output
    assert "Hidden input is unavailable" in result.stderr
    assert not Path(".env").exists()


def test_wizard_does_not_show_credential_bearing_url(terminal: None) -> None:
    Path("input.yaml").write_text(
        "dataset_source: skip\noutput_path: https://user:secret-marker@example.com\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app, ["setup", "--config", "input.yaml", "--dry-run"], input="\n\noutputs\n" + "\n" * 4
    )

    assert result.exit_code == 0, result.output
    assert "secret-marker" not in result.output
    assert "output_path: outputs" in result.stdout
    assert not Path("configs").exists()


def test_entered_credential_cannot_be_previewed_as_a_path(
    terminal: None, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(wizard.getpass, "getpass", lambda _: "secret-marker")
    result = runner.invoke(
        app, ["setup", "--dataset-source", "aihub", "--output-path", "test/secret-marker"],
        input="\n" * 6,
    )

    assert result.exit_code == 2, result.output
    assert "output_path contains a credential" in result.stderr
    assert "secret-marker" not in result.output
    assert not Path(".env").exists()
    assert not Path("configs").exists()


def test_headless_process_with_closed_stdin(tmp_path: Path) -> None:
    base = [
        sys.executable, "-c", "from ict_pipeline.cli import app; app()",
        "setup",
    ]
    options = dict(
        stdin=subprocess.DEVNULL, capture_output=True, text=True,
        cwd=tmp_path, env=os.environ.copy(), timeout=15, check=False,
    )
    interactive = subprocess.run(base, **options)
    assert interactive.returncode == 2
    assert "--non-interactive" in interactive.stderr

    headless_result = subprocess.run(
        [*base, "--non-interactive", "--dataset-source", "skip"], **options
    )
    assert headless_result.returncode == 0, headless_result.stderr
    assert "Saved configs/local.yaml." in headless_result.stdout
