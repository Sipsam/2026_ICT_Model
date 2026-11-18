import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.pipeline.model import PipelineError
from ict_pipeline.pipeline.store import RunStore
from ict_pipeline.test_artifacts import artifact_root, delete_session, open_test_session, preview_cleanup


def test_preview_exact_confirmation_and_retention(tmp_path: Path) -> None:
    root = artifact_root(tmp_path)
    unrelated = root / "sessions" / ("1" * 32)
    unrelated.mkdir(parents=True)
    (unrelated / "keep.pt").write_bytes(b"persistent")
    with open_test_session(root, keep_failed=True) as session:
        session.status = "failed"
        (session.payload / "diagnostic").write_bytes(b"failed")
        session_id = session.store.run_id

    runner = CliRunner()
    args = ["clean", "--root", str(tmp_path)]
    preview = runner.invoke(app, [*args, "--dry-run"])
    assert preview.exit_code == 0 and "too recent" in preview.stdout
    assert all(value in preview.stdout for value in (session_id, "6 logical bytes", "failed"))
    assert unrelated.name not in preview.stdout
    assert runner.invoke(app, [*args, "--delete", session_id]).exit_code == 1
    assert runner.invoke(app, [*args, "--delete", session_id, "--confirm"]).exit_code == 1
    assert session.payload.is_dir()
    deletion = runner.invoke(app, [*args, "--older-than-hours", "0", "--delete", session_id, "--confirm"])
    assert deletion.exit_code == 0, deletion.output
    assert not session.store.directory.exists()
    assert (unrelated / "keep.pt").read_bytes() == b"persistent"
    assert runner.invoke(app, [*args, "--delete", str(unrelated), "--confirm"]).exit_code == 1


def test_active_session_and_nested_job_are_protected(tmp_path: Path) -> None:
    root = artifact_root(tmp_path)
    with open_test_session(root, keep_failed=True) as session:
        nested = RunStore(session.payload, "detached")
        with nested.lock():
            session_id = session.store.run_id
            [candidate] = list(preview_cleanup(root, age_hours=0))
            assert not candidate.eligible and candidate.detail == "active session"
            with pytest.raises(PipelineError, match="active session"):
                delete_session(root, session_id, age_hours=0)
        session.status = "failed"
    with nested.lock():
        [candidate] = list(preview_cleanup(root, age_hours=0))
        assert not candidate.eligible and "holds a run lock" in candidate.detail
        with pytest.raises(PipelineError, match="run lock"):
            delete_session(root, session_id, age_hours=0)
    assert session.payload.is_dir()


def test_cleanup_refuses_a_linked_payload(tmp_path: Path) -> None:
    root = artifact_root(tmp_path)
    outside = tmp_path / "persistent"
    outside.mkdir()
    weight = outside / "best.pt"
    weight.write_bytes(b"keep")
    with open_test_session(root, keep_failed=True) as session:
        session.status = "failed"
    session.payload.rmdir()
    try:
        session.payload.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Directory symlinks unavailable: {getattr(exc, 'winerror', exc.errno)}")
    [candidate] = list(preview_cleanup(root, age_hours=0))
    assert not candidate.eligible
    with pytest.raises(PipelineError, match="symbolic link"):
        delete_session(root, session.store.run_id, age_hours=0)
    assert weight.read_bytes() == b"keep"


def test_interruption_is_bounded_and_can_be_recovered(tmp_path: Path) -> None:
    root = artifact_root(tmp_path)
    script = """
import os, sys
from pathlib import Path
from ict_pipeline.test_artifacts import open_test_session
with open_test_session(Path(sys.argv[1]), keep_failed=False) as session:
    (session.payload / 'partial').write_bytes(b'checkpoint')
    os._exit(17)
"""
    result = subprocess.run([sys.executable, "-c", script, str(root)], timeout=30, check=False)
    assert result.returncode == 17
    [candidate] = list(preview_cleanup(root, age_hours=0))
    assert candidate.eligible and candidate.status == "interrupted" and candidate.logical_bytes == 10
    with pytest.raises(PipelineError, match="limit reached"):
        with open_test_session(root, keep_failed=False, limit=1):
            pytest.fail("An abandoned session must count toward the limit")
    delete_session(root, candidate.path.name, age_hours=0)
    with open_test_session(root, keep_failed=False, limit=1) as restarted:
        restarted.status = "passed"
    assert list(preview_cleanup(root, age_hours=0)) == []


@pytest.mark.parametrize(("failure", "keep"), [(False, False), (True, False), (True, True)])
def test_pytest_lifecycle_on_repeated_invocations(tmp_path: Path, failure: bool, keep: bool) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "conftest.py").write_bytes(Path(__file__).with_name("conftest.py").read_bytes())
    (project / "test_fixture.py").write_text(
        "import tempfile\n"
        "from pathlib import Path\n"
        "def test_pass(tmp_path, tmp_path_factory):\n"
        "    (tmp_path / 'passed').write_text('ok')\n"
        "    (tmp_path_factory.mktemp('shared') / 'shared').write_text('shared')\n"
        "def test_final(tmp_path):\n"
        "    assert not list(tmp_path.parent.glob('test_pass*/passed'))\n"
        "    assert list(tmp_path.parent.glob('shared*/shared'))\n"
        "    (Path(tempfile.gettempdir()) / 'scratch-marker').write_text('scratch')\n"
        "    (tmp_path / 'result').write_text('diagnostic')\n"
        f"    assert {not failure}\n",
        encoding="utf-8",
    )
    parent = tmp_path / "switch-drive"
    parent.mkdir()
    (parent / "existing").write_bytes(b"stay")
    env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "ICT_TEST_ARTIFACT_ROOT": str(parent)}
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"]
    if keep:
        args.append("--ict-keep-failed")
    for attempt in range(2):
        options = ["--basetemp", str(parent)] if attempt else []
        result = subprocess.run([*args, *options], env=env, cwd=project, capture_output=True, timeout=30, check=False)
        assert result.returncode == int(failure), result.stdout.decode(errors="replace") + result.stderr.decode(errors="replace")
        assert (parent / "existing").read_bytes() == b"stay"
        retained = list(preview_cleanup(artifact_root(parent), age_hours=0))
        assert len(retained) == (attempt + 1 if keep else 0)
        for item in retained:
            assert (item.path / "files/scratch/scratch-marker").read_text() == "scratch"
            assert list((item.path / "files/pytest").glob("test_final*/result"))
            assert not list((item.path / "files/pytest").glob("test_pass*/passed"))
