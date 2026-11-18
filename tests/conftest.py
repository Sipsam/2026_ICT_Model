import os
import tempfile
from collections.abc import Generator
from contextlib import ExitStack
from pathlib import Path
from warnings import warn

import pytest

from ict_pipeline.data.preprocessing.console import protect_source_data
from ict_pipeline.environment.model import Graphics, Host, Resources
from ict_pipeline.pipeline.model import PipelineError
from ict_pipeline.reporting import metadata
from ict_pipeline.test_artifacts import TestSession, artifact_root, measure_test_files, open_test_session


_SESSION = pytest.StashKey[TestSession]()
_TEST_PATH = pytest.StashKey[Path]()
_FAILED = pytest.StashKey[bool]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ICT test artifacts")
    group.addoption("--ict-artifact-root", type=Path, help="Parent directory for managed ICT test sessions.")
    group.addoption("--ict-keep-failed", action="store_true", help="Keep failed or interrupted test artifacts.")
    group.addoption("--ict-artifact-summary", action="store_true", help="Report artifact sizes before test cleanup.")


@pytest.hookimpl(tryfirst=True)
def pytest_configure(config: pytest.Config) -> None:
    if config.option.help:
        return
    if config.getoption("numprocesses", default=0):
        raise pytest.UsageError("ICT test artifact leases require a single pytest process. Omit xdist -n.")
    parent = config.getoption("--ict-artifact-root") or config.option.basetemp
    with ExitStack() as resources:
        try:
            root = artifact_root(Path(parent) if parent is not None else None)
            protect_source_data(root, config.rootpath)
            session = resources.enter_context(open_test_session(
                root,
                keep_failed=config.getoption("--ict-keep-failed"),
                limit=int(os.environ.get("ICT_TEST_SESSION_LIMIT", "8")),
            ))
            scratch = session.payload / "scratch"
            scratch.mkdir()
        except (OSError, ValueError, PipelineError) as exc:
            raise pytest.UsageError(f"Cannot initialize ICT test artifacts. {exc}") from None
        config.stash[_SESSION] = session
        # An explicit, fresh child avoids pytest's numbered-directory cleanup
        # and its unconditional removal of an existing --basetemp.
        config.option.basetemp = str(session.payload / "pytest")
        env = resources.enter_context(pytest.MonkeyPatch.context())
        for name in ("TEMP", "TMP", "TMPDIR"):
            env.setenv(name, str(scratch))
        env.setattr(tempfile, "tempdir", str(scratch))
        cleanup = resources.pop_all()

        def close_session() -> None:
            try:
                cleanup.close()
            except (OSError, PipelineError) as exc:
                warn(pytest.PytestWarning(f"Deferred cleanup for {session.store.directory}: {exc}"), stacklevel=1)

        config.add_cleanup(close_session)


def pytest_report_header(config: pytest.Config) -> str:
    return f"ICT disposable test session: {config.stash[_SESSION].store.directory}"


@pytest.fixture
def tmp_path(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    request.node.stash[_TEST_PATH] = tmp_path
    return tmp_path


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item: pytest.Item) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if report.failed:
        item.stash[_FAILED] = True
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item) -> Generator[None, object, object]:
    result = yield
    path = item.stash.get(_TEST_PATH, None)
    session = item.config.stash[_SESSION]
    if path is not None and path.is_dir():
        if item.config.getoption("--ict-artifact-summary"):
            logical, unique, checkpoints, checkpoint_bytes = measure_test_files(path)
            reporter = item.config.pluginmanager.get_plugin("terminalreporter")
            if reporter:
                reporter.write_line(
                    f"artifact {item.nodeid} logical={logical} unique={unique} "
                    f"checkpoints={checkpoints} checkpoint_bytes={checkpoint_bytes}",
                )
        if not (session.keep_failed and item.stash.get(_FAILED, False)):
            try:
                session.discard_test(path)
            except (OSError, PipelineError):
                warn(pytest.PytestWarning(f"Deferred cleanup for {path}."), stacklevel=1)
    return result


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    artifacts = session.config.stash[_SESSION]
    if exitstatus in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
        artifacts.status = "passed"
    else:
        artifacts.status = "interrupted" if exitstatus == pytest.ExitCode.INTERRUPTED else "failed"
        if artifacts.keep_failed:
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter:
                reporter.write_line(f"Retained ICT test artifacts: {artifacts.store.directory}")


@pytest.fixture(autouse=True)
def isolate_reporting_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep run-tracking fixtures independent of OS tools and accelerators."""
    def hardware(host: Host, workspace: Path) -> tuple[Resources, Graphics]:
        return Resources("Fixture CPU", 2, None, None), Graphics("unknown", (), "unavailable", None)

    monkeypatch.setattr(metadata, "probe_hardware", hardware)
    monkeypatch.setattr(metadata, "run_probe", lambda _command: ("unknown", ""))
