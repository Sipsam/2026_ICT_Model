"""Console helpers for the standalone preprocessing scripts."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from time import monotonic
from typing import TextIO

from ..model import Progress
from .common import require_separate_output


def protect_source_data(out: Path, workspace: Path) -> None:
    require_separate_output(workspace / "data" / "raw", out)
    require_separate_output(workspace / "data" / "extracted", out)


@contextmanager
def fresh_output(out: Path, source: Path) -> Iterator[Path]:
    """Publish a complete new output directory after its files have been written."""
    if out.is_symlink():
        raise ValueError("Choose an output directory that is not a link.")
    out = out.resolve()
    require_separate_output(source, out)
    protect_source_data(out, Path.cwd())
    if out.exists():
        raise ValueError("Output already exists. Choose a fresh directory.")
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ict-integrity-", dir=out.parent) as directory:
        working = Path(directory) / "publishing"
        working.mkdir()
        yield working
        if out.exists():
            raise ValueError("Output already exists. Choose a fresh directory.")
        working.rename(out)


@contextmanager
def report_output(
    out: Path, names: tuple[str, ...], *, overwrite: bool, sources: tuple[Path, ...],
) -> Iterator[Path]:
    out = out.resolve()
    protect_source_data(out, Path.cwd())
    for source in sources:
        require_separate_output(source, out)
    targets = [out / name for name in names]
    if any(path.is_symlink() or path.is_dir() for path in targets):
        raise ValueError("Choose an output directory without links or directories at the report paths.")
    if not overwrite and any(path.exists() for path in targets):
        raise ValueError(f"Reports already exist in {out}. Choose another --out or approve replacement with --overwrite.")
    out.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".ict-report-", dir=out) as directory:
        working = Path(directory)
        yield working
        for target in targets:
            temporary = working / target.name
            if overwrite:
                temporary.replace(target)
            else:
                with target.open("xb") as destination, temporary.open("rb") as source:
                    copyfileobj(source, destination)


def console_progress(stage: str, *, file: TextIO | None = None) -> Progress:
    started = monotonic()
    last = float("-inf")

    def report(done: int, total: int | None) -> None:
        nonlocal last
        now = monotonic()
        if done == total or now - last >= 1:
            elapsed = now - started
            eta = f"{elapsed * (total - done) / done:.1f}s" if done and total is not None else "unknown"
            print(f"{stage} {done}/{total if total is not None else '?'}   elapsed {elapsed:.1f}s   ETA {eta}",
                  file=file, flush=True)
            last = now

    return report
