"""Dataset adapters for the resumable stage runner."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

from ..pipeline.model import StageContext, StageFailure, StageResult
from .ingestion import Acquisition, ArchiveSet, ExtractProvider, copy_dataset, storage_error, verify_dataset
from .model import DataError, Progress
from .providers import DownloadProvider


def limited_progress(report: Progress) -> Progress:
    last = -float("inf")

    def update(done: int, total: int | None) -> None:
        nonlocal last
        now = monotonic()
        if done == 0 or done == total or now - last >= 1:
            report(done, total)
            last = now

    return update


@dataclass(frozen=True)
class DataStages:
    acquisition: Acquisition
    provider: DownloadProvider
    existing: Path | None = None
    approved: bool = False
    show_plan: Callable[[tuple[str, ...]], None] | None = None

    def download(self, context: StageContext) -> StageResult:
        target = context.output_dir / "archives"
        selection = self.acquisition.selection
        try:
            if self.existing is not None:
                verify_dataset(context.inputs["existing_dataset"], selection)
                target.mkdir()
                (target / "reuse.json").write_text('{"source":"existing"}\n', encoding="utf-8")
            else:
                archives = self.acquisition.acquire(
                    self.provider, lambda _: limited_progress(context.progress), approved=self.approved,
                    show_plan=self.show_plan,
                )
                archives.publish(target, limited_progress(context.progress))
        except DataError as exc:
            raise StageFailure(f"Download stage. {exc}") from None
        except OSError as exc:
            raise StageFailure(f"Download stage. {storage_error(exc)}") from None
        return StageResult.SUCCESS

    def extract(self, context: StageContext) -> StageResult:
        selection = self.acquisition.selection
        target = context.output_dir / "dataset"
        progress = limited_progress(context.progress)
        try:
            if self.existing is not None:
                copy_dataset(context.inputs["existing_dataset"], target, selection, progress)
            else:
                archives = ArchiveSet.from_artifact(context.inputs["archives"], selection)
                ExtractProvider().extract(archives, target, selection, progress)
        except DataError as exc:
            raise StageFailure(f"Extract stage. {exc}") from None
        except OSError as exc:
            raise StageFailure(f"Extract stage. {storage_error(exc)}") from None
        return StageResult.SUCCESS
