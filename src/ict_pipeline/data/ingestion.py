"""Read-only verification, archive checkpoints, and atomic dataset publication."""

import errno
import json
import os
import shutil
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import sleep
from uuid import uuid4

from ..pipeline.store import RunStore, _is_link, digest_json, fingerprint_path
from .archives import Archive, IncompleteArchive, bundle_archives, folder_name, require_coverage
from .model import DataError, Progress, Selection
from .providers import BLOCK, DownloadProvider, RetryableDownload

MANIFEST = ".ict-ingestion.json"
CACHE = ".ict-aihub"


def check_destination(path: Path, required: int) -> None:
    """Check the target filesystem and test write access before a large operation."""
    parent = path
    while not parent.exists() and not parent.is_symlink():
        parent = parent.parent
    if not parent.is_dir() or _is_link(parent.lstat()):
        raise DataError("Ingestion needs a regular destination directory.")
    available = shutil.disk_usage(parent).free
    if available < required:
        raise DataError(f"Insufficient disk space for ingestion ({required} bytes required, {available} free).")
    try:
        with NamedTemporaryFile(dir=parent, prefix=".ict-write-") as stream:
            stream.write(b"\0")
            stream.flush()
    except OSError as exc:
        raise storage_error(exc) from None


def storage_error(exc: OSError) -> DataError:
    if exc.errno in {errno.ENOSPC, errno.EDQUOT}:
        return DataError("Ingestion ran out of disk space or exceeded the filesystem quota.")
    return DataError("Ingestion could not access a file. Check path permissions and available storage.")


def _regular(path: Path, *, directory: bool = False) -> bool:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return False
    return not _is_link(info) and (stat.S_ISDIR if directory else stat.S_ISREG)(info.st_mode)


def _walk_archives(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    if not _regular(root, directory=True):
        raise DataError("The archive source must be a regular directory.")
    for parent, dirs, filenames in os.walk(root, followlinks=False, onerror=_raise_walk_error):
        folder = Path(parent)
        dirs[:] = sorted(name for name in dirs if name != CACHE and _regular(folder / name, directory=True))
        for name in sorted(filenames):
            if name.lower().endswith(".zip"):
                path = folder / name
                if not _regular(path):
                    raise DataError("An archive source is a link or special file.")
                yield path


def _raise_walk_error(exc: OSError) -> None:
    raise exc


@dataclass(frozen=True)
class ArchiveSet:
    by_split: Mapping[str, tuple[Archive, ...]]

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(item.path for group in self.by_split.values() for item in group))

    def publish(self, root: Path, progress: Progress) -> None:
        total = sum(path.stat().st_size for path in self.files)
        check_destination(root, total)
        root.mkdir()
        copied = 0
        for split, group in self.by_split.items():
            directory = root / split
            directory.mkdir()
            for path in dict.fromkeys(item.path for item in group):
                before = path.stat()
                name = "bundle.tar" if path.suffix == ".tar" else path.name
                with path.open("rb") as source, (directory / name).open("xb") as destination:
                    for chunk in iter(lambda: source.read(BLOCK), b""):
                        destination.write(chunk)
                        copied += len(chunk)
                        progress(copied, total)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_ino) != (
                    after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_ino,
                ):
                    raise DataError("An archive source changed while it was being copied.")

    @classmethod
    def from_artifact(cls, root: Path, selection: Selection) -> "ArchiveSet":
        sources = {}
        for split in selection.splits:
            directory = root / split.name
            bundle = directory / "bundle.tar"
            if _regular(bundle):
                sources[split.name] = bundle_archives(bundle, selection, split.name)
            else:
                archives = [Archive(path, path.name) for path in _walk_archives(directory)]
                require_coverage(archives, selection, split.name)
                sources[split.name] = tuple(archives)
        return cls(sources)


def _validate_zip(archive: Archive, selection: Selection, *, crc: bool) -> None:
    with archive.open() as source:
        members = archive.members(source, selection.folders[archive.folder])
        if crc:
            for entry in members.values():
                with source.open(entry) as data:
                    while data.read(BLOCK):
                        pass


@dataclass(frozen=True)
class Acquisition:
    selection: Selection
    raw_root: Path

    @property
    def cache(self) -> Path:
        return self.raw_root / CACHE / self.selection.fingerprint

    def find(self, *, crc_local: bool = False) -> ArchiveSet:
        """Prefer complete, valid local ZIPs, then completed download checkpoints."""
        paths = tuple(_walk_archives(self.raw_root))
        groups = {}
        for split in self.selection.splits:
            name = split.name
            local = [Archive(path, path.name) for path in paths
                     if (folder := folder_name(path.name)) in self.selection.folders
                     and folder.startswith(f"{name}_")]
            try:
                require_coverage(local, self.selection, name)
                for archive in local:
                    _validate_zip(archive, self.selection, crc=crc_local)
            except IncompleteArchive:
                pass
            else:
                groups[name] = tuple(local)
                continue

            directory = self.cache / name
            for parent in (self.raw_root / CACHE, self.cache, directory):
                if (parent.exists() or parent.is_symlink()) and not _regular(parent, directory=True):
                    raise DataError("The archive checkpoint directory must be a regular directory.")
            if not directory.is_dir():
                continue
            for candidate in sorted(directory.glob("*.tar")):
                if not _regular(candidate):
                    raise DataError("The archive checkpoint must be a regular file.")
                expected, _, _ = candidate.stem.partition("-")
                if len(expected) != 64 or fingerprint_path(candidate) != expected:
                    continue
                try:
                    archives = bundle_archives(candidate, self.selection, name)
                except IncompleteArchive:
                    continue
                groups[name] = archives
                break
        return ArchiveSet(groups)

    def missing(self, archives: ArchiveSet) -> tuple[str, ...]:
        return tuple(split.name for split in self.selection.splits if split.name not in archives.by_split)

    def acquire(
        self, provider: DownloadProvider, progress_for: Callable[[str], Progress], *, approved: bool = False,
        show_plan: Callable[[tuple[str, ...]], None] | None = None,
    ) -> ArchiveSet:
        sources = self.find(crc_local=True)
        missing = self.missing(sources)
        if not missing:
            return sources
        if show_plan is not None:
            show_plan(missing)
        if not approved:
            raise DataError("Selected archives are missing or incomplete. Review the download plan and approve transfer.")

        remaining = [split for split in self.selection.splits if split.name in missing]
        check_destination(self.raw_root, sum(split.max_download_bytes for split in remaining))

        # This lock coordinates all runs targeting the same selection and raw root.
        lock = RunStore(self.raw_root / CACHE, self.selection.fingerprint)
        with lock.lock():
            sources = self.find(crc_local=True)
            groups = dict(sources.by_split)
            if self.missing(sources):
                provider.preflight()
            for split in remaining:
                if split.name in groups:
                    continue
                target = self.cache / split.name
                target.mkdir(exist_ok=True)
                for attempt in range(3):
                    check_destination(target, split.max_download_bytes)
                    partial = target / f"{uuid4().hex}.tar.partial"
                    progress = progress_for(split.name)
                    progress(0, None)
                    try:
                        provider.download(self.selection.dataset_key, split, partial, progress)
                        archives = bundle_archives(partial, self.selection, split.name)
                        for archive in archives:
                            _validate_zip(archive, self.selection, crc=True)
                    except (RetryableDownload, IncompleteArchive) as exc:
                        if attempt == 2:
                            raise DataError(f"Download failed for {split.name}. {exc}") from None
                        sleep(2 ** attempt)
                    else:
                        final = target / f"{fingerprint_path(partial)}-{uuid4().hex}.tar"
                        if final.exists() or final.is_symlink():
                            raise DataError("A conflicting final archive exists. Preserve it and choose another raw directory.")
                        partial.rename(final)
                        groups[split.name] = bundle_archives(final, self.selection, split.name)
                        break
        return ArchiveSet(groups)


@dataclass(frozen=True)
class DatasetReport:
    samples: dict[str, int]
    fingerprint: str | None


def has_dataset(root: Path, selection: Selection) -> bool:
    if (root.exists() or root.is_symlink()) and not _regular(root, directory=True):
        raise DataError("The dataset source must be a regular directory.")
    return root.is_dir() and any((root / name).exists() or (root / name).is_symlink()
                                 for name in selection.folders)


def verify_dataset(
    root: Path, selection: Selection, *, checksum: bool = False, progress: Progress | None = None,
) -> DatasetReport:
    if not _regular(root, directory=True):
        raise DataError("The extracted dataset directory is missing or is a link.")
    counts = {split.name: 0 for split in selection.splits}
    groups: dict[tuple[str, str], list[set[str]]] = {}
    fingerprints = {}
    marker = root / MANIFEST
    has_marker = marker.exists() or marker.is_symlink()
    total = len(selection.folders)
    if progress is not None:
        progress(0, total)
    for completed, (name, extensions) in enumerate(selection.folders.items(), 1):
        folder = root / name
        if not _regular(folder, directory=True):
            raise DataError(f"Missing or unsafe selected folder {name}.")
        split, kind, category = name.split("_", maxsplit=2)
        stems = set()
        for path in folder.iterdir():
            if not _regular(path) or path.suffix.lower() not in extensions or not path.stat().st_size:
                raise DataError(f"Unexpected, unreadable, or empty sample in {name}.")
            if checksum and not _has_sample_signature(path, kind):
                raise DataError(f"A sample in {name} has an invalid file signature.")
            if path.stem in stems:
                raise DataError(f"Duplicate sample stem in {name}.")
            stems.add(path.stem)
        if not stems:
            raise DataError(f"Empty selected folder {name}.")
        groups.setdefault((split, category), []).append(stems)
        if kind == "annotations":
            counts[split] += len(stems)
        if checksum or has_marker:
            fingerprints[name] = fingerprint_path(folder)
        if progress is not None:
            progress(completed, total)
    if any(any(stems != group[0] for stems in group[1:]) for group in groups.values()):
        raise DataError("Image, annotation, and sensor sample names do not match within a selected group.")
    fingerprint = digest_json(fingerprints) if fingerprints else None
    report = DatasetReport(counts, fingerprint)
    if has_marker:
        if not _regular(marker):
            raise DataError("The extraction completion manifest must be a regular file.")
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            raise DataError("The extraction completion manifest is invalid.") from None
        if recorded != _manifest(selection, report):
            raise DataError("The extracted dataset differs from its completion manifest. Use a new destination.")
    return report


def _has_sample_signature(path: Path, kind: str) -> bool:
    with path.open("rb") as stream:
        prefix = stream.read(64)
    if kind == "images":
        return prefix.startswith(b"\xff\xd8\xff")
    if kind == "sensors":
        return prefix.startswith(b"PK\x03\x04")
    return prefix.removeprefix(b"\xef\xbb\xbf").lstrip().startswith(b"{")


def _manifest(selection: Selection, report: DatasetReport) -> dict[str, object]:
    return {"schema_version": 1, "dataset_key": selection.dataset_key, "selection_sha256": selection.fingerprint,
            "tree_sha256": report.fingerprint, "samples": report.samples}


def _write_manifest(root: Path, selection: Selection) -> DatasetReport:
    report = verify_dataset(root, selection, checksum=True)
    (root / MANIFEST).write_text(json.dumps(_manifest(selection, report), indent=2) + "\n", encoding="utf-8")
    return report


class ExtractProvider:
    """Extract only selected flat samples into a new dataset directory."""

    def extract(self, archives: ArchiveSet, target: Path, selection: Selection, progress: Progress) -> DatasetReport:
        total = 0
        for split in selection.splits:
            group = archives.by_split.get(split.name, ())
            require_coverage(list(group), selection, split.name)
            for archive in group:
                with archive.open() as source:
                    total += sum(info.file_size for info in archive.members(
                        source, selection.folders[archive.folder],
                    ).values())
        check_destination(target, total)
        target.mkdir()
        done = 0
        for group in archives.by_split.values():
            for archive in group:
                folder = target / archive.folder
                folder.mkdir()
                with archive.open() as source:
                    for name, info in archive.members(source, selection.folders[archive.folder]).items():
                        with source.open(info) as data, (folder / name).open("xb") as output:
                            for chunk in iter(lambda: data.read(BLOCK), b""):
                                output.write(chunk)
                                done += len(chunk)
                            progress(done, total)
        return _write_manifest(target, selection)


def publish_dataset(
    root: Path, target: Path, archives: ArchiveSet, selection: Selection, progress: Progress,
) -> DatasetReport:
    with RunStore(root / "runs" / ".ingestion", digest_json(target.as_posix())).lock():
        if has_dataset(target, selection):
            return verify_dataset(target, selection, checksum=True)
        if target.exists():
            raise DataError("Extraction destination already exists. Choose a new directory.")
        check_destination(target.parent, 0)
        target.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".ict-extract-", dir=target.parent) as working:
            staged = Path(working) / "dataset"
            report = ExtractProvider().extract(archives, staged, selection, progress)
            if target.exists() or target.is_symlink():
                raise DataError("Extraction destination appeared during extraction. Choose a new directory.")
            staged.rename(target)
    return report


def copy_dataset(source: Path, target: Path, selection: Selection, progress: Progress) -> None:
    verify_dataset(source, selection)
    total = sum(path.stat().st_size for name in selection.folders for path in (source / name).iterdir())
    check_destination(target, total)
    target.mkdir()
    done = 0
    for name in selection.folders:
        original = source / name
        destination = target / name
        destination.mkdir()
        for path in original.iterdir():
            shutil.copyfile(path, destination / path.name)
            done += path.stat().st_size
            progress(done, total)
    _write_manifest(target, selection)
