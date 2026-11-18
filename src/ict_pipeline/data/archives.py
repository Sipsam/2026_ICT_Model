"""Validate AI Hub bundle layout and safely read their selected ZIP files."""

import bisect
import io
import re
import stat
import tarfile
import unicodedata
import zipfile
import zlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import BinaryIO

from .model import DataError, Selection
from .naming import translated

_PART = re.compile(r"(?P<name>.+\.zip)(?:\.part(?P<index>[0-9]{1,8}))?", re.IGNORECASE)
_LEGACY_SAMPLE = re.compile(r"/SD_(?:NOR|DEF)_(?:CC|ML|SH|MS|ST)_[A-Za-z0-9_-]+\.[A-Za-z]+", re.ASCII)


class IncompleteArchive(DataError):
    """A bundle is incomplete or failed its integrity check."""


def _parts(name: str) -> list[str]:
    parts = name.split("/")
    if any(
        not part or part in {".", ".."} or part.endswith((".", " "))
        or not part.isprintable() or any(char in part for char in ':\\<>"|?*')
        or PureWindowsPath(part).is_reserved()
        for part in parts
    ):
        raise DataError("Unsafe archive member path.")
    return parts


def _portable_key(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


def folder_name(name: str) -> str:
    stem = Path(name).stem
    return translated(stem) or stem


class _BundleZIP(io.RawIOBase):
    """Seekable view of ZIP parts inside an uncompressed TAR."""

    def __init__(self, stream: BinaryIO, parts: tuple[tarfile.TarInfo, ...]) -> None:
        super().__init__()
        self.stream = stream
        self.parts = parts
        self.ends = [0]
        for part in parts:
            self.ends.append(self.ends[-1] + part.size)
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.ends[-1]}[whence]
        position = base + offset
        if position < 0:
            raise OSError("ZIP seek preceded archive start.")
        self.position = position
        return position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        view = memoryview(buffer)
        done = 0
        while done < len(view) and self.position < self.ends[-1]:
            index = bisect.bisect_right(self.ends, self.position) - 1
            offset = self.position - self.ends[index]
            length = min(len(view) - done, self.parts[index].size - offset)
            self.stream.seek(self.parts[index].offset_data + offset)
            count = self.stream.readinto(view[done:done + length])
            if not count:
                raise IncompleteArchive("TAR archive ended inside a ZIP part.")
            self.position += count
            done += count
        return done


@dataclass(frozen=True)
class Archive:
    path: Path
    name: str
    parts: tuple[tarfile.TarInfo, ...] = ()

    @property
    def folder(self) -> str:
        return folder_name(self.name)

    @contextmanager
    def open(self) -> Iterator[zipfile.ZipFile]:
        try:
            if self.parts:
                with self.path.open("rb") as stream, io.BufferedReader(_BundleZIP(stream, self.parts)) as reader:
                    with zipfile.ZipFile(reader) as archive:
                        yield archive
            else:
                with zipfile.ZipFile(self.path) as archive:
                    yield archive
        except (zipfile.BadZipFile, EOFError, zlib.error):
            raise IncompleteArchive("ZIP is partial, corrupt, or failed its CRC check.") from None
        except (NotImplementedError, RuntimeError):
            raise DataError("ZIP uses unsupported compression or encryption.") from None

    def members(self, archive: zipfile.ZipFile, extensions: tuple[str, ...]) -> dict[str, zipfile.ZipInfo]:
        members = {}
        seen = set()
        for entry in archive.infolist():
            name = entry.orig_filename
            # All observed legacy 71900 ZIPs wrap a flat SD sample name in a
            # leading slash. Strip only this exact wrapper from known Korean ZIPs.
            if translated(Path(self.name).stem) == self.folder and _LEGACY_SAMPLE.fullmatch(name):
                name = name[1:]
            parts = _parts(name)
            mode = stat.S_IFMT(entry.external_attr >> 16)
            if len(parts) != 1 or mode not in {0, stat.S_IFREG} or entry.flag_bits & 1:
                raise DataError("ZIP contains a nested path, link, special file, or encrypted member.")
            if Path(name).suffix.lower() not in extensions or not entry.file_size:
                raise DataError("ZIP contains a file outside the selected modality or an empty sample.")
            key = _portable_key(name)
            if key in seen:
                raise DataError("ZIP contains duplicate or case-colliding sample names.")
            seen.add(key)
            members[name] = entry
        if not members:
            raise IncompleteArchive("ZIP contains no selected samples.")
        return members


def bundle_archives(path: Path, selection: Selection, split: str) -> tuple[Archive, ...]:
    groups: dict[str, dict[int | None, tarfile.TarInfo]] = {}
    seen = set()
    try:
        with tarfile.open(path, "r:") as bundle:
            for member in bundle:
                parts = _parts(member.name)
                key = _portable_key(member.name)
                if key in seen:
                    raise DataError("TAR contains duplicate or case-colliding member paths.")
                seen.add(key)
                if member.isdir():
                    continue
                if not member.isreg() or member.issparse():
                    raise DataError("TAR contains a link or unsupported special file.")
                match = _PART.fullmatch(parts[-1])
                if not match:
                    continue
                zip_name = match["name"]
                folder = folder_name(zip_name)
                if folder not in selection.folders or folder.partition("_")[0] != split:
                    raise DataError("TAR contains a ZIP outside the selected split or file groups.")
                declared = {part.casefold() for part in parts[:-1]} & {"training", "validation"}
                if declared and declared != {{"train": "training", "validation": "validation"}[split]}:
                    raise DataError("TAR ZIP path conflicts with its split prefix.")
                index = int(match["index"]) if match["index"] is not None else None
                if not member.size:
                    raise IncompleteArchive("TAR contains an empty ZIP part.")
                group = groups.setdefault("/".join([*parts[:-1], zip_name]), {})
                if index in group:
                    raise DataError("TAR contains duplicate ZIP part numbers.")
                group[index] = member
            size = path.stat().st_size
            assert bundle.fileobj is not None
            bundle.fileobj.seek(bundle.offset)
            if size % tarfile.BLOCKSIZE or bundle.fileobj.read(1024) != bytes(1024):
                raise IncompleteArchive("TAR is partial or lacks its end-of-archive blocks.")
    except (tarfile.TarError, EOFError):
        raise IncompleteArchive("TAR is partial or has an invalid header.") from None

    archives = []
    for name, group in groups.items():
        if None in group:
            if len(group) != 1:
                raise DataError("TAR contains both a ZIP and parts for that ZIP.")
            ordered = (group[None],)
        else:
            indexes = sorted(index for index in group if index is not None)
            if indexes != list(range(len(indexes))):
                raise IncompleteArchive("TAR contains a gap in its ZIP parts.")
            ordered = tuple(group[index] for index in indexes)
        archives.append(Archive(path, name.rsplit("/", 1)[-1], ordered))
    require_coverage(archives, selection, split)
    return tuple(archives)


def require_coverage(archives: list[Archive], selection: Selection, split: str) -> None:
    expected = {name for name in selection.folders if name.startswith(f"{split}_")}
    actual = {archive.folder for archive in archives}
    if len(archives) != len(actual):
        raise DataError("Multiple archives map to the same selected English folder.")
    if actual != expected:
        raise IncompleteArchive(f"Archive selection for {split} is incomplete "
                                f"({len(actual & expected)}/{len(expected)} selected folders).")
