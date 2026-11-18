"""Bounded, deterministic readers for local replay fixtures."""

import csv
import hashlib
import json
import stat
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from threading import Event

from pydantic import ValidationError

from ..data.preprocessing.common import safe_component
from .contracts import CameraFrame, SampleReference, SensorObservation, SourceIssue, SENSOR_COLUMNS, SMT_UNITS
from .settings import DirectorySettings, SensorReplaySettings

INDEX_BYTES = 16 * 1024**2
IMAGE_FORMATS = {".jpg": "JPEG", ".jpeg": "JPEG", ".png": "PNG"}
REFERENCE_FIELDS = frozenset({"timestamp", "sequence_id", "sample_id", "production_id"})


def _read(path: Path, limit: int) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise OSError
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise SourceIssue("input_limit")
    return data


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _json(data: str) -> object:
    return json.loads(data, object_pairs_hook=_unique_keys)


class DirectoryReplay:
    kind = "directory"

    def __init__(self, settings: DirectorySettings, *, max_image_bytes: int) -> None:
        self.settings = settings
        self.source_id = settings.source_id
        self.max_image_bytes = max_image_bytes
        self._entries: list[tuple[Path, SampleReference]] | None = None

    def _index(self, stop: Event) -> list[tuple[Path, SampleReference]]:
        images: dict[str, Path] = {}
        try:
            for entry in self.settings.directory.iterdir():
                if stop.is_set():
                    return []
                if entry.suffix.lower() in IMAGE_FORMATS:
                    images[entry.name] = entry
                    if len(images) > self.settings.max_frames:
                        raise SourceIssue("input_limit")
        except OSError:
            raise SourceIssue("source_unavailable") from None
        if not images:
            raise SourceIssue("source_unavailable")

        manifest: dict[str, SampleReference] = {}
        try:
            if self.settings.manifest is not None:
                raw = _json(_read(self.settings.manifest, INDEX_BYTES).decode("utf-8-sig"))
                if not isinstance(raw, dict) or any(not safe_component(name) or name not in images for name in raw):
                    raise ValueError
                manifest = {name: SampleReference.model_validate(item) for name, item in raw.items()}
            index = [(images[name], manifest.get(name, SampleReference())) for name in sorted(images)]
            if self.settings.order == "timestamp":
                if any(reference.instant is None for _, reference in index):
                    raise ValueError
                index.sort(key=lambda item: item[1].instant)
            elif self.settings.order == "reverse_name":
                index.reverse()
            return index
        except (OSError, ValueError, TypeError, OverflowError, RecursionError):
            raise SourceIssue("invalid_manifest") from None

    def _catalog(self, stop: Event) -> list[tuple[Path, SampleReference]]:
        if self._entries is None:
            self._entries = self._index(stop)
        return self._entries

    def references(self, stop: Event) -> Iterator[SampleReference]:
        for _, reference in self._catalog(stop):
            yield reference

    def frames(self, stop: Event) -> Iterator[CameraFrame | SourceIssue]:
        entries = self._catalog(stop)
        self._entries = None
        for index, (path, reference) in enumerate(entries):
            rate = self.settings.frames_per_second
            if stop.is_set() or (index and rate and stop.wait(1 / rate)):
                return
            try:
                data = _read(path, self.max_image_bytes)
                yield CameraFrame(
                    source_id=self.source_id, sequence_id=hashlib.sha256(path.name.encode("utf-8")).hexdigest(),
                    kind=self.kind, format=IMAGE_FORMATS[path.suffix.lower()], data=data,
                    **reference.model_dump(),
                )
            except SourceIssue as issue:
                yield issue
            except (OSError, ValueError):
                yield SourceIssue("invalid_frame")


def _csv_rows(data: str) -> Iterator[dict[str, object]]:
    reader = csv.DictReader(StringIO(data, newline=""), strict=True)
    header = reader.fieldnames or []
    canonical = [SENSOR_COLUMNS.get(name, name) for name in header]
    if (len(set(canonical)) != len(header) or not {"sequence_id", "timestamp"} <= set(canonical)
            or set(canonical) - REFERENCE_FIELDS - SMT_UNITS.keys()):
        raise ValueError
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError
        fields = {key: row[key] or None for key in REFERENCE_FIELDS if key in row}
        values = {
            name: float(row[column]) if row[column].strip() else None
            for column, name in zip(header, canonical, strict=True) if name in SMT_UNITS
        }
        yield {**fields, "values": values}


class SensorReplay:
    def __init__(self, settings: SensorReplaySettings) -> None:
        self.settings = settings
        self.kind = settings.kind
        self.source_id = settings.source_id

    def observations(self, stop: Event) -> Iterator[SensorObservation]:
        try:
            data = _read(self.settings.path, INDEX_BYTES).decode("utf-8-sig")
        except OSError:
            raise SourceIssue("source_unavailable") from None
        except UnicodeError:
            raise SourceIssue("invalid_sensor_data") from None
        try:
            rows = _csv_rows(data) if self.kind == "csv" else _json(data)
            if self.kind == "json" and not isinstance(rows, list):
                raise ValueError
            for index, row in enumerate(rows):
                if stop.is_set():
                    return
                if index >= self.settings.max_observations:
                    raise SourceIssue("input_limit")
                if not isinstance(row, dict) or row.keys() - REFERENCE_FIELDS - {"values", "units"}:
                    raise ValueError
                yield SensorObservation(source_id=self.source_id, kind=self.kind, **row)
        except (ValidationError, ValueError, csv.Error, OverflowError, RecursionError):
            raise SourceIssue("invalid_sensor_data") from None
