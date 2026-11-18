"""Read and verify the allowlisted sensor measurements."""

import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Literal, TypeVar

from ..data.preprocessing.audit import XLSX_FIELDS
from ..data.preprocessing.policy import SENSOR_FIELDS
from .contracts import Reading

_Reading = TypeVar("_Reading")


class SampleIssue(ValueError):
    """A stable diagnostic code for an individual sample."""


def parse_timestamp(value: object) -> datetime:
    if isinstance(value, str) and ("T" in value or " " in value.strip()):
        try:
            value = datetime.fromisoformat(value.strip())
        except ValueError:
            pass
    if not isinstance(value, datetime):
        raise SampleIssue("invalid_sensor_timestamp")
    return value.astimezone(UTC) if value.tzinfo is not None else value


def reading(timestamp: object, values: tuple[object, ...]) -> Reading:
    if any(isinstance(value, bool) for value in values):
        raise SampleIssue("invalid_sensor_value")
    try:
        return Reading(timestamp=parse_timestamp(timestamp), values=values)
    except SampleIssue:
        raise
    except ValueError:
        raise SampleIssue("invalid_sensor_value") from None


def check_label(value: object, label: int) -> None:
    if value is not None and (type(value) not in (int, float, bool) or value != label):
        raise SampleIssue("sensor_label_conflict")


def json_readings(raw: object, label: int) -> tuple[Reading, ...]:
    if raw is None or raw == []:
        return ()
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        raise SampleIssue("invalid_json_sensor_sequence")
    series = []
    for row in raw:
        check_label(row.get("defect_detected"), label)
        series.append(reading(row.get("timestamp"), tuple(row.get(key) for key in SENSOR_FIELDS)))
    return tuple(series)


def xlsx_readings(path: Path, label: int | None) -> tuple[Reading, ...]:
    return read_xlsx(path, reading, label)


def read_xlsx(
    source: Path | BinaryIO, parse: Callable[[object, tuple[object, ...]], _Reading],
    label: int | None = None,
) -> tuple[_Reading, ...]:
    """Read the five allowlisted columns with a caller-selected reading contract."""
    from openpyxl import load_workbook

    try:
        book = load_workbook(source, read_only=True, data_only=True)
        try:
            rows = book.active.iter_rows(values_only=True)
            header = tuple(str(value).strip() for value in next(rows, ()))
            columns = []
            for names in (("timestamp",), *zip(SENSOR_FIELDS, XLSX_FIELDS)):
                matches = [index for index, column in enumerate(header) if column in names]
                if len(matches) != 1:
                    raise SampleIssue("invalid_xlsx_sensor_columns")
                columns.append(matches[0])
            labels = [index for index, column in enumerate(header) if column == "defect_detected"]
            if len(labels) > 1:
                raise SampleIssue("invalid_xlsx_sensor_columns")
            series = []
            for row in rows:
                if labels and label is not None:
                    check_label(row[labels[0]], label)
                series.append(parse(row[columns[0]], tuple(row[index] for index in columns[1:])))
        finally:
            book.close()
    except SampleIssue:
        raise
    except Exception:
        raise SampleIssue("invalid_sensor_workbook") from None
    if not series:
        raise SampleIssue("empty_sensor_workbook")
    return tuple(series)


def aligned_sequence(
    embedded: tuple[Reading, ...], workbook: tuple[Reading, ...],
    capture: datetime | None, tolerance: float,
) -> tuple[tuple[Reading, ...], Literal["json", "xlsx", "both"] | None]:
    for series in (embedded, workbook):
        if any(not math.isfinite(value) for row in series for value in row.values):
            raise SampleIssue("invalid_sensor_value")
        try:
            ordered = all(first.timestamp < second.timestamp for first, second in zip(series, series[1:]))
        except TypeError:
            ordered = False
        if not ordered:
            raise SampleIssue("non_increasing_sensor_timestamps")
    if embedded and workbook:
        if len(embedded) != len(workbook):
            raise SampleIssue("sensor_length_mismatch")
        if any(left.timestamp != right.timestamp for left, right in zip(embedded, workbook)):
            raise SampleIssue("sensor_timestamp_mismatch")
        if any(
            abs(a - b) > tolerance
            for left, right in zip(embedded, workbook)
            for a, b in zip(left.values, right.values)
        ):
            raise SampleIssue("sensor_value_mismatch")
    sequence = embedded or workbook
    if sequence and capture is not None:
        first = sequence[0].timestamp
        # SMT filename capture timestamps are local wall times without a zone.
        # A zoned source requires an explicit zoned capture timestamp.
        if first != capture:
            raise SampleIssue("sensor_capture_timestamp_mismatch")
    source = "both" if embedded and workbook else "json" if embedded else "xlsx" if workbook else None
    return sequence, source
