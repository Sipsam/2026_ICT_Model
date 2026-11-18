"""Shared history queries and results for inspection repositories."""

from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import AbstractContextManager
from datetime import UTC, date, datetime, timedelta
from typing import Annotated, Literal, Protocol, Self, runtime_checkable

from pydantic import UUID4, AfterValidator, AwareDatetime, BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from ..evaluation.contracts import CLASSES
from ..inspection.contracts import Contract, Outcome, SchemaVersion
from ..models.contracts import ModelName
from .schemas import StoredInspection
from .settings import ModelID

Mode = Literal["all", "real", "mock"]
Order = Literal["newest", "oldest"]
ExportFormat = Literal["json", "csv"]


def _canonical_class(value: str) -> str:
    if value not in CLASSES.values():
        raise ValueError("Use a canonical defect class.")
    return value


def _parse_timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    raise ValueError("Use an ISO 8601 timestamp with a UTC offset.")


HistoryTimestamp = Annotated[AwareDatetime, BeforeValidator(_parse_timestamp)]


class HistoryFilter(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    created_from: HistoryTimestamp | None = None
    created_before: HistoryTimestamp | None = None
    status: Outcome | None = None
    defect_class: Annotated[str, AfterValidator(_canonical_class)] | None = None
    model_id: ModelID | None = None
    model: ModelName | None = None
    mode: Mode = "all"
    inspection_id: UUID4 | None = None

    @model_validator(mode="after")
    def check_dates(self) -> Self:
        start = utc_timestamp(self.created_from) if self.created_from is not None else None
        end = utc_timestamp(self.created_before) if self.created_before is not None else None
        if start is not None and end is not None and start >= end:
            raise ValueError("created_from must precede created_before.")
        return self


class StatisticsFilter(HistoryFilter):
    mode: Mode = "real"


class HistoryQuery(HistoryFilter):
    order: Order = "newest"
    limit: int = Field(default=100, ge=1, le=500)
    offset: int = Field(default=0, ge=0, le=2**63 - 1)


class ExportQuery(HistoryQuery):
    format: ExportFormat = "json"


class HistoryPage(Contract):
    schema_version: SchemaVersion = 1
    total: int
    limit: int
    offset: int
    items: tuple[StoredInspection, ...]


class StatisticsSummary(Contract):
    schema_version: SchemaVersion = 1
    mode: Mode
    total_inspections: int
    total_detections: int
    by_status: dict[Outcome, int]
    by_model: dict[str, int]


class DefectFrequency(Contract):
    class_name: str
    inspection_count: int
    detection_count: int


class DefectStatistics(Contract):
    schema_version: SchemaVersion = 1
    mode: Mode
    items: tuple[DefectFrequency, ...]


class DailyCount(Contract):
    date: date
    inspection_count: int


class TrendStatistics(Contract):
    schema_version: SchemaVersion = 1
    mode: Mode
    items: tuple[DailyCount, ...]


@runtime_checkable
class InspectionHistory(Protocol):
    def list(self, query: HistoryQuery) -> HistoryPage: ...

    def records(
        self, filters: HistoryFilter, order: Order,
    ) -> AbstractContextManager[Iterator[StoredInspection]]: ...

    def summary(self, filters: StatisticsFilter) -> StatisticsSummary: ...

    def defects(self, filters: StatisticsFilter) -> DefectStatistics: ...

    def trends(self, filters: StatisticsFilter) -> TrendStatistics: ...


def utc_timestamp(value: datetime) -> str:
    try:
        if value.utcoffset() is None:
            raise ValueError
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    except (OverflowError, ValueError):
        raise ValueError("Timestamp cannot be represented in UTC.") from None


def retention_cutoff(seconds: int | None) -> datetime | None:
    if seconds is None:
        return None
    try:
        return datetime.now(UTC) - timedelta(seconds=seconds)
    except OverflowError:
        return None


def matches(record: StoredInspection, filters: HistoryFilter, cutoff: datetime | None) -> bool:
    result = record.result
    # Normalize before comparison, including during ambiguous local DST hours.
    created = record.created_at.astimezone(UTC)
    return (
        (cutoff is None or created > cutoff)
        and (filters.created_from is None or created >= filters.created_from.astimezone(UTC))
        and (filters.created_before is None or created < filters.created_before.astimezone(UTC))
        and (filters.status is None or filters.status == result.outcome)
        and (filters.inspection_id is None or filters.inspection_id == record.inspection_id)
        and (filters.mode == "all" or filters.mode == result.provenance.mode)
        and (filters.model_id is None or filters.model_id == record.model_id)
        and (filters.model is None or filters.model == result.provenance.model)
        and (filters.defect_class is None or any(d.class_name == filters.defect_class for d in result.detections))
    )


def summarize(records: Iterable[StoredInspection], mode: Mode) -> StatisticsSummary:
    statuses: Counter[Outcome] = Counter()
    models: Counter[str] = Counter()
    detections = 0
    for record in records:
        statuses[record.result.outcome] += 1
        models[record.model_id] += 1
        detections += len(record.result.detections)
    return StatisticsSummary(
        mode=mode, total_inspections=sum(statuses.values()), total_detections=detections,
        by_status=dict(sorted(statuses.items())), by_model=dict(sorted(models.items())),
    )


def count_defects(records: Iterable[StoredInspection], mode: Mode) -> DefectStatistics:
    inspections: Counter[str] = Counter()
    detections: Counter[str] = Counter()
    for record in records:
        classes = [d.class_name for d in record.result.detections]
        inspections.update(set(classes))
        detections.update(classes)
    return DefectStatistics(mode=mode, items=tuple(
        DefectFrequency(class_name=name, inspection_count=inspections[name], detection_count=count)
        for name, count in sorted(detections.items())
    ))


def count_days(records: Iterable[StoredInspection], mode: Mode) -> TrendStatistics:
    counts = Counter(record.created_at.astimezone(UTC).date() for record in records)
    return TrendStatistics(mode=mode, items=tuple(
        DailyCount(date=day, inspection_count=count) for day, count in sorted(counts.items())
    ))
