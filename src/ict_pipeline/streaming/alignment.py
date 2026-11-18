"""Conservative one-to-one association within a bounded source snapshot."""

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

from .contracts import SampleReference, SensorObservation
from .settings import AlignmentSettings

EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
Method = Literal["sample_id", "production_id", "timestamp"]
AlignmentReason = Literal[
    "no_sensor_source", "sensor_unavailable", "missing_reference", "unmatched", "ambiguous",
    "conflicting_identifiers", "outside_time_window", "incompatible_timestamps", "already_matched",
    "missing_measurements",
]


def _identities(event: SampleReference) -> Iterable[tuple[Method, str]]:
    if event.sample_id is not None:
        yield ("sample_id", event.sample_id)
    if event.production_id is not None:
        yield ("production_id", event.production_id)


def _clock_us(event: SampleReference) -> int | None:
    timestamp = event.instant
    if event.sample_id is not None or event.production_id is not None or timestamp is None or timestamp.tzinfo is None:
        return None
    delta = timestamp - EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


@dataclass(frozen=True)
class Alignment:
    method: Method | None = None
    observation: SensorObservation | None = None
    reason: AlignmentReason | None = None
    skew_seconds: float | None = None


class SampleAligner:
    def __init__(
        self, settings: AlignmentSettings, observations: Iterable[SensorObservation],
        camera_references: Iterable[SampleReference] = (),
    ) -> None:
        self.settings = settings
        self._by_identity: dict[tuple[Method, str], list[SensorObservation]] = defaultdict(list)
        timed: list[tuple[int, SensorObservation]] = []
        self._sequences: Counter[tuple[str, str]] = Counter()
        self._used: set[tuple[str, str]] = set()
        self._count = 0
        for observation in observations:
            self._count += 1
            self._sequences[observation.source_id, observation.sequence_id] += 1
            for identity in _identities(observation):
                self._by_identity[identity].append(observation)
            if settings.shared_clock and (instant := _clock_us(observation)) is not None:
                timed.append((instant, observation))
        timed.sort(key=lambda item: item[0])
        self._sensor_times = [instant for instant, _ in timed]
        self._timed = [observation for _, observation in timed]
        self._camera_identities: Counter[tuple[Method, str]] = Counter()
        self._camera_times: list[int] = []
        for reference in camera_references:
            for identity in _identities(reference):
                self._camera_identities[identity] += 1
            if settings.shared_clock and (instant := _clock_us(reference)) is not None:
                self._camera_times.append(instant)
        self._camera_times.sort()

    @property
    def duplicate_observations(self) -> int:
        return sum(count for count in self._sequences.values() if count > 1)

    @property
    def unused_observations(self) -> int:
        return self._count - len(self._used)

    def _window(self, timeline: list[int], center: int) -> tuple[int, int]:
        radius = int(self.settings.max_skew_seconds * 1_000_000)
        return bisect_left(timeline, center - radius), bisect_right(timeline, center + radius)

    def match(self, frame: SampleReference) -> Alignment:
        identities = tuple(_identities(frame))
        if identities:
            identity = identities[0]
            if len(identities) == 2 and identity not in self._by_identity:
                identity = identities[1]
            method = identity[0]
            candidates = self._by_identity.get(identity, ())
            if self._camera_identities[identities[0]] > 1 or (
                self._camera_identities[identity] > 1 or len(candidates) > 1
            ):
                return Alignment(reason="ambiguous")
            if not candidates:
                return Alignment(reason="unmatched")
            observation = candidates[0]
            if ((frame.production_id is not None and observation.production_id is not None
                    and frame.production_id != observation.production_id)
                    or (frame.sample_id is not None and observation.sample_id is not None
                        and frame.sample_id != observation.sample_id)):
                return Alignment(reason="conflicting_identifiers")
        elif self.settings.shared_clock and (instant := _clock_us(frame)) is not None:
            method = "timestamp"
            start, end = self._window(self._sensor_times, instant)
            if start == end:
                return Alignment(reason="outside_time_window")
            first, last = self._window(self._camera_times, self._sensor_times[start])
            if end - start != 1 or last - first != 1:
                return Alignment(reason="ambiguous")
            observation = self._timed[start]
        else:
            return Alignment(reason="missing_reference")

        key = (observation.source_id, observation.sequence_id)
        if self._sequences[key] != 1:
            return Alignment(reason="ambiguous")
        if key in self._used:
            return Alignment(reason="already_matched")
        skew = None
        capture, measured = frame.instant, observation.instant
        if capture is not None and measured is not None:
            if (capture.tzinfo is None) != (measured.tzinfo is None):
                return Alignment(reason="incompatible_timestamps")
            skew = (measured - capture).total_seconds()
            if abs(skew) > self.settings.max_skew_seconds:
                return Alignment(reason="outside_time_window", skew_seconds=skew)
        self._used.add(key)
        return Alignment(method=method, observation=observation,
                         reason=None if observation.measurements else "missing_measurements", skew_seconds=skew)
