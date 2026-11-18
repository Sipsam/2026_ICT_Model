"""JSON and spreadsheet-safe CSV serialization for inspection history."""

import csv
import json
from collections.abc import Iterable
from typing import TextIO

from ..inspection.records import csv_identifier
from .history import ExportFormat, utc_timestamp
from .schemas import StoredInspection

CSV_COLUMNS = (
    "schema_version", "inspection_id", "created_at", "model_id", "model", "mode", "status",
    "checkpoint_fingerprint", "request_id", "source_id", "sensor_source_id",
    "detection_count", "detections_json", "record_json",
)


def write_history(stream: TextIO, records: Iterable[StoredInspection], format: ExportFormat) -> None:
    if format == "json":
        stream.write('{"schema_version":1,"items":[\n')
        separator = ""
        for record in records:
            stream.write(separator + record.model_dump_json())
            separator = ",\n"
        stream.write("\n]}\n")
    else:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(CSV_COLUMNS)
        for record in records:
            result = record.result
            provenance = result.provenance
            writer.writerow((
                record.schema_version, str(record.inspection_id), utc_timestamp(record.created_at), record.model_id,
                provenance.model, provenance.mode, result.outcome, provenance.checkpoint_fingerprint,
                csv_identifier(result.request_id), csv_identifier(result.source_id),
                csv_identifier(result.sensor_source_id), len(result.detections),
                json.dumps(result.model_dump(mode="json")["detections"], ensure_ascii=False, separators=(",", ":")),
                record.model_dump_json(),
            ))
