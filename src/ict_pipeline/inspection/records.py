"""Publish stable inspection records and checkpoint batch progress."""

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from ..data.preprocessing.common import require_separate_output
from ..data.preprocessing.console import fresh_output, protect_source_data
from ..evaluation.runner import evaluation_code, write_result
from ..models.contracts import Progress
from ..pipeline.store import RunStore, digest_json, fingerprint_path, timestamp
from .contracts import ImageSource, InspectionError, InspectionRequest, InspectionResult
from .service import InspectionService

MAX_MANIFEST_BYTES = 32 * 1024**2
CSV_COLUMNS = (
    "schema_version", "request_id", "source_id", "input_line", "mode", "model", "mock_scenario",
    "checkpoint_fingerprint", "outcome", "reason", "image_sha256", "width", "height",
    "sensor_source_id", "confidence_threshold", "nms_iou", "detection_count", "detections_json",
)


def _json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _csv_id(value: str | None) -> str | None:
    return "'" + value if value and value[0] in "=+-@" else value


def write_records(output: Path, records: Iterable[InspectionResult]) -> bool:
    """Write into a staging directory. Return whether all requests ran successfully."""
    complete = True
    with (
        (output / "inspection.json").open("x", encoding="utf-8", newline="\n") as document,
        (output / "inspection.csv").open("x", encoding="utf-8", newline="") as table,
    ):
        writer = csv.writer(table, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        document.write('{"schema_version":1,"records":[\n')
        separator = ""
        for record in records:
            encoded = record.model_dump(mode="json")
            document.write(separator + _json(encoded))
            separator = ",\n"
            image = record.image
            provenance = record.provenance
            writer.writerow((
                record.schema_version, _csv_id(record.request_id), _csv_id(record.source_id), record.input_line,
                provenance.mode, provenance.model, provenance.mock_scenario, provenance.checkpoint_fingerprint,
                record.outcome, record.reason, image.sha256 if image else None, image.width if image else None,
                image.height if image else None, _csv_id(record.sensor_source_id), record.thresholds.confidence,
                record.thresholds.nms_iou, len(record.detections), _json(encoded["detections"]),
            ))
            complete = complete and record.outcome != "inference_failed"
        document.write("\n]}\n")
    return complete


@dataclass(frozen=True)
class BatchManifest:
    path: Path
    content: bytes = field(repr=False)
    fingerprint: str
    total: int


def read_manifest(path: Path) -> BatchManifest:
    """Read one bounded, immutable snapshot of the UTF-8 JSON Lines manifest."""
    try:
        if not path.is_file():
            raise OSError
        with path.open("rb") as stream:
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
    except OSError:
        raise InspectionError("Cannot read the batch manifest. Check its path and permissions.") from None
    if len(raw) > MAX_MANIFEST_BYTES:
        raise InspectionError("The batch manifest exceeds 32 MiB. Split it into smaller batches.")
    content = raw.removeprefix(b"\xef\xbb\xbf")
    if not content.strip():
        raise InspectionError("The batch manifest must contain at least one inspection request.")
    total = content.count(b"\n") + int(not content.endswith(b"\n"))
    return BatchManifest(path, content, hashlib.sha256(raw).hexdigest(), total)


def _cached_record(path: Path, line: int, image: ImageSource) -> InspectionResult | None:
    try:
        if path.is_symlink():
            raise ValueError
        document = json.loads(path.read_bytes())
        result = document["result"]
        if digest_json(result) != document["fingerprint"]:
            raise ValueError
        cached = InspectionResult.model_validate_json(_json(result))
        if cached.input_line != line or cached.image != image or cached.outcome not in ("defect_detected", "review_required"):
            raise ValueError
        return cached
    except FileNotFoundError:
        return None
    except (OSError, ValueError, TypeError, KeyError):
        raise InspectionError("An inspection recovery record is invalid. Start a fresh batch.") from None


def _batch_records(
    service: InspectionService, manifest: BatchManifest, journal: Path, progress: Progress,
) -> Iterable[InspectionResult]:
    progress("inspect images", 0, manifest.total)
    for number, line in enumerate(BytesIO(manifest.content), 1):
        try:
            request = InspectionRequest.model_validate_json(line)
        except ValidationError:
            result = service.invalid_request(number)
        else:
            # Scope cache hits to this input line and the exact bytes decoded by the service.
            def reuse(image: ImageSource) -> InspectionResult | None:
                return _cached_record(journal / f"{number}-{image.sha256}.json", number, image)

            result = service.inspect(request, base=manifest.path.parent, input_line=number, reuse=reuse)
            if result.outcome != "inference_failed" and result.image is not None:
                target = journal / f"{number}-{result.image.sha256}.json"
                if not target.exists():
                    write_result(target, result.model_dump(mode="json"))
        yield result
        progress("inspect images", number, manifest.total)


def publish_batch(
    service: InspectionService, manifest: BatchManifest, output: Path, progress: Progress, *,
    resume_from: Path | None = None,
) -> tuple[Path, bool]:
    """Publish a JSON/CSV pair after completing every line, reusing verified checkpoint records."""
    attempt = output / uuid4().hex if resume_from is None else resume_from
    protect_source_data(attempt, Path.cwd())
    require_separate_output(manifest.path, attempt)
    store = RunStore(attempt.parent, attempt.name, service.credentials)
    try:
        decoder = version("Pillow")
    except PackageNotFoundError:
        raise InspectionError("Inspection requires Pillow from the existing preprocess or detector extra.") from None
    code = evaluation_code()
    code.update({path.name: fingerprint_path(path) for path in Path(__file__).parent.glob("*.py")})
    identity = digest_json({
        "manifest": manifest.fingerprint, "code": code, "provenance": service.provenance.model_dump(mode="json"),
        "pillow": decoder,
        "thresholds": service.thresholds.model_dump(mode="json"),
    })
    marker = attempt / "batch.json"
    journal = attempt / "recovery"
    reports = attempt / "reports"
    if resume_from is not None and not marker.is_file():
        raise InspectionError("Pass an existing incomplete inspection batch to --resume-from.")
    if resume_from is None:
        attempt.mkdir(parents=True)
    with store.lock():
        if reports.exists() or journal.resolve() != journal or marker.resolve() != marker:
            raise InspectionError("Inspection batch is already published or its recovery layout is invalid.")
        if resume_from is None:
            journal.mkdir()
            store.write_json(marker.name, {"schema_version": 1, "fingerprint": identity})
        elif (not journal.is_dir()
              or json.loads(marker.read_bytes()) != {"schema_version": 1, "fingerprint": identity}):
            raise InspectionError("Resume requires the original manifest, model, settings and inspection code.")
        progress(f"inspection batch {attempt}", 0, 0)
        store.log({"time": timestamp(), "stage": "inspect images", "status": "running", "total": manifest.total})
        try:
            with fresh_output(reports, manifest.path) as working:
                complete = write_records(working, _batch_records(service, manifest, journal, progress))
        except KeyboardInterrupt:
            store.log({"time": timestamp(), "stage": "inspect images", "status": "interrupted"})
            raise
        except Exception:
            store.log({"time": timestamp(), "stage": "inspect images", "status": "failed"})
            raise
        store.log({"time": timestamp(), "stage": "publish records", "status": "completed",
                   "all_requests_succeeded": complete})
    return reports, complete
