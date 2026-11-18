"""Run an isolated, synthetic inspection demonstration and write a JSON report."""

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import tracemalloc
from collections.abc import Callable
from contextlib import closing, redirect_stdout
from io import StringIO
from pathlib import Path
from shutil import copyfile
from tempfile import mkdtemp
from time import monotonic
from typing import Any

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from ict_pipeline.api.app import create_app
from ict_pipeline.api.settings import APISettings
from ict_pipeline.cli import app as cli
from ict_pipeline.data.preprocessing.common import write_json
from ict_pipeline.data.preprocessing.console import protect_source_data
from ict_pipeline.inspection.contracts import InspectionResult
from ict_pipeline.streaming.runner import StreamRunner

HISTORY = "/v1/inspections"
SOURCE_STATUS = "/v1/sources/status"
FRAME_TIME = "2026-10-30T10:00:00+09:00"
SENSOR_TIME = "2026-10-30T01:00:00.125Z"
EXPECTED_BOX = [20.0, 10.0, 60.0, 30.0]
JsonObject = dict[str, Any]


class CheckFailed(Exception):
    """A failed demonstration check with a public diagnostic."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def make_inputs(root: Path, with_sensor: bool) -> tuple[APISettings, Path]:
    inputs = root / "inputs"
    images = inputs / "images"
    staged = root / "originals"
    images.mkdir(parents=True)
    staged.mkdir()
    names = ("linked.png", "unlinked.png", "ambiguous.png")
    for name, color in zip(names, ("#eea765", "#f2d778", "#7ed3b2"), strict=True):
        with Image.new("RGB", (80, 40), "#183731") as image:
            draw = ImageDraw.Draw(image)
            draw.rectangle((10, 8, 70, 32), fill="#296658")
            draw.rectangle((26, 12, 53, 28), fill=color)
            image.save(images / name)
        data = (images / name).read_bytes()
        (staged / (hashlib.sha256(data).hexdigest() + ".png")).write_bytes(data)
    manifest = inputs / "frames.json"
    write_json(manifest, {name: {"sample_id": Path(name).stem, "timestamp": FRAME_TIME} for name in names})
    sensor = inputs / "observations.json"
    if with_sensor:
        with sensor.open("x", encoding="utf-8") as stream:
            json.dump([
                {"sequence_id": sequence, "sample_id": sample, "timestamp": SENSOR_TIME,
                 "values": {"temperature": 19.625, "vibration": 0}, "units": {"temperature": "°C", "vibration": "m/s²"}}
                for sequence, sample in (("reading-23", "linked"), ("reading-24", "ambiguous"),
                                         ("reading-25", "ambiguous"))
            ], stream, ensure_ascii=False)
    settings = APISettings.model_validate_json(json.dumps({
        "storage_backend": "sqlite", "storage_root": str(root / "storage"),
        "image_root": str(staged), "model_root": str(root / "models"),
        "input_stream": {
            "stream_id": "synthetic-replay", "model_id": "mock-defect",
            "camera": {"kind": "directory", "source_id": "demo-camera",
                       "directory": str(images), "manifest": str(manifest)},
            "sensor": {"kind": "json", "source_id": "demo-sensor", "path": str(sensor)} if with_sensor else None,
            "alignment": {"allow_image_only": True},
            "processing": {"queue_capacity": 1},
        },
    }))
    (root / "api.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    return settings, images / "linked.png"


def client_for(settings: APISettings) -> TestClient:
    token = os.environ.get("ICT_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    return TestClient(create_app(settings), base_url="http://127.0.0.1", client=("127.0.0.1", 50000),
                      headers=headers, raise_server_exceptions=False)


def replay_status(client: TestClient) -> JsonObject:
    runner = client.app.state.input_stream
    require(isinstance(runner, StreamRunner) and runner.wait(5), "Replay did not complete.")
    response = client.get(SOURCE_STATUS)
    require(response.status_code == 200, "Source status was unavailable.")
    return response.json()


def check_simulated(result: JsonObject) -> None:
    InspectionResult.model_validate_json(json.dumps(result))
    require(result["schema_version"] == 1 and result["provenance"]["mode"] == "mock"
            and result["provenance"]["mock_scenario"] == "defect"
            and result["outcome"] == "review_required" and result["reason"] == "simulated_prediction",
            "Mock decision or version differed.")
    require(result["detections"] == [
        {"class_id": 1, "class_name": "cold_solder", "source_category_id": 23,
         "bbox_xyxy": EXPECTED_BOX, "confidence": 0.95},
    ], "Canonical detection or original pixel coordinates differed.")


def inspect_cli(root: Path, image: Path) -> JsonObject:
    with redirect_stdout(StringIO()) as output:
        exit_code = cli(args=["inspect", str(image), "--mock", "defect", "--output-dir", str(root / "cli")],
                        prog_name="ict", standalone_mode=False)
    require(exit_code in (None, 0), "CLI inspection failed.")
    report = Path(output.getvalue().strip())
    document = json.loads((report / "inspection.json").read_bytes())
    require(document["schema_version"] == 1 and len(document["records"]) == 1, "CLI report shape differed.")
    result = document["records"][0]
    check_simulated(result)
    with (report / "inspection.csv").open(encoding="utf-8", newline="") as stream:
        row, = csv.DictReader(stream)
    require(json.loads(row["detections_json"]) == result["detections"], "CLI CSV detection differed from JSON.")
    return result


def inspect_api(client: TestClient, data: bytes, cli_result: JsonObject) -> JsonObject:
    details = {"model_id": "mock-defect", "request_id": "demo-upload", "source_id": "demo-http"}
    response = client.post(HISTORY, data={"details": json.dumps(details)}, files={"image": ("ignored.png", data, "image/png")})
    require(response.status_code == 201, "Multipart upload did not commit.")
    record = response.json()
    require(record["schema_version"] == 1, "Stored record version differed.")
    check_simulated(record["result"])
    for field in ("image", "thresholds", "detections", "provenance", "outcome", "reason"):
        require(record["result"][field] == cli_result[field], f"CLI and upload differed in {field}.")
    location = response.headers["Location"]
    require(client.get(location).json() == record, "Upload detail did not round-trip.")
    require(client.get(f"{location}/image").content == data, "Staged original differed from the upload.")
    return record


def check_history(client: TestClient, upload: JsonObject, with_sensor: bool, root: Path) -> list[JsonObject]:
    response = client.get(HISTORY, params={"mode": "mock"})
    require(response.status_code == 200, "Mock history was unavailable.")
    page = response.json()
    require(page["total"] == 4 and upload in page["items"], "History did not contain three replay records and the upload.")
    reasons = {"linked": None, "unlinked": "unmatched", "ambiguous": "ambiguous"}
    samples = set()
    for item in page["items"]:
        result = item["result"]
        check_simulated(result)
        path = f"{HISTORY}/{item['inspection_id']}"
        require(client.get(path).json() == item, "History detail differed from its page.")
        if item == upload:
            continue
        metadata = result["metadata"]["streaming"]
        sample = metadata["camera"]["sample_id"]
        samples.add(sample)
        alignment = metadata["alignment"]
        require(metadata["schema_version"] == 1 and metadata["camera"]["timestamp"] == FRAME_TIME,
                "Replay lost the original camera timestamp.")
        require(alignment["reason"] == (reasons[sample] if with_sensor else "no_sensor_source"),
                "Replay alignment reason differed.")
        if with_sensor and sample == "linked":
            sensor = metadata["sensor_observation"]
            require(alignment["method"] == "sample_id" and alignment["confidence"] == "explicit"
                    and alignment["sensor_status"] == "available" and alignment["skew_seconds"] == 0.125
                    and result["sensor_source_id"] == "demo-sensor" and sensor["sequence_id"] == "reading-23"
                    and sensor["timestamp"] == SENSOR_TIME
                    and sensor["units"] == {"temperature": "°C", "vibration": "m/s²"}
                    and metadata["missing_measurements"] == ["humidity", "acceleration", "noise"]
                    and "values" not in sensor, "Linked sensor provenance or units differed.")
        else:
            require(alignment["sensor_status"] == "unavailable" and result["sensor_source_id"] is None
                    and metadata["sensor_observation"] is None, "An unavailable observation was attached.")
    require(samples == reasons.keys(), "Replay history lost an expected sample.")
    json_export = client.get(f"{HISTORY}/export", params={"mode": "mock"})
    csv_export = client.get(f"{HISTORY}/export", params={"mode": "mock", "format": "csv"})
    require(json_export.status_code == csv_export.status_code == 200 and json_export.json() == page,
            "History JSON export differed from the page.")
    rows = list(csv.DictReader(StringIO(csv_export.text)))
    require([json.loads(row["record_json"]) for row in rows] == page["items"],
            "History CSV lost the versioned record or alignment evidence.")
    (root / "history.json").write_bytes(json_export.content)
    (root / "history.csv").write_bytes(csv_export.content)
    return page["items"]


def check_dashboard(client: TestClient) -> JsonObject:
    for asset in ("/dashboard", "/dashboard/dashboard.css", "/dashboard/dashboard.mjs", "/dashboard/client.mjs"):
        require(client.get(asset).status_code == 200, "A dashboard asset was unavailable.")
    summary_path = "/v1/statistics/summary"
    real = client.get(summary_path).json()
    mock = client.get(summary_path, params={"mode": "mock"}).json()
    defects = client.get("/v1/statistics/defects", params={"mode": "mock"}).json()
    trends = client.get("/v1/statistics/trends", params={"mode": "mock"}).json()
    require(real["total_inspections"] == 0 and client.get(HISTORY, params={"mode": "real"}).json()["total"] == 0
            and mock["total_inspections"] == mock["total_detections"] == 4
            and mock["by_status"] == {"review_required": 4}
            and defects["items"] == [{"class_name": "cold_solder", "inspection_count": 4, "detection_count": 4}]
            and sum(row["inspection_count"] for row in trends["items"]) == 4,
            "Dashboard data endpoints returned inconsistent totals or modes.")
    return mock


def backup_database(database: Path, destination: Path) -> None:
    destination.touch(exist_ok=False, mode=0o600)
    with (
        closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as source,
        closing(sqlite3.connect(destination)) as backup,
    ):
        source.backup(backup)
        require(backup.execute("PRAGMA integrity_check").fetchall() == [("ok",)]
                and backup.execute("PRAGMA foreign_key_check").fetchall() == [], "SQLite backup integrity check failed.")


def run_demo(parent: Path, *, with_sensor: bool, progress: Callable[[str], None] | None = None) -> tuple[Path, bool]:
    parent = parent.resolve()
    protect_source_data(parent, Path.cwd())
    parent.mkdir(parents=True, exist_ok=True)
    root = Path(mkdtemp(prefix="inspection-e2e-", dir=parent))
    checks: dict[str, dict[str, object]] = {}
    stage = ""
    started = monotonic()
    completed = False

    def start(name: str) -> None:
        nonlocal stage
        stage = name
        if progress:
            progress(name)

    def passed(**evidence: object) -> None:
        checks[stage] = {"status": "passed", **evidence}

    tracing_owned = not tracemalloc.is_tracing()
    if tracing_owned:
        tracemalloc.start()
    try:
        start("fixture_and_cli_export")
        settings, image = make_inputs(root, with_sensor)
        data = image.read_bytes()
        cli_result = inspect_cli(root, image)
        passed(schema_version=1, detection=cli_result["detections"][0])

        start("replay_upload_history_and_exports")
        database = settings.storage_root / settings.database_name
        with (
            client_for(settings) as client,
            closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as reader,
        ):
            status = replay_status(client)
            require(status["state"] == "completed" and status["counters"]["persisted"] == 3,
                    "Fresh replay did not persist three inspections.")
            # Pin the pre-upload snapshot so the backup must read the new WAL commit.
            reader.execute("BEGIN")
            require(reader.execute("SELECT count(*) FROM inspections").fetchone() == (3,),
                    "The pre-upload SQLite snapshot differed from the replay status.")
            upload = inspect_api(client, data, cli_result)
            records = check_history(client, upload, with_sensor, root)
            passed(replay=status["counters"], upload_id=upload["inspection_id"], persisted_records=len(records))
            start("dashboard_assets_and_data")
            passed(mock_summary=check_dashboard(client))
            start("sqlite_online_backup")
            backup = root / "snapshot.sqlite3"
            wal_bytes = database.with_name(database.name + "-wal").stat().st_size
            require(wal_bytes > 32, "The test database did not retain committed WAL pages.")
            backup_database(database, backup)
            passed(integrity_check="ok", foreign_key_violations=0, live_wal_bytes=wal_bytes)

        start("restart_and_replay_deduplication")
        with client_for(settings) as restarted:
            status = replay_status(restarted)
            require(status["state"] == "completed" and status["counters"]["reused"] == 3
                    and restarted.get(HISTORY, params={"mode": "mock"}).json()["items"] == records,
                    "Replay restart changed committed records.")
            passed(replay=status["counters"], unchanged_records=len(records))

        start("sqlite_restore")
        restored_root = root / "restored"
        restored_root.mkdir()
        copyfile(backup, restored_root / settings.database_name)
        restored = settings.model_copy(update={"storage_root": restored_root, "input_stream": None})
        with client_for(restored) as client:
            require(client.get(HISTORY, params={"mode": "mock"}).json()["items"] == records,
                    "Restored database differed from the original.")
            passed(restored_records=len(records))

        start("temporary_artifacts")
        scratch = list(settings.storage_root.glob(".work-*")) + list(restored_root.glob(".work-*"))
        require(not scratch, "A server work directory survived normal shutdown.")
        passed(remaining_server_workspace_directories=len(scratch))
        completed = True
    except Exception as exc:
        # HTTP/library exceptions can embed credentials, request bodies, or paths.
        checks[stage] = {"status": "failed", "error": str(exc) if isinstance(exc, CheckFailed) else type(exc).__name__}
    finally:
        peak = tracemalloc.get_traced_memory()[1] if tracing_owned else None
        if tracing_owned:
            tracemalloc.stop()
    write_json(root / "integration-report.json", {
        "schema_version": 1, "status": "passed" if completed else "failed",
        "inference_mode": "mock", "real_model_inference": "not_run", "sensor_fixture": with_sensor,
        "transport": "ASGI TestClient with lifespan", "checks": checks,
        "runtime": {"platform": sys.platform, "python": sys.version.split()[0], "sqlite": sqlite3.sqlite_version},
        "elapsed_seconds": round(monotonic() - started, 3),
        "resources": {"python_traced_peak_bytes": peak,
                      "retained_artifact_bytes_before_report": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())},
        "manual_or_separate": {"browser_rendering": "not_run", "physical_input_devices": "not_run",
                               "native_model_resource_usage": "not_run"},
    })
    return root, completed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/inspection-e2e"))
    parser.add_argument("--image-only", action="store_true", help="Replay without a sensor snapshot.")
    args = parser.parse_args()
    root, passed = run_demo(args.output_dir, with_sensor=not args.image_only,
                            progress=lambda label: print(label, file=sys.stderr, flush=True))
    print(root / "integration-report.json")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
