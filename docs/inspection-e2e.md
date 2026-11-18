# Inspection service integration audit

## Run the demonstration

From the repository root, use an environment with the project CLI and its `api` and `test-api` extras already available. The demonstration uses Pillow, HTTPX, and the public FastAPI application factory. Both variants run on CPU with synthetic 80 × 40 PNGs and deterministic `mock-defect` predictions.

```text
python -m scripts.inspection_e2e
python -m scripts.inspection_e2e --image-only
```

The script prints the report path and exits with status 0 when its checks pass or 1 when a check fails. Use `--output-dir` to choose a parent for fresh run directories. The default parent is `outputs/inspection-e2e`. The app runs in-process with ASGI lifespan enabled. If `ICT_API_TOKEN` is set, the demo uses it to authenticate in memory.

| Artifact relative to the run directory | Contents |
| --- | --- |
| `integration-report.json` | Versioned outcome, per-check evidence, runtime, elapsed time, Python allocation peak, and retained file size |
| `inputs/`, `originals/`, `api.json` | Image and optional sensor fixtures, hash-staged dashboard originals, and configuration for serving the same run |
| `cli/<attempt>/inspection.json` and `inspection.csv` | Standalone CLI result |
| `storage/inspections.sqlite3`, `history.json`, `history.csv` | SQLite history and the API's exported page |
| `snapshot.sqlite3`, `restored/inspections.sqlite3` | Online backup and the backup copy opened by a fresh service |

Both variants produce four stored records, three from replay and one from upload. The CLI report is a separate artifact. Every record requires review for `simulated_prediction` and contains `cold_solder` (class 1, source category 23), confidence 0.95, and box `[20, 10, 60, 30]` in original pixels. The sensor variant links one observation by sample ID, records the 0.125-second skew, and keeps both original timestamp strings, the measurement units, and the missing channels in the versioned result. The unmatched and ambiguous frames proceed with image-only provenance. The strict-alignment regression verifies that both are skipped when image-only fallback is disabled.

## Open the persisted results in a browser

Start the usual API with the `api.json` next to the printed report. Replace the example path with the generated path. The loopback port defaults to 8000 and must be available.

```powershell
$env:ICT_API_CONFIG = "outputs/inspection-e2e/<run>/api.json"
python -m ict_pipeline.api
```

```sh
ICT_API_CONFIG="outputs/inspection-e2e/<run>/api.json" python -m ict_pipeline.api
```

1. Open the dashboard on loopback. If the API requires a bearer token, enter it under **API access** and apply the filters. [Local dashboard](http://127.0.0.1:8000/dashboard)
2. Check that **Real inspections** is empty. Select **Simulated inspections** and apply the filters. Expect four rows and four detections. For an older run, set the date range to include the UTC creation dates in `history.json`.
3. Open a replay record. Check the simulated label, `cold_solder` detection, original image, and centered overlay. With sensor fixtures, open **View versioned JSON** on the linked record and look for `reading-23` and the alignment evidence.
4. Check **Input sources → Check source status**. An unchanged replay completes with three reused records. Stop the server when finished.

The demo checks dashboard assets and data endpoints automatically. The steps above exercise rendering and controls in a browser. A headless Chromium smoke test exercised the same controls for this audit.

## Verification coverage

| Workflow | Evidence |
| --- | --- |
| CLI, upload, replay, history, and exports | The demo compares version 1 results, image descriptors, threshold settings, original-pixel boxes, and decision policy. The regression suite verifies all five defect labels through SQLite filters and export. |
| Sensor provenance | The demo verifies explicit linkage, original timestamp offsets and units, image-only fallback, and JSON and CSV round-trips. A predictor spy receives only a pixel snapshot, its dimensions, and an empty annotation tuple. Raw readings stay out of the stored record. |
| Recovery and concurrency | Repeated HTTP requests get new IDs. A forced duplicate inspection UUID cannot replace a committed row. An unchanged replay reuses committed IDs across restarts. A saturated one-slot queue stays bounded while a competing HTTP request times out waiting for the shared predictor. CLI batch failures and HTTP partial batch failures preserve successful results. |
| Uploads, paths, and model lifecycle | Tests cover malformed uploads, size and MIME restrictions, request-supplied path rejection, staged-original byte validation, token redaction, approved model IDs, fingerprint and detector checks, and one load per configured model during startup. Checkpoint tests use fake adapters and local fixtures. |
| SQLite recovery | The demo holds a pre-upload read snapshot, backs up the live database with committed WAL pages, checks database and foreign-key integrity, and opens a fresh service on the backup. All four records match. |
| Resource cleanup | Tests verify that temporary image snapshots disappear after inference and cancellation. The demo checks server scratch cleanup after shutdown. Three multipart abort regressions reproduced unclosed spools on the previous implementation. The parser now closes unfinished files on truncation, body-limit rejection, and timeout. |

Run the regression suite from the project environment with the existing test extras and Node.js.

```text
python -m pytest -q -rs -m "not cpu_backend and not real_data_smoke" -k "not real_data_regression"
node --test tests/dashboard.test.mjs
```

## Observed results

| Check | Status | Observed outcome |
| --- | --- | --- |
| Lightweight pytest command above | **Passed** | 284 passed, 3 skipped, 18 deselected. Skips covered Windows symlink privileges, an opt-in real-data integrity test, and a Linux-only detached worker. An installed AnyIO plugin emitted one deprecation warning. |
| JavaScript test command above | **Passed** | 6 passed on Node.js 24.22.1. |
| Both demo commands | **Passed** | Four simulated records in each variant, three reused on restart, and all four recovered from backup. The sensor and image-only runs retained 179,105 and 177,912 logical bytes before writing their reports, respectively. Their traced Python peaks were about 69 MB each. |
| Installed CLI and authenticated loopback HTTP | **Passed** | `ict inspect` exported both formats. Over Uvicorn, an unauthenticated request returned 401, an authenticated upload returned 201, and detail and history matched the redacted SQLite record. |
| Chromium dashboard over loopback HTTP | **Passed** | Chrome 157.0.8103.79 rendered the empty real view, four simulated rows, a staged image and its overlay, the linked sensor JSON, and replay status. Zero uncaught JavaScript errors. |
| Multipart cleanup against the previous implementation | **Failed as expected** | All three abort cases left an open upload spool. The same cases passed in the updated suite. |
| Real model and target hardware validation | **Not run** | Native model execution, physical devices, and full production load remain unverified. |

The Python checks ran on Windows with Python 3.11.9 and SQLite 3.45.1. The local test launcher appended an existing API dependency directory to the virtual environment's import path. [Machine-readable audit results](inspection-e2e-verification.json)

## Operational limits

- Replay uses an insert-only key derived from the input and settings. Keep those inputs stable for restart recovery. HTTP callers should track the returned inspection UUID and plan for distinct rows on retries. The supplied request ID is for correlation.
- HTTP uploads and replay share the predictor lock. Plan for upload timeouts under replay pressure. A native prediction can outlive the stream shutdown deadline, and queued frames depend on replaying the original inputs for recovery. [Queue and recovery behavior](input-streaming.md#queue-recovery-and-status)
- Keep SQLite and its sidecars on private, writable local storage. Back up with SQLite's backup API and restore into a fresh directory. Keep staged originals with the appropriate source material if an operator needs them after restoration. [Inspection database operations](inspection-database.md)
- The demonstration measures only Python allocations for a tiny synthetic run and counts logical bytes within its output directory before writing the report. Browser files, process RSS, native image decoding, accelerator memory, and sustained load are outside those measurements.
- USB and RTSP operation, real factory sensor timing, trusted trained model inference, detector accuracy, and production throughput require target-environment validation. Training, real-data smoke tests, and internal Test evaluation were outside this audit. The service routes no sensor values into its pixel detector.

Clients can use `ict inspect` and `ict history`, `InspectionService.inspect` and `InspectionService.inspect_bytes`, and the existing `/v1/inspections`, `/v1/statistics/*`, and `/v1/sources/status` contracts. Full alignment evidence is in `result.metadata.streaming`. [Inspection result contract](inspection-interface.md) · [REST API](rest-api.md) · [Replay inputs](input-streaming.md)
