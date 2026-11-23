# Inspection dashboard

The dashboard provides both interactive image inspection and read-only history
for the local inspection API. It uses the existing version 1 upload, history,
detail, statistics, model catalog, and export endpoints with either the file or
SQLite repository. FastAPI serves the HTML, CSS, and JavaScript with the API.
No frontend build is required.

## Start locally

With the existing API extra available, run this command from the project environment.

```text
python -m ict_pipeline.api
```

Open [the dashboard](http://127.0.0.1:8000/dashboard). Use the configured port if `ICT_API_CONFIG` changes it. The dashboard is served on loopback binds only. If you configured `ICT_API_TOKEN`, expand **API access**, enter the token, and select **Apply filters**. The page sends it in the Authorization header and keeps it only in page memory. API access to all records and staged originals follows the server's access rules. Keep the service and data roots private. See the [REST API access controls](rest-api.md#storage-and-access-controls).

The default model catalog includes `mock-empty` and `mock-defect`. Submit a local JPEG or PNG through the [API upload examples](rest-api.md#curl-examples), choose **Simulated inspections**, and apply the filters. Mock mode uses Pillow and runs without trained weights. The overview and history open in the **Real inspections** view, which excludes mock records. The simulated view contains only mock records, labels the rows and details as simulated, and leaves their stored outcomes and reasons intact.

## Run an inspection

Use **Inspection workspace** to choose or drag and drop a JPG or PNG, preview
its dimensions, select an available model returned by `GET /v1/models`, and run
the existing single or multimodal upload endpoint. The page blocks duplicate
submissions while inference is active and does not show failed HTTP requests as
stored history.

Faster R-CNN and image-only inputs use the uploaded image directly. Image-only
and Sensor Concat results render their sigmoid score bars separately from
detection boxes; scores are explicitly uncalibrated sigmoid outputs. Sensor
Concat additionally accepts the five documented channel values and capture
timestamp. Sending sensor evidence requires a matching server-owned
PairingReceipt and capture-time sensor policy. Missing or mismatched evidence
returns the existing API validation error. The UI does not bypass pairing or
present unverified fusion as a paired result.

The dashboard sends `schema_version`, `model_id`, `request_id`, and `source_id`
on every inspection. It adds `sample_id` only to multimodal FPN requests;
Faster R-CNN requests use the ordinary `/v1/inspections` endpoint and do not
include multimodal fields. Browser `datetime-local` values are interpreted in
the browser's local timezone and converted with `Date`/`toISOString()` before
upload. Invalid or nonexistent local times are rejected instead of being
appended with an ambiguous `Z` suffix.

**Run available models and compare** submits the available finalized detector
and FPN arms sequentially. Sensor Concat is included only when sensor evidence
is complete. A failed model does not hide successful comparison results.

## Browse inspections

- Filter by UTC date, outcome, canonical defect class, detector family, exact public model ID, or exact version 4 inspection UUID. The settings apply to both overview and history. Mock records have no detector family.
- Filter by submitted image, sensor, or paired evidence. Product details show the scored branch, sensor usability, weight status, and pairing fingerprint separately from the original-image detection overlay. Sensor-only records display their product evidence without an image. Untrained and unavailable product rows remain visible in history and are excluded from production statistics.
- The default range includes today and the previous 29 UTC dates. **From** and **Through** are inclusive calendar dates. Choose up to 366 days per query. The trend includes zero-count days for nonempty results.
- Total inspections counts records. Total detections counts retained boxes. Defect distributions show both counts for each class. A class filter selects matching inspections, and the overview includes every class detected on those inspections. The detector chart groups by public model ID.
- Use **Rows**, **Sort**, **Previous**, and **Next** to page through history. Equal timestamps are ordered by inspection UUID. JSON and CSV buttons export the current filtered page from the server. Use `ict history export` for a full export. The JSON export and detail view preserve both schema versions.
- Select **View** on a record to see its outcome, reason, identifiers, image descriptor, detection coordinates, thresholds, and available model and checkpoint provenance. Expand **View versioned JSON** for the complete stored record. Use Tab to move through controls and Escape to close the detail dialog. When the image is available, focusing a detection highlights its box.
- **Refresh data** reruns the applied query. Auto-refresh defaults to Off and can run every 1, 2, or 5 minutes while the tab is visible and no detail or export is open. The same date range remains active across refreshes.
- **Input sources → Check source status** requests one snapshot of the configured stream, current queue depth, and source health. It uses the applied API token and has no polling timer. Replay records carry alignment evidence under **View versioned JSON**. [Input streaming](input-streaming.md)

## Original images

The API keeps image dimensions, format, and SHA-256 in each successful record.
Set `image_root` in the JSON file selected by `ICT_API_CONFIG` to opt in to
automatic original-byte retention. If omitted, record behavior remains unchanged
and the dashboard reports the original image as unavailable.

For example, use the following settings with a dedicated private staging
directory writable only by trusted operators.

```json
{
  "storage_backend": "sqlite",
  "storage_root": "outputs/inspection-api",
  "image_root": "outputs/inspection-originals"
}
```

When enabled, the server verifies the decoded image, writes the upload atomically
by lower-case SHA-256 basename, and uses `.jpg` for JPEG or `.png` for PNG.
Duplicate SHA-256 files are reused only after exact byte verification. The image
is retained only when record persistence succeeds; partial temporary files are
removed, and a conflicting staged file prevents a new record from being
published. Client-provided filenames and paths are never used. An inspection UUID
is the only identifier used by the browser's image request. The server resolves
that UUID through the repository and checks the staged file's hash, format, and
original dimensions before serving it. Staged originals require private storage
capacity and backup management separate from record retention.

Boxes follow the original encoded pixel grid, including when a JPEG has EXIF orientation metadata. The image and overlay resize together with the same aspect ratio. The detection list retains class names, confidence scores, and coordinates when the original is unavailable. Failed records may have no image descriptor.

## Limits

The dashboard displays persisted records. Failed HTTP uploads or inference requests that only returned an error do not appear in history. Standalone `ict inspect` reports are separate artifacts. Statistics and pages come from separate reads, so totals or page membership can change during live writes or retention. File-backed history scans records for most queries. For larger collections, use the [indexed SQLite repository](inspection-database.md).

The dashboard uses standard browser modules and dialogs in recent desktop browsers. Run the lightweight backend and frontend tests from the repository root.

```text
python -m pytest -m "not cpu_backend and not real_data_smoke" -k "not real_data_regression"
node --test tests/dashboard.test.mjs
```

Node.js is used for the frontend tests only. The backend dashboard tests need the existing API and test extras.
