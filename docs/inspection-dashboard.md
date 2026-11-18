# Inspection dashboard

The dashboard is a read-only browser interface for the local inspection API. It uses the existing version 1 history, detail, statistics, and export endpoints with either the file or SQLite repository. FastAPI serves the HTML, CSS, and JavaScript with the API. No frontend build is required.

## Start locally

With the existing API extra available, run this command from the project environment.

```text
python -m ict_pipeline.api
```

Open [the dashboard](http://127.0.0.1:8000/dashboard). Use the configured port if `ICT_API_CONFIG` changes it. The dashboard is served on loopback binds only. If you configured `ICT_API_TOKEN`, expand **API access**, enter the token, and select **Apply filters**. The page sends it in the Authorization header and keeps it only in page memory. API access to all records and staged originals follows the server's access rules. Keep the service and data roots private. See the [REST API access controls](rest-api.md#storage-and-access-controls).

The default model catalog includes `mock-empty` and `mock-defect`. Submit a local JPEG or PNG through the [API upload examples](rest-api.md#curl-examples), choose **Simulated inspections**, and apply the filters. Mock mode uses Pillow and runs without trained weights. The overview and history open in the **Real inspections** view, which excludes mock records. The simulated view contains only mock records, labels the rows and details as simulated, and leaves their stored outcomes and reasons intact.

## Browse inspections

- Filter by UTC date, outcome, canonical defect class, detector family, exact public model ID, or exact version 4 inspection UUID. The settings apply to both overview and history. Mock records have no detector family.
- The default range includes today and the previous 29 UTC dates. **From** and **Through** are inclusive calendar dates. Choose up to 366 days per query. The trend includes zero-count days for nonempty results.
- Total inspections counts records. Total detections counts retained boxes. Defect distributions show both counts for each class. A class filter selects matching inspections, and the overview includes every class detected on those inspections. The detector chart groups by public model ID.
- Use **Rows**, **Sort**, **Previous**, and **Next** to page through history. Equal timestamps are ordered by inspection UUID. JSON and CSV buttons export the current filtered page from the server. Use `ict history export` for a full export. The JSON export and detail view preserve both schema versions.
- Select **View** on a record to see its outcome, reason, identifiers, image descriptor, detection coordinates, thresholds, and available model and checkpoint provenance. Expand **View versioned JSON** for the complete stored record. Use Tab to move through controls and Escape to close the detail dialog. When the image is available, focusing a detection highlights its box.
- **Refresh data** reruns the applied query. Auto-refresh defaults to Off and can run every 1, 2, or 5 minutes while the tab is visible and no detail or export is open. The same date range remains active across refreshes.

## Original images

The API keeps image dimensions, format, and SHA-256 in each successful record. It does not retain uploaded image bytes. To display an original, set `image_root` in the JSON file selected by `ICT_API_CONFIG`, then restart the API. For example, use the following settings with a dedicated private staging directory writable only by trusted operators.

```json
{
  "storage_backend": "sqlite",
  "storage_root": "outputs/inspection-api",
  "image_root": "outputs/inspection-originals"
}
```

Copy the exact original bytes into the staging directory using the lower-case SHA-256 from `result.image.sha256` as the basename. Use `.jpg` for JPEG or `.png` for PNG, regardless of the original filename. Leave source files in place. An inspection UUID is the only identifier used by the browser's image request. The server resolves that UUID through the repository and checks the staged file's hash, format, and original dimensions before serving it. Only stage images appropriate for the users who can access the API. Staged originals are managed separately from record retention and backups.

Boxes follow the original encoded pixel grid, including when a JPEG has EXIF orientation metadata. The image and overlay resize together with the same aspect ratio. The detection list retains class names, confidence scores, and coordinates when the original is unavailable. Failed records may have no image descriptor.

## Limits

The dashboard displays persisted records. Failed HTTP uploads or inference requests that only returned an error do not appear in history. Standalone `ict inspect` reports are separate artifacts. Statistics and pages come from separate reads, so totals or page membership can change during live writes or retention. File-backed history scans records for most queries. For larger collections, use the [indexed SQLite repository](inspection-database.md).

The dashboard uses standard browser modules and dialogs in recent desktop browsers. Run the lightweight backend and frontend tests from the repository root.

```text
python -m pytest -m "not cpu_backend and not real_data_smoke" -k "not real_data_regression"
node --test tests/dashboard.test.mjs
```

Node.js is used for the frontend tests only. The backend dashboard tests need the existing API and test extras.
