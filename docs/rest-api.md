# Inspection REST API

The optional FastAPI server runs the same `InspectionService` as `ict inspect`. It returns version 1 inspection results with the same canonical class IDs, original-image pixel boxes, thresholds, and provenance. See the [shared inspection contracts](inspection-interface.md).

## Local startup

```text
python -m pip install -e ".[api]"
python -m ict_pipeline.api
```

The launcher binds to `127.0.0.1:8000` and starts one worker. It serves the interactive OpenAPI UI at `http://127.0.0.1:8000/docs` and the JSON specification at `/openapi.json`. The specification includes multipart request examples and structured errors. The default model IDs are `mock-empty` and `mock-defect`. Mock mode runs on CPU with Pillow and needs no model weights or dataset.

## Routes

| Method and path | Successful response |
| --- | --- |
| `GET /health` | `200` with `status: ok` when all configured models loaded, or `status: degraded` when some loaded. If none loaded, the API returns `503 model_unavailable`. |
| `GET /v1/models` | `200` with model IDs, mock or real mode, the mock scenario or detector family, and availability. |
| `POST /v1/inspections` | `201` with a stored result and a relative `Location` header. |
| `POST /v1/inspections/batch` | `200` with one item per upload, in upload order, including per-item status and a stored inspection or error. |
| `GET /v1/inspections/{inspection_id}` | `200` with the stored result for a server-issued UUID. |

Send a `multipart/form-data` body with a text field named `details` containing JSON. Send one `image` file field for a single inspection or repeat `images` in upload order for a batch. Set each file part's MIME type to `image/jpeg` or `image/png`. The server verifies the declared MIME against the decoded format and accepts only single-frame images. The filename is ignored, and coordinates refer to the encoded image grid.

A single request uses this `details` object.

```json
{
  "schema_version": 1,
  "model_id": "mock-defect",
  "request_id": "request-1",
  "source_id": "camera-3",
  "expected_size": {"width": 1280, "height": 720},
  "sensor": {"source_id": "session-19", "values": {"temperature": 23.4, "vibration": 0.03}}
}
```

`model_id`, `request_id`, and `source_id` are required. Version defaults to 1. `expected_size` and `sensor` are optional. Sensor values accept the five finite SMT measurement channels `temperature`, `humidity`, `vibration`, `acceleration`, and `noise`. Inference uses pixels only. The store retains the sanitized sensor source ID for correlation. Use non-sensitive identifiers. Arbitrary metadata and paths are rejected.

A batch selects one model for all images. Each item supplies the same image metadata as a single request, with the model selected at the batch level.

```json
{
  "schema_version": 1,
  "model_id": "mock-empty",
  "items": [
    {"request_id": "request-1", "source_id": "camera-3"},
    {"request_id": "request-2", "source_id": "camera-3"}
  ]
}
```

The batch validates the whole metadata object and requires one metadata item per file. The response has `schema_version: 1` and `items`. Each item has a zero-based `index`, an HTTP-style `status`, and either `inspection` or `error`, with the other field set to null. Per-item failures allow other items to finish and be stored. A request-level rejection or processing timeout produces a top-level HTTP error. A timed-out batch publishes no inspection records.

## Results and errors

A stored inspection includes `schema_version`, `inspection_id`, UTC `created_at`, `model_id`, and the shared versioned `result`. `result.detections` carries canonical class IDs and names, original AI Hub category IDs, confidence, and `bbox_xyxy` in absolute pixels. Mock results always carry `provenance.mode: "mock"` and a scenario, with `outcome: "review_required"` and `reason: "simulated_prediction"`. A real prediction with retained detections returns `defect_detected`. An empty real prediction requires review. The HTTP service never issues a `pass` decision.

Top-level errors have the following structure. The server-generated ID also appears in the `X-Request-ID` response header and structured access log. It is separate from the caller's `request_id` in inspection metadata.

```json
{
  "request_id": "679de798cf784be6b55acd91d5ec1602",
  "error": {
    "code": "dimension_mismatch",
    "message": "Image dimensions differ from expected_size."
  }
}
```

| HTTP status | Typical causes |
| --- | --- |
| `400` | Malformed multipart data, missing parts, or too many parts. |
| `401`, `403` | Missing or invalid bearer token, or a request blocked by local access rules. |
| `404` | Unknown model or missing or expired inspection. |
| `405` | Unsupported HTTP method. |
| `408` | Upload deadline elapsed. |
| `413` | Request body or image exceeds the configured byte limit. |
| `415` | Unsupported request or file MIME type. |
| `422` | Invalid metadata, image corruption, MIME/format mismatch, dimension limit, or mismatch with `expected_size`. |
| `429` | Request capacity or inference worker occupied. The response carries `Retry-After: 1`. |
| `500` | Prediction or other internal failure. |
| `503` | The selected model or inspection store is unavailable. |
| `504` | The inference deadline elapsed. |

## curl examples

These commands use shell single quotes for JSON. In Windows PowerShell, call `curl.exe` and use a curl form text file such as `details=<details.json` if PowerShell alters the quotes.

```sh
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models

curl -i http://127.0.0.1:8000/v1/inspections \
  -F 'image=@part.png;type=image/png' \
  -F 'details={"model_id":"mock-defect","request_id":"request-1","source_id":"camera-3"}'

curl http://127.0.0.1:8000/v1/inspections/batch \
  -F 'images=@part.png;type=image/png' \
  -F 'images=@other.jpg;type=image/jpeg' \
  -F 'details={"model_id":"mock-empty","items":[{"request_id":"1","source_id":"camera-3"},{"request_id":"2","source_id":"camera-3"}]}'

# Replace INSPECTION_ID with the UUID returned by a successful request.
curl http://127.0.0.1:8000/v1/inspections/INSPECTION_ID
```

## Python example

This example uses HTTPX, available through the `test-api` extra.

```python
import json
import os
from pathlib import Path

import httpx

token = os.environ.get("ICT_API_TOKEN")
headers = {"Authorization": f"Bearer {token}"} if token else {}
details = {"model_id": "mock-defect", "request_id": "request-1", "source_id": "camera-3"}
with httpx.Client(base_url="http://127.0.0.1:8000", headers=headers, timeout=60.0) as client:
    with Path("part.png").open("rb") as image:
        submission = client.post(
            "/v1/inspections",
            data={"details": json.dumps(details)},
            files={"image": ("part.png", image, "image/png")},
        )
        submission.raise_for_status()

    stored = client.get(submission.headers["Location"])
    stored.raise_for_status()
    result = stored.json()["result"]
    print(result["provenance"]["mode"], result["outcome"], result["detections"])
```

## Server settings and trusted checkpoints

Set `ICT_API_CONFIG` to a local JSON configuration file. Paths are relative to the server's working directory unless absolute. Model `run` paths are relative to `model_root`. A supplied `models` object replaces the default catalog. Restart the server to load a changed configuration or replace a checkpoint.

```json
{
  "host": "127.0.0.1",
  "port": 8010,
  "storage_root": "outputs/inspection-api",
  "model_root": "runs/training",
  "retention_seconds": 604800,
  "models": {
    "simulated": {"mode": "mock", "scenario": "defect"},
    "line-3": {
      "mode": "real",
      "model": "faster_rcnn",
      "run": "completed-run/faster_rcnn",
      "trust_run": true,
      "checkpoint": "best",
      "device": "cpu"
    }
  }
}
```

Real mode supports `yolo`, `faster_rcnn`, and `rt_detr` with the matching detector extra already installed. Give each model a distinct public ID. Real models require an explicitly trusted completed ICT run and canonical label policy. Native YOLO deserialization can execute code, so restrict the model root to administrator-controlled files and trust the entire run. An optional `expected_fingerprint` pins the ICT checkpoint fingerprint from a trusted channel. The server also accepts `checkpoint: "last"`, a supported `device`, an `image_size` override, and `thresholds` with `confidence` and `nms_iou`. An incompatible run is listed as unavailable. Requests for its ID fail with `503`. Each model loads once at startup and the server keeps up to `max_loaded_models` real models resident (default 3). Select this limit according to available host or accelerator memory.

| Limit | Default |
| --- | --- |
| `max_request_bytes` | 24 MiB including the multipart fields and framing |
| `max_image_bytes` | 8 MiB per encoded file |
| `max_width`, `max_height`, `max_pixels` | 6000 × 6000 bounds with a 16 million pixel cap |
| `max_batch_items` | 8 images |
| `max_concurrent_requests` | 4 admitted HTTP requests |
| `upload_timeout_seconds` | 15 seconds to read the request |
| `inference_timeout_seconds` | 30 seconds for image decode and inference for the entire request or batch |
| `retention_seconds` | null, so stored records remain available indefinitely |

The API has one inference worker per process and rejects work when that worker is occupied. The deadline stops waiting for the result. Native inference already in progress can continue, and the worker remains occupied until it returns. Pending batch items are skipped after cancellation. Orderly shutdown waits for the worker. Request bodies are bounded and parsed in memory, so size and concurrency limits also determine upload memory usage.

## Storage and access controls

The default file repository publishes one versioned JSON record per successful inspection under `storage_root`. It uses a same-directory atomic, insert-only write on a filesystem that supports hard links, such as NTFS. Each batch item is committed independently. Uploaded bytes may be copied to a private temporary working directory for the detector and are cleaned up after the operation. The service does not retain sensor readings, upload filenames, or raw request bodies as records. Keep the output and model roots private and separate from source data.

With a configured retention period, expired records return `404`. Expiration controls API visibility. Operators own physical removal of expired records and any temporary files left after a process crash. This local repository has no disk quota or automatic purge.

Without a token the server requires a loopback peer, loopback Host header, and same-origin browser requests. Set a strong `ICT_API_TOKEN` in the process environment to require bearer auth on every route, including the OpenAPI endpoints. The launcher requires a token for non-loopback binds. For token-protected OpenAPI, fetch `/openapi.json` with an authenticated client. Keep the server off public networks unless it sits behind TLS, authenticated access controls, and rate limits. The built-in bearer token grants access to the entire model catalog and result store. For shared deployment, use an authenticated gateway with appropriate per-user authorization, disable raw access logs, and keep direct access to the application private.

## Python integration and tests

Use `ict_pipeline.api.app.create_app(APISettings(...))` with ASGI lifespan enabled. The factory accepts an `InspectionRepository` and a predictor factory for integration and test use. A future SQLite repository can implement the same `add(record)` and `get(inspection_id)` interface. Existing CLI callers can keep using `ict inspect` or `InspectionService.inspect` with local paths.

```text
python -m pip install -e ".[api,test,test-api]"
python -m pytest -m "not cpu_backend and not real_data_smoke"
```

API tests generate small images and checkpoint fixtures and replace detector inference with deterministic predictors. Model execution, calibration, and deployment performance require separate validation with trusted checkpoints.
