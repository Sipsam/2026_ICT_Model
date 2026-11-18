# Inspection interface

`ict inspect` processes local JPEG and PNG images with one initialized detector. It writes `inspection.json` and `inspection.csv` to a fresh report directory and prints that path on standard output. The default parent is `outputs/inspection`. Batch progress goes to standard error.

## Single image

```text
ict inspect part.png --mock empty
ict inspect part.png --mock defect --request-id request-001 --source-id camera-03 --output-dir outputs/line-3
ict inspect part.png --run path/to/completed/model-run --trust-run --device cpu --checkpoint best
```

In single-image mode both identifiers default to the image filename. Mock mode needs Pillow from the existing preprocessing or test environment. The `defect` scenario produces a simulated `cold_solder` box centered in the image at confidence 0.95. The `empty` scenario produces an empty list. Every mock record has `provenance.mode: "mock"` and names its scenario.

Real inference uses the existing local detector dependencies and the same predictors as evaluation. Pass a completed model directory containing its `run.json` and selected `best` or `last` checkpoint. `--trust-run` attests that you trust the entire run and is required because native YOLO checkpoint deserialization can execute code. Inspection checks the completed run, canonical label policy, checkpoint format, recorded path and fingerprint. It verifies a private copy before invoking the adapter. An optional `--expected-fingerprint` pins the ICT fingerprint obtained through an independent trusted channel. These fingerprints use the ICT file fingerprint algorithm and differ from the raw file SHA-256 shown by tools such as `Get-FileHash`.

CPU and FP32 are the defaults. Use `--device cuda`, `rocm`, or `mps` with a compatible installed backend and device. Real inference uses the saved training image size by default. `--image-size` overrides it with a multiple of 32 subject to model limits. `--confidence` defaults to 0.25 and keeps scores at or above the threshold. `--nms-iou` defaults to 0.7 and suppresses overlapping boxes of the same class when IoU exceeds the threshold. YOLO and Faster R-CNN also use their native NMS. RT-DETR uses its existing query selection followed by inspection NMS.

## Batch requests

```text
ict inspect --batch requests.jsonl --mock defect --output-dir outputs/line-3
```

Provide one UTF-8 version 1 request per physical line, up to 32 MiB per manifest. A UTF-8 BOM and a final newline are accepted. Relative image paths resolve from the manifest's directory. This example shows one line.

```json
{"schema_version":1,"request_id":"request-001","source_id":"camera-03","image_path":"images/part.png","expected_size":{"width":1280,"height":720},"sensor":{"source_id":"session-19","values":{"temperature":23.4,"vibration":0.03}},"metadata":{"lot":"A37"}}
```

`request_id`, `source_id`, and `image_path` are required. The batch preserves line order and records the one-based `input_line` even for malformed rows. `expected_size` is optional and compares against encoded width and height. Inspection checks image contents, decodes the image, and accepts single-frame JPEG and PNG up to 64 MiB encoded and 25 million pixels. Coordinates refer to the original encoded pixel grid. EXIF orientation is left as encoded.

`sensor` and `metadata` are optional. Sensor values support the canonical SMT measurements `temperature`, `humidity`, `vibration`, `acceleration`, and `noise`. Image detectors receive only pixels. The result carries the sensor source ID and sanitized metadata for correlation. Sensor measurements stay with the request, and sensor alignment and quality require upstream validation.

To use a verified offline sensor step as inspection context, send the observed
source-unit values in the existing `sensor.values` dictionary and omit missing
channels. Padded or scaled dataset values are reserved for downstream model
inputs. [Multimodal sequence contracts](multimodal-data.md)

## Results

The JSON document contains `schema_version: 1` and a `records` array. Each record includes request and source identifiers when available, outcome and reason, image dimensions and content SHA-256 when a validated image is available, provenance, thresholds, and detections. Real provenance includes the model, checkpoint selection and fingerprints, resolved device, resize setting, and backend package versions. Detection boxes are `[x_min, y_min, x_max, y_max]` in absolute original-image pixels and retain their model confidence. IDs follow the canonical mapping.

| Class | Canonical ID | AI Hub source category |
| --- | ---: | ---: |
| `cold_solder` | 1 | 23 |
| `misalignment` | 2 | 24 |
| `short` | 3 | 25 |
| `missing_component` | 4 | 27 |
| `tombstone` | 5 | 30 |

| Outcome | Meaning |
| --- | --- |
| `defect_detected` | Real inference retained at least one detection at the configured thresholds. |
| `review_required` | Real inference retained no detections (`no_detections`), or the run was simulated (`simulated_prediction`). Inspect the reason and provenance before acting. |
| `inference_failed` | The request, image, or prediction failed. The reason is a stable code and the detection list is empty. |
| `pass` | Reserved for independent verification of a good product. A valid record requires `verification_reference`. The service and CLI always require review for an empty real prediction. |

The CSV has one row per request, including failures and empty detections, with a fixed header and a `detections_json` cell containing the detection array. Identifier cells that start with a spreadsheet formula prefix have a leading apostrophe in CSV. Use the JSON for the complete structured record and exact sanitized identifiers. An exit status of 0 means all requests executed, including review and defect outcomes. Status 1 signals at least one failed request or an execution error. A finished batch with failed rows still publishes the report pair. Invalid CLI options exit with 2 and interruption exits with 130.

## Recovery and Python use

Batch attempts report their directory on standard error before processing images. If interrupted before publication, repeat the same batch command with `--resume-from <attempt-directory>`. Resume verifies the manifest, provenance, runtime and settings fingerprint, and the recovery records. It rechecks each image and reruns that line if the bytes changed or if the previous request failed. Reports are published together in the attempt's `reports/` directory after all lines are processed. Completed attempts are immutable through this command. `batch.json`, `events.jsonl`, and `recovery/` track the attempt.

Python callers can use `InspectionRequest`, `Detection`, `ModelProvenance`, and `InspectionResult` from `ict_pipeline.inspection.contracts`. Construct `InspectionService.mock(...)` or `InspectionService.from_run(..., trust_run=True)` once and call `service.inspect(request)` for each image. Pydantic's `model_json_schema()` exposes the version 1 structures. Image inspection runs locally and reads the supplied image and checkpoint files. Integration with factory verification, model calibration, and production acceptance remains separate from this interface.

Camera and sensor streams use the same inspection service and record format. Replay and device adapters send encoded frames with `InspectionMetadataInput` and add their alignment evidence under `result.metadata.streaming`. [Input streaming and alignment](input-streaming.md)

The standalone Python product model can carry existing detector predictions
alongside view-level risk scores. It records unavailable modalities and
uncalibrated score status. Factory acceptance still depends on the
inspection verification policy described above.
[Multimodal prediction contract](multimodal-architecture.md)
