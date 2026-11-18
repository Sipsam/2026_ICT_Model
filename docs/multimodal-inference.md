# Multimodal inspection and evaluation

## Integration audit and schema plan

The existing image inspection service verifies image bytes and reuses the detector evaluation adapters for original-image boxes. Its version 1 sensor field carries source metadata. It supplies neither a sensor window nor evidence of a pairing. Product inference already has a label-free `ProductInput`, branch availability masks, and a `MultimodalModel.predict` method. Its tensor-only checkpoint loader validates the architecture, preprocessing fingerprint, and parameter shapes. The product integration will reuse these interfaces and keep detection filtering and coordinate validation in the existing service.

The version 1 inspection result and REST envelope retain their existing fields. An optional, independently versioned `multimodal` extension carries product scores, branch availability, model identity, and alignment evidence. The legacy image endpoint and batch contracts keep their sensor metadata semantics. The product request contract accepts ordered sensor samples and uses a content-bound pairing receipt for image and sensor requests.

SQLite keeps the full validated record JSON alongside indexed detector fields and detections. Modality queries read the extension from JSON. The dashboard renders product evidence in its own section and uses the existing original-image overlay for detector evidence.

The offline dataset sensor window can extend from capture into a later decision interval. Capture-time inference accepts a zero-second window or an explicitly causal Train-fitted state. Previously serialized states retain their fingerprints.

## Run a local fixture

Save a product model specification as `product-fixture.json`. This deterministic fixture requires no checkpoint.

```json
{"kind":"mock"}
```

Save the following as `product-request.json`. The five values in each step follow the declared channel order. JSON `null` means unobserved. Numeric zero is observed. This request carries one captured temperature and vibration measurement.

```json
{
  "schema_version": 1,
  "request_id": "request-1",
  "source_id": "station-1",
  "sample_id": "part-1",
  "capture_timestamp": "2026-01-01T00:00:05Z",
  "sensor_evidence": {
    "source_id": "sensor-1",
    "sequence": {
      "channels": ["temperature", "humidity", "vibration", "acceleration", "noise"],
      "units": ["°C", "%", "m/s²", "g", "dB"],
      "steps": [
        {"timestamp": "2026-01-01T00:00:05Z", "values": [23.4, null, 0, null, null]}
      ]
    }
  }
}
```

```text
ict inspect --product-model product-fixture.json --product-request product-request.json
```

For image-only product scoring, omit `sensor_evidence` and pass the local JPEG or PNG as the positional image. For paired scoring, supply both and configure a pairing registry. The CLI publishes a fresh attempt directory. Use `inspection.json` for the product extension. The existing `inspection.csv` columns describe detector evidence and the inspection outcome. The legacy commands `ict inspect IMAGE --mock …`, `ict inspect IMAGE --run … --trust-run`, and image batch inspection use the existing contracts. [Inspection CLI reference](inspection-interface.md)

## Serve product inspections

Add a server-owned entry to the API settings `models` map. This example lists the product fixture alongside one existing image fixture.

```json
{
  "storage_backend": "sqlite",
  "models": {
    "product-fixture": {"mode": "multimodal", "product": {"kind": "mock"}},
    "mock-defect": {"mode": "mock", "scenario": "defect"}
  }
}
```

Send `POST /v1/inspections/multimodal` with multipart form data. The `details` field is the product request JSON plus `"model_id": "product-fixture"`. Use a text multipart field even for sensor-only requests. Include the `image` file part for image-only or paired inference. Success returns HTTP 201 and the usual stored inspection envelope and `Location`. The existing request size, upload timeout, access control, inference deadline, and persistence rules apply. For example, save the REST request with its model ID as `product-http.json`.

```sh
curl -F 'details=<product-http.json' http://127.0.0.1:8000/v1/inspections/multimodal
```

`/v1/models` identifies these entries with `input_kind: "multimodal"`. Clients select a configured model ID. Unknown IDs return 404 and invalid input or unverified pairs return 422. Details are schema checked and reject checkpoint paths, raw preprocessing windows, free-form sensor feature names, and caller-provided pairing evidence. The existing single and batch image endpoints continue to accept their original request formats. The camera and sensor replay runner still feeds the image inspection service with observational sensor metadata. [REST API reference](rest-api.md)

## Verify pairing and availability

The service matches a request against a server-owned `PairingRegistry`. Each version 1 `PairingReceipt` binds the sample ID, image and sensor source IDs, original image SHA-256, normalized capture timestamp, and fingerprint of the ordered measurements at or before capture. Identifiers use letters, digits, underscores, dots, and hyphens. Use opaque station and sample IDs.

Generate receipts in a trusted preparation step. For AI Hub records use `PairingReceipt.from_audited_record(record, image_source_id=..., sensor_source_id=...)`. It requires a verified, non-quarantined pair with image bytes and a capture-time sequence. For a controlled local fixture, build the receipt from an independently verified fixture relationship and the same content hashes. Serialize a version 1 registry with `PairingRegistry(receipts=(receipt, ...)).model_dump_json()`, place it alongside the model specification or under the server `model_root`, and set `"pairings": "pairings.json"` in the product model specification. Receipt verification binds the supplied bytes and readings to that trusted registry. Nearby timestamps alone cannot establish pairing.

Supply measurements available to the relevant capture. Timestamps must increase strictly and use a comparable time basis. Zoned timestamps are normalized to UTC for pairing. Local timestamps require a shared clock basis. Future-timestamped readings are excluded before preprocessing and receipt matching. Trusted acquisition must verify availability by capture when sources can deliver backdated readings. Raw readings, file paths and original filenames are absent from the result.

## Load a local checkpoint

Use a product specification with these fields. Replace the placeholder fingerprints with the actual values.

```json
{
  "kind": "checkpoint",
  "checkpoint": "product.pt",
  "checkpoint_sha256": "<64 lowercase hex digits for the checkpoint bytes>",
  "architecture_fingerprint": "<64 lowercase hex digits from MultimodalModel.architecture_fingerprint>",
  "preprocessing": "sensor-state.json",
  "device": "cpu"
}
```

CLI artifact paths are relative to the local product specification. API artifact paths are confined to the configured `model_root`. The loader snapshots the checkpoint, verifies SHA-256 and the pinned architecture fingerprint, and uses the existing tensor-only checkpoint contract to verify the saved architecture, exact tensor state, and preprocessing fingerprint. Use only trusted checkpoints and provenance when assigning a trained status. A hash identifies bytes. Training status comes from trusted branch metadata in those verified bytes.

The preprocessing file is the fingerprinted `PreprocessingState.payload()` artifact. Fit it on eligible Train data only. Use `PreprocessingConfig(window_policy="at-or-before-capture-v1", window_seconds=...)` to fit a causal lookback or the legacy zero-second configuration for capture-only data. The architecture must bind that exact frozen state. A positive capture-through-decision window can be used offline for its declared decision interval and is rejected for online sensor requests. The service derives masks from nulls and reuses the stored channel statistics. Train-median preprocessing preserves the mask on filled cells.

`InspectionService.from_product` loads one model and reuses it across requests. The existing device selection checks CPU, CUDA, ROCm, or MPS capability where the encoder permits it. RGB product scoring can run without a detector. Add `--run ... --trust-run` with a checkpoint product specification, or add a server-owned `detector` object using the existing trusted run settings, to include spatial evidence. A Faster R-CNN FPN product checkpoint requires a matching configured Faster R-CNN run and transform. Its product backbone state comes from the verified product checkpoint. Any returned boxes come from the separate detector predictor.

## Read the result

| Field | Meaning |
| --- | --- |
| `detections` | Detector class IDs, confidence, and original-image xyxy boxes. Empty for sensor-only and product-only requests. |
| `multimodal.input_mode` | Submitted evidence, one of `image`, `sensor` or `paired`. History uses this field for modality filtering. |
| `multimodal.branch` | Branch actually scored. A verified pair with an unusable window may fall back to `image`. A sensor-only request with no usable observations has a null branch and empty scores. |
| `multimodal.sensor_status` | `absent`, `no_observations`, `insufficient_steps`, or `available`. |
| `multimodal.scores` | Ordered product-level scores with a versioned label policy, threshold, and threshold-crossing flags. Sigmoid scores are uncalibrated. |
| `multimodal.weight_status` | `simulated`, `untrained`, `trusted_trained`, or `unavailable`. |
| `multimodal.model` | For checkpoints, format and SHA-256, architecture and preprocessing fingerprints, encoder and fusion versions, device, and seed. |
| `multimodal.alignment` | `verified_receipt` for a pair, with pairing fingerprint and capture time, or `not_applicable` for single-modality input. |

Successful product inspections require review, including when a score crosses the threshold or a companion detector returns boxes. A requested image that fails validation or a pair that fails verification produces a failed inspection. A frozen `reject` policy also rejects missing values within the selected sensor window. Explicitly omitting a modality selects the eligible branch. The database and dashboard retain these distinctions. Default production statistics exclude mock, untrained and unavailable product records. Detection counters continue to count retained boxes. [Dashboard](inspection-dashboard.md) · [Database](inspection-database.md)

## Freeze and compare Validation experiments

Use the Python interfaces in `ict_pipeline.evaluation.product_protocol` and `ict_pipeline.evaluation.product`.

1. Use `FrozenValidation.from_audit(cohort, sample_ids)` to select eligible, verified paired Validation samples before examining model results. Freeze the same IDs, dataset and split fingerprints, original-image hashes, causal sensor fingerprints, and annotations for each arm. Choose samples with enough causal sensor observations for every planned sensor encoder. Training and internal Test rows are ineligible.
2. Define a `ProductProtocol` with the frozen identity, target policy, image and sensor preprocessing fingerprints, and common threshold. Generate one image, sensor or paired inspection result per sample for each arm. `evaluate_products` verifies coverage, input evidence, branch, target policy and threshold, and requires matching frozen preprocessing and resolved fusion settings for checkpoint reports. It rejects fallback records for the requested comparison arm.
3. Use `DetectionProtocol` and `evaluate_detector_artifact` to score the same cohort from a native `validate_models` artifact and a trusted completed detector run. It verifies the native Validation images, COCO annotations, checkpoint, metric protocol, and cohort boxes before subsetting. `evaluate_detector` also accepts raw adapter predictions from a trusted caller that supplies the verified cohort identity. Use the existing `ict evaluate` and `ict compare` workflows for standalone detector benchmarks.
4. Use `compare_experiments` for reports with the same task, frozen cohort, code and scoring protocol. Product comparisons can vary the branch, checkpoint or fusion. Compare detector mAP in a separate group. `pending_experiment` records planned evaluations before checkpoints exist. `write_artifact` and `read_artifact` publish and verify deterministic fingerprinted JSON for both cohorts and reports. Use a fresh output path for each artifact.

Product metrics are binary presence metrics per output, using either any canonical target defect or the five canonical class-presence labels. For each output the confusion matrix uses true negative/positive rows and predicted negative/positive columns. Unknown labels are excluded for that output and counted. Precision, recall and F1 are null when their denominators are zero. AUROC uses the uncalibrated ranking scores, gives ties half credit, and is null unless both positive and negative labels are present. A checkpoint with no known labels yields `unscorable`. Simulated and untrained reports contain provenance and prediction fingerprints with no accuracy metrics. Detection mAP and the original detector confusion matrix use the unchanged COCO-box evaluator and require complete box annotations and at least one target object.

Future training should fit and tune on the designated Train and Validation splits respectively, record per-branch training status in checkpoints, and use the frozen common Validation protocol above. Use a separately approved final protocol for internal Test after model and thresholds are fixed. The current interfaces are Validation-only. Full-scale training, accuracy, calibration, GPU execution, real-data alignment coverage under the capture-time cutoff, and inference with real trained multimodal checkpoints remain unverified by this integration.
