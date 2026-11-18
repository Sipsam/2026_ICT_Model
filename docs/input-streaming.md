# Camera and sensor input streaming

The inspection API can run one explicitly configured input stream per process. The default server and `ict inspect` continue to accept their existing inputs. The stream sends JPEG or PNG bytes and validated sensor context to `InspectionService`. The current detector uses pixels for inference. Source alignment is recorded as evidence and does not change the decision policy. Mock results and empty real predictions still require review.

## Local replay

Place a local JPEG or PNG named `part.png` in `fixtures/replay/images/`. Use a writable output root separate from the inputs. Save the following server configuration as `replay-api.json`.

```json
{
  "storage_backend": "sqlite",
  "storage_root": "outputs/replay-inspections",
  "input_stream": {
    "stream_id": "bench-replay",
    "model_id": "mock-defect",
    "camera": {
      "kind": "directory",
      "source_id": "replay-camera",
      "directory": "fixtures/replay/images",
      "manifest": "fixtures/replay/frames.json",
      "order": "name",
      "frames_per_second": 0
    },
    "sensor": {
      "kind": "csv",
      "source_id": "replay-sensor",
      "path": "fixtures/replay/readings.csv"
    },
    "alignment": {"allow_image_only": false, "max_skew_seconds": 1},
    "processing": {"queue_capacity": 8, "workers": 1, "max_retries": 2}
  }
}
```

Save this as `fixtures/replay/frames.json`. The keys are exact filenames. A frame without a manifest entry has an unavailable timestamp and sample identifiers. Each filename supplies a stable sequence ID through its SHA-256 hash. File modification times are ignored.

```json
{
  "part.png": {
    "sample_id": "part-01",
    "production_id": "item-01",
    "timestamp": "2026-10-30T10:00:00+09:00"
  }
}
```

Save this as `fixtures/replay/readings.csv`. The sensor sequence ID and camera sequence ID identify events in their own sources and can differ.

```csv
sequence_id,timestamp,sample_id,production_id,temperature,humidity,vibration,acceleration,noise
reading-23,2026-10-30T10:00:00.100+09:00,part-01,item-01,23.4,48.0,0,,35
```

With the existing API environment loaded, set the config path and start the server.

```powershell
$env:ICT_API_CONFIG = "replay-api.json"
python -m ict_pipeline.api
```

```sh
ICT_API_CONFIG=replay-api.json python -m ict_pipeline.api
```

Check the stream at `GET /v1/sources/status`. In the dashboard, check **Input sources**, select **Simulated inspections**, and apply the filters. History dates reflect inspection commit time. The recorded input timestamps remain in `result.metadata.streaming`. Run `ict history --mode mock list` with the same config for the CLI view.

Directory replay reads JPEG and PNG files in a single directory. The configurable orders are `name`, `reverse_name`, and `timestamp`. Timestamp order requires a timestamp on every frame and comparable timezone representations. Ties follow filename order. The default `frames_per_second: 0` runs at the speed allowed by the processing queue. A positive value from 0.001 to 240 inserts a cancellable delay between inputs. Pacing leaves acquisition timestamps intact. Replay indexes default to 10,000 frames and 10,000 observations and can be raised to 100,000 each. The manifest and sensor file each have a 16 MiB limit, and camera frames share the API's encoded size and decoded dimension limits.

## Sensor formats and units

JSON replay uses an array of observation objects. Change the sensor `kind` to `json` and use a path with this format.

```json
[
  {
    "sequence_id": "reading-23",
    "timestamp": "2026-10-30T10:00:00.100+09:00",
    "sample_id": "part-01",
    "production_id": "item-01",
    "values": {"temperature": 23.4, "vibration": 0, "acceleration": null},
    "units": {"temperature": "°C", "vibration": "m/s²", "acceleration": "g"}
  }
]
```

Both formats preserve the original ISO timestamp. CSV accepts either bare canonical measurement names or the exact SMT XLSX headers. Missing columns, JSON nulls, and blank measurement cells mean unavailable. Zero is a measured value. JSON `units` defaults to the SMT units below. If provided, supply the canonical unit for each measurement key. The adapter rejects unknown fields, invalid values, mismatched units, and duplicate JSON keys or CSV columns.

| Measurement | Unit |
| --- | --- |
| `temperature` | °C |
| `humidity` | % |
| `vibration` | m/s² |
| `acceleration` | g |
| `noise` | dB |

These are replay interchange formats. For SMT exports, verify the image-to-sensor filename relationship, capture time, and available JSON/XLSX timestamp and value agreement first. Use a trusted per-item reference in both exports. SMT image and sensor numeric IDs may differ. Export the intended observation for each frame and omit the target `defect_detected` and any fields derived from it. The stream reuses the SMT timestamp parser and measurement contracts.

The offline multimodal audit verifies AI Hub records before exposing canonical
sequences. A chosen raw step can populate `SensorObservation.values` using the
same five channel names, units, and nullable values. A decision-window tensor
uses separately frozen Train preprocessing. [Offline sequence and dataset contracts](multimodal-data.md)

## Alignment and missing input

- The matcher checks the shared `sample_id` first. If it has no match, a shared `production_id` can identify the same item when it is unique and the known identifiers agree. Use a per-item production identifier. Lot IDs and session IDs describe groups and are unsuitable for an item join.
- With comparable explicit timestamps present on both sides, the absolute skew must be at most `max_skew_seconds` (default 1, inclusive). Timezone offsets are normalized for comparison and original strings are retained. Mixing a zoned timestamp and a local wall timestamp makes the pair unavailable. Local wall times in explicitly linked records must come from the same time basis.
- Timestamp-only matching is off by default. To enable it for replay, set `alignment.shared_clock` to a public clock ID after verifying both capture sources use the same time basis. Eligible records have timezone-aware timestamps and no sample or production ID. The time window must contain exactly one candidate in each direction. The adapter does not infer a link from filename order, arrival time, file modification time, or nearest timestamp across ambiguous candidates.
- Duplicate observation sequence IDs and ambiguous item references block pairing. An observation can be consumed once. Readings outside the inclusive window, including late event timestamps, are unavailable to that frame. A sensor snapshot is sealed before frames are processed. Later arrivals require another snapshot and cannot change committed records. Status reports unused observations and alignment reason counts.
- If the snapshot is malformed or unavailable, the whole snapshot is rejected. With `allow_image_only: true`, unmatched frames can proceed with `sensor_status: unavailable` and a reason. With the default `false`, unmatched frames are skipped, and an unreadable sensor snapshot stops that stream. A matched observation with only missing values also follows this rule.

The record contains source kinds, public identifiers, original timestamps, alignment method, categorical confidence, signed skew when known, and missing channel names. Use non-sensitive source, sample, and production identifiers. Images and raw sensor values are held only for processing. The stores retain the existing image descriptor and sanitized metadata.

## USB, RTSP, and sensor adapters

Camera access requires a configured `input_stream` with an explicitly enabled device. Replace the directory `camera` object with one of the following. Set `alignment.allow_image_only` to `true` for a camera with no verified sensor association.

```json
{"kind": "usb", "source_id": "bench-usb", "enabled": true, "device_index": 0}
```

```json
{"kind": "rtsp", "source_id": "bench-rtsp", "enabled": true, "url_env": "ICT_RTSP_BENCH"}
```

Supply an RTSP or RTSPS URL through the named environment variable. Keep it out of JSON and logs. The adapter uses an already available `cv2` and uses OpenCV's FFmpeg backend for RTSP. Actual USB and network support depend on the OS, driver, and OpenCV build. The capture helper process uses bounded open/read waits and can be terminated when a native call stalls. The default timeout is 5 seconds. A timeout or disconnect gets up to two reconnect attempts per stream run, with a one-second default delay. A successful reconnect starts a new sequence namespace. Live timestamps are host receipt times. The built-in live adapters do not have production or sample IDs and are intended for image-only inspection. [OpenCV video I/O timeout properties and backend limits](https://docs.opencv.org/4.12.0/d4/d15/group__videoio__flags__base.html)

Python adapters implement `ImageSource` or `SensorSource` from `ict_pipeline.streaming.contracts`. Produce `CameraFrame` or `SensorObservation` events and honor the supplied stop event with bounded I/O. Raise `SourceIssue` with a stable code for a failed source. A camera can also yield a recoverable issue and continue. A custom sensor adapter provides a finite snapshot and must verify its own source protocol, timestamps, and item identifiers. A custom indexed camera can supply the complete reference index through `IndexedImageSource` for timestamp matching. Connect adapters to `StreamRunner` and `FrameProcessor` for Python integration. The application ships no configured factory sensor protocol or automatic device discovery.

## Queue, recovery, and status

The default queue holds eight pending frames and has one worker. Configure `queue_capacity` from 1 to 128 and `workers` from 1 to 8. Memory also includes one frame held by the producer, active workers, decoded images, and source indexes. When the queue fills, the producer waits and stops requesting frames. A live camera or its driver may drop frames while reads pause. Concurrent workers can overlap decoding and storage. The shared inspection backend serializes predictor calls, and multiple workers can finish out of input order.

The default processing retry policy allows two retries, with a 0.1-second delay between attempts. SQLite lock contention and explicitly transient predictor errors can retry. Corrupt frames, invalid measurements, and terminal predictor failures are counted without retrying. Successful records use stable inspection UUIDs derived from full input and inspection configuration fingerprints. File and SQLite stores publish them with insert-only writes. Replaying the same unchanged source and model reuses each committed result, including retained records hidden by read-time retention. Explicitly purging a record removes that deduplication evidence. Input bytes, sensor context, alignment policy, or model configuration changes can produce a new record. Keep replay inputs unchanged for a consistent restart.

Shutdown stops admission, drops queued work, and waits up to `shutdown_seconds` (default 3) for stream threads. Work already inside a native predictor can run past the wait. An insert that has already begun may commit. Start the same replay again to resume from what was committed. Live acquisition is best-effort, with no durable spool for uncommitted frames. Source health and counters describe the current run, while successful inspections are durable.

`GET /v1/sources/status` returns a version 1 snapshot with stream state, source kinds and health, error codes and event counts, elapsed time, input and new-inspection rates, queue depth/capacity, active workers, mean/max latency, frame progress, and outcome counters. A running directory replay can estimate remaining time from the rate of completed frames. Counters appear once observed. Latency measures receipt through processing and includes queue wait and retry delays. The endpoint excludes filesystem paths, URLs, credentials, raw sample identifiers, and raw payloads. It uses the existing API authentication policy. The dashboard fetches it only when an operator checks source status.

Automated fixtures exercise replay, mock inference, timeouts, reconnect control, queue pressure, and persistence without camera or sensor hardware. Physical USB/RTSP capture and actual factory sensor integration require hardware validation with the target environment.
