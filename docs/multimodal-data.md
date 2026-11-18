# Multimodal data preparation

## Source and code audit

The workspace stores AI Hub 71900 in English-layout provider folders. A read-only
inventory counted 13,664 JPG/JSON/XLSX files per modality in Train and 1,708 per
modality in Validation. The sample relationship is represented by the shared
filename stem and sibling provider folders. The JSON `images[].file_name`
names the JPG and the single `sensor_data[]` group contains its sensor sequence.
The image and sensor numerical identifiers are retained independently.

| JSON field | XLSX field | Source unit |
| --- | --- | --- |
| `temperature` | `temperature(°C)` | °C |
| `humidity` | `humidity(%)` | % |
| `vibration` | `vibration(m/s²)` | m/s² |
| `acceleration` | `acceleration(g)` | g |
| `noise` | `noise(dB)` | dB |

The observed timestamp format is a local wall-clock string such as
`2025-08-07 09:03:40`. The corresponding filename embeds `20250807-090340`.
The JSON `defect_detected` field is target information and must stay outside
model inputs. The source gives no timezone or independently verified shared
clock for recovering relationships between different records.

The existing deep audit compares workbook and JSON timestamps and values.
The classification preparation stage also checks capture time and retains
alignment evidence, but requires complete measurements and prepares small
RGB thumbnails for binary classification. Its fitted classifier bundles its
own feature scaling. Detection loaders consume the canonical box labels.
Streaming matches bounded snapshots by explicit references and only allows
timestamp-only matching with a configured shared clock. Inspection accepts
individual measurement dictionaries. The detector evaluator consumes image
predictions against fixed cohorts.

The missing pieces were a nullable sequence contract, a reusable frozen sensor
transform, per-channel and padding masks, detection targets beside sensor
inputs, and an audit of eligible multimodal coverage that can run without
entering any model stage.

## Run the audit and prepare an index

```text
ict multimodal audit --root data/extracted --manifest data/processed/manifest.csv --path-base . --preview
ict multimodal audit --root data/extracted --manifest data/processed/manifest.csv --path-base . --out outputs/multimodal --export-index --checkpoint outputs/multimodal-scan.sqlite
```

Omit `--path-base .` for a manifest whose paths are relative to the extracted
root. The command prints JSON to stdout and sends scan progress, elapsed time,
and ETA to stderr. The first form writes no artifacts. The second publishes
`audit.json` and a checksummed `cohort.json` to a fresh directory. The index
stores verified sensor readings and source-relative paths. RGB images are
loaded lazily from the original files, with a SHA-256 check on access.
With `--checkpoint`, a retry verifies input bytes, file ambiguity, settings,
and implementation before using completed observations. Use a fresh
checkpoint for a different configuration or manifest.

For a published group-aware assignment, pass its `manifest.csv` and
`--revision <bundle>/split.json`. The audit checks the revision, label policy,
manifest, and source-byte fingerprints. Records carry both the original
provider split and the assigned split. An internal Test assignment can be
indexed under that contract. It stays outside statistic fitting and
augmentation.

### Alignment rules

The version 1 provider rule requires a single image record and an unambiguous
sibling workbook in the same provider split, folder group and sample stem.
The first reading must agree with the capture timestamp. The embedded JSON
and the XLSX must have equal lengths, strictly ordered comparable timestamps,
matching null locations, and values within an absolute tolerance of 0.001 in
source units. Explicit filename conflicts or repeated modality files block
attachment. A missing or invalid sensor association leaves an eligible
image-only record when its image and labels are valid. A verified sensor
sequence can also survive a missing image. Two local wall-clock readings from
different records cannot establish a relationship, even when their text is
identical.

The index records raw timestamps, separate image and linked sensor IDs,
paths, file hashes, split, source group, class-policy decision, method, and
issue codes. A null channel value is distinct from a recorded zero. Boolean,
string, non-finite, and out-of-range measurements are rejected. The magnitude
limit of 10³⁰ is a numeric guardrail. It carries no claim about calibration or
device operating limits. Raw measurements and timestamps must remain in
their source units and clock representation. Inspection can use the observed
values from one chosen raw step in its existing `SensorInput` dictionary.

The dataset views quarantine cross-assignment groups with repeated sample
stems, identical images or sensor windows, identical workbooks, shared
production or capture groups, shared source/capture timestamps, or shared
revision groups. Connected groups can use held and excluded rows as
evidence. Repeated stems are also ambiguous within one split. Quarantine is
local to these dataset views. Read the audit's raw pairing counts and its
post-quarantine availability together. The group-aware split workflow can
incorporate near-duplicate image and sensor evidence. Manufacturer groups
and date tokens are incomplete proxies for real session provenance.
[Dataset integrity and split revisions](dataset-integrity.md)

## Fit and reuse sensor preprocessing

```python
from pathlib import Path

from ict_pipeline.data.preprocessing.common import write_json
from ict_pipeline.multimodal.audit import Cohort
from ict_pipeline.multimodal.dataset import MultimodalDataset, collate
from ict_pipeline.multimodal.features import statistical_features
from ict_pipeline.multimodal.preprocessing import (
    PreprocessingConfig, PreprocessingState, fit_preprocessor, training_fingerprint,
)

root = Path("data/extracted")
cohort = Cohort.load(Path("outputs/multimodal/cohort.json"))
config = PreprocessingConfig(missing="mask", scaling="robust")
state_path = Path("outputs/sensor-state.json")  # fresh destination
write_json(state_path, fit_preprocessor(cohort, config).payload())
state = PreprocessingState.load(
    state_path, expected_config=config,
    expected_training_fingerprint=training_fingerprint(cohort),
)
validation = MultimodalDataset(
    cohort, root, "validation", mode="paired", missing="preserve", preprocessing=state,
)
batch = collate([validation[i] for i in range(min(2, len(validation)))])
features = [statistical_features(sample.sensor) if sample.sensor else None for sample in batch.samples]
```

The default window includes only readings timestamped at capture. The
`window_seconds` option includes readings through that many seconds after
capture. Set it according to the time the downstream decision can actually
wait for the sensor. The files record measurement times. Production
ingestion delay and clock synchronization still need verification. Fitting,
validation, internal Test, and `transform_sensor` at inference use the
same configured window and channel ordering.

Preprocessing pools only observed measurements from eligible, isolated Train
windows. It supports identity scaling, mean/population-standard-deviation
scaling, and median/interquartile-range scaling. Constant channels use a
unit scale. An unobserved Train channel has identity scaling and no learned
fill. Missing values can be masked, filled with the Train median when
available, or rejected. Filled values retain a false observation mask.
Optional per-step L2 normalization follows channel scaling and uses observed
measurements to compute the norm. Source identifiers and labels stay out of
statistics and input arrays. The state preserves version, channel units,
configuration, and Train-data fingerprint.

`statistical_features` produces 30 channel-major values ordered by mean,
population standard deviation, minimum, maximum, first-to-last difference,
and difference per second. It ignores masked and imputed values. Mean,
minimum and maximum require one observation. The other features require
two observations, and temporal features use actual elapsed time. Unavailable
features return `None` and a false feature mask. Pass a decision-window
`SensorWindow` for model inputs and a raw `SensorSequence` for exploratory
source-unit statistics.

## Dataset and batch contracts

| Interface | Contract |
| --- | --- |
| `MultimodalDataset` | Supports `mode="image"`, `"sensor"`, and `"paired"`. Use `missing="drop"` for complete requested inputs, `"preserve"` for records with at least one requested input, or `"error"` to require complete inputs. Selection follows deterministic cohort order within a single split. |
| `DatasetSample` | Carries an optional RGB image and sensor window, source image width/height, and original absolute XYWH boxes with canonical one-based COCO classes. Normal candidates have empty target tuples. Train image augmentation reuses the existing policy and exposes corresponding boxes under `image_targets`. Validation and internal Test use the original image and boxes. |
| `collate` | Keeps RGB images ragged and pads sensor sequences right with zeros to the longest window. It returns observation masks `[B,T,5]`, timestep validity `[B,T]`, unpadded lengths `[B]`, and image/sensor presence flags `[B,2]`. Zero values in padded or masked positions have false validity. A batch shares a split and preprocessing fingerprint. |
| `Batch.sensor_tensors()` | With an existing PyTorch installation, returns CPU float64 sensor values, Boolean masks, and int64 lengths. All-image batches have a zero-length sensor axis. The default data API uses Python floats, Boolean masks, and no tensor dependency. |

The cohort includes only verified AI Hub sensor windows for model-facing
views. The Python product models consume these batches and the frozen
preprocessing fingerprint. Live sensor integration and performance evaluation
require separate validation. [Encoders and fusion contracts](multimodal-architecture.md)

## Measured coverage on local AI Hub 71900

The provider audit read every manifest row and verified every embedded
sequence against its workbook. It found zero unmatched, ambiguous, or
timestamp-inferred pairs. Every window had five readings and each of the
61,488 adjacent intervals was one second. All 76,860 measurement timestamps
were local wall times. The verified channels contained zero null values and
zero recorded zeros.

| Provider split | Verified source pairs | Detection candidates | Quarantined candidates | Paired dataset samples |
| --- | ---: | ---: | ---: | ---: |
| Train | 13,664 | 13,062 | 11 | 13,051 |
| Validation | 1,708 | 1,656 | 11 | 1,645 |
| Total | 15,372 | 14,718 | 22 | 14,696 |

The 22 quarantined candidates belonged to 11 cross-split source/capture
groups. The provider manifest retained 645 held and 9 excluded rows. The
audit also found 136 held or excluded rows that could not supply canonical
detection targets, despite their verified sensor relationships. Image and
sensor IDs differed in 4,479 records. All four manufacturer tokens appeared
in both splits, and none of the 15,372 records had an explicit production
session ID. The provider split therefore leaves uncertainty about independent
production sessions and near-duplicate imagery.

The local `outputs/smt-multimodal-audit/audit.json` includes class and
manufacturer coverage, channel summaries, and an example Train window.
`outputs/smt-multimodal-datasets/cohort.json` contains the same checksummed
cohort, and `sensor-state.json` beside it contains pooled robust scaling
fit on one capture-time observation per channel for each of the 13,051
retained Train pairs. The index and state fingerprints were verified after
reload, and a two-sample Validation batch had lengths `[1, 1]`. These
results describe data preparation only.
