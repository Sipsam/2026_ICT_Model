# ICT Pipeline

The ICT Pipeline CLI is a Python entry point for the AI Hub SMT manufacturing defect detection project.

## Install

Use Git and Python 3.11 or newer.

```text
git clone https://github.com/Sipsam/2026_ICT_Model.git
cd 2026_ICT_Model
```

Create a virtual environment and activate it in your shell.

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux, WSL2, or macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install the CLI in editable mode and check the entry point.

```text
python -m pip install -e .
ict --help
ict --version
```

The CLI runs without PyTorch, GPU libraries, or dataset files.

## First local run

Use an already extracted SMT tree for an offline run. Install the preprocessing extra, run diagnostics, and point setup at the source tree. See the [data access guide](docs/data-access.md) if you need the archives or dataset access.

```text
python -m pip install -e ".[preprocess]"
ict doctor
ict setup --non-interactive --dataset-source skip --data-path data/extracted
ict run --dry-run
ict run --run-id smt
ict status --run-id smt
ict resume --run-id smt
```

Use the path printed on the `Detection dataset` line for training and evaluation. With a trusted completed model run for that dataset, pass its directory containing `run.json` as `--run`. If you need a checkpoint, follow [Train detection models](#train-detection-models) and use the path printed on the `Model run` line. Replace the quoted example paths with the printed values.

```text
ict evaluate --dataset "path/to/detection" --run "path/to/model-run" --device cpu
ict report
```

The benchmark uses Validation by default. The report prints the path to its HTML file. See [troubleshooting](docs/troubleshooting.md) and the [manual platform checklist](docs/platforms.md).

## Available commands

`ict --help` lists the registered commands, and `ict --version` prints the installed package version. Both exit with status `0`.

`clean` is a placeholder and exits with status `1`.

Use `ict runtime` to run preprocessing, training, or Validation evaluation on the local machine, in Docker, or on a Linux SSH server. The [runtime guide](docs/runtime.md) covers setup, job recovery, device checks, and platform limitations.

## Acquire and verify dataset 71900

Obtain dataset approval and keep the API key in an environment variable or local `.env`. See [data access](docs/data-access.md) for approval, offline reuse, and source layout.

```text
ict data status
ict data verify
ict data download
ict data download --approve-download
ict data extract
```

The packaged [SMT dataset selection](src/ict_pipeline/data/aihub_smt.yaml) records the 60 file keys from [the original download script](scripts/download_data.sh). It covers Training and Validation for normal and defect cold solder, misalignment, short, missing component, and tombstone groups, with JPG images, JSON labels, and XLSX sensors. Use `ict data --raw-dir <archive-root> --data-path <dataset-root> <command>` to override the default archive and dataset directories. The dataset root can also come from setup.

`ict data download` checks local data and previews any missing split transfers. The approval flag authorizes the listed transfers. The provider checks free space, write permissions, API reachability, and the local `AIHUB_API_KEY` supplied through `.env` or the environment. AI Hub checks the key and dataset access on the authorized request. Linux, WSL, and macOS use curl. On Windows the provider discovers an installed WSL distribution with curl and streams files into the Windows archive root. Set `ICT_WSL_DISTRIBUTION` to select a distribution explicitly. Existing English-layout data and local ZIPs work independently of WSL, curl, and AI Hub credentials. The transport uses the v0.6 request format verified in [AI Hub's aihubshell workflow](https://www.aihub.or.kr/devsport/apishell/list.do).

Validated bundles are kept under `data/raw/.ict-aihub/` by selection and split. Interrupted transfers remain as `.partial` files. A retry starts a fresh transfer for the affected split and reuses any verified completed split. The downloader checks bundle integrity and ZIP CRCs before publishing a checkpoint. The file keys and size caps are based on the repository's historical XLSX selection. If AI Hub changes the selection or format, ingestion reports a failed validation.

`ict data extract` publishes a new English-layout dataset directory with `.ict-ingestion.json`, which records the selection fingerprint, selected file-tree hash, and counts per split. Source archives are retained. `ict data verify` checks selected folder coverage, matched sample stems, basic file signatures, and the completion hash when present. The `audit` stage checks annotation and sensor content.

## Plan, inspect, and resume a pipeline run

```text
ict run --dry-run
ict run --dry-run --until validate_detection
ict status
ict resume --run-id <id>
```

By default, `ict run` completes preprocessing through `validate_detection`. With `dataset_source: skip`, it starts at `audit` and reads the configured extracted tree in place, defaulting to `data/extracted`. With `dataset_source: aihub` or `dataset_source: existing` it starts at `download`. Ingestion publishes `archives/` and `dataset/` under the corresponding attempt directories. If an extracted dataset is selected for ingestion, the download stage records reuse and the extract stage copies the selected files into the run artifact.

`ict run` and `ict resume` work with closed stdin. For missing transfers, review `ict data download` and pass `--approve-download` to authorize acquisition for that invocation. Pass the same flag to `ict resume` if transfers are still missing. Pass `--run-id <id>` to choose a name or reuse that run's cache. `--from` and `--until` select an inclusive range in dependency order. Repeat `--only` to select individual stages. Any omitted prerequisite must have a valid result in the same run. `--force` recomputes the selection in new artifact directories.

`ict status` shows the latest recorded attempt per stage for a pipeline run. Use `--run-id <id>` to select it when training or evaluation has a newer record. `ict resume` defaults to the latest pipeline run and rechecks settings, inputs, stage versions, and artifact bytes before reuse. If you ran with `--config`, pass the same file to `resume` to use that profile again.

Each run stores an atomic snapshot in `runs/<id>/state.json` and a structured event log in `events.jsonl`. A stage writes to `work/<stage>/<attempt>/`. After output verification, the runner publishes the directory under `artifacts/<stage>/<attempt>/` and commits the successful attempt to state. Failed and abandoned attempts remain available for inspection. Only an attempt recorded as completed and verified against its fingerprints is eligible for cache reuse. A lock allows one writer per run, and the OS releases it when the process exits. Cancellation returns status `130`. Progress includes elapsed time and an ETA when the stage reports measurable work.

The Python adapter interface is `Stage`, `StageInput`, `StageContext`, and `StageResult` in `ict_pipeline.pipeline`. An adapter reads its declared inputs, writes publishable artifacts to `StageContext.output_dir`, and reports progress as completed and total units. `StageContext.artifact_dir` is the future published path for files that must refer to it. `StageContext.input_fingerprints` provides the verified content hash for each declared input. The optional `StageContext.recovery_dir` provides run-local storage for verified checkpoint recovery across attempts. Adapters own data-format validation and must keep secrets out of outputs and console messages. Bump `Stage.version` when changes to an adapter affect the validity of its results.

## Run end-to-end detection

Use an already extracted fixture and model-specific YAML profiles in `configs/models`. For a bounded CPU smoke run, set `epochs: 1`, `batch_size: 2`, `image_size: 64`, and `device: cpu` in each selected profile. Include at least one labeled object in each split and use distinct images across splits. Packaged profiles use 50 epochs.

```text
ict setup --non-interactive --dataset-source skip --data-path path/to/extracted-fixture
ict run --run-id detector-fixture --model faster_rcnn --config-dir configs/models --evaluation-image-size 64 --dry-run
ict run --run-id detector-fixture --model faster_rcnn --config-dir configs/models --evaluation-image-size 64 --approve-detection-train
ict status --run-id detector-fixture
ict resume --run-id detector-fixture
```

Use `--model yolo`, `--model faster_rcnn`, `--model rt_detr`, or `--all` to run through `report`. With `--until report` and no model flag, the run uses the setup model. For AI Hub or existing-archive profiles, the graph starts at `download` and `extract`. With `dataset_source: skip`, it starts at `audit`. If a run already completed preprocessing, append `--from train` to the detector command and use that run ID to reuse verified prerequisites. The selected YAML controls each training device. `--evaluation-device` and `--evaluation-image-size` set the shared Validation inference options and default to CPU and 640.

To run individual stages against a saved detector run, repeat the original model, profile directory, and evaluation options with `ict run --only <stage>`. Use `ict resume` to repeat the saved selection and options.

`--approve-detection-train` authorizes the selected detector profiles for that invocation. Use the dry run to review the models, resolved settings, YAML hashes, and output root. A resume restores the saved model selection, profile directory, and evaluation settings. It reads the current YAMLs again. Pass `--approve-detection-train` again when a model still needs to train or a profile has changed. Completed stages and verified completed models can be reused without training approval. Classification training and missing dataset transfers have their own approval flags. `--force` with `train` in the selection starts a fresh training generation and requires approval.

| Stage | Published artifacts |
|---|---|
| `validate_detection` | `receipt.json` binds the successful YOLO/COCO label equivalence check to the conversion, manifest, and label policy. |
| `train` | `checkpoints/<model>/run.json`, best and last checkpoints, and fingerprints for the verified dataset, both split label formats, model YAML, code, and runtime. |
| `validate_models` | `predictions/<model>.json` with a schema version, content fingerprint, sample IDs, original-image boxes, classes, scores, preprocessing, timing, and checkpoint and dataset provenance. Validation includes background images. |
| `evaluate` | `metrics.json` and `model_reports/*.json` with the scored Validation metrics and prediction fingerprints. Pass the individual JSON reports to `ict compare`. |
| `report` | `report.html` and `report.json` for the selected results. The CLI prints the HTML path. |

Training resumes from each model's latest verified epoch checkpoint. The runner stores the underlying attempts under `runs/<id>/recovery/train/`. By default this includes the private YOLO dataset copy. Set `ICT_TRAIN_CACHE_ROOT` on the machine doing the training to put the private copy and its caches on another volume. A model without a verified recovery checkpoint needs a fresh approved attempt. Validation inference restarts its stage if interrupted. Published outputs use new attempt directories when a stage needs to be rebuilt.

## Train detection models

Use `ict recommend --all` on the training host to inspect model readiness and starting batch sizes, precision, workers, and resource estimates. Use `ict pilot prepare` to derive a bounded cohort from a sealed split bundle, then preview training with `ict pilot run`. Adding `--approve-train` authorizes only the displayed pilot. Pilots keep their own checkpoint and Validation reports under the cohort's `attempts/` directory. [Hardware recommendations and pilot workflow](docs/gpu-training-pilot.md)

```text
ict train --model faster_rcnn --dry-run
ict train --model faster_rcnn --config-dir configs/models --approve-train
ict train --all --dataset <conversion-attempt>/detection --config-dir configs/models --dry-run
```

Choose `yolo`, `faster_rcnn`, `rt_detr`, or `--all` for sequential training. Training starts only with `--approve-train`. Profiles default to `initialization: scratch`. To fine-tune, set `initialization: pretrained` and pin a trusted local checkpoint by SHA-256. The same profiles work with `ict train`, `ict run`, and `ict runtime`. Dry runs show the architecture, initialization, checkpoint path, selected device and precision, and resource inputs. [Pretrained profiles, sources, and licensing](docs/models.md)

The default input is the standalone converter's `data/processed/detection`. Pass `--dataset` for a pipeline conversion attempt. The command preserves the generated Train and Validation splits, uses the canonical five classes, and checks that both formats have the same split membership and image bytes with `--all`. Keep the conversion artifacts stable during training.

Install the extra for the selected backend, for example `python -m pip install -e ".[train-faster-rcnn]"`. Select a compatible PyTorch and TorchVision build for your OS and accelerator. The project requires Python 3.11 or newer and the training extras require PyTorch 2.6+ and TorchVision 0.21+. Ultralytics 8.3.228 declares Python 3.8+ and Transformers 4.57.6 declares Python 3.9+. For accelerator availability, use `ict doctor`. ROCm uses PyTorch's `cuda` device namespace. [PyTorch and TorchVision compatibility](https://github.com/pytorch/vision#installation), [Ultralytics package metadata](https://github.com/ultralytics/ultralytics/blob/v8.3.228/pyproject.toml), [Transformers package metadata](https://github.com/huggingface/transformers/blob/v4.57.6/setup.py).

| Model | Extra and implementation | Library license | Devices and precision | Checkpoint |
|---|---|---|---|---|
| YOLO | `train-yolo`, Ultralytics 8.3.228 YOLO11n from YAML | [AGPL-3.0 / Ultralytics Enterprise](https://www.ultralytics.com/license) | CPU, CUDA, ROCm, MPS. FP32 or accelerator FP16 on CUDA/ROCm. | Native Ultralytics `.pt` with training state, selected by validation fitness. |
| Faster R-CNN | `train-faster-rcnn`, TorchVision MobileNetV3-Large FPN | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | CPU, CUDA, ROCm. FP32 or CUDA/ROCm FP16 and native BF16. | PyTorch state dictionary `.pt`, selected by minimum validation loss. |
| RT-DETR | `train-rt-detr`, Transformers 4.57.6 RT-DETR with a local ResNet-18 config | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.57.6/LICENSE) | CPU, CUDA, ROCm. FP32 or CUDA/ROCm FP16 and native BF16. | PyTorch state dictionary `.pt`, selected by minimum validation loss. |

The adapters construct models offline. Scratch profiles initialize locally, while pretrained profiles transfer compatible tensors and initialize the class-dependent heads for SMT. See the model guide for the architecture and freezing rules. [Model guide](docs/models.md)

For a one-epoch check with official weights and real AI Hub images, use the sealed 36-image cohort and opt-in tests in the [real-data smoke guide](docs/real-data-smoke-test.md). It records checkpoint recovery and shared Validation results for all three detectors.

Copy the packaged [YOLO](src/ict_pipeline/models/yolo.yaml), [Faster R-CNN](src/ict_pipeline/models/faster_rcnn.yaml), or [RT-DETR](src/ict_pipeline/models/rt_detr.yaml) YAML to the directory specified by `--config-dir`. Name the files `yolo.yaml`, `faster_rcnn.yaml`, and `rt_detr.yaml` and supply one for each selected model. Each accepts `epochs`, `batch_size`, `image_size` (multiple of 32, at least 64), `optimizer` (`SGD`, `Adam`, or `AdamW`), `lr`, `seed`, `device`, `precision`, and `save_period` (0 disables periodic snapshots). The default device is CPU and `device: auto` checks for an available accelerator. YOLO uses private copies of the converted images and labels for its caches and JPEG checks. Both COCO adapters read the original converted files directly.

`runs/training/<id>/<model>/run.json` records status, config, initialization, library and code versions, split and annotation fingerprints, the label policy, and relative paths and fingerprints for `best.pt` and `last.pt`. Pretrained runs also record source provenance, the verified weight hash, and the transfer diagnostics. A new run gets a new ID. To resume an unfinished model, pass `--resume-from <model-run-directory>`, the original dataset and config, and `--approve-train`. Keep the pinned pretrained file available for verification. Resume starts after the last saved epoch in a new directory. COCO checkpoints restore the model, optimizer and scaler and reconstruct the epoch seed. YOLO uses native training-state restoration from its EMA checkpoint and may produce a different trajectory from an uninterrupted run. Resume only from your own trusted runs because native Ultralytics checkpoints contain pickled objects.

Use `ict train --dry-run` with the model and dataset options to see measured free space and estimated write requirements. Use `--runs-dir` to choose a checkpoint volume. Training checks available space before execution, during progress and before checkpoint commits. Treat checkpoint files as read-only. Byte-identical snapshots within a run can share filesystem storage. [Storage estimates, cache paths, and recovery](docs/artifact-management.md)

The Python entry points are `ict_pipeline.models.contracts.ModelAdapter`, `TrainingRequest`, and `Checkpoints`, and `ict_pipeline.models.registry.REGISTRY`. The stage runner uses the same training implementation.

Use `ict training-data imbalance` for class support by split, manufacturer, and group. Use `ict training-data preview` to inspect Train-only SMT transforms and `ict training-data ablation --dry-run` to validate a pending pretrained comparison. The same opt-in YAML settings work with `ict train` and `ict run`. [Training optimization and experiment planning](docs/training-optimization.md)

## Evaluate and compare detectors

```text
ict evaluate --run runs/training/<id>/yolo --run runs/training/<id>/faster_rcnn --run runs/training/<id>/rt_detr
ict compare outputs/evaluation/<attempt>/01-yolo.json outputs/evaluation/<attempt>/02-faster_rcnn.json outputs/evaluation/<attempt>/03-rt_detr.json
```

Pass one or more trusted, completed ICT model run directories and use the same backend extras as training. Ultralytics checkpoints can contain pickled objects. Evaluation verifies each checkpoint against its training record and defaults to `--checkpoint best`. Use `--checkpoint last` to evaluate the final saved weights. Every model infers on all images listed in the same COCO annotation file, including normal images that have zero target boxes. COCO files must use the canonical five classes, original-image pixel coordinates, and non-crowd boxes. Evaluation requires at least one ground-truth object in the selected split.

The default `--split val` reads `coco/annotations/instances_val.json` and `coco/images/val/` under `--dataset`. Image names and bytes must match the Validation split recorded by every model run. Checkpoints from the integrated DAG also require the exact converted dataset revision recorded at training. Reports identify Validation as `validation_for_tuning` and record the evaluation annotation fingerprint. For a separately prepared Test set, put its labels and images in the corresponding `instances_test.json` and `coco/images/test/` paths. Use `--split test --held-out` only after independently holding out Test and freezing model and protocol choices using Train and Validation. Provide the original detection directory with `--reference-dataset` if Test lives in a different directory. The command verifies the reference image fingerprints against every model's recorded Train and Validation and rejects shared filenames or byte-identical images. The holdout attestation covers production groups, near duplicates and prior tuning history, which these file checks cannot establish.

The evaluator uses COCO bbox AP across all sizes, with 101 interpolated recall points, IoU thresholds from 0.50 to 0.95 in steps of 0.05, and at most 100 predictions per image and class. AP uses scores at or above `--score-floor` (default 0.001). Classes absent from ground truth have null AP and are excluded from the mAP average. Precision, recall and F1 use class-aware, score-ordered matching at `--confidence` (default 0.25) and `--match-iou` (default 0.5), with per-class counts and micro-averaged totals. The confusion matrix matches by IoU regardless of predicted class at that same operating point. Its rows are ground truth and its columns are predictions. The background row counts unmatched predictions and the background column counts missed objects. [COCO reference evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

| Benchmark setting | Behavior |
|---|---|
| Image sizing | `--image-size 640` by default. YOLO letterboxes with bilinear resize and fill 114. Faster R-CNN preserves aspect ratio within the size bound and pads to its tensor stride. RT-DETR stretches with bilinear resize. Reported boxes are mapped back to original-image pixels. |
| Postprocessing | YOLO selects the highest-scoring class at each location and uses class-aware NMS. Faster R-CNN uses its proposal NMS and class-aware ROI NMS, with up to 1,000 retained ROI detections before the common COCO cap. The default ROI/YOLO `--nms-iou` is 0.7. RT-DETR ranks query/class scores and keeps the checkpoint architecture's query count without NMS. |
| Warmup and timing | `--warmup 3` by default. Warmup uses the first samples in ID order and cycles through the split if needed, then every image is measured once at batch size 1. Timing includes image read/decode, resizing, host/device copies, inference, postprocessing and normalization. Model load, warmup, label matching and report I/O are excluded. FPS is timed images divided by summed latency. Latency is reported as mean, median and nearest-rank p95 in milliseconds. |
| Synchronization and precision | CPU is synchronous. CUDA/ROCm and MPS synchronize around each timed image. Default is `--device cpu --precision fp32`. CUDA/ROCm FP16 uses half-precision weights and input for YOLO and autocast for the other adapters. The report records TF32 settings for CUDA. |
| Hardware | The report records the available CPU and accelerator identities, device backend, OS, PyTorch thread count, driver and library versions. Cache state, thermals and host load can affect runtimes. `ict compare` flags whether reported hardware and shared runtime settings match. |

Evaluation writes `outputs/evaluation/<attempt>/evaluation.json`, `events.jsonl`, and an atomic schema-versioned JSON report per completed model. Report paths go to stdout and progress to stderr. Reports include the checkpoint fingerprint and file size in bytes, image and annotation fingerprints, the split role, class mapping, evaluation settings, and code fingerprints. Checkpoint size includes training state if the saved file contains it. Resume with `--resume-from <attempt-dir>` and the original arguments in the same order. Completed reports are verified and reused. An interrupted model starts its inference from the beginning. When given explicit paths, `ict compare` requires reports from the same labeled split, code and evaluation settings.

## Inspect production images

`ict inspect` writes versioned JSON and CSV for one image or a JSON Lines batch. Mock mode works on CPU with Pillow and marks every record as simulated. Real mode uses a trusted, completed YOLO, Faster R-CNN, or RT-DETR run.

```text
ict inspect part.png --mock defect --output-dir outputs/inspection
ict inspect --batch requests.jsonl --run outputs/pilot/training/faster_rcnn --trust-run --device cpu
```

The command prints the new report directory. An empty real prediction produces `review_required`. The `pass` outcome requires a separate verification record. [Inspection contracts, batch recovery, and examples](docs/inspection-interface.md)

## Serve inspections over HTTP

Install the optional API extra and start the local FastAPI server.

```text
python -m pip install -e ".[api]"
python -m ict_pipeline.api
```

The server defaults to `http://127.0.0.1:8000` with two explicitly simulated models and requires no detector weights for mock mode. Submit JPEG or PNG images singly or in batches and retrieve stored results by ID. Use the [REST API guide](docs/rest-api.md) for curl and Python examples, OpenAPI, trusted model configuration, and deployment limits.

Open [the local dashboard](http://127.0.0.1:8000/dashboard) to view inspection counts and trends, filter history, export a page, and review detection boxes on available originals. The dashboard starts with real inspections. Select **Simulated inspections** and apply the filters to review mock submissions. See the [inspection dashboard guide](docs/inspection-dashboard.md) for image setup, filters, and access controls.

The API also lists and filters inspection history, exports JSON or CSV, and reports UTC daily counts and defect frequencies. Statistics count real inspections by default. The file repository remains the default. Set `"storage_backend": "sqlite"` in the JSON file selected by `ICT_API_CONFIG` to use indexed SQLite history. The API and CLI share this configuration.

```text
ict history --mode real list --limit 25
ict history --mode real export outputs/line-history --format csv
ict history import-files outputs/legacy-inspections --dry-run
```

The history commands work without starting inference. See the [inspection database guide](docs/inspection-database.md) for storage setup, migration, backup, restore, and retention.

## Replay camera and sensor inputs

Set `input_stream` in the JSON file selected by `ICT_API_CONFIG` to run an image-directory replay with an optional CSV or JSON sensor snapshot. Select a mock model to test ordering, sample alignment, backpressure, and persistence on CPU. The stream writes to the same inspection history as HTTP uploads, and rerunning the same replay skips committed records. Use **Input sources → Check source status** in the dashboard or query `GET /v1/sources/status`. Streaming defaults to disabled. USB and RTSP require explicit device settings and an existing OpenCV installation. [Replay example, alignment rules, and device limits](docs/input-streaming.md)

## Run the synthetic inspection demonstration

Run these commands from the repository root with the API and HTTPX test extras available.

```text
python -m scripts.inspection_e2e
python -m scripts.inspection_e2e --image-only
```

Each command creates synthetic images, verifies CLI export, API upload, replay, SQLite history and restoration, and writes `integration-report.json` in a fresh output directory. The generated `api.json` opens the same simulated records in the dashboard. [Inspection demonstration, browser steps, and verification results](docs/inspection-e2e.md)

## Inspect runs and export local reports

```text
ict runs list
ict runs show training/<id>
ict status --run-id evaluation/<id>
ict compare
ict report
ict report --format json
ict report --format csv
```

The catalog uses the pipeline ID as saved and prefixes standalone run IDs with `training/` or `evaluation/`. Use `--training-root` or `--evaluation-root` to read custom roots supplied to training or evaluation. New standalone attempts record lifecycle and host metadata in `attempt.json` beside the original model records and logs. Pipeline runs save that metadata in `state.json`. Older records show the metadata available in their saved artifacts. An abandoned run can appear as interrupted with no recorded end time.

`ict compare` prints the same versioned document as the JSON export, with run history and completed evaluations grouped by matching split, data fingerprints, code and protocol. Each group flags whether the reported speed environments match. Failed, interrupted and incomplete runs stay visible in the history with any saved results. Training metrics remain available in the model records. CSV has one row per model or one row for a run without model records. It includes common evaluation metrics and JSON columns for detailed metadata. `ict report` writes HTML by default and prints the saved path. Every export gets a fresh directory under `outputs/reports` or the supplied `--output-dir`.

Local recording and reports work without MLflow. To also publish the JSON snapshot as an artifact, use the optional `ict-pipeline[mlflow]` extra, set `MLFLOW_TRACKING_URI` in the environment or `.env`, and run `ict report --mlflow --experiment-id <id>`. The flag sends the report, including run metadata and artifact paths, to the configured MLflow service. Each export creates a new MLflow run. [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)

Python consumers can use `ict_pipeline.evaluation.contracts.Prediction` and `Predictor` for normalized, one-based detections with preprocessing metadata, and `ict_pipeline.evaluation.metrics.detection_metrics` for scoring. The report reader is `ict_pipeline.evaluation.runner.read_report`. The standalone `ict report` reads saved preprocessing, training, and evaluation records.

## Preprocess and validate detections

Install the CPU preprocessing extra for image decoding and XLSX checks.

```text
python -m pip install -e ".[preprocess]"
ict run --until validate_detection --run-id smt
ict resume --run-id smt
```

The command uses the configured ingestion source. To audit an extracted dataset in place, use a profile with `dataset_source: skip` and `data_path` pointing to the extracted dataset. Run `ict run --config <profile> --from audit --until validate_detection --run-id smt` to start from that tree.

| Stage | Published artifacts |
|---|---|
| `audit` | `samples.csv`, `issues.csv`, and `summary.json` with split counts, sensor ID offsets, JSON/XLSX checks, data fingerprints, and the versioned label policy |
| `manifest` | `manifest.csv` and `manifest_summary.json` with audit flags, decision reasons, label policy, and input hashes. Paths in the pipeline manifest are relative to the dataset root. |
| `convert` | `detection/yolo/data.yaml`, YOLO images and labels, COCO images and annotations, and `detection/conversion_summary.json` |
| `validate_detection` | `issues.csv` and `summary.json`. If issues are found, the run fails and retains the reports in the failed work directory. |

Conversion keeps the original Train and Validation assignments and exports only `candidate` rows eligible for detection. AI Hub categories `23/24/25/27/30` map to YOLO `0/1/2/3/4` and COCO `1/2/3/4/5`. Normal boxes are omitted, and eligible normal images retain empty YOLO labels. The audit compares sensor timestamps and measurement values independently of the reported sensor/image ID offsets. The manifest records sensor alignment metadata and keeps sensor training approval pending.

The four commands under `scripts/` retain their standalone arguments and default paths. Standalone manifests use the legacy repository-relative paths. Pass `--overwrite` to authorize replacing existing standalone reports. Standalone conversion requires a fresh output directory and keeps its legacy hard-link default. Pipeline conversion copies images.

Use `ict integrity analyze` to measure overlap in the original AI Hub assignments and `ict integrity split --dry-run` to preview a group-aware revision. Published split bundles are opt-in through `ict run --split-bundle` or the standalone commands' `--dataset` option. The [dataset integrity and evaluation protocol](docs/dataset-integrity.md) describes the evidence, publication options, and Test roles.

## Audit and prepare multimodal data

```text
ict multimodal audit --root data/extracted --manifest data/processed/manifest.csv --path-base . --preview
```

The audit reports verified JPG/JSON/XLSX relationships, split and source coverage, missing measurements, and channel statistics without fitting a model. Use `--out <fresh-directory> --export-index` to prepare a portable cohort for image, sensor, and paired dataset views. Sensor preprocessing uses frozen Train statistics and a capture-time window by default. [Source schema, measured coverage, and Python examples](docs/multimodal-data.md)

Python callers can pass these batches to the multimodal architecture for CPU forward checks. It provides an MLP sensor encoder, an optional GRU for verified multi-step windows, RGB and Faster R-CNN feature adapters, and late, concatenation, and gated fusion. Product risk scores carry branch and weight status. Original detector boxes can accompany those scores. The architecture ships with untrained heads and no measured accuracy. [Multimodal architecture, inference and checkpoint interfaces](docs/multimodal-architecture.md)

## Sensor and image-level classification

Use the preprocessing dependencies for JPG and XLSX decoding. The classification models run on CPU with no pretrained weights. Put this configuration in a file such as `classification.yaml` and point `data_path` at the extracted dataset.

```yaml
dataset_source: existing
data_path: data/extracted
classification:
  mode: fusion            # sensor, unimodal, or fusion
  sensor_model: logistic  # logistic or centroid
  missing: available      # available or paired
  group_policy: session   # session, source_day, or source
  sensor_window_seconds: 0
  sensor_value_atol: 0.001
  seed: 42
  epochs: 20
  learning_rate: 0.05
```

```text
ict run --config classification.yaml --run-id classifier --dry-run
ict run --config classification.yaml --run-id classifier --until classification_prepare
ict run --config classification.yaml --run-id classifier --approve-train
ict resume --config classification.yaml --run-id classifier --approve-train
```

Omit `classification` for the default detection workflow. An explicit detection range such as `--from audit --until validate_detection` also works with the configuration above after the existing detection prerequisites are available. Pass training approval again on resume if training needs to be retried. Completed stages reuse fingerprinted artifacts. An interrupted training stage restarts its fit from the configured seed. Run state records the Git revision and runtime.

### Samples and leakage controls

- Classification reads the original SMT Train and Validation annotation, image, and sensor directories. It pairs sibling files by stem, verifies JPEG metadata, and accepts embedded JSON `sensor_sequence`, XLSX, or both. Each sequence uses ordered ISO timestamps and five finite values in this order: temperature (°C), humidity (%), vibration (m/s²), acceleration (g), and noise (dB). Two available copies must agree row for row on length, timestamps, and measurement values within the configured absolute tolerance. The first sensor timestamp must agree with the capture time when the filename or image metadata provides one. Sensor/image ID offsets are tracked as metadata. With a single sensor source, cross-file value agreement is unavailable.
- Image label policy `smt-target-defect-presence-v1` assigns 1 for the presence of a target defect box and 0 for a normal sample with no target defect box. The SMT filename and annotation checks screen for contradictions and unsupported categories. An available `defect_detected` value may corroborate the label. Features are restricted to the five measured channels and RGB pixels. Boxes, label fields, filenames, source and session IDs, and absolute timestamps stay out of model input. Detection keeps its five-class object label policy.
- Both original splits are preserved. The audit quarantines cross-split duplicate stems, exact JPEG bytes, exact timestamped sequences, explicit production sessions, and exact source/capture overlaps. Optional `production_session_id` values in annotation or sensor metadata are scoped by source. The audit reports source/day and source overlaps, which can also be excluded with `group_policy`. Filename groups are proxies for production sessions. Incomplete session metadata, near duplicates, and source-specific signatures limit claims about generalization to new production lines.
- Sensor features summarize the mean, population standard deviation, and rate of change of measurements within the configured cutoff, measured in seconds after capture. The default uses the first reading only. Without a capture time the first reading anchors the window. Confirm that the chosen readings and JPG are available at the intended decision time, including whether images or measurements were produced after inspection. Standardization and model fitting use Train only.

### Model and report artifacts

The `sensor` mode fits a sensor-only classifier. `unimodal` adds a logistic classifier on 8×8 RGB thumbnails. `fusion` also averages the two classifier scores. With `missing: available`, each model uses samples with its input, and fusion uses the available score on single-modality samples. With `missing: paired`, both training and scoring require both inputs. Conflicting or corrupt files are flagged rather than treated as absent.

| Stage | Outputs |
| --- | --- |
| `classification_prepare` | `samples.jsonl` with versioned normalized samples, `alignment.json` with split and grouping checks, and `alignment_issues.csv` with sample dispositions |
| `classification_train` | `checkpoint.json` with feature order, Train normalization and fitted weights, model configuration, and data and implementation fingerprints |
| `classification_evaluate` | `comparison.json`, `comparison.csv`, and `predictions.csv` for Validation |

Use the `paired` report rows to compare modalities on the same Validation samples. The `available`, `sensor_only`, and `image_only` rows show coverage and behavior under missing inputs. Available-input training cohorts can differ by modality. The comparison uses a fixed threshold of 0.5 and reports confusion counts, accuracy, balanced accuracy, precision, recall, and F1. Undefined metrics are empty in CSV and null in JSON. Scores are uncalibrated.

## Environment diagnostics

`ict doctor` displays the local report. `ict doctor --json` writes one JSON document to stdout using the [version 1 schema](docs/doctor.schema.json). A completed report exits with status `0` even when optional tools or backends are unavailable. Missing measurements use `null`. Capabilities use `available`, `unavailable`, or `unknown`. Memory and disk measurements are in bytes, and free disk refers to the filesystem containing the current directory. RAM and logical CPU counts describe the environment as reported by the OS and may exceed container limits.

GPU inventory is best effort. PyTorch availability comes from PyTorch itself, and readiness requires a small tensor operation and a successful scalar readback. CUDA and ROCm test the first visible device and measure its free and total memory when available. MPS reports PyTorch's shared-memory recommendation when available. The report lists tested precisions and selected detector operators, and reports available host RAM separately from total RAM. Tool statuses reflect executable discovery and, for the repository's AI Hub helper, file presence. Container detection relies on visible runtime markers. [Hardware checks and pilot execution](docs/gpu-training-pilot.md)

`local_cpu` covers CLI diagnostics and CPU preprocessing. The `torch_*` modes describe the tested device operations. The WSL and container modes show whether the current process is already running in those environments or their readiness is unverified. External hardware commands have individual timeouts of at most eight seconds, and the isolated PyTorch process has a 20-second timeout.

## Configure the pipeline

Run `ict setup` from the workspace root in a terminal. Press Enter to accept a default or Ctrl+C to cancel. The wizard previews the settings and asks before saving. Setup records runtime preferences for later commands. Dataset and device readiness checks, downloads, installs, training, and remote jobs are separate operations.

```text
ict setup --dry-run
ict setup --non-interactive --dataset-source skip
ict setup --non-interactive --config profile.yaml --model rt_detr --dry-run
```

Headless setup requires an explicit dataset source from a flag, environment variable, or YAML. It exits with status `2` on invalid or missing settings and uses `--replace` to authorize changing an existing `configs/local.yaml`. Dry runs validate and preview the effective settings without writing files. An interrupted wizard exits with status `130`.

| YAML key | Accepted values and defaults |
|---|---|
| `execution_mode` | `local` (default), `wsl`, `docker`, `remote` |
| `dataset_source` | `aihub`, `existing`, `skip` (wizard default) |
| `data_path` | Required for `aihub` and `existing`. The wizard suggests `data/extracted`. |
| `output_path` | Filesystem path, defaults to `outputs` |
| `model` | `yolo` (default), `faster_rcnn`, `rt_detr` |
| `device` | `cpu` (default), `auto`, `cuda`, `rocm`, `mps` |
| `training_profile` | `smoke` (default), `standard`, `full` |
| `tracking` | `"off"` (default), `local`, `mlflow` |

CLI flags take precedence over process environment variables such as `ICT_MODEL` and `ICT_OUTPUT_PATH`. Environment variables take precedence over the selected YAML, which takes precedence over the packaged defaults in `src/ict_pipeline/config/default.yaml`. The wizard uses the resolved settings as prompt defaults and accepts answers for fields without explicit CLI flags. `--config` selects an input YAML in place of `configs/local.yaml`. Successful saves write the effective settings to `configs/local.yaml`. Relative paths use the workspace root. Setup validates path syntax and leaves filesystem and device availability checks to execution.

Use the process environment or copy `.env.example` to `.env` for credentials. A nonblank environment value takes precedence over the corresponding value in `.env`. The AI Hub source requires `AIHUB_API_KEY`. MLflow tracking requires `MLFLOW_TRACKING_URI`. Remote mode requires `REMOTE_HOST` and `REMOTE_USER` and accepts `REMOTE_SSH_KEY_PATH` for key-based authentication. The wizard uses hidden input for missing credentials and saves those entries to `.env` after confirmation. Headless setup reads credentials and leaves `.env` intact.

## Run tests

```text
python -m pip install -e ".[test]"
python -m pytest -m "not cpu_backend and not real_data_smoke"
```

The lightweight tests generate tiny JPG, JSON, and XLSX fixtures and require no AI Hub dataset, credentials, or GPU. Use `python -m pytest -m "e2e and not cpu_backend"` for fixture-based CLI workflows with mocked detectors. With training approval and the backend extras already installed, use `python -m pytest -m cpu_backend` to run real CPU training, checkpoint reload, interruption and resume, and DAG workflows for each installed detector.

Set `ICT_TEST_ARTIFACT_ROOT` or use `--ict-artifact-root=D:/ict-test-storage` to move disposable test files off the Windows system drive. Pytest cleans up completed fixtures by default. Use `--ict-keep-failed` to retain failure diagnostics and `--ict-artifact-summary` to measure fixture bytes before cleanup. `ict clean --dry-run` previews registered sessions in the selected root. [Retention and exact-target cleanup](docs/artifact-management.md)

See the [integration audit](docs/integration-audit.md) for executed checks, backend coverage, and the remaining work for a real-data model run.

To opt in to the real-data regression, set `ICT_REAL_DATA_ROOT` to an already extracted AI Hub 71900 dataset and run `python -m pytest -q tests/test_preprocessing.py -k real_data_regression`. The test reads that tree and writes derived artifacts under pytest's temporary directory. It checks the historical expectations of 15,372 audited samples, 14,718 candidates, 645 holds, 9 exclusions, 13,062 Train and 1,656 Validation candidates, 16,877 boxes, and zero validation issues.

See [code, dependency, and dataset licensing](docs/licenses.md) before distributing project materials.
