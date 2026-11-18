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

`ict run` and `ict resume` work with closed stdin. For missing transfers, review `ict data download` and pass `--approve-download` to authorize acquisition for that invocation. Pass the same flag to `ict resume` if transfers are still missing. Pass `--run-id <id>` to choose a name or reuse that run's cache. `--from` and `--until` select an inclusive range in dependency order. Repeat `--only` to select individual stages. Any omitted prerequisite must have a valid result in the same run. `--force` recomputes the selection in new artifact directories. The DAG reserves `train`, `validate_models`, `evaluate`, and `report`. Use the standalone commands for model work and reporting.

`ict status` shows the latest recorded attempt per stage for a pipeline run. Use `--run-id <id>` to select it when training or evaluation has a newer record. `ict resume` defaults to the latest pipeline run and rechecks settings, inputs, stage versions, and artifact bytes before reuse. If you ran with `--config`, pass the same file to `resume` to use that profile again.

Each run stores an atomic snapshot in `runs/<id>/state.json` and a structured event log in `events.jsonl`. A stage writes to `work/<stage>/<attempt>/`. After output verification, the runner publishes the directory under `artifacts/<stage>/<attempt>/` and commits the successful attempt to state. Failed and abandoned attempts remain available for inspection. Only an attempt recorded as completed and verified against its fingerprints is eligible for cache reuse. A lock allows one writer per run, and the OS releases it when the process exits. Cancellation returns status `130`. Progress includes elapsed time and an ETA when the stage reports measurable work.

The Python adapter interface is `Stage`, `StageInput`, `StageContext`, and `StageResult` in `ict_pipeline.pipeline`. An adapter should read the declared inputs, write artifacts exclusively to `StageContext.output_dir`, and report progress as completed and total units. `StageContext.artifact_dir` is the future published path for files that must refer to it. `StageContext.input_fingerprints` provides the verified content hash for each declared input. Adapters own data-format validation and must keep secrets out of outputs and console messages. Bump `Stage.version` when changes to an adapter affect the validity of its results.

## Train detection models

```text
ict train --model faster_rcnn --dry-run
ict train --model faster_rcnn --config-dir configs/models --approve-train
ict train --all --dataset <conversion-attempt>/detection --config-dir configs/models --dry-run
```

Choose `yolo`, `faster_rcnn`, `rt_detr`, or `--all` for sequential training. Training starts only with `--approve-train` and initializes from architecture definitions with random weights. The default input is the standalone converter's `data/processed/detection`. Pass `--dataset` for a pipeline conversion attempt. The command preserves the generated Train and Validation splits, uses the canonical five classes, and checks that both formats have the same split membership and image bytes with `--all`. Keep the conversion artifacts stable during training.

Install the extra for the selected backend, for example `python -m pip install -e ".[train-faster-rcnn]"`. Select a compatible PyTorch and TorchVision build for your OS and accelerator. The project requires Python 3.11 or newer and the training extras require PyTorch 2.6+ and TorchVision 0.21+. Ultralytics 8.3.228 declares Python 3.8+ and Transformers 4.57.6 declares Python 3.9+. For accelerator availability, use `ict doctor`. ROCm uses PyTorch's `cuda` device namespace. [PyTorch and TorchVision compatibility](https://github.com/pytorch/vision#installation), [Ultralytics package metadata](https://github.com/ultralytics/ultralytics/blob/v8.3.228/pyproject.toml), [Transformers package metadata](https://github.com/huggingface/transformers/blob/v4.57.6/setup.py).

| Model | Extra and implementation | Library license | Devices and precision | Checkpoint |
|---|---|---|---|---|
| YOLO | `train-yolo`, Ultralytics 8.3.228 YOLO11n from YAML | [AGPL-3.0 / Ultralytics Enterprise](https://www.ultralytics.com/license) | CPU, CUDA, ROCm, MPS. FP32 or accelerator FP16 on CUDA/ROCm. | Native Ultralytics `.pt` with training state, selected by validation fitness. |
| Faster R-CNN | `train-faster-rcnn`, TorchVision MobileNetV3-Large FPN | [BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE) | CPU, CUDA, ROCm. FP32 or CUDA/ROCm FP16 and native BF16. | PyTorch state dictionary `.pt`, selected by minimum validation loss. |
| RT-DETR | `train-rt-detr`, Transformers 4.57.6 RT-DETR with a local ResNet-18 config | [Apache-2.0](https://github.com/huggingface/transformers/blob/v4.57.6/LICENSE) | CPU, CUDA, ROCm. FP32 or CUDA/ROCm FP16 and native BF16. | PyTorch state dictionary `.pt`, selected by minimum validation loss. |

The adapters use Ultralytics' detection trainer, the TorchVision model builder with both weight arguments disabled, and the Transformers RT-DETR model constructor with an explicit local backbone config. [Ultralytics training API](https://docs.ultralytics.com/modes/train/), [TorchVision Faster R-CNN API](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html), [Transformers RT-DETR API](https://huggingface.co/docs/transformers/v4.57.6/model_doc/rt_detr).

Copy the packaged [YOLO](src/ict_pipeline/models/yolo.yaml), [Faster R-CNN](src/ict_pipeline/models/faster_rcnn.yaml), or [RT-DETR](src/ict_pipeline/models/rt_detr.yaml) YAML to the directory specified by `--config-dir`. Name the files `yolo.yaml`, `faster_rcnn.yaml`, and `rt_detr.yaml` and supply one for each selected model. Each accepts `epochs`, `batch_size`, `image_size` (multiple of 32, at least 64), `optimizer` (`SGD`, `Adam`, or `AdamW`), `lr`, `seed`, `device`, `precision`, and `save_period` (0 disables periodic snapshots). The default device is CPU and `device: auto` checks for an available accelerator. YOLO uses private copies of the converted images and labels for its caches and JPEG checks. Both COCO adapters read the original converted files directly.

`runs/training/<id>/<model>/run.json` records status, config, library and code versions, split and annotation fingerprints, the label policy, and relative paths and fingerprints for `best.pt` and `last.pt`. A new run gets a new ID. To resume an unfinished model, pass `--resume-from <model-run-directory>`, the original dataset and config, and `--approve-train`. Resume starts after the last saved epoch in a new directory. COCO checkpoints restore the model, optimizer and scaler and reconstruct the epoch seed. YOLO uses native training-state restoration from its EMA checkpoint and may produce a different trajectory from an uninterrupted run. Resume only from your own trusted runs because native Ultralytics checkpoints contain pickled objects.

The Python entry points are `ict_pipeline.models.contracts.ModelAdapter`, `TrainingRequest`, and `Checkpoints`, and `ict_pipeline.models.registry.REGISTRY`. Training within `ict run` remains planned.

## Evaluate and compare detectors

```text
ict evaluate --run runs/training/<id>/yolo --run runs/training/<id>/faster_rcnn --run runs/training/<id>/rt_detr
ict compare outputs/evaluation/<attempt>/01-yolo.json outputs/evaluation/<attempt>/02-faster_rcnn.json outputs/evaluation/<attempt>/03-rt_detr.json
```

Pass one or more trusted, completed ICT model run directories and use the same backend extras as training. Ultralytics checkpoints can contain pickled objects. Evaluation verifies each checkpoint against its training record and defaults to `--checkpoint best`. Use `--checkpoint last` to evaluate the final saved weights. Every model infers on all images listed in the same COCO annotation file, including normal images that have zero target boxes. COCO files must use the canonical five classes, original-image pixel coordinates, and non-crowd boxes. Evaluation requires at least one ground-truth object in the selected split.

The default `--split val` reads `coco/annotations/instances_val.json` and `coco/images/val/` under `--dataset`. Reports identify Validation as `validation_for_tuning`. For a separately prepared Test set, put its labels and images in the corresponding `instances_test.json` and `coco/images/test/` paths. Use `--split test --held-out` only after independently holding out Test and freezing model and protocol choices using Train and Validation. Provide the original detection directory with `--reference-dataset` if Test lives in a different directory. The command verifies the reference image fingerprints against every model's recorded Train and Validation and rejects shared filenames or byte-identical images. The holdout attestation covers production groups, near duplicates and prior tuning history, which these file checks cannot establish.

The evaluator uses COCO bbox AP across all sizes, with 101 interpolated recall points, IoU thresholds from 0.50 to 0.95 in steps of 0.05, and at most 100 predictions per image and class. AP uses scores at or above `--score-floor` (default 0.001). Classes absent from ground truth have null AP and are excluded from the mAP average. Precision, recall and F1 use class-aware, score-ordered matching at `--confidence` (default 0.25) and `--match-iou` (default 0.5), with per-class counts and micro-averaged totals. The confusion matrix matches by IoU regardless of predicted class at that same operating point. Its rows are ground truth and its columns are predictions. The background row counts unmatched predictions and the background column counts missed objects. [COCO reference evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py).

| Benchmark setting | Behavior |
|---|---|
| Image sizing | `--image-size 640` by default. YOLO letterboxes with bilinear resize and fill 114. Faster R-CNN preserves aspect ratio within the size bound and pads to its tensor stride. RT-DETR stretches with bilinear resize. Reported boxes are mapped back to original-image pixels. |
| Postprocessing | YOLO selects the highest-scoring class at each location and uses class-aware NMS. Faster R-CNN uses its proposal NMS and class-aware ROI NMS, with up to 1,000 retained ROI detections before the common COCO cap. The default ROI/YOLO `--nms-iou` is 0.7. RT-DETR ranks query/class scores and keeps the checkpoint architecture's query count without NMS. |
| Warmup and timing | `--warmup 3` by default. Warmup uses the first samples in ID order and cycles through the split if needed, then every image is measured once at batch size 1. Timing includes image read/decode, resizing, host/device copies, inference, postprocessing and normalization. Model load, warmup, label matching and report I/O are excluded. FPS is timed images divided by summed latency. Latency is reported as mean, median and nearest-rank p95 in milliseconds. |
| Synchronization and precision | CPU is synchronous. CUDA/ROCm and MPS synchronize around each timed image. Default is `--device cpu --precision fp32`. CUDA/ROCm FP16 uses half-precision weights and input for YOLO and autocast for the other adapters. The report records TF32 settings for CUDA. |
| Hardware | The report records the available CPU and accelerator identities, device backend, OS, PyTorch thread count, driver and library versions. Cache state, thermals and host load can affect runtimes. `ict compare` flags whether reported hardware and shared runtime settings match. |

Evaluation writes `outputs/evaluation/<attempt>/evaluation.json`, `events.jsonl`, and an atomic schema-versioned JSON report per completed model. Report paths go to stdout and progress to stderr. Reports include the checkpoint fingerprint and file size in bytes, image and annotation fingerprints, the split role, class mapping, evaluation settings, and code fingerprints. Checkpoint size includes training state if the saved file contains it. Resume with `--resume-from <attempt-dir>` and the original arguments in the same order. Completed reports are verified and reused. An interrupted model starts its inference from the beginning. When given explicit paths, `ict compare` requires reports from the same labeled split, code and evaluation settings.

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

## Environment diagnostics

`ict doctor` displays the local report. `ict doctor --json` writes one JSON document to stdout using the [version 1 schema](docs/doctor.schema.json). A completed report exits with status `0` even when optional tools or backends are unavailable. Missing measurements use `null`. Capabilities use `available`, `unavailable`, or `unknown`. Memory and disk measurements are in bytes, and free disk refers to the filesystem containing the current directory. RAM and logical CPU counts describe the environment as reported by the OS and may exceed container limits.

GPU inventory is best effort. PyTorch availability comes from PyTorch itself, and readiness requires a small tensor operation and a successful scalar readback. CUDA and ROCm test the first visible device. Apple unified memory and unreported dedicated VRAM appear as unknown. Tool statuses reflect executable discovery and, for the repository's AI Hub helper, file presence. Docker daemon access, WSL distributions, AI Hub credentials, and remote hosts are outside the probe. Container detection relies on visible runtime markers.

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
python -m pytest
```

The tests generate tiny JPG, JSON, and XLSX fixtures in temporary directories and require no AI Hub dataset, credentials, GPU, or hosted runner. Run `python -m pytest -m e2e` for the fixture-based CLI journey. It uses local mock acquisition and model adapters with real preprocessing, checkpoint verification, evaluation scoring, and report export. Optional backend smoke tests use one tiny CPU epoch when compatible training dependencies are already installed.

To opt in to the real-data regression, set `ICT_REAL_DATA_ROOT` to an already extracted AI Hub 71900 dataset and run `python -m pytest -q tests/test_preprocessing.py -k real_data_regression`. The test reads that tree and writes derived artifacts under pytest's temporary directory. It checks the historical expectations of 15,372 audited samples, 14,718 candidates, 645 holds, 9 exclusions, 13,062 Train and 1,656 Validation candidates, 16,877 boxes, and zero validation issues.

See [code, dependency, and dataset licensing](docs/licenses.md) before distributing project materials.
