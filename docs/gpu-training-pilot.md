# Hardware-adaptive detector pilots

Run hardware checks in the Python environment that will execute the models. Use the local shell on Windows, Linux, WSL2 or macOS, an SSH session on a remote host, or a shell inside a provisioned container. Use paths visible there and mount input bundles and weight caches read-only where possible. The commands use existing packages, data, and weights. [Runtime guide](runtime.md)

## Check the execution host

```text
ict doctor --json
ict recommend --all --output outputs/pilots
ict recommend --model rt_detr --device cuda --image-size 640 --batch-size 2 --workers 0 --gradient-accumulation 4
```

`doctor` reports graphics inventory separately from PyTorch discovery and tensor execution. It reports host RAM, available RAM and free disk. For CUDA and ROCm, the current single-device adapters use the visible logical device 0, and the probe measures its total and free memory when the driver exposes those values. MPS uses shared system memory and can report PyTorch's recommended working-set limit. On Windows available RAM is physical memory reported by the OS. On Linux it uses `MemAvailable` and accounts for detectable cgroup v2 memory limits. On macOS it counts free and inactive pages where available.

`recommend` checks installed package versions, FP32 execution, the requested precision, and model-specific operators. It selects a probe-passing device by default and otherwise reports why the requested settings are blocked. All defaults are overridable. For the first pilot, FP32 isolates model and operator issues from mixed precision. FP16 and BF16 are offered only when the selected adapter exposes them and a small autocast forward and backward probe succeeds. An approved pilot tests actual detector execution, numerical stability, and throughput.

| Detector | Device paths in the adapter | Readiness checks beyond base tensor execution | Initial batch cap |
| --- | --- | --- | ---: |
| YOLO11n | CPU, CUDA, ROCm, MPS | FP32 NMS; FP16 only on CUDA or ROCm | 8 |
| Faster R-CNN MobileNetV3 | CPU, CUDA, ROCm | FP32 NMS and ROI Align backward; FP16 or native BF16 on CUDA or ROCm | 4 |
| RT-DETR R18 | CPU, CUDA, ROCm | Pinned Transformers and SciPy dependency metadata; FP16 or native BF16 on CUDA or ROCm | 4 |

The batch model uses fixed and per-image allowances, a 65% planning budget, and the configured resolution. It allows extra headroom for YOLO's larger native Validation batch and uses FP32 memory allowances across all precision modes. It recommends zero workers on CPU or MPS and at most two initially on an accelerator with enough host RAM and CPUs. Unknown free device memory starts at batch one. An estimate above 85% of measured available memory produces an actionable preflight failure. Real memory use also depends on image dimensions and decoding, proposal counts, optimizer state, and concurrent workloads. Resource reports identify measured free bytes separately from estimated requirements.

Faster R-CNN and RT-DETR support `--gradient-accumulation`. The nominal effective batch is `batch_size × gradient_accumulation`, with one step for a partial group at epoch end. This does not change activation memory per microbatch. Use the flag with an individual COCO model. YOLO keeps native warmup and accumulation, and its requested GPU worker count is an upper bound that the library may clamp. [Detector profiles](models.md)

## Prepare a bounded cohort

Use a reviewed group-aware bundle published with `ict integrity split --publish --export-detection`. A previously prepared real SMT smoke bundle also works. The parent must contain its sealed detection export and matching `manifest.csv`. [Dataset integrity and publication](dataset-integrity.md) · [Existing real SMT smoke bundle and official weight profiles](real-data-smoke-test.md)

```text
ict pilot prepare --bundle outputs/reviewed-split --out outputs/pilots/smt-01 --train-images 64 --val-images 32 --min-per-stratum 1 --seed 42
```

The limits are per split, up to 256 each, and the Train limit must be smaller than parent Train. The seeded sampler selects whole groups, first covering all five defect classes and background in each split and then filling within the budget. A multi-label image counts toward each of its represented classes. Whole-group selection can undershoot a limit. If the coverage heuristic cannot fit the requested groups, adjust the seed, limits or minimum. Only existing candidate images enter the pilot, with both parent and original assignments retained. The parent seal verification checks the integrity of all parent files, including any reserved Test bytes. Pilot selection, copying, model fitting and evaluation use only Train and Validation.

The destination must be fresh. Preparation copies the chosen YOLO and COCO data into a separate dataset, validates both formats, and seals the output. The result contains `cohort.json`, a membership `manifest.csv`, `detection/split_revision.json`, and validation reports. Preserve the parent revision and seed alongside the experiment.

## Preview, run and recover

Point the pilot at existing pretrained model YAML files and locally staged checkpoints. The weights manager checks path, architecture, library version, format and SHA-256. The pilot preserves the model profile's initialization, freezing and optimizer settings and resolves execution settings through the recommendation and any CLI overrides. All models use the cohort seed and the same image-size override. Default image size is 320, with a maximum of 1280. Batch size is capped at 32 and the Train count, and periodic checkpoints are disabled. The default is one epoch, with a hard maximum of three.

```text
ict pilot run --cohort outputs/pilots/smt-01 --config-dir profiles --all --device cpu --image-size 128 --batch-size 2 --dry-run
ict pilot run --cohort outputs/pilots/smt-01 --config-dir profiles --all --device cpu --image-size 128 --batch-size 2 --approve-train
```

Use `--model yolo`, `--model faster_rcnn` or `--model rt_detr` for an individual model. Review the preview's `checks_passed`, `issues`, `storage_ready`, scope, and checkpoint sources before authorizing training. The worker rechecks storage and pretrained files before running. Existing storage checks also protect private YOLO cache copies and atomic checkpoint writes by volume. The validated best checkpoint is evaluated on the same selected Validation cohort using shared COCO bbox AP and FP32 inference.

Each execution creates a fresh `attempts/<id>/`. The atomic `pilot.json` records status, options, scope, cohort provenance, and model results as they become available. For an interrupted model with a committed checkpoint, rerun with the same profile and epoch count, `--model`, `--resume-from <cohort>/attempts/<id>/training/<model>`, and fresh approval. Omitted adaptive options inherit the unfinished run's settings. The training runner verifies input, config, code, runtime and checkpoint fingerprints, then writes to a new attempt. To recover a failed Validation step after training completed, evaluate its completed checkpoint via `ict evaluate --split val` with the same dataset, device, image size and `--warmup 1`.

## Use the measurements

Use `ict training-data ablation --dry-run` for an offline comparison plan with pinned checkpoints and a shared Validation protocol. Pilot hardware findings can inform a detector's common base profile. Keep those settings fixed across that detector's ablation variants. [Controlled training experiments](training-optimization.md)

- The pilot summary includes preparation, adapter and Validation wall times, per-epoch training and native Validation metrics, and the shared Validation report path and metrics. Per-model `run.json` contains checkpoint hashes, transfer diagnostics, package versions, and code fingerprints.
- RAM is sampled process RSS at 100 ms intervals for the main process. CUDA and ROCm peaks describe the PyTorch allocator on the selected device. MPS allocation is sampled. Missing measurements are null or absent. Process reuse and other activity can affect measurements.
- Loader wait is host time blocked obtaining batches. Batch preparation is the adapter's timed host-side work and may overlap GPU transfers; Faster R-CNN's internal model transforms remain in model wall time. YOLO also reports loader setup, private input copy time, and generated cache bytes. Checkpoint logical sizes count each filename, even when best and last share storage.
- The next-settings record keeps the measured pilot settings or reduces the batch when the observed peak approaches the available budget. Recheck representative image sizes, box sizes, group coverage and longer-run throughput before increasing workload. Estimating production accuracy calls for a separately designed benchmark.

## Verification

Run the lightweight suite before using the existing CPU backend fixtures. Real SMT training still requires explicit approval, and the real GPU acceptance case is an opt-in pilot on the selected target. Review free disk before running real backend cases. [Managed pytest artifacts and storage checks](artifact-management.md)

```text
python -m pytest -q -rs -m "not cpu_backend and not real_data_smoke"
python -m pytest -q -rs --ict-artifact-summary
```
