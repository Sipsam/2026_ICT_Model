# Real AI Hub SMT detector smoke test

This experiment checks official checkpoint transfer, one CPU training epoch, epoch checkpoint recovery, inference, and shared Validation evaluation for YOLO11n, Faster R-CNN MobileNetV3 Large 320 FPN, and RT-DETR R18. The cohort is too small and the inputs too reduced to estimate defect-detection accuracy or rank detectors.

## Stage the runtime and weights

Run from the repository root in an existing project environment. Review the source terms and obtain explicit approval before installing packages or downloading checkpoints. Use matching PyTorch and TorchVision builds. The local experiment used Python 3.11.9 and CPU builds of PyTorch 2.15.0 and TorchVision 0.29.2. The optional tests also use `psutil` on Windows for process peak working set, supplied by the YOLO extra.

```text
python -m pip install -e ".[preprocess,test,train-yolo,train-faster-rcnn,train-rt-detr,pretrained-safetensors]"
ict doctor
python -m pip check
```

Use an empty local cache directory such as `outputs/smt-smoke-official-weights`. Download the files from the linked publisher locations after approval, verify each full SHA-256, then place the verified file under its listed name. Use `Get-FileHash -Algorithm SHA256 <file>` on PowerShell or `shasum -a 256 <file>` on Unix. Keep the file local for training and recovery. The YOLO release asset metadata did not provide a digest. The digest below identifies the publisher download observed in this experiment. TorchVision publishes the digest prefix in the checkpoint URL and the full digest below was computed from those bytes. RT-DETR's full digest matches the publisher's LFS metadata for the pinned revision.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| [yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt) | 5,613,764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |
| [fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth](https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth) | 77,844,807 | `907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5` |
| [model.safetensors](https://huggingface.co/PekingU/rtdetr_r18vd/resolve/ac77a11ff0170a41b771c03264987f8ce2b0d753/model.safetensors) | 80,904,152 | `fe87a5a30f5daf298d10794c7682a63b6107986f97d6a770ba948d89e4340093` |

The YOLO profile trusts the exact pinned native pickle. Ultralytics lists AGPL-3.0 and Enterprise terms for its YOLO models. TorchVision's library has a BSD-3-Clause license and the project directs users to check applicable weight and training-data terms. The Peking University model card lists Apache-2.0. AI Hub's data access terms apply to the SMT inputs and must be reviewed before distributing data, derived artifacts, or models. [Ultralytics terms](https://www.ultralytics.com/license) · [TorchVision weight terms](https://github.com/pytorch/vision#pre-trained-model-license) · [COCO terms](https://cocodataset.org/#termsofuse) · [RT-DETR model card](https://huggingface.co/PekingU/rtdetr_r18vd) · [Project data permissions](licenses.md)

## Prepare a separate cohort and freeze the protocol

Use an already extracted dataset, its original candidate manifest, and a fingerprinted `leakage.json` produced by `ict integrity analyze`. The integrity guide shows how to generate the audit and preview the provider and group-aware assignments. Use an unused output path. [Dataset integrity guide](dataset-integrity.md)

```text
python -m ict_pipeline.smoke --root data/extracted --manifest data/processed/manifest.csv --path-base . --analysis outputs/integrity-ef0ef1dd/leakage.json --out outputs/smt-smoke-cohort --cache outputs/smt-smoke-official-weights
ict train --all --dataset outputs/smt-smoke-cohort/detection --config-dir outputs/smt-smoke-cohort/profiles --dry-run
```

Preparation rechecks the manifest and all audited source bytes, then draws two images per class and two normal images per split from singleton candidate groups in the seed-42 capture-group preview. The preview groups exact matches, known sessions, matching source capture timestamps, and both near-image and near-sensor components. The smoke export copies 12 Train, 12 Validation, and 12 internal Test images to each detection format. Derivation preserves original split names and the full parent preview fingerprints. `preparation.json` includes both preview summaries and validation results. `split.json` records per-sample evidence and image and source annotation fingerprints, and the detection seal binds the exported files. The original provider manifest, split, and source data remain available as the baseline.

All three profiles are pinned to one epoch, batch size 2, 128-pixel input, CPU FP32, seed 42, AdamW at 0.0001, and a frozen backbone. `protocol.json` fixes best-checkpoint Validation evaluation at 128 pixels, score floor 0.001, confidence cutoff 0.25, match IoU 0.5, NMS IoU 0.7 where applicable, one warmup call, and shared COCO bbox AP. The internal Test is reserved. It belongs to the same AI Hub cohort and its production-session independence is unknown.

## Local split evidence

The latest local audit scanned 15,372 records and identified 14,718 candidates. Its dataset and manifest fingerprints match the preceding XLSX-verified audit. The original provider assignment had 13,062 Train and 1,656 Validation candidates, including 45 near-image components and 11 same-source capture groups crossing the split. The rerun of the group-aware preview assigned 11,038 Train, 2,208 Validation, and 1,472 internal Test candidates with all five defect classes and background represented in every split. The largest candidate component had four members. None of the enabled grouping methods crossed splits. Broader overlaps remained for capture timestamps (121 groups), source days (13), and manufacturers (4). There was no usable explicit production-session identifier for any of the 15,372 records. The full group-aware assignment was only previewed.

| Audit fingerprint | SHA-256 |
| --- | --- |
| Latest leakage report | `8c4b39770dc6e6ac5785305308945850646462a1c221d595770834d6bc3b5180` |
| Normalized provider manifest | `1cfec200be60f863bbe933fc5226f0b445200802076ab9443c763999b56d066c` |
| Audited source bytes | `7cc59935a64ac7b889b9f8834ad1083ebd41106c2566d2719f32b5a19d148011` |
| Smoke revision ID | `28ffa3075e65de5e7c0c8c6128bb2e61791c71d4645a851a28f7cea2ce98b39c` |
| Selected source images | `de41de9d20420820c50f7eb6cd8fde4ea7e344238b1d5f4ae4d78bac6f2bb02c` |
| Selected source annotations | `4dd4166f5b241f98b827e9171ca473ac64fb6b67529e843640a3448a2c5382b9` |

All 36 selected images have distinct SHA-256 digests and group IDs. YOLO/COCO conversion validation reported zero issues. The exported COCO labels contain 14 Train, 20 Validation, and 22 internal Test target boxes, with two background images in each split.

## Run the approved experiment

Use one process per model to obtain separate memory peaks. Set the approval variable only after authorizing the shown workload. This PowerShell example uses a fresh output directory.

```powershell
$env:ICT_REAL_SMOKE_COHORT = 'outputs/smt-smoke-cohort'
$env:ICT_REAL_SMOKE_OUTPUT = 'outputs/smt-smoke-results'
$env:ICT_REAL_SMOKE_APPROVE_TRAIN = '1'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
foreach ($detector in 'yolo','faster_rcnn','rt_detr') {
    python -m pytest -q -s tests/test_real_data_smoke.py -m real_data_smoke -k $detector
    if ($LASTEXITCODE) { throw "Smoke test failed for $detector" }
}
Remove-Item Env:ICT_REAL_SMOKE_APPROVE_TRAIN
```

Each test blocks Python network connections, verifies the local cache, fine-tunes one epoch, then injects a cooperative interruption after the checkpoint and metadata are committed. The regular training runner recovers and finalizes the saved epoch in a fresh attempt. The test compares checkpoint fingerprints, reloads the last checkpoint, checks head dimensions, and uses the normal evaluator with the recovered best checkpoint. This exercises recovery after a committed final epoch. Optimizer continuation across a remaining epoch is covered by the two-epoch fixture tests.

Read `smoke.json` in each completed model directory for timing, peak process memory, dimensions, and links to `run.json` and the Validation result. The training record provides transfer keys, full provenance, epoch loss, and best and last checkpoint fingerprints. The parent attempt metadata records runtime versions and hardware. Use `ict compare` with the three Validation reports to verify that they share the evaluation cohort and protocol. Leave training approval unset for the full CPU fixture suite.

## Local experiment results

**Passed** on Windows build 26100, AMD Ryzen 7 4700U with eight logical CPUs, and 7.42 GiB RAM. The installed PyTorch and TorchVision CPU builds were `2.15.0+cpu` and `0.29.2+cpu`. Additional packages included Ultralytics 8.3.228, Transformers 4.57.6, safetensors 0.8.0, OpenCV 5.1.0, NumPy 2.4.6, Pillow 13.0.0, and SciPy 1.17.1. CUDA, ROCm, and MPS were unavailable to this runtime.

| Detector | Train loss | Native Validation loss | Train / recover / evaluate (s) | Peak process RSS (MiB) | Shared Val AP50 / AP50–95 |
| --- | --- | --- | --- | ---: | --- |
| YOLO11n | 2.14022 / 4.05021 / 1.62584 | 1.80185 / 3.91357 / 1.37966 | 44.72 / 1.97 / 3.80 | 473 | 0.04000 / 0.03406 |
| Faster R-CNN | 1.62596 | 1.69555 | 29.17 / 30.19 / 18.17 | 658 | 0.11906 / 0.05701 |
| RT-DETR R18 | 150.13839 | 45.00056 | 62.05 / 13.34 / 12.95 | 798 | 0.00168 / 0.00038 |

YOLO loss components are box, classification, and distribution focal loss in that order. Wall times include local setup, loading, verification, and writes. Memory is the high-water mark for each separate test process. The run used a shared machine during regression checks. These timings and scores document workflow behavior for this cohort.

| Detector | Raw missing / unexpected keys | Accepted mappings | Incompatible / reset keys | Reloaded head |
| --- | --- | --- | --- | --- |
| YOLO11n | 0 / 0 | None | 0 / 78 | 5 class outputs |
| Faster R-CNN | 10 / 10 | 10 TorchVision FPN/RPN key renames | 0 / 4 | 6 classifier rows including background, 24 box regression rows |
| RT-DETR R18 | 24 / 0 | 24 restored tied aliases | 0 / 15 | 5 decoder and encoder class outputs, 6 denoising embedding rows |

The first Faster R-CNN tensor check failed on the older key names in the genuine COCO_V1 artifact. The adapter now applies exactly the matching TorchVision renames and validates the resulting tensor names, shapes, and dtypes. Regression tests reject malformed shapes and collisions between a legacy key and its modern counterpart. Each real-data test also rejected a deliberately incorrect SHA-256, loaded the correct local file with Python network connections blocked, and recovered byte-identical best and last checkpoints after the saved epoch. All three evaluated the same 12 Validation images and 20 boxes. Validation image fingerprint is `0f10b843c9abb4ece199e10659da539d6c91ce110e69ff30c564091aa1db06f5`. Validation annotation fingerprint is `593829a9dd1d1138beb2a66d26d48de08e62f752b57c6d8ef14168ad09a59d26`.

The local consolidated results are under `outputs/smt-smoke-results`, including `experiment.json`, `comparison.json`, and per-model training and evaluation records. Internal Test scoring, continuation into a second real-data epoch, and GPU runs were **Not run**.

| Check | Result |
| --- | --- |
| `ict doctor`, `python -m pip check`, and offline TorchVision NMS | **Passed** for the CPU environment |
| `ict integrity split --dry-run`, `python -m ict_pipeline.smoke`, and `ict train --all --dry-run` with the inputs above | **Passed** |
| `pytest -q -s tests/test_real_data_smoke.py -m real_data_smoke -k <detector>` | **Passed** for all three detectors, one process each |
| `ict compare <three-validation-reports>` and `ict report --format json` against the smoke result roots | **Passed** |
| `python -m pytest -q -rs` | **Passed** with 180 passed and 6 skipped. The skips were three approval-gated smoke cases run separately, two existing opt-in dataset regressions, and one Linux-only detached-worker test. |

## Full-scale GPU training

Use a dedicated supported Linux GPU host and verify detector operators, GPU memory, and package versions before committing to a full run. Stage the reviewed data and hash-verified checkpoints locally on that host, use separate writable run storage, and preserve the same approved split revision across models. Repeat a short GPU pilot at production input resolutions to verify that small SMT boxes survive resizing and to measure memory, storage throughput, and sustained batch time. Begin with FP32, validate mixed precision and finite losses in the pilot, and then set per-model batch size from the measured headroom. Select checkpoints on Validation and keep Test sealed until the final evaluation protocol and checkpoints are frozen. An external production claim requires separately collected and labeled boards with traceable acquisition provenance.
