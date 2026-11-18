# Detector initialization and fine-tuning

Omitting `initialization` uses `scratch`. The packaged profiles specify that mode explicitly. To fine-tune, put a checkpoint from a provider you trust in a local cache outside the dataset and pipeline runs directories. Stage the file only after approval for any large download, and obtain its full SHA-256 from trusted publisher metadata or an independently verified copy. ICT reads the file from disk and performs no checkpoint downloads.

## Supported checkpoint layouts

| Model YAML | Architecture | Published checkpoint | Format | Class-dependent tensors initialized for SMT |
| --- | --- | --- | --- | --- |
| `yolo.yaml` | `yolo11n.yaml` | Ultralytics `yolo11n.pt` from the YOLO11 release, 5.6 MB. Use `library_version: "8.3.228"`. [Release](https://github.com/ultralytics/assets/releases/tag/v8.3.0) | `ultralytics_pickle` with `trusted_pickle: true`. Trusted flat tensor exports also work. | The classification tower is reset for five classes. Compatible box regression and backbone weights are transferred. |
| `faster_rcnn.yaml` | `fasterrcnn_mobilenet_v3_large_320_fpn` | TorchVision `FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1`, listed as 74.2 MB. Use the exact TorchVision build version on the target. [TorchVision model and weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html) | `state_dict`, read with `torch.load(weights_only=True)`. | Both ROI prediction layers are reset for five classes plus background. The MobileNet body, FPN, proposals and ROI box features are transferred. |
| `rt_detr.yaml` | `rt-detr-resnet18` | Peking University `PekingU/rtdetr_r18vd`, 80.9 MB. Use `library_version: "4.57.6"`. [Model card](https://huggingface.co/PekingU/rtdetr_r18vd) | `safetensors` | Encoder and decoder class projections and the denoising class embedding are reset. Compatible backbone, encoder, decoder and box tensors are transferred. |

The supported source label counts default to 80 for YOLO and RT-DETR, and 91 including background for Faster R-CNN. For trusted checkpoints with the same architecture and a different label count, set `checkpoint.source_classes`. This changes the expected source tensor shapes. The target class order always comes from the SMT label policy.

The Faster R-CNN pretrained profile uses the official FrozenBatchNorm backbone layout. It applies TorchVision's legacy FPN and RPN key renames before strict tensor validation and records the applied renames in `run.json`. The RT-DETR pretrained profile uses three decoder layers and the official R18 encoder expansion of 0.5. Historical scratch profiles retain their original BatchNorm and RT-DETR encoder expansion of 1.0. These differences matter when interpreting a scratch-versus-pretrained comparison.

The [real-data smoke guide](real-data-smoke-test.md) records the exact official files and full digests used by the bounded SMT experiment. `python -m ict_pipeline.smoke` generates all three pinned profiles. The experiment freezes each backbone and records transfer diagnostics and training loss in `run.json`.

## Example RT-DETR profile

Copy the packaged profile to `profiles/rt_detr.yaml` and update it as shown. The hash below is the published file digest for the pinned model revision.

```yaml
epochs: 1
batch_size: 2
image_size: 64
device: cpu
initialization: pretrained
checkpoint:
  source: weights/rtdetr-r18vd/model.safetensors
  origin: https://huggingface.co/PekingU/rtdetr_r18vd
  revision: ac77a11ff0170a41b771c03264987f8ce2b0d753
  sha256: fe87a5a30f5daf298d10794c7682a63b6107986f97d6a770ba948d89e4340093
  license: Apache-2.0
  architecture: rt-detr-resnet18
  library_version: "4.57.6"
  format: safetensors
freeze_backbone: true
fine_tuning:
  optimizer: AdamW
  lr: 0.0001
  weight_decay: 0.0001
  momentum: 0.9
```

For Faster R-CNN, use `revision: COCO_V1` and the official checkpoint from the TorchVision weights enum. For YOLO, use `revision: v8.3.0` and set `trusted_pickle: true` only for a file whose publisher and digest you trust. Supply `origin`, the full SHA-256, the checkpoint's applicable license terms and the architecture and library version from the table.

`checkpoint.source` is an absolute local path or a path relative to its model YAML directory. `origin` is a public HTTPS URL or publisher identifier and is recorded as provenance. `library_version` pins the version of the consuming library. `revision` identifies the artifact release or commit. Every fresh initialization checks the requested architecture, library version, file digest, tensor names, shapes, and dtype compatibility before transfer. A native YOLO `.pt` can execute pickle code on load. The digest establishes byte identity and the trust decision belongs to the operator. Prefer safetensors or a flat tensor-only state dictionary obtained from a trusted publisher. The optional `pretrained-safetensors` extra provides the reader when it is absent from the backend environment.

`freeze_backbone` defaults to `false`. When enabled, YOLO freezes the YAML backbone layers and their batch-normalization statistics, Faster R-CNN freezes the MobileNet body, and RT-DETR freezes the ResNet body. FPN layers, transformer components outside the backbone, and heads can still train. Fine-tuning optimizer and LR overrides fall back to the model profile's `optimizer` and `lr`. Optional decay and momentum fall back to backend defaults. `momentum` controls SGD momentum or Adam/AdamW's first beta. YOLO SGD requires positive momentum and retains native Nesterov, warmup, accumulation and decay scaling.

Model YAML supports `workers` (default 0). Ultralytics forces zero workers on CPU and MPS and may cap requested workers on GPU. Set `gradient_accumulation` for Faster R-CNN or RT-DETR to the number of microbatches per optimizer step (default 1). COCO adapters weight microbatch losses by image count and step on a partial group at epoch end. YOLO uses its native accumulation schedule. Changing workers or accumulation requires a new training run. [Hardware recommendations and pilot workflow](gpu-training-pilot.md)

## Preview, run, and resume

```text
ict train --model rt_detr --config-dir profiles --dataset path/to/fixture/detection --dry-run
ict train --model rt_detr --config-dir profiles --dataset path/to/fixture/detection --approve-train
ict run --run-id fixture-transfer --model rt_detr --config-dir profiles --evaluation-image-size 64 --dry-run
ict run --run-id fixture-transfer --model rt_detr --config-dir profiles --evaluation-image-size 64 --approve-detection-train
ict runtime --workspace path/to/jobs preflight transfer --workflow train --model rt_detr --config-dir profiles --dataset path/to/fixture/detection
```

For Docker, put the cache under the profile directory and use a relative checkpoint path so both files are visible in the read-only profile mount. For SSH, stage the profile directory and cache on the target and use the target's paths. The runtime executor overrides the training device from its own `--device` option. Its run command uses `--approve-train` and supports an explicit prior model directory through `--model-run`.

`run.json` records `initialization_mode`, `pretrained_weights`, and `weight_transfer`. The latter includes source omissions and unexpected keys, incompatible keys, restored tied aliases, and the SMT tensors that were reset. The runner binds training to the initialization settings, verified file, code, runtime, and selected dataset and split revision. Keep the pinned file and YAML available for cache reuse and resume. Resume restores an ICT epoch checkpoint after verifying compatibility, retaining the original transfer diagnostics. Evaluation reconstructs the trained architecture from the saved configuration and reads the run's own checkpoint. It uses the same Train/Validation provenance, five classes, and shared COCO evaluation protocol for both initialization modes.

## Licensing

- Ultralytics publishes its YOLO models under AGPL-3.0 by default. Review source-disclosure obligations and the Enterprise terms before integrating or distributing a proprietary deployment or its fine-tuned weights. [Ultralytics model and Enterprise licensing](https://www.ultralytics.com/license)
- TorchVision's code uses BSD-3-Clause. TorchVision cautions that pretrained weights may have additional terms arising from the training data. Review the publisher's weight and COCO terms and record the applicable terms before use or redistribution. [TorchVision license](https://github.com/pytorch/vision/blob/main/LICENSE) · [TorchVision pretrained model terms](https://github.com/pytorch/vision#pre-trained-model-license) · [COCO terms](https://cocodataset.org/#termsofuse)
- Peking University's RT-DETR model card lists Apache-2.0. Retain applicable notices and license terms when redistributing, and review the training data terms for the intended use. [Checkpoint model card](https://huggingface.co/PekingU/rtdetr_r18vd) · [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

AI Hub usage conditions also apply when training on SMT data. [Project licensing and data permissions](licenses.md)
