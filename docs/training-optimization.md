# SMT training optimization

## Audit and implementation plan

- Reuse the generated COCO absolute boxes and canonical target IDs. The conversion validator already checks YOLO and COCO label equivalence. Empty target annotations are valid backgrounds.
- Use the published split membership as input. The group-aware exporter already records connected components and seals the converted formats. Join the existing candidate manifest for manufacturer and group reporting. Keep internal Test out of policy fitting, preview, and experiment scoring.
- Add a shared deterministic image and box transform with conservative geometry rejection and a training cohort selector based on Train image frequencies. Keep the current loaders and native YOLO augmentation when the new configuration is absent.
- Apply the shared policy on read for Faster R-CNN and RT-DETR. For controlled YOLO runs, stage selected Train images in the existing private working directory and use Ultralytics preprocessing with native random augmentation disabled. Keep the Validation path intact.
- Extend the current per-detector YAML, run metadata, and stage recovery fingerprints. Reuse checkpoint formats and exact predecessor checks for legacy runs.
- Provide read-only class reports and experiment previews, plus fresh output directories for pending ablations and annotated image examples. Verify with synthetic fixtures and mocked adapters.

## Configure a detector profile

Add an `optimization` section to a copy of the selected detector's YAML. These are conservative starting settings for fixed-camera RGB board images with sufficiently resolved defects and small fixture or lighting variations.

```yaml
optimization:
  augmentation:
    photometric:
      brightness: 0.08
      contrast: 0.08
      noise: 3
      saturation: 0.0
    geometric:
      rotation_degrees: 3
      translation_fraction: 0.02
      scale_delta: 0.03
      crop_fraction: 0.02
      min_visibility: 1.0
  balancing:
    method: weighted
    strength: 0.5
    max_weight_ratio: 4
```

| Setting | Starting range and assumptions | Valid max |
| --- | --- | --- |
| Brightness and contrast | 0–0.08 each. Sample independent multiplicative factors around 1. Compare shadowed joints and specular highlights. | 0.12 each |
| Noise | 0–3 digital levels on the 8-bit image, sampled per pixel and shared across color channels. This approximates mild intensity noise. | 6 |
| Saturation | Keep at 0 for AOI workflows that depend on calibrated lighting or color cues. Try 0–0.03 after confirming color is incidental to labeling. | 0.10 |
| Rotation | 0–3° about the image center for small camera or board offsets. | 7° |
| Translation and scale | 0–0.02 of each axis and 0–0.03 scale about center. Use settings consistent with observed fixture variation. | 0.05 and 0.08 |
| Crop | 0–0.02 per edge after affine geometry. Crop dimensions become the transformed dimensions and feed the backend's normal resize. | 0.05 |
| Visible box area | Default 1.0. For reviewed cases, values down to 0.95 permit clipping when nearly the entire transformed box remains. | 1.0 |

Horizontal flips default to zero. Enable `horizontal_flip_probability` only with `flip_policy: verified_mirror_invariant` after reviewing the entire Train cohort for component orientation, polarity markings, pad layout, viewing direction, and orientation-dependent labeling. A mixed cohort requires evidence covering every relevant source. Vertical flips and arbitrary orientation changes are unsupported.

Photometric proposals are rejected for the whole image when a target box gains more than 2% newly saturated RGB channel positions or its luminance standard deviation falls by more than 15% from a starting value of at least 2 levels. Geometry transforms all four corners and clips the polygon to the image before deriving its axis-aligned box. If any object would disappear, lose more visible area than allowed, or shrink below the size guard, the image keeps its original geometry. A valid photometric change can still apply. Malformed annotations fail validation, and normal images retain empty annotations. The padding color is neutral gray and images use bilinear resampling. These checks use box coverage and simple pixel statistics. Review small, specular, and edge-of-frame defects visually before adopting a policy.

An explicit `optimization: {}` selects the controlled baseline with no augmentation or balancing. Omitting `optimization` retains the existing backend behavior, including YOLO's native augmentation. With an explicit policy, YOLO disables its native random transforms and writes lossless Train PNGs and normalized labels in its private dataset directory. Repeated draws receive distinct private filenames. Faster R-CNN and RT-DETR apply the policy lazily over COCO samples. Both paths keep their respective deterministically resized Validation inputs. PNG staging can consume more disk than the original JPEGs. YOLO's native checkpoint and EMA resume rules still apply.

Each photometric and geometric proposal has its own deterministic seed derived from the configured seed, sample ID, and corresponding settings. Repeated visits to a sample, including oversampling and later epochs, replay the same proposal. Use matched, explicitly recorded seeds across variants when running repeated experiments. The training and stage fingerprints include the opt-in configuration, implementation, source annotation revision, and selected cohort or the inputs needed to derive it. `run.json` records the ordered Train cohort fingerprint and before/after class counts.

## Inspect class imbalance

```text
ict training-data imbalance --dataset data/processed/detection --manifest data/processed/manifest.csv --out outputs/provider-imbalance
ict training-data imbalance --dataset <split-bundle>/detection --out outputs/group-aware-imbalance
```

The command counts each target class once per image and counts every target box at object level. Multi-label images contribute to multiple image counts. A background is an image with zero target boxes. The report uses canonical class names and breaks out each split by manufacturer and group when available. For a sealed bundle, the command discovers the sibling `manifest.csv` and checks its digest and membership. Unsealed manifests provide descriptive metadata. Missing fields fall under `(unknown)`. Omit `--out` to print JSON. The default report covers Train and Validation. Use `--include-test` only for a final dataset audit of Test annotation counts. Keep Test counts and predictions out of experiment selection.

### Counts from the local provider conversion

The audit read 13,062 Train images and 1,656 Validation images, including 518 and 70 multi-label images respectively.

| Class | Train images | Train boxes | Validation images | Validation boxes |
| --- | ---: | ---: | ---: | ---: |
| Cold solder | 1,177 | 2,572 | 149 | 316 |
| Misalignment | 2,097 | 3,381 | 261 | 401 |
| Short | 1,451 | 1,661 | 192 | 210 |
| Missing component | 2,042 | 4,044 | 258 | 538 |
| Tombstone | 1,867 | 3,342 | 243 | 412 |
| Background | 4,950 | 0 | 624 | 0 |

Cold solder had the fewest positive Train images, and short had the fewest Train boxes. All 3,194 manufacturer B Train images were backgrounds. Manufacturer D contributed 1,410 of the 1,867 tombstone-positive Train images. Manufacturer-specific label support makes it important to check which sources a balancing draw amplifies. The provider manifest has no group IDs. A separate audit of the sealed 36-image smoke bundle verified 12 Train groups and 12 Validation groups.

### Balancing semantics

Both balancing methods draw a single fixed Train cohort of the original Train size, with replacement. They preserve every selected image's full label set. Samples may repeat or be omitted, so check the resulting per-class object and image counts and the unique-image count.

- `weighted` gives an image the largest rarity factor among its classes. A class's factor is `(largest class image count / class image count) ** strength`, capped by `max_weight_ratio`. The background bucket participates on the same terms. Multi-label images can increase exposure for common classes that co-occur with rare classes.
- `class_aware` multiplies each class bucket's image count by the same rarity factor, draws a bucket, then chooses a uniform member image. Multi-label images can be reached through multiple buckets. Class exposure after the draw need not match the bucket probabilities because each image keeps every target annotation.

Both methods estimate frequencies from Train only. They leave group and split assignments intact and can amplify the influence of producers or sessions concentrated in a rare class. They do not add new independent boards or guarantee class coverage. Validation performance, minority recall, and false positives on backgrounds remain empirical questions.

## Preview images and plan ablations

```text
ict training-data ablation --dataset <split-bundle>/detection --config-dir <pretrained-profiles> --dry-run
ict training-data ablation --dataset <split-bundle>/detection --config-dir <pretrained-profiles> --out outputs/smt-ablation
ict training-data preview --dataset <split-bundle>/detection --config-dir outputs/smt-ablation/combined --model yolo --out outputs/smt-preview --limit 5
ict train --all --dataset <split-bundle>/detection --config-dir outputs/smt-ablation/combined --dry-run
```

Use `--model` one or more times to restrict the ablation to specific detectors. The base profiles must pin staged local pretrained checkpoints and omit `optimization`. The planner verifies checkpoint bytes without loading weights or requiring an accelerator. It checks the label mapping and cross-format parity where both formats are present. Pass `--policy <file>` with the contents of the example's `optimization` section at the top level to customize the combined variant. Pass `--evaluation-config <file>` with the fields of `EvaluationConfig` to pin a common Validation protocol.

`experiment.json` records the selected split and annotation fingerprints, initialization weights, code, seed, detector-specific checkpoint selection rules, and common Validation protocol. Each of the five variant directories contains detector YAMLs accepted by `ict train --config-dir` and `ict run --config-dir`. These profiles store resolved local checkpoint paths. Preserve their recorded SHA-256 and update paths if moving to a new host. Each row is `pending` with null results. Publishing a manifest does not execute a run.

Compare each detector against its own pretrained baseline with the same seed, initialization, resize, batch, epoch budget, checkpoint selection rule, and evaluation settings. Use the common evaluator and the same Validation samples and labels across detectors. Use identical evaluation flags from the manifest when evaluating outside the pipeline. Keep Test reserved for final evaluation after freezing choices. The ablations measure the combined policy and each individual intervention against baseline. They do not isolate interaction effects. Realized accuracy, runtime cost, and generalization effects have not been measured.

Preview images show original and transformed boxes side by side. Use repeated `--sample-id` for targeted Train examples, including normal images, edge-of-frame defects, and samples from different manufacturers. `preview.json` records applied clipping, rejected proposals, original and transformed boxes, and policy provenance. If a source annotation is malformed, the command reports its index and refuses to publish a partial preview. Use a fresh destination outside the conversion and source directories.
