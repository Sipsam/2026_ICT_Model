# Product training with images and sensors

`ict multimodal train` runs product classification on a reviewed SMT dataset revision. It uses the existing `MultimodalDataset`, `collate`, `MultimodalModel`, `product_targets`, and `product_loss` interfaces. It adds split-scoped batch iteration, AdamW and scheduling, approval and resource checks, epoch recovery, frozen Validation scoring, and experiment manifests. The detector adapters retain ownership of spatial training, detector checkpoints, original-image boxes, and mAP.

## Prepare inputs

1. Select a reviewed split bundle already published with `split.json`, `manifest.csv`, and `detection/split_revision.json`. Use the exact revision ID. If a new bundle is needed, review the split preview and approve publication separately. [Dataset integrity and revision rules](dataset-integrity.md)
2. Have the data custodian audit the same revision. Use the extracted source root and a fresh index directory. Keep the full audit index in custody when it contains internal Test. Distribute the training index from the next step.

   ```text
   ict multimodal audit --root data/extracted --manifest outputs/reviewed-bundle/manifest.csv --revision outputs/reviewed-bundle/split.json --export-index --out outputs/reviewed-product-index
   ```

3. Fit preprocessing on eligible Train records and save the fingerprinted state. Use the same state for experiments that share a scoring protocol. Supply the view of the dataset you plan to deploy, including the window policy.
4. If scoring Validation, choose verified paired Validation IDs before seeing model results. Use the same IDs for sensor, image, and paired comparison arms. The runner requires complete product labels and every requested modality for these IDs. Omit the `validation` block for a fixed run with no Validation scoring.

```python
from pathlib import Path

from ict_pipeline.evaluation.product import write_artifact
from ict_pipeline.evaluation.product_protocol import FrozenValidation
from ict_pipeline.multimodal.audit import Cohort
from ict_pipeline.multimodal.preprocessing import PreprocessingConfig, fit_preprocessor
from ict_pipeline.data.preprocessing.common import write_json

index_dir = Path("outputs/reviewed-product-index")
cohort = Cohort.load(index_dir / "cohort.json").for_product_training()
write_json(index_dir / "training-cohort.json", {
    "cohort": cohort.model_dump(mode="json"),
    "fingerprint": cohort.fingerprint,
})
state = fit_preprocessor(cohort, PreprocessingConfig(scaling="robust"))
write_json(index_dir / "sensor-state.json", state.payload())
print(cohort.revision_id, cohort.fingerprint, state.fingerprint)

# Supply one reviewed Validation sample ID per line.
ids = tuple(Path("validation-ids.txt").read_text(encoding="utf-8").splitlines())
frozen = FrozenValidation.from_audit(cohort, ids)
write_artifact(index_dir / "frozen-validation.json", frozen)
print(frozen.identity.cohort_fingerprint)
```

Planning checks the publication receipt against the split and manifest, verifies the pinned cohort, recomputes source and assignment fingerprints from the index, and hashes the source bytes used by eligible Train and explicitly selected Validation records. The sealed detector conversion files are outside this product input path. The training index retains held-out assignment, group and hash metadata for isolation checks. The planner rejects an index that includes Test measurements or annotations. It opens no internal Test source or converted files, creates no Test dataset, and never scores Test. Register any future held-out evaluation separately after model selection and thresholds are frozen. [Split and label policy](pretraining-readiness.md#training-and-holdout-policy)

## Profiles and approval

Use JSON or YAML. Resolve relative paths from the working directory. Replace the fingerprint placeholders with the values prepared above. Set an epoch budget and threshold before viewing benchmark results.

```yaml
schema_version: 1
source: data/extracted
bundle: outputs/reviewed-bundle
revision_id: "<revision-id>"
cohort: outputs/reviewed-product-index/training-cohort.json
cohort_fingerprint: "<cohort-fingerprint>"
preprocessing: outputs/reviewed-product-index/sensor-state.json
branch: sensor
architecture:
  preprocessing_fingerprint: "<sensor-state-fingerprint>"
  seed: 42
  target_policy: smt-target-defect-presence-v1
  sensor:
    kind: mlp
    features: normalized_mean
  image:
    kind: rgb_grid
  fusion:
    kind: late
    threshold: 0.5
missing_train: error
optimizer:
  algorithm: adamw
  learning_rate: 0.001
  weight_decay: 0.0001
scheduler:
  kind: constant
epochs: 5
batch_size: 4
gradient_accumulation: 2
device: cpu
cpu_threads: 1
checkpoint_policy: final_epoch
validation:
  cohort: outputs/reviewed-product-index/frozen-validation.json
  fingerprint: "<frozen-validation-fingerprint>"
  frequency: final
```

| Branch | Settings and behavior |
| --- | --- |
| Sensor MLP | `branch: sensor`, `sensor.kind: mlp`. Use `normalized_mean` or `statistics` on observed channels. Start with a zero-second window. |
| Sensor GRU | `sensor.kind: gru`, `sensor.features: normalized_mean`, and `min_steps` of at least two. Each admitted window must have that many distinct timestamps with observed measurements. Padding and imputation do not count. |
| Image-only | `branch: image` with the RGB grid and default MLP settings. Uses verified RGB images and the original annotations to project product labels. A frozen Train sensor state is still required by the shared checkpoint contract. |
| Paired late | `branch: paired`, `fusion.kind: late`. The configured fixed-weight logit average trains both unimodal heads jointly on the paired objective, with no learned fusion weights. |
| Paired concat or gated | `branch: paired`, `fusion.kind: concat` or `gated`. The paired objective updates the encoders and fusion path. Unimodal fallback heads are marked untrained. |

Use `missing_train: drop` to explicitly omit rows with missing modalities, invalid targets, or too few observed timestamps. The manifest records used and candidate counts. Ambiguous pairing is rejected for eligible sensor or paired rows. Audited quarantines are excluded. `smt-target-defect-presence-v1` projects the presence of any canonical target class. `smt-view-class-presence-v1` projects five independent class-presence labels. Empty canonical annotations represent the negative case. Inputs exclude `defect_detected`, identifiers, and bounding boxes. Product checkpoints emit presence scores. Spatial predictions still come from the existing detector.

`scheduler.kind: step` multiplies the learning rate by `gamma` every `step_size` epochs. Accumulated gradients are weighted by microbatch row counts, including the last partial group. The current runner uses deterministic FP32 operations, seeded initialization and shuffling, and RGB-grid image encoding. Pooled Faster R-CNN training would require a separately pinned product initialization and transform contract. Its existing encoder and inference interfaces remain available.

For live capture-time GRU use a causal lookback with verified availability by capture. A positive capture-through-decision window allows a separate offline experiment at the declared decision horizon. Its report binds the audited source and frozen window, and uses a distinct protocol from capture-time serving. The existing provider capture-time state yields only one observed step. [Sensor window and batch contracts](multimodal-data.md)

```text
ict multimodal train --config sensor-mlp.yaml --out runs/product/sensor-mlp --dry-run
ict multimodal train --config sensor-mlp.yaml --out runs/product/sensor-mlp --approve-train
ict multimodal train --config sensor-mlp.yaml --out runs/product/sensor-mlp --resume --dry-run
ict multimodal train --config sensor-mlp.yaml --out runs/product/sensor-mlp --resume --approve-train
```

The approval commands are for future authorized execution. Dry-run prints the resolved pending manifest and readiness as JSON and writes no run directory. Device, memory, and disk checks run on the execution host. For a future remote pilot, stage immutable inputs and the repository on that host, use target-visible paths, and run the same CLI there. The detector-oriented `ict runtime` job protocol has no product job type. For deterministic CUDA execution, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching. The device probe checks generic operators and estimates storage and memory. A separately approved pilot must verify the actual product workload on the intended accelerator. [Hardware pilot and runtime context](gpu-training-pilot.md)

Start with a reviewed development cohort and fixed-epoch sensor MLP and RGB baselines. Then compare late, concat, and gated runs using the same frozen Validation protocol. Where a controlled comparison calls for identical Train samples, use an approved common paired revision with identical branch eligibility. Consider a GRU only after confirming observed temporal coverage and decision time. Keep final benchmark Validation independent of model-selection experiments.

## Checkpoints and experiment reports

- **Final epoch** selects the last configured epoch. Validation is optional. Set `frequency: final` to score only after the last Train epoch, or explicitly choose `each_epoch` for diagnostics.
- **Best Validation** requires `frequency: each_epoch`. Select the highest `macro_f1` or `macro_auroc` via `selection_metric`. Macro scores average defined class metrics. Undefined epochs are ineligible and ties select the earlier epoch. Selection fails if every epoch is undefined. Use this policy only where the frozen Validation set is designated for selection.
- `run.json` records the resolved profile, dataset and split digests, source and preprocessing identities, class policy, seed, code and runtime, readiness, status, selected checkpoint, and report hashes. Unexecuted plans report `pending`. Completed runs without Validation report `completed_unscored`.
- Each committed `epoch-NNNNNN/` contains inference `model.pt`, a separate tensor-only `resume.pt` with optimizer, scheduler and RNG, and a chained `receipt.json`. Scored epochs also contain a fingerprinted `validation.json` report. The run directory stores the exact `sensor-state.json`. Use a writable local filesystem with atomic rename and hard links.
- Resume appends to the same run under a lock. It checks the pinned profile, dataset, code, runtime, device, frozen preprocessing, receipt chain and artifact hashes. It reloads the latest committed epoch even when an earlier epoch is selected for inference. Incomplete directories do not count as checkpoints, so an interrupted epoch is replayed. Changed training settings require a fresh run.
- Product reports use the canonical per-output confusion counts, precision, recall, F1 and AUROC on the exact frozen IDs. Undefined metrics are JSON `null`. Detector box mAP uses its own report. Use `compare_experiments` for compatible product reports. Inference takes the selected `model.pt`, matching state and manifest hashes. [Checkpoint inference and report comparison](multimodal-inference.md)

The runner and artifact contracts are available through `ProductTrainingConfig`, `plan_training`, `inspect_recovery`, and `execute_training` in `ict_pipeline.multimodal.training_plan` and `ict_pipeline.multimodal.training`. This implementation was checked with CPU fixtures and mocked execution. Native fitting, optimizer trajectories across an actual interruption, trained-model metrics, remote GPU operation, and production inference with trained weights remain unverified.
