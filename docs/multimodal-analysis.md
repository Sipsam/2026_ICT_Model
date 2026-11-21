# FPN leakage and sensor analysis

Run from the Ubuntu checkout after the analysis code is available there. Use the
existing ICT environment with the Faster R-CNN runtime and the exact cohort and
FrozenValidation artifacts used by the two completed runs. The paths for those
inputs below follow the FPN training profiles. Adjust them if the server stored
the files elsewhere.

```bash
cd /home/sipsam/2026_ICT_Model
results=/home/sipsam/ict-results
index="$results/smt-aihub-sensor-index-v1"

ict multimodal analyze \
  --root /home/sipsam/2026_ICT_Model/data/extracted \
  --audit "$results/sensor-integrity-analysis/leakage.json" \
  --cohort "$index/training-cohort.json" \
  --frozen-validation "$index/frozen-validation.json" \
  --image-run "$results/product-models/aihub-fpn-image-30e-v1" \
  --concat-run "$results/product-models/aihub-fpn-concat-30e-v1" \
  --mode all --device cuda --seed 42 --batch-size 1 \
  --checkpoint-dir "$results/fpn-generalization-v1-batches" \
  --out "$results/fpn-generalization-v1"
```

The command uses the selected checkpoint and selected Validation report recorded
in each completed run's `run.json`. For the runs above, the expected selected
epochs are 30 and 14 respectively. Check `analysis.json` for the selected epochs,
checkpoint hashes, frozen protocol and environment. The command reconstructs and
checks each run's Train and Validation views, uses the checkpoint's saved
preprocessor, and runs evaluation-only inference on the selected device. It uses
one device per invocation. No training or pretrained-weight download occurs.

`--mode leakage` verifies both runs and projects the existing audit without
loading the tensor runtime. `--mode groups` adds baseline scoring for both
models. `--mode ablation` scores the concat run only and accepts just
`--concat-run`. For these modes, choose a separate fresh output path. To resume
an interrupted command, rerun it with the same inputs, options and checkpoint
directory. Complete score batches are fingerprinted and published atomically.
Images are still checked against the cohort when a batch is reused. The final
report directory is published only after all requested reports are complete.

## Verified group-aware runs

Set `SOURCE`, `BUNDLE`, `INDEX`, and `VERIFY` to the paths used for group
training. Set `DETECTOR_BEST` to the original Detector checkpoint with its
adjacent `run.json` or `finalization.json`. Set `GROUP_RUNS` to the group
experiment directory and `results` to the directory containing the original
sensor integrity audit.

```bash
ict multimodal analyze \
  --root "$SOURCE" --audit "$results/sensor-integrity-analysis/leakage.json" \
  --cohort "$INDEX/training-cohort.json" \
  --frozen-validation "$INDEX/frozen-validation.json" \
  --group-verification "$VERIFY/verification.json" --group-bundle "$BUNDLE" \
  --detector-initialization "$DETECTOR_BEST" \
  --concat-run "$GROUP_RUNS/product/fpn-sensor-concat" \
  --mode ablation --device cuda --batch-size 1 \
  --checkpoint-dir "$GROUP_RUNS/sensor-ablation-batches" \
  --out "$GROUP_RUNS/sensor-ablation"
```

Add one `--prior-test-manifest /path/to/manifest.csv` per prior reservation
manifest recorded in the group verification receipt. For subgroup metrics, use
`--mode groups --image-run "$GROUP_RUNS/product/fpn-image-only"` and choose a
fresh output directory. Use `--mode all` with both runs to produce both reports
in one command.

The command verifies the group receipt against each run, rechecks the sealed
assignments and scoped Train/Validation source hashes, and checks the original
Detector provenance. Test contributes reservation metadata only. The input
`--audit` must be the original integrity audit. The `fpn-generalization-v1/leakage.json`
file is an analysis result. `analysis.json` records the audit fingerprint,
the provider split used for source identity, and the verified revision used
to assign Train and Validation roles.

## Analysis outputs

- `leakage.json` and `leakage.csv` report cross-split group membership and counts
  per model. In the JSON, `any_method_ids` is the union across findings.
  `image_similarity` and `sensor_similarity` are projected memberships in
  provider-audit connected components. Two retained members may have an
  indirect connection through excluded records. Direct cohort pair counts
  require recomputing edges and are unavailable from the existing audit.
- `groups.json` and `groups.csv` cover overall Validation, the two similarity
  strata, manufacturers, and capture dates. `manufacturers.json` and
  `manufacturers.csv` contain the manufacturer rows. JSON includes the exact
  membership for every row. Undefined metrics are JSON null and blank CSV
  cells. Confusion matrix rows are true negative/positive classes, and columns
  are predicted negative/positive classes. Macro values average the defined
  class metrics.
- `ablation.json` and `ablation.csv` compare native paired inputs, a seeded
  within-manufacturer shuffle, and a seeded global shuffle over the same
  Validation images and labels. The JSON records the recipient-to-donor map.
  Unknown manufacturers and single-member pools retain their sensor for the
  within-manufacturer condition. Removal uses the trained image fallback
  when available. A checkpoint with only trained paired weights records the
  removal condition as skipped.
- `summary.json` collects overlap counts and headline metrics, including
  concat intervention deltas from the paired baseline.

Capture date comes from the audited capture timestamp. Verified categorical
process conditions are unavailable in the current artifacts. Sensor swapping
can create out-of-distribution combinations. These runs selected their best
epochs on Validation, so the analysis quantifies sensitivity and variation on
that split. Independent generalization requires a separate held-out evaluation.
