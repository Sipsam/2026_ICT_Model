# Group revision retraining

Use the existing published full split bundle and the matching sanitized product index on Ubuntu. The previous profiles name `~/ict-results/smt-aihub-split-v1` and `~/ict-results/smt-aihub-sensor-index-v1`. Confirm the actual server paths and pin the expected identifiers from the publication record and cohort receipt. The local 36-image smoke revision and the historical full split preview are compatibility and planning artifacts, respectively.

The local custody review at `outputs/dataset-release-review-custody/release-candidate.json` marks the source-day proposal `5f6aabd00368128e87cfd1e4abb0b019efa326a4ef08d4312c34220153ce66a2` blocked. Its comparison records eight previously reserved samples assigned to development. Treat this proposal as blocked and use the confirmed published revision.

## Verify the existing inputs

Set `BUNDLE`, `SOURCE`, and `INDEX` to the confirmed bundle, extracted source, and index directories. Set `REVISION` and `COHORT_SHA` to the confirmed `split.json` revision ID and `training-cohort.json` fingerprint. Use a fresh verification destination and a dedicated output root for this revision.

```bash
GROUP_RUNS="$HOME/ict-results/group-retraining/$REVISION"
VERIFY="$GROUP_RUNS/verification"
PROFILES=docs/profiles/group-retraining

ict integrity verify-group \
  --root "$SOURCE" --bundle "$BUNDLE" --revision-id "$REVISION" \
  --cohort "$INDEX/training-cohort.json" --cohort-fingerprint "$COHORT_SHA" \
  --frozen-validation "$INDEX/frozen-validation.json" \
  --preprocessing "$INDEX/sensor-state.json" --out "$VERIFY" \
  --checkpoint "$GROUP_RUNS/verification-observations.sqlite"
```

Add `--prior-test-manifest /path/to/prior/manifest.csv` for each earlier Test reservation applicable to these data. This checks sample IDs and recorded source paths against group Train/Validation before opening any development source files.

On retry, reuse the scoped SQLite checkpoint. The scan rechecks source bytes before reusing observations. Similarity comparisons and final COCO checks run again.

Review `verification.json` and proceed only after a zero exit status and `verification.passed: true`. The report includes publication and cohort identities, class support and candidate counts, isolation rules, exact matches by evidence type, Train/Validation near-match pairs, and manufacturer and capture-date distributions. Shared manufacturer or date and heuristic similarity have different interpretations from exact byte matches. Similarity gates isolation when the method is part of the published policy. The live scan uses the existing default thresholds and XLSX verification. Test contributes published counts and aggregate class support, permitted manifest metadata, sanitized hashes and group keys, and the publication's recorded residual grouping summary. Live Test similarity and the Test byte seal remain unverified. Missing production-session metadata limits production independence claims.

The command compares the exact group Train and Validation membership, image content, dimensions and boxes against the scoped COCO exports. It refits the saved preprocessing policy on Train to compare the statistics and reconstructs FrozenValidation from its pinned IDs. The receipt binds the applicable artifacts and scoped COCO bytes for subsequent runs. If artifacts are missing, obtain the existing sanitized export from the custodian. The broader analysis, multimodal audit, and ordinary full-seal Detector path open Test payloads.

## Train the group Detector after approval

Use the group receipt option on both the preview and the training invocation. The CLI uses the existing Faster R-CNN adapter with the selected COCO Train and Validation roles. The supplied YAML uses the existing scratch defaults with CUDA selected. If using an already approved official pretrained initialization profile, keep its existing pin and transfer policy. Both aligned product profiles must record that run's actual initialization layout and image size.

```bash
ict train --model faster_rcnn --dataset "$BUNDLE/detection" \
  --group-verification "$VERIFY/verification.json" \
  --config-dir "$PROFILES" --runs-dir "$GROUP_RUNS/detector" --dry-run

ict train --model faster_rcnn --dataset "$BUNDLE/detection" \
  --group-verification "$VERIFY/verification.json" \
  --config-dir "$PROFILES" --runs-dir "$GROUP_RUNS/detector" --approve-train
```

## Bind the two FPN profiles

After Detector completion, record the full path printed for the completed run's `best.pt` and calculate `sha256sum "$BEST"`. Keep the run's `run.json` beside it. Copy both YAML templates to the group experiment's configuration directory. Fill in the confirmed source, bundle, index and receipt paths and the fingerprints from `verification.json`. Use the **same** `best.pt` and SHA-256 in both profiles. Match `architecture.image.detector.layout` to `initialization_mode` and `image_size` to `configuration.image_size` in the completed Detector's `run.json`. The existing FPN loader checks the SHA-256, architecture and tensor layout on CPU. The group provenance check additionally validates the completed best checkpoint record, split revision and verification receipt.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for arm in fpn-image-only fpn-sensor-concat; do
  ict multimodal train --config "$GROUP_RUNS/profiles/$arm.yaml" \
    --out "$GROUP_RUNS/product/$arm" --dry-run || break
done
```

Once both previews pass, run the same commands with `--approve-train` in place of `--dry-run` after training approval. Both profiles retain the frozen FPN backbone, 30 epochs, and the existing optimizer and FrozenValidation selection settings. The paired arm retains the normalized-mean sensor MLP and concatenation.
