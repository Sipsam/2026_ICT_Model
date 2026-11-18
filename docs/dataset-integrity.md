# Dataset integrity and detection evaluation

The default detection workflow keeps the original AI Hub 71900 Train and Validation assignments. The integrity commands read the original manifest and its extracted JPG, JSON, and XLSX files. A split revision changes only the assignment of detection candidates in a new output directory. The five canonical target classes, background treatment, and the policy decisions `candidate`, `hold_review`, and `exclude` continue to apply under `aihub-smt-71900-v1`.

## Analyze the provider split

From the repository root, with the existing standalone manifest:

```text
ict integrity analyze --root data/extracted --manifest data/processed/manifest.csv --path-base . --verify-xlsx --out outputs/smt-leakage
```

For a pipeline manifest with paths relative to the extracted dataset, omit `--path-base .`. Omit `--out` to see summary counts without writing a report. Use a fresh output directory. The command scans all manifest rows, including holds and exclusions, and reports both provider-wide and candidate-only exposure. Paths in the report are relative to the extracted root.

For a resumable scan, add `--checkpoint outputs/smt-leakage.sqlite` and reuse that path on retry. The checkpoint records scan status and completed sample observations. The retry verifies the manifest, configuration, implementation and source bytes before reusing an observation. Similarity comparisons restart from the verified observations. Keep the checkpoint outside the report directory.

| Method | Evidence and interpretation |
| --- | --- |
| Exact JPG bytes | A cross-split SHA-256 match confirms image overlap. Inspect the associated annotations before deciding how to retain duplicates. |
| Sample ID | A case-insensitive filename stem match is potential overlap. IDs can be reused and need corroboration. |
| Near images | Matches require a 64-bit grayscale difference-hash distance of at most four and an RGB thumbnail mean absolute error of at most eight on the 0–255 scale. Both thresholds are configurable. Review the qualifying pair before treating it as a recapture or duplicate. Similar boards and acquisition setups may produce false positives. Cropped, rotated, or substantially altered images may escape the scan. |
| Production session | Repeated explicit `production_session_id` scoped by source confirms shared recorded session membership, subject to the correctness of the metadata. Conflicting values count as unknown. |
| Capture time and groups | Group by timestamp alone, source plus capture timestamp, source plus capture day, or explicit source-scoped `capture_group_id` when available. Filename tokens can supply source and local wall time. Capture matches are potential overlap and require production context. |
| Manufacturer and source | A shared manufacturer token or explicit `source_id` or `manufacturer_id` is an informational potential risk. These identifiers alone do not establish shared boards, captures, sessions, or leakage. |
| Sensor sequences | Exact matches use ordered timestamps and all five measurement channels. Near matches require equal relative timing and every measurement within its channel-specific absolute tolerance. Default tolerances for temperature, humidity, vibration, acceleration, and noise are 0.05, 0.1, 0.001, 0.001, and 0.1 in source units. `--sensor-scale` multiplies them together. The scan compares whole windows with the same cadence. It can miss partially overlapping windows or lagged measurements. |

Sensor similarity uses the allowlisted measurements and excludes `defect_detected`. The default uses the embedded sequence and hashes the workbook bytes. `--verify-xlsx` parses the workbook, allows XLSX-only evidence, and checks timestamps and values against the JSON when both are available. Timing disagreement or malformed measurements produce an evidence gap. Sensor/image ID offsets are normal metadata and do not serve as duplicate evidence.

`leakage.json` includes an integrity digest, normalized source and manifest fingerprints, method settings, per-sample coverage and issues, and findings for both within-split and cross-split groups. Group members index the `samples` array. Similarity findings are connected components, with qualifying pair counts, direct candidate cross-split pair counts, and an example pair with its distances. Component membership can be transitive. The report includes severity, evidence, and recommended handling. Unknown session provenance is recorded separately.

## Preview and publish assignments

```text
ict integrity split --root data/extracted --manifest data/processed/manifest.csv --path-base . --analysis outputs/smt-leakage/leakage.json --strategy aihub --dry-run
ict integrity split --root data/extracted --manifest data/processed/manifest.csv --path-base . --analysis outputs/smt-leakage/leakage.json --seed 42 --val-fraction 0.15 --test-fraction 0.10 --grouping capture --dry-run
```

The AI Hub strategy reproduces provider assignments and reports their remaining overlaps. Use a separate baseline destination when publishing. Group-aware planning assigns connected components together. It always connects exact images, repeated IDs, exact timestamped sensor sequences, explicit sessions and capture groups, and same-source exact capture times. `--grouping source-day` also connects source/day groups, and `--grouping manufacturer` additionally connects source and manufacturer identifiers. Review the similarity report before enabling `--group-near-images` or `--group-near-sensors`. Heuristic edges can join many samples through chains. Non-candidate observations can connect candidate groups, while their labels and original assignments stay outside the detection export.

The seeded allocator tries to cover each target class, then balance split sizes and the class and background sample distributions. It reports requested fractions, actual counts, groups supporting each class, the largest component, missing classes, and remaining cross-split findings. A class concentrated in fewer groups than requested splits cannot appear in all of them. The allocation is heuristic, so try alternate seeds and review group constraints if feasible coverage is missed. Publishing refuses empty splits and requires an explicit `--allow-missing-classes` if a group-aware split is missing any target class. Use `--test-fraction 0` for an internal Train and Validation revision without an internal Test.

After reviewing a dry run, repeat the chosen command with `--publish --out <fresh-directory>` in place of `--dry-run`. Add `--export-detection` to copy and validate both model formats. Publishing rechecks source and manifest fingerprints and writes a new `manifest.csv` and `split.json`. The CSV retains original split, decision, label metadata and source-relative paths. The JSON records assignments, seed, grouping, source and manifest fingerprints, policy and revision ID. The optional export contains `detection/split_revision.json`, which verifies both formats, including all exported Test images and labels. Use the revision ID and the content seal alongside a benchmark. Re-run analysis if source files change. Default preprocessing and standalone commands continue to accept their existing baseline artifacts.

## Evaluate the same cohort across detectors

1. Freeze one reviewed revision before training. For the DAG, select it explicitly with `ict run --split-bundle <bundle> --all --dry-run`. A saved run retains the bundle path for `ict resume`. For standalone workflows use `ict train --all --dataset <bundle>/detection --dry-run`. Only Train and Validation enter model fitting and checkpoint selection. The selected split and label fingerprints are checked against cached training results and checkpoint metadata.
2. Tune model settings, checkpoint rules, and evaluation thresholds with Train and Validation. Use one labeled COCO evaluation set for YOLO, Faster R-CNN, and RT-DETR, with identical sample membership, annotations, metric settings and class mapping. Report class support and any classes absent from a split. AP is undefined for a class with no ground truth.
3. Freeze checkpoints and evaluation settings before opening Test. An internal Test comes from the same 71900 cohort and can support an internal group-holdout estimate. Use `ict evaluate --dataset <bundle>/detection --split test --held-out` with one `--run <model-directory>` per detector. Reports identify this as `internal_group_holdout`. The attestation confirms that Test was kept out of tuning. It does not certify production-session independence.
4. For an external independent Test, obtain separately collected and labeled production data with traceable acquisition records. Apply the same five-class label policy, review exact matches, near matches, and session overlap against the training reference, and document the provenance. Submit the separately prepared labeled COCO Test directory through `ict evaluate --split test --held-out --dataset <external-test> --reference-dataset <training-detection>`. The evaluator verifies reference fingerprints, Test filename and byte disjointness, and available training label provenance. The input's independent collection and production-session boundaries require external evidence.
5. Pass the three saved JSON model reports to `ict compare`. Compare reports with matching sample and annotation fingerprints, role, and evaluation protocol. Include the label and split revision, test origin, timings and hardware, and model-specific checkpoint selection rules when reporting results.

Changing a sealed image, label file, split revision, or label policy invalidates the corresponding selection. Unsealed legacy checkpoints remain supported through their available recorded split and label fingerprints. Incomplete historical records carry less provenance than a sealed revision.

`ict training-data imbalance` joins the published manifest to the converted annotation counts and reports support by split, manufacturer and group. Train-only balancing leaves split assignments intact. Keep Test closed during ablation planning and tuning. [Class counts and Train cohort selection](training-optimization.md)

`ict multimodal audit` accepts the provider manifest or a published
`manifest.csv` and `split.json`. It binds verified sensor windows to that
assignment and records group quarantines before fitting sensor preprocessing.
[Multimodal coverage and dataset API](multimodal-data.md)

## Verification

Run the fixture and optional local regression tests with the existing environment:

```text
python -m pytest -q tests/test_integrity.py
```

For the real-data regression, set `ICT_REAL_DATA_ROOT` and `ICT_REAL_DATA_MANIFEST` to the extracted dataset and its manifest. If its paths are repository-relative, set `ICT_REAL_DATA_PATH_BASE` to the repository root. The integrity regression is read-only and uses embedded sensor sequences. The XLSX analysis is available through the CLI command above.

### Local 71900 findings

The XLSX-verified scan inspected all 15,372 manifest rows and used 14,718 eligible candidates for planning. The original candidate counts were 13,062 Train and 1,656 Validation.

| Cross-split evidence | All manifest rows | Groups with candidates on both sides |
| --- | ---: | ---: |
| Exact JPG, repeated sample ID, explicit session or capture group, or exact or near sensor sequence | 0 groups per method | 0 groups per method |
| Near images | 45 components and 52 direct pairs | 45 components with 54 Train and 46 Validation candidates |
| Capture timestamp | 40 groups | 38 groups |
| Source and capture timestamp | 11 groups | 11 groups |
| Source and capture day | 7 groups | 7 groups |
| Manufacturer | 4 groups | 4 groups |

All 15,372 rows lack a usable explicit production-session identifier. The four shared manufacturer groups cover all candidates. The image matches need visual and production-record review. Actual production-session independence cannot be established from this metadata.

The seed-42 capture-group preview with both near-match options enabled assigned 11,038 Train, 2,208 Validation, and 1,472 internal Test candidates. Every split contained all five classes and background images. The largest selected group contained four candidates, and the enabled grouping methods left no candidate component spanning splits. Broad manufacturer and source/day overlaps remained. Grouping by manufacturer left only two groups supporting each of `cold_solder` and `short`, which precludes all five classes appearing in three splits under that strategy. These were dry-run results. The real-data assignments were kept in the existing AI Hub split.
