# ICT Pipeline maintenance instructions

This repository maintains the AI Hub 71900 PCB SMT detector and multimodal
classification pipeline. Read the relevant implementation, public command,
artifact contract, and tests before editing. Record Git status, preserve existing
edits, and consult local `PLANS.md` when available as historical context.

## Follow the existing module boundaries

- `cli.py` and domain `commands.py` modules parse arguments, load configuration,
  present progress, and map errors to exit codes. Keep typed execution in the
  domain modules and platform operations in the existing adapters.
- `data/preprocessing/` owns audit, Manifest, leakage evidence, group assignments,
  YOLO/COCO conversion and split publication. `data/` also owns archive handling,
  group verification, detector finalization and Test Release Gate receipts.
- `models/` owns the detector registry and backend training. `classification/`
  contains the earlier classification workflows. `multimodal/` owns verified
  pairs, Train-fitted preprocessing, encoders, training, checkpoint validation,
  and Validation analysis. `evaluation/` owns evaluation protocols and metrics.
- `pipeline/`, `runtime/`, and `reporting/` own stage state, execution adapters,
  and run metadata. `inspection/`, `api/`, and `streaming/` provide inference
  interfaces. Preserve the installed `ict` entry point and supported script or
  Python interfaces. Reuse existing code before moving or generalizing it.
- Type new and changed Python interfaces. Use `pathlib.Path`, portable relative
  paths where the artifact contract permits them, and explicit I/O roots. Keep
  command help and CPU fixture workflows available without optional backends.

## Preserve scientific records and isolate splits

- Treat `data/raw/`, source files in `data/extracted/`, and all of
  `outputs/final/` as read-only. Never modify an image through a hard link.
  Write derived files under separately configured roots. Keep final artifacts,
  source sample IDs, checkpoints, raw data and machine-specific metadata out
  of commits. Cite public aggregate results and artifact fingerprints.
- Preserve canonical class order, source and revised split identity,
  `candidate`/`hold_review`/`exclude`, and normal images with empty target
  annotations. Pair files with verified names, timestamps and value agreement.
  Image IDs and sensor IDs may legitimately differ.
- Keep `defect_detected` and derived or unavailable-at-inference fields out of
  model inputs. Fit preprocessing, balancing, and feature selection on Train.
  Freeze Validation membership and thresholds. Compare models on the same
  samples and metric protocol. Inspect explicit session and capture-group
  evidence, exact hashes and similarity across splits.
- Use the existing published revision and verify scoped Train/Validation
  inputs. Preserve membership, publication and cohort fingerprints, frozen
  preprocessing, resolved non-secret settings, seed, code/runtime identity,
  and checkpoint hashes. Load only trusted, pinned checkpoint formats and
  verify the original run or finalization receipt. Preserve finalized
  checkpoint bytes, thresholds, metrics and receipts.
- Keep Test out of development and tuning. The strict Test Release Gate requires
  its existing preflight and trusted historical seal evidence. Limited Internal
  Holdout is a separate same-source protocol with explicit Test access approval.
  Preserve historical reservation and seal status. Never open Test payloads
  without approval or use Test outcomes to choose checkpoints, thresholds,
  preprocessing, model settings or membership. Preserve Test access history.

## Operate and verify safely

- Use environment variables or an ignored `.env` for credentials. Sanitize
  subprocess output, diagnostics, reports and saved settings. Use placeholders
  in templates and command examples.
- Require explicit approval before installs, large downloads, training,
  deletion, overwrites, or applying data migrations. Preview scope and
  destinations. Verify exact paths before any authorized cleanup and preserve
  recoverable state. A research publication task uses stored metrics.
- Keep long stages observable and resumable. Report progress, elapsed time and
  ETA when estimable. Check input, configuration and artifact fingerprints,
  publish atomically where supported, and return failure on errors.
- Add focused synthetic `pytest` coverage for changed behavior. Preserve existing
  contracts around filesystem safety, alignment, leakage, checkpoint
  provenance, deterministic scores, batch recovery and both Test protocols.
  Default fixture checks use CPU, mocks and `tmp_path`. Treat real-data and
  experimental training as separate approved activities.
- Validate the affected commands, compilation, focused tests, full applicable
  suite, links and scope. Maintain the [documentation index](docs/README.md) and
  update example commands with interface changes. List actual checks as
  **Passed**, **Failed**, or **Not run**, with skipped platforms and limitations.
  Review the complete staged diff for scientific payloads and unintended files.
  Use English Conventional Commits and record actual test results.
