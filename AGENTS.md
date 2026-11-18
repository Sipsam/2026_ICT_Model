# ICT pipeline contributor instructions

These rules apply throughout the repository. The cross-platform Python pipeline covers AI Hub dataset download and extraction, dataset audit, manifest generation, YOLO/COCO conversion and validation, model training and validation, evaluation, and reporting. The dataset is AI Hub SMT manufacturing dataset `71900`, with images, JSON annotations, and XLSX sensor files.

## Inspect and reuse existing work

- Before editing, read `PLANS.md`, check `git status`, and inspect the relevant entry points, helpers, and data contracts. Preserve in-progress edits and keep changes within the requested phase and file scope.
- Start with the existing implementations. Inspect `scripts/download_data.sh` and `tools/aihubshell` for acquisition, `scripts/migrate_dataset_paths.py` for path mapping, `scripts/validate_dataset.py` for audit, `scripts/build_manifest.py` for manifest generation, `scripts/convert_detection.py` for YOLO/COCO conversion, and `scripts/validate_detection.py` for conversion validation. Check `src/data/validate.py` and the sensor and visualization scripts when their behavior is relevant.
- Reuse working logic before refactoring. Preserve existing scripts and their entry points, arguments, and artifact contracts. When extracting reusable functions, keep compatibility wrappers and verify behavior on small inputs.
- Treat the `ict` CLI, packaged pipeline, and later training and evaluation interfaces in `PLANS.md` as roadmap items until their implementations are verified in the working tree.

## Architecture and portability

- Keep CLI handlers focused on argument parsing, configuration, presentation, and exit codes. Put typed stage logic in importable Python modules. Isolate shell commands, external tools, OS and device probing, and container or remote execution in adapters.
- Use `pathlib.Path` for filesystem paths. Resolve configurable roots explicitly and use portable relative paths in manifests where the existing contract permits. Avoid hard-coded drive letters, mount points, home directories, and shell assumptions. Apply OS-specific path translation at adapter boundaries.
- Design for Windows, Linux, WSL2, and macOS. Keep CLI help, diagnostics, preprocessing, and fixture tests usable on CPU without importing optional training dependencies. Report unavailable platform or device capabilities explicitly.

## Protect data and credentials

- Treat source archives and images, JSON annotations, and XLSX sensor files as read-only. Preserve `data/raw` and source files under `data/extracted`. Write derived datasets, run state, and reports under configured output roots such as `data/processed`, `runs`, and `outputs`. Never edit an image through a hard link to a source file.
- Load credentials from environment variables or a local untracked `.env`. Use placeholders in examples. Keep credentials out of Git, command output, logs, errors and tracebacks, artifacts, and reports. Sanitize subprocess diagnostics and saved config or run metadata before emitting or persisting them. Keep secret values out of command-line arguments where the adapter supports a safer channel.
- Require explicit user approval before starting large downloads, including datasets or pretrained weights, any installs, model training, deletion, or other destructive operations such as overwriting existing artifacts or applying dataset path migrations. This applies to direct calls, subprocesses, retries, and non-interactive runs. Default to safe previews where supported and show scope and destinations before a gated action.

## Make long tasks observable and resumable

- Show the active stage, completed and total work when known, elapsed time, and an ETA when estimable. Use structured logs and actionable errors with the failed stage and relevant non-secret context. Return a failure status when execution fails.
- Persist stage status and checkpoints so interrupted jobs can resume. Verify output integrity and check input and configuration fingerprints before reusing completed work. Use temporary outputs and finalize them atomically where possible. Keep partial artifacts distinguishable from completed results and handle cancellation and retries without corrupting completed outputs.

## Preserve split and label integrity

- Carry the original Train and Validation assignments through audit, manifest generation, conversion, training, and evaluation. Fit preprocessing, feature selection, and resampling on Train only. Check for cross-split duplicates and shared production groups or sensor sessions when provenance permits. Keep any independent Test split out of training and tuning and identify the split in reported metrics.
- Never include `defect_detected` or any feature derived from it in model inputs, including sensor-only and multimodal inputs. Use it only as a label where the task defines it as such. Exclude other target-derived data and signals unavailable at inference time.
- Preserve the five target defect classes and existing class mappings, normal images with empty target annotations, and the manifest decisions `candidate`, `hold_review`, and `exclude`. A `candidate` row is eligible for conversion under the existing policy and may still require ground-truth review. Validate sensor alignment with filenames, timestamps, and values where available. Sensor and image IDs can legitimately differ.
- Record the data and split fingerprints, class and label policies, seeds, resolved non-secret configuration, code version, runtime, and checkpoints needed to reproduce training and evaluation. Compare models on the same evaluation samples and metrics.

## Test and report actual results

- Add type annotations to new or changed Python interfaces. Write focused `pytest` tests for changed behavior with tiny synthetic fixtures, `tmp_path`, and mocks for network, subprocess, and accelerator calls. Keep default tests CPU-only and independent of real datasets, credentials, and full training. Cover relevant failures and regressions, including split preservation, restricted feature selection, and interrupted-stage recovery when affected.
- Report checks as **Passed**, **Failed**, or **Not run**. Include executed commands and observed outcomes, and explain skipped checks or unavailable environments. Treat historical dataset baselines in `PLANS.md` as comparison references. Never invent test results, dataset counts, or metrics for the current work.
- Close each task with changed files, test outcomes, limitations, and next-step interfaces. Name the CLI entry points, Python interfaces, or artifact contracts the next phase can rely on and identify any that remain planned. If a commit is requested, use an English Conventional Commit and record the actual test outcomes.
