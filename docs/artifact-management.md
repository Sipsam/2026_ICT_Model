# Artifact management

Treat the entire local `outputs/final/` tree as a read-only research archive.
Keep both compressed and extracted artifacts outside Git. Use a different
destination for generated figures, test outputs and analysis runs. The final
report cites preserved receipts and publishes aggregate values.
[Final research report](final-report.md)

## What filled the Windows temporary drive

The reported audit found about 4.6 GiB free on a 292 GiB C: volume, with about 31.6 GiB in pytest temporary directories and another 9.8 GiB in ICT integration and E2E directories. Individual RT-DETR checkpoints were about 302 MiB. The current audit found no matching historical pytest or ICT directories under the Windows account's `TEMP`. The original inventory is user-reported and the earlier directories could not be attributed to specific tests from their files.

- The former pytest configuration used pytest's defaults. Its numbered `pytest-of-<user>/pytest-N` layout keeps the most recent three sessions under its default `all` policy. Each retained session can be gigabytes. Active and recently interrupted sessions, separate users and roots, and manually supplied `--basetemp` locations have separate lifetimes. A stock pytest invocation clears an existing explicit `--basetemp` at startup. [pytest temporary directory documentation](https://docs.pytest.org/en/stable/how-to/tmp_path.html)
- Standalone tiny training, interrupted resume, the real detector DAG, and synthetic fine-tuning each run YOLO, Faster R-CNN, and RT-DETR when their extras are present. The shared test profile used `save_period: 1` even where a test only needed best and last. The interruption test kept two model runs and a private pair of resume input copies. The DAG kept recovery weights and another published pair, plus YOLO's private dataset copy. Mocked training tests already used small placeholder files.
- Repository tests originally used function-scoped `tmp_path`. No existing test called `tmp_path_factory` directly. Factory-created shared paths now have session lifetime, and the regression suite exercises both scopes. Real SMT smoke runs intentionally write to `ICT_REAL_SMOKE_OUTPUT`, which holds persistent experiment results outside pytest's lifecycle.
- The runtime executor stores job state and working outputs in its configured target workspace. Pipeline `work/`, `recovery/`, and `artifacts/` support retries and reproducibility. The report exporter uses a unique directory under its explicit output root. Extraction, integrity reports, pretrained weight staging, evaluation scratch, and runtime snapshot staging use scoped temporary directories or sibling temporary files. A process kill can bypass their cleanup. The historical manual ICT integration and E2E directories have no registration in the repository and are outside the cleanup interface.

## Test directories

On PowerShell, start with the lightweight suite and choose a parent directory on a volume that can hold the largest synthetic model test.

```powershell
$env:ICT_TEST_ARTIFACT_ROOT = 'D:\ict-test-storage'
python -m pytest -m "not cpu_backend and not real_data_smoke"
python -m pytest tests/test_training.py tests/test_e2e.py -m cpu_backend --ict-artifact-summary
```

Run the second command after authorizing tiny CPU fixture training. It performs one- and two-epoch training on synthetic images, verifies real checkpoint deserialization and periodic snapshot bytes, interrupts and resumes training, and exercises publication and cache reuse through the DAG. Packages must already be available. The `cpu_backend` marker also covers the synthetic fine-tuning cases in `tests/test_pretrained.py`. A plain `pytest` still includes these tests when the backend packages are installed.

Pytest chooses the artifact parent from `--ict-artifact-root=PATH`, `--basetemp`, `ICT_TEST_ARTIFACT_ROOT`, `PYTEST_DEBUG_TEMPROOT`, or the OS temporary directory, in that order. The default parent on POSIX is an account-specific subdirectory of the shared temporary directory. Under this repository's test configuration, `--basetemp` is a parent and keeps its existing contents. Use the equals form of the custom flag when omitting test paths so pytest can discover the repository configuration. Every invocation gets a fresh child under `ict-test-artifacts-v1/sessions/`. The session owns its pytest fixtures and Python scratch directory, including subprocesses that honor `TEMP`, `TMP`, or `TMPDIR`. Tests also use the repository's ordinary lightweight `.pytest_cache` for pytest bookkeeping. Managed sessions currently use one pytest process and reject xdist parallel workers. Separate pytest invocations get separate leases.

| Outcome | Retention |
| --- | --- |
| Completed fixture | Delete its `tmp_path` after teardown. |
| Successful session | Delete its session-scoped fixture and scratch files. |
| Failed or gracefully interrupted session | Delete its disposable files by default. With `--ict-keep-failed`, keep failed fixtures and session scratch. |
| Process killed or cleanup blocked by an active job or file handle | Keep the registered directory for inspection. The OS releases the process lease. The preview identifies an abandoned running session as interrupted after the lock is released. |

`ICT_TEST_SESSION_LIMIT` defaults to eight concurrent or retained sessions per parent. It bounds session accumulation after repeated kills. When the limit is reached, a new invocation reports how to inspect retained sessions. Use a private local filesystem directory for managed storage. A single invocation's size still depends on its tests and filesystem capacity. `--ict-artifact-summary` reports per-fixture logical file bytes, bytes counted once per file identity, and checkpoint counts and logical bytes. Filesystem allocation blocks, compression, and other processes are outside that measurement.

## Training storage and persistent runs

Use `ict train --runs-dir PATH` for standalone model artifacts. Integrated runs are stored under the selected workspace's `runs/` directory, and runtime jobs use their target workspace. To move YOLO's writable copy and label caches, set `ICT_TRAIN_CACHE_ROOT` on the training host. A relative value resolves against the training workspace. Local execution inherits the environment. SSH reads target configuration, and containers use paths available to the worker. Standalone examples use PowerShell.

```powershell
$env:ICT_TRAIN_CACHE_ROOT = 'D:\ict-training-scratch'
ict train --model yolo --dataset 'D:\datasets\detection' --runs-dir 'D:\ict-runs' --dry-run
```

YOLO copies the images and labels because its decoder can repair JPEGs in place. An external cache uses a new private directory which is released at normal exit. A hard-killed training worker may leave that external scratch directory behind. Inspect it with the owning run stopped and obtain approval for the exact path before removing it manually. Without an external cache, the private copy remains in the original run. Dataset inputs, weights, and previous runs are treated as read-only.

The preflight compares measured free bytes with an estimate on each involved filesystem. It counts the YOLO input files and available resume or initialization files by measured byte size. It uses planning allowances of 64 MiB per YOLO checkpoint, 320 MiB per Faster R-CNN checkpoint, and 512 MiB per RT-DETR checkpoint. It budgets for best, last, staging, configured periodic saves, DAG publication, YOLO cache metadata, and a 64 MiB reserve. It assumes checkpoints consume separate space even when hard links are available. The estimate is deliberately conservative and concurrent writers can still consume available space.

`ict train --dry-run` prints storage estimates with the model scope. `ict runtime preflight` reports storage readiness separately from device readiness and exits unsuccessfully when either check fails. Docker previews use the host space for the mounted job output. Training checks the actual destination volumes again inside the worker.

Controlled YOLO profiles stage lossless Train PNGs in the same private cache and budget for each oversampled occurrence. The `ict training-data` commands publish imbalance reports, augmentation previews, and pending experiment profiles into fresh output directories. [Training optimization artifacts](training-optimization.md)

Training checks storage during progress and before epoch checkpoint writes. Files are serialized and snapshots are staged before replacing prior checkpoint names. An interrupted publish still has to pass fingerprint validation before recovery. YOLO retains the native checkpoint format and optimizer state. Its pinned writer serializes once and writes both temporary names on a best epoch. After serialization the final names can share an inode. Faster R-CNN and RT-DETR serialize one file per epoch and stage identical best or periodic files from it. The adapters replace names atomically, so later updates to last leave earlier epochs intact. On filesystems where hard links are unavailable, each snapshot gets a copy. Treat checkpoint files as immutable. To edit or transfer a checkpoint independently, first make a separate copy.

DAG publication keeps a separate copy from recovery storage. Best and last in the published attempt can share storage when fingerprints match. Resume uses a private verified input copy within the new run and removes that temporary staging directory at normal exit. Recovery roots, periodic checkpoints requested in a training profile, published reports, and opt-in SMT experiment outputs remain persistent.

## Preview and exact-target cleanup

```powershell
ict clean --dry-run --root 'D:\ict-test-storage'
ict clean --root 'D:\ict-test-storage' --older-than-hours 0
ict clean --root 'D:\ict-test-storage' --older-than-hours 0 --delete <exact-session-id> --confirm
```

Omit `--root` when `ICT_TEST_ARTIFACT_ROOT` already selects the parent. The preview is the default. It shows the registered session path, recorded owner, state, logical byte size when accessible, and eligibility. The default age threshold is 24 hours. Inspect the listed path before providing its ID and explicit confirmation. A deletion rechecks registration, owner, age, session lease, and nested ICT job locks. It refuses unrecognized directories and junctions. It only removes the explicitly selected session in managed pytest storage.

Historical unmanaged pytest directories require separate review. First close the pytest sessions or custom jobs that could own them. Use File Explorer or a read-only directory listing to identify and measure individual `pytest-N` or manually created integration folders. Check ownership, file dates, and whether each directory holds weights or experiment results worth preserving. Obtain approval for the exact inactive directory before deleting any historical files. There is no wildcard or historical import mode in `ict clean`.

## Windows measurements

The current read-only baseline measured 76,569,264,128 free bytes on C: and 172,358,275,072 on D:. The initial inventory of the historical name patterns in `TEMP` found no directories. The persistent `outputs/smt-smoke-results` tree contained six checkpoint names per model, with logical checkpoint sizes of 10,454,563 bytes each for YOLO, 204,125,271 for Faster R-CNN, and 151,922,561 for RT-DETR. These are frozen-backbone smoke results and have different optimizer state sizes from scratch fixture checkpoints.

One approved backend run on D: measured the following after each test returned and before its fixture was removed. File identity bytes count a hard-linked file once per fixture. The approximate legacy checkpoint counts come from the previous save settings and copy paths. No legacy backend rerun was performed.

| Test | Model | Legacy checkpoint count (estimated) | Checkpoint count | Checkpoint logical MiB | All fixture logical MiB | File identity MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| One-epoch training and reload | YOLO | 3 | 3 | 45.8 | 45.9 | 15.3 |
| One-epoch training and reload | Faster R-CNN | 3 | 3 | 652.4 | 652.4 | 217.5 |
| One-epoch training and reload | RT-DETR | 3 | 3 | 907.1 | 907.1 | 302.4 |
| Interrupted two-epoch resume | YOLO | 8 | 4 | 61.1 | 61.1 | 30.6 |
| Interrupted two-epoch resume | Faster R-CNN | 8 | 4 | 869.9 | 869.9 | 435.0 |
| Interrupted two-epoch resume | RT-DETR | 8 | 4 | 1209.4 | 1209.4 | 604.7 |
| Detector DAG | YOLO | 5 | 4 | 61.1 | 61.3 | 30.8 |
| Detector DAG | Faster R-CNN | 5 | 4 | 869.9 | 870.2 | 435.2 |
| Detector DAG | RT-DETR | 5 | 4 | 1209.4 | 1209.7 | 605.0 |

The largest observed single fixture contained 634,374,183 bytes counted by file identity. The nine fixtures totaled 6,173,010,803 logical bytes and 2,806,466,297 bytes by file identity when added across tests. The sums describe separate fixtures at teardown. They exclude transient writes and are not peak disk usage.

Applying the old copy counts to the measured checkpoint sizes yields an estimated 8.36 GiB of checkpoint files per nine-case session before cleanup. The revised session's fixtures totaled 2.61 GiB by file identity before per-test cleanup and retained zero fixture bytes at exit. The older size is an estimate. The synthetic fine-tuning cases were outside this measurement.

| Volume | Free bytes immediately before the CPU run | Free bytes immediately after |
| --- | ---: | ---: |
| C: | 81,842,438,144 | 82,680,451,072 |
| D: | 170,166,611,968 | 169,390,039,040 |

Other activity, including a concurrent lightweight pytest session on D:, affected those volume readings. When both sessions finished, the managed sessions directory had zero entries. No existing temporary folders, datasets, weights, or experiment outputs were removed.

The final read-only check measured 84,303,159,296 free bytes on C: and 167,743,520,768 on D:. It also found one registered interrupted session from a later cancelled follow-up check. The preview reported 11,591 logical payload bytes. That session was left in place.

## Verification on Windows

Commands below used the existing repository virtual environment and the same D: artifact parent as the measured run.

| Check | Result |
| --- | --- |
| `python -m pytest -q tests/test_training.py tests/test_e2e.py -m cpu_backend --ict-artifact-summary` | **Passed**. Nine real CPU cases passed in 207.73 seconds. |
| `python -m pytest -q -m "not cpu_backend and not real_data_smoke"` before labeling the synthetic fine-tuning cases | **Passed**. 179 passed, three skipped, and 12 deselected. The selector still included the synthetic fine-tuning cases. One Windows test cleanup warning exposed an open SQLite handle. |
| `python -m pytest -q -m "not cpu_backend and not real_data_smoke"` after labeling fine-tuning and closing the SQLite handle | **Passed**. 179 passed, three skipped, and 15 deselected in 84.70 seconds with no cleanup warnings. |
| `python -m pytest -q -rs tests/test_artifacts.py tests/test_training_storage.py tests/test_integrity.py tests/test_runtime.py tests/test_pretrained.py -m "not cpu_backend and not real_data_smoke"` | **Passed**. 37 passed, three skipped, and three deselected. Windows symlink creation lacked permission. The Linux detached-worker test and opt-in dataset regression also skipped. |
| `python -m pytest -q -rs tests/test_artifacts.py tests/test_training_storage.py` and `ict clean --dry-run --older-than-hours 0` on D: | **Passed**. Eleven passed and one Windows symlink case skipped. The preview listed the interrupted session and performed no deletion. |
| Real SMT training, GPU training, and native Linux, WSL and macOS runs | **Not run**. The requested run used synthetic fixtures on Windows CPU. |
