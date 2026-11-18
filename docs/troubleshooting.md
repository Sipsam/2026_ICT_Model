# Troubleshooting the local workflow

| Symptom | What to check |
| --- | --- |
| `ict` missing after install | Run `python -m pip show ict-pipeline` with the intended virtual environment's Python. Activate that environment or invoke its installed launcher directly. |
| Setup exits with code 2 | For closed stdin, pass `--non-interactive` and an explicit `--dataset-source`. Use `--dry-run` for validation and `--replace` to authorize replacing a saved profile. |
| Missing or rejected AI Hub key | Check your approval for dataset 71900 and the source of `AIHUB_API_KEY`. An environment value takes precedence over `.env`. HTTP 401 or 403 means the provider rejected the key or dataset access. |
| AI Hub endpoint unreachable | Check connectivity and curl. On Windows, verify curl inside the chosen WSL distribution. To work offline, use verified local archives or run on an extracted tree. |
| Ingestion reports insufficient space | Check space and permissions on both the raw and run filesystems. Move or configure storage separately from the source tree. Partial transfers are kept for inspection, and verified completed splits can be reused. |
| Archive is incomplete or corrupt | Use `ict data verify` to check coverage and ZIP integrity. Supply verified archives in another directory or review the download plan before approving missing transfers. |
| Audit cannot verify its dataset input | Check the extracted path and expected layout. With `dataset_source: skip`, the default source is `data/extracted`. Install the `preprocess` extra for deep JPG and XLSX audit. |
| Conversion fails on bad annotations | Inspect the completed audit's `issues.csv` and manifest's `manifest.csv` in the selected run's artifact directories. Point the profile at a verified source tree and resume. |
| A run was interrupted or a cache was invalidated | Use `ict status --run-id <id>` and `ict runs show <id>`. Resume with the same run ID and the original `--config` if supplied. Changed inputs, relevant settings, or damaged artifacts cause a fresh attempt for affected stages. If an older run selected unimplemented model stages, use `ict run --run-id <id>` to select preprocessing and reuse verified results. |
| GPU or backend unavailable | Check both inventory and readiness with `ict doctor`. For a small CPU model run, explicitly use `device: cpu` and `precision: fp32` in the model YAML and `--device cpu` for evaluation. |
| Evaluation cannot find a checkpoint | Pass the model directory printed by training, which contains `run.json`. Check that the `best` or `last` checkpoint exists and matches the recorded fingerprint. Use the original compatible detection dataset and a trusted completed run. |

`ict run`, `ict resume`, `ict train`, and `ict evaluate` return `1` for execution failures and `130` for interruption. Command syntax errors and invalid setup or evaluation options return `2`. A successfully displayed `ict status` can show a failed run and still return `0`. Read the saved status to decide whether downstream work should proceed.
