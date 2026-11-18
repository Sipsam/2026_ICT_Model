# Manual platform compatibility checklist

Run the checks in a fresh virtual environment in a disposable clone. Record the OS, architecture, Python version, and actual result for each check. Use **Passed**, **Failed**, or **Not run**. The fixture suite can run without AI Hub credentials, optional ML dependencies, or access to any external execution service.

| Platform | Platform-specific checks |
| --- | --- |
| Windows PowerShell | Use native paths. Try a workspace with spaces and run the installed `ict.exe`. If script activation is restricted, invoke `.venv\Scripts\python.exe` and `.venv\Scripts\ict.exe` directly. Test WSL2 separately if you use it for downloads. |
| Linux | Activate `.venv/bin/activate` and run the installed `ict`. Verify permissions on the chosen output filesystem. The local suite includes the Linux detached-worker test. |
| macOS | Record whether Python runs on Intel or Apple Silicon. Use the matching interpreter and run `ict` through its virtual environment. Record MPS readiness independently when PyTorch is present. |
| WSL2, if used | Check `ict doctor --json` from inside the distribution. Use that distribution's Python and Linux paths. Test the filesystem where you intend to keep run artifacts. |

For each platform you have available:

1. Follow the [install instructions](../README.md#install). Confirm `ict --version`, `ict --help`, and `ict doctor --json` exit successfully. Save the diagnostic report with the environment details.
2. Install the test extra and run `python -m pytest -m "not cpu_backend and not real_data_smoke"`. The fixture suite covers headless setup and run, local reuse, interrupted-stage recovery, cache invalidation, and evaluation and report generation. [Configure test storage and run approved backend checks](artifact-management.md)
3. With a small existing extracted tree, run the [local walkthrough](../README.md#first-local-run) using a workspace path with spaces. Check that `ict status --run-id smt` shows a completed run, that `ict resume --run-id smt` prints `reused` for verified stages, and that it prints a usable `Detection dataset` path. Run setup with `--non-interactive` and run the pipeline with stdin closed to check headless operation.
4. If a compatible ML runtime is already provisioned, evaluate a trusted fixture checkpoint with `--device cpu`. For an approved training smoke test, use a tiny fixture with both splits and a model YAML containing `epochs: 1`, `batch_size: 1`, `image_size: 64`, `device: cpu`, and `precision: fp32`. Pass its directory via `ict train --config-dir` and review `--dry-run` before passing `--approve-train`. Verify that `ict evaluate` writes a Validation result and that `ict report` includes it.
5. Record accelerator, Docker, SSH, and notebook results only when you ran the relevant checks on the target. See the [runtime guide](runtime.md) for those checks.
