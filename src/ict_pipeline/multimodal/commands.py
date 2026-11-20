"""Multimodal audit and approval-gated training entry points."""

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
import yaml

from ..config.settings import Credentials, SetupError
from ..config.storage import SetupFiles
from ..data.model import Progress
from ..data.preprocessing.common import require_separate_output, write_json
from ..data.preprocessing.console import console_progress, fresh_output, protect_source_data
from ..disk import StorageError
from ..models.contracts import TrainingError
from ..pipeline.model import PipelineError
from .audit import audit_cohort, audit_report

multimodal_app = typer.Typer(no_args_is_help=True, help="Prepare SMT data and run approved product training.")


@multimodal_app.command("audit")
def audit(
    root: Annotated[Path, typer.Option(help="Extracted AI Hub SMT data root.")],
    manifest: Annotated[Path, typer.Option(help="Original or published split manifest CSV.")],
    path_base: Annotated[Path | None, typer.Option(help="Manifest path base. Defaults to --root.")] = None,
    revision: Annotated[Path | None, typer.Option(help="Published split.json for a revised manifest.")] = None,
    out: Annotated[Path | None, typer.Option(help="Fresh directory for audit.json and optional cohort.json.")] = None,
    export_index: Annotated[bool, typer.Option(help="Include the reusable sensor and detection cohort index.")] = False,
    checkpoint: Annotated[Path | None, typer.Option(help="Optional resumable SQLite scan checkpoint.")] = None,
    preview: Annotated[bool, typer.Option(help="Show a small Train sensor window in the report.")] = False,
) -> None:
    """Verify source relationships and report data quality without fitting or evaluation."""
    credentials = Credentials()
    try:
        credentials = SetupFiles.read(Path.cwd(), None).credentials(os.environ)
        for path in (root, manifest, path_base, revision, out, checkpoint):
            if path is not None:
                credentials.protect_path(str(path.resolve()), "Multimodal audit path")
        if out is None and export_index:
            raise ValueError("Use --out with --export-index.")
        if out is not None and (out.exists() or out.is_symlink()):
            raise ValueError("Output already exists. Choose a fresh directory.")
        if out is not None:
            require_separate_output(root, out)
            protect_source_data(out, Path.cwd())
        if out is not None and checkpoint is not None and checkpoint.resolve().is_relative_to(out.resolve()):
            raise ValueError("Choose a checkpoint path outside the report directory.")
        credentials.protect_path(manifest.read_text(encoding="utf-8-sig"), "Multimodal manifest")
        cohort = audit_cohort(root, manifest, path_base=path_base, revision_path=revision, checkpoint=checkpoint,
                              progress=console_progress("Verify SMT source samples", file=sys.stderr))
        report = audit_report(cohort, preview=preview)
        payload = json.dumps(report, indent=2, allow_nan=False)
        credentials.protect_path(payload, "Multimodal report")
        if out is not None:
            with fresh_output(out, root) as working:
                write_json(working / "audit.json", report)
                if export_index:
                    with (working / "cohort.json").open("x", encoding="utf-8") as stream:
                        json.dump({"fingerprint": report["cohort_fingerprint"],
                                   "cohort": cohort.model_dump(mode="json")}, stream, separators=(",", ":"))
        typer.echo(payload)
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except ImportError:
        typer.echo("Multimodal source verification requires the existing Pillow and openpyxl preprocessing extra.", err=True)
        raise typer.Exit(2) from None
    except (KeyError, TypeError, IndexError):
        typer.echo("Multimodal audit failed. Check the manifest and split revision schema.", err=True)
        raise typer.Exit(2) from None
    except (ValueError, OSError, SetupError) as exc:
        typer.echo(credentials.redact_text(f"Multimodal audit failed. {exc}"), err=True)
        raise typer.Exit(2) from None


@multimodal_app.command("train")
def train(
    config: Annotated[Path, typer.Option(help="JSON or YAML product training profile.")],
    out: Annotated[Path, typer.Option(help="Fresh run directory, or the existing run when resuming.")],
    dry_run: Annotated[bool, typer.Option(help="Validate inputs and print a pending experiment and preflight.")] = False,
    resume: Annotated[bool, typer.Option(help="Recover the same run from its latest committed epoch.")] = False,
    approve_train: Annotated[bool, typer.Option("--approve-train", help="Authorize product optimizer execution.")] = False,
) -> None:
    """Plan an SMT product experiment or execute it with explicit training approval."""
    from .training import execute_training, inspect_recovery
    from .training_plan import ProductTrainingConfig, plan_training, preflight

    credentials = Credentials()
    try:
        workspace = Path.cwd()
        credentials = SetupFiles.read(workspace, None).credentials(os.environ)
        for path in (config, out):
            credentials.protect_path(str(path.resolve()), "Product training path")
        raw = config.read_text(encoding="utf-8-sig")
        credentials.protect_path(raw, "Product training profile")
        try:
            profile = ProductTrainingConfig.model_validate_json(json.dumps(yaml.safe_load(raw), allow_nan=False))
        except (yaml.YAMLError, TypeError):
            raise ValueError("Use a valid JSON or YAML product training profile.") from None
        plan = plan_training(
            profile, out, workspace=workspace, progress=console_progress("Verify product source bytes", file=sys.stderr),
        )
        credentials.protect_path(json.dumps(plan.manifest), "Resolved product training profile")
        if dry_run or not approve_train:
            receipts = inspect_recovery(plan, resume=resume)
            readiness = preflight(plan, remaining_epochs=profile.epochs - len(receipts))
            typer.echo(json.dumps(credentials.redact({
                **plan.manifest, "output": str(plan.output), "completed_epochs": len(receipts),
                "experiment_status": "pending", "compatibility": plan.compatibility, "preflight": readiness,
            }), indent=2, allow_nan=False))
            if not dry_run:
                raise TrainingError("Review the scope above and pass --approve-train to execute.")
            if not readiness["ready"]:
                raise typer.Exit(1)
            return
        typer.echo(credentials.redact_text(json.dumps({
            "output": str(plan.output), "branch": profile.branch, "epochs": profile.epochs,
            "device": profile.device, "revision_id": profile.revision_id,
            "checkpoint_policy": profile.checkpoint_policy, "samples": plan.manifest["samples"],
        })), err=True)
        result = execute_training(
            plan, approved=approve_train, resume=resume, credentials=credentials,
            progress=lambda stage: console_progress(stage, file=sys.stderr),
        )
        typer.echo(credentials.redact_text(str(result)))
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (ValueError, OSError, SetupError, StorageError, PipelineError, TrainingError) as exc:
        typer.echo(credentials.redact_text(f"Product training failed. {exc}"), err=True)
        raise typer.Exit(2) from None
    except ImportError:
        typer.echo("Product training requires an installed PyTorch and Pillow runtime.", err=True)
        raise typer.Exit(2) from None
    except (RuntimeError, KeyError, TypeError, IndexError):
        typer.echo("Product training failed. Check the input contracts and the active stage in run.json.", err=True)
        raise typer.Exit(2) from None


@multimodal_app.command("analyze")
def analyze_command(
    root: Annotated[Path, typer.Option(help="Extracted SMT source root.")],
    audit: Annotated[Path, typer.Option(help="Existing integrity leakage.json.")],
    cohort: Annotated[Path, typer.Option(help="Exact checksummed cohort used for training.")],
    frozen_validation: Annotated[Path, typer.Option(help="Exact FrozenValidation artifact used for selection.")],
    out: Annotated[Path, typer.Option(help="New analysis output directory.")],
    image_run: Annotated[Path | None, typer.Option(help="Completed FPN image-only run.")] = None,
    concat_run: Annotated[Path | None, typer.Option(help="Completed FPN sensor concat run.")] = None,
    mode: Annotated[str, typer.Option(help="all, leakage, groups, or ablation.")] = "all",
    seed: Annotated[int, typer.Option(min=0)] = 42,
    batch_size: Annotated[int, typer.Option(min=1)] = 1,
    device: Annotated[str, typer.Option(help="Evaluation device: cpu, cuda, rocm, or mps.")] = "cpu",
    checkpoint_dir: Annotated[Path | None, typer.Option(help="Separate directory for resumable score batches.")] = None,
) -> None:
    """Analyze cohort leakage, Validation subgroups and sensor dependence."""
    from .analysis import analyze

    credentials = Credentials()
    active_stage = "Check analysis options and paths"

    def progress(stage: str) -> Progress:
        nonlocal active_stage
        active_stage = credentials.redact_text(stage)
        return console_progress(active_stage, file=sys.stderr)

    try:
        credentials = SetupFiles.read(Path.cwd(), None).credentials(os.environ)
        if mode not in ("all", "leakage", "groups", "ablation") or device not in ("cpu", "cuda", "rocm", "mps"):
            raise ValueError("Choose a supported --mode and --device. See analyze --help.")
        runs = {}
        if image_run is not None:
            runs["image"] = image_run
        if concat_run is not None:
            runs["concat"] = concat_run
        analyze(root, audit, cohort, frozen_validation, runs, out, mode=mode, seed=seed, batch_size=batch_size,
                device=device, credentials=credentials, progress=progress,
                checkpoint_dir=checkpoint_dir)
        typer.echo(credentials.redact_text(f"Published multimodal analysis to {out}"))
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except ImportError:
        typer.echo(f"Multimodal analysis failed during '{active_stage}'. "
                   "Analysis scoring requires the existing PyTorch, torchvision and Pillow runtime.", err=True)
        raise typer.Exit(2) from None
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        # Unexpected exceptions may include raw artifact values or keys.
        typer.echo(f"Multimodal analysis failed during '{active_stage}' ({type(exc).__name__}). "
                   "Check the inputs for this stage.", err=True)
        raise typer.Exit(2) from None
    except (OSError, ValueError, SetupError, TrainingError, RuntimeError) as exc:
        typer.echo(credentials.redact_text(f"Multimodal analysis failed during '{active_stage}'. {exc}"), err=True)
        raise typer.Exit(2) from None
