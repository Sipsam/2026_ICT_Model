"""Commands for the dataset integrity and split revision workflow."""

import json
import os
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, cast

import typer

from ..config.settings import Credentials, SetupError
from ..config.storage import SetupFiles
from ..pipeline.store import digest_json
from ..pipeline.model import PipelineError
from ..models.contracts import Progress, TrainingError
from .preprocessing.common import write_json
from .preprocessing.console import console_progress, fresh_output
from .preprocessing.leakage import AnalysisConfig, analyze_leakage
from .preprocessing.split import Grouping, SplitConfig, Strategy, plan_split, publish_split

integrity_app = typer.Typer(no_args_is_help=True, help="Analyze leakage and preview or publish split revisions.")
SourceRoot = Annotated[Path, typer.Option(help="Root of the extracted source dataset.")]
Manifest = Annotated[Path, typer.Option(help="Original detection manifest, including held and excluded rows.")]
PathBase = Annotated[Path | None, typer.Option(help="Base for manifest paths. Defaults to --root. Use . for legacy repository-relative manifests.")]


@contextmanager
def _command(*paths: Path | None) -> Iterator[Credentials]:
    credentials = Credentials()
    try:
        credentials = SetupFiles.read(Path.cwd(), None).credentials(os.environ)
        for path in paths:
            if path is not None:
                credentials.protect_path(str(path.resolve()), "integrity command path")
        yield credentials
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except ImportError:
        typer.echo("Required dependencies are missing. Check the extra for this integrity command.", err=True)
        raise typer.Exit(2) from None
    except (KeyError, TypeError, IndexError):
        typer.echo("Invalid manifest, source metadata, or leakage report schema.", err=True)
        raise typer.Exit(2) from None
    except (ValueError, OSError, SetupError, PipelineError, TrainingError) as exc:
        typer.echo(credentials.redact_text(str(exc)), err=True)
        raise typer.Exit(2) from None


@integrity_app.command()
def verify_group(
    root: SourceRoot,
    bundle: Annotated[Path, typer.Option(help="Existing published group-aware split bundle.")],
    revision_id: Annotated[str, typer.Option(help="Expected published revision SHA-256.")],
    cohort: Annotated[Path, typer.Option(help="Existing sanitized product training cohort.")],
    cohort_fingerprint: Annotated[str, typer.Option(help="Expected product cohort SHA-256.")],
    frozen_validation: Annotated[Path, typer.Option(help="Existing frozen Validation artifact.")],
    preprocessing: Annotated[Path, typer.Option(help="Existing group Train preprocessing state.")],
    out: Annotated[Path, typer.Option(help="Fresh directory for the group verification result.")],
    prior_test_manifest: Annotated[list[Path] | None, typer.Option(
        help="Earlier manifest with reserved Test rows. Repeat for each applicable reservation.",
    )] = None,
    checkpoint: Annotated[Path | None, typer.Option(help="Resume scoped development observations from SQLite.")] = None,
) -> None:
    """Verify group retraining inputs without opening Test payloads."""
    from ..evaluation.product import read_artifact
    from ..evaluation.product_protocol import FrozenValidation
    from ..multimodal.preprocessing import PreprocessingState
    from .group_retraining import GroupInputs, verify_group_retraining

    root, bundle, cohort, frozen_validation, preprocessing, out = (
        path.expanduser().resolve() for path in (root, bundle, cohort, frozen_validation, preprocessing, out)
    )
    prior = tuple(path.expanduser().resolve() for path in (prior_test_manifest or ()))
    checkpoint = checkpoint.expanduser().absolute() if checkpoint else None
    with _command(root, bundle, cohort, frozen_validation, preprocessing, out, checkpoint, *prior) as credentials:
        if out.exists():
            raise ValueError("Output already exists. Choose a fresh directory.")
        if out.is_relative_to(bundle):
            raise ValueError("Write the verification result outside the published split bundle.")
        if checkpoint is not None and any(checkpoint.resolve().is_relative_to(path) for path in (bundle, out)):
            raise ValueError("Choose a checkpoint outside the published split and report directories.")
        result = verify_group_retraining(
            GroupInputs(root, bundle, revision_id, cohort, cohort_fingerprint),
            read_artifact(frozen_validation, FrozenValidation), PreprocessingState.load(preprocessing), credentials,
            lambda stage: console_progress(stage, file=sys.stderr), prior_manifests=prior, checkpoint=checkpoint,
        )
        payload = {"verification": result, "fingerprint": digest_json(result)}
        credentials.protect_path(json.dumps(payload), "group verification report")
        with fresh_output(out, root) as temporary:
            write_json(temporary / "verification.json", payload)
        typer.echo(json.dumps(payload, indent=2))
        if not result["passed"]:
            raise typer.Exit(1)


@integrity_app.command("test-release-preflight")
def test_release_preflight(
    root: SourceRoot,
    bundle: Annotated[Path, typer.Option(help="Pinned published group-aware split bundle.")],
    revision_id: Annotated[str, typer.Option(help="Expected published revision SHA-256.")],
    group_verification: Annotated[Path, typer.Option(help="Passing verify-group verification.json.")],
    cohort: Annotated[Path, typer.Option(help="Sanitized audited product training cohort.")],
    frozen_validation: Annotated[Path, typer.Option(help="Frozen Validation artifact.")],
    preprocessing: Annotated[Path, typer.Option(help="Fitted Train-only sensor preprocessing state.")],
    image_run: Annotated[Path, typer.Option(help="Completed Image-only product training run directory.")],
    concat_run: Annotated[Path, typer.Option(help="Completed Sensor Concat product training run directory.")],
    out: Annotated[Path, typer.Option(help="Fresh directory for the preflight receipt.")],
    prior_test_manifest: Annotated[list[Path] | None, typer.Option(
        help="Earlier reserved Test manifest. Repeat if multiple reservations are known.",
    )] = None,
) -> None:
    """Preflight Test evaluation metadata and checkpoints without opening Test payloads."""
    from .test_release_gate import preflight_test_release

    paths = tuple(path.expanduser().resolve() for path in (
        root, bundle, group_verification, cohort, frozen_validation, preprocessing,
        image_run, concat_run, out,
    ))
    root, bundle, group_verification, cohort, frozen_validation, preprocessing, image_run, concat_run, out = paths
    prior = tuple(path.expanduser().resolve() for path in (prior_test_manifest or ()))
    with _command(root, bundle, group_verification, cohort, frozen_validation, preprocessing,
                  image_run, concat_run, out, *prior) as credentials:
        if out.exists():
            raise ValueError("Output already exists. Choose a fresh directory.")
        receipt = preflight_test_release(
            root=root, bundle=bundle, revision_id=revision_id,
            group_verification=group_verification, cohort=cohort,
            frozen_validation=frozen_validation, preprocessing=preprocessing,
            image_run=image_run, concat_run=concat_run, prior_manifests=prior,
        )
        credentials.protect_path(json.dumps(receipt), "Test preflight receipt")
        with fresh_output(out, root) as temporary:
            write_json(temporary / "preflight.json", receipt)
        typer.echo(json.dumps(receipt, indent=2))
        if not receipt["passed"]:
            raise typer.Exit(1)


@integrity_app.command("test-release-seal")
def test_release_seal(
    root: SourceRoot,
    bundle: Annotated[Path, typer.Option(help="Pinned published split bundle containing the sealed Test split.")],
    preflight: Annotated[Path, typer.Option(help="Passing preflight.json from test-release-preflight.")],
    public_index: Annotated[Path, typer.Option(help="Public sanitized Test sample index with original hashes.")],
    out: Annotated[Path, typer.Option(help="Fresh directory for the seal verification receipt.")],
    approve_seal_verification: Annotated[bool, typer.Option(
        "--approve-seal-verification",
        help="Explicitly authorize reading Test images, annotations and sensor payloads.",
    )] = False,
) -> None:
    """Reverify Test payload hashes only after explicit user approval."""
    from .test_release_gate import verify_test_payload_seal

    root, bundle, preflight, public_index, out = (
        path.expanduser().resolve() for path in (root, bundle, preflight, public_index, out)
    )
    with _command(root, bundle, preflight, public_index, out) as credentials:
        if not approve_seal_verification:
            raise ValueError("Seal Verification requires --approve-seal-verification.")
        if out.exists():
            raise ValueError("Output already exists. Choose a fresh directory.")
        payload = json.loads(preflight.read_bytes())
        receipt = verify_test_payload_seal(
            root=root, bundle=bundle, preflight_receipt=payload,
            preflight_path=preflight, public_index=public_index,
        )
        credentials.protect_path(json.dumps(receipt), "Test seal verification receipt")
        with fresh_output(out, root) as temporary:
            write_json(temporary / "seal-verification.json", receipt)
        typer.echo(json.dumps(receipt, indent=2))


@integrity_app.command()
def finalize_detector(
    source_run: Annotated[Path, typer.Option(help="Original failed Faster R-CNN run directory.")],
    bundle: Annotated[Path, typer.Option(help="Published group-aware split bundle.")],
    group_verification: Annotated[Path, typer.Option(help="Passing group verification.json.")],
    cohort: Annotated[Path, typer.Option(help="Sanitized group training-cohort.json.")],
    out: Annotated[Path, typer.Option(help="Fresh directory for the finalized best checkpoint and receipt.")],
    best_epoch: Annotated[int, typer.Option(min=1, help="Expected selected epoch.")],
    completed_epoch: Annotated[int, typer.Option(min=1, help="Expected last fully completed epoch.")],
    validation_device: Annotated[str | None, typer.Option(
        help="Run the optional Validation-only mAP and inference check on cpu, cuda, or rocm.",
    )] = None,
) -> None:
    """Finalize a failed group Detector's best checkpoint for FPN initialization."""
    from .detector_finalization import Device, finalize_detector as publish

    source_run, bundle, group_verification, cohort, out = (
        path.expanduser().absolute() for path in (source_run, bundle, group_verification, cohort, out)
    )
    with _command(source_run, bundle, group_verification, cohort, out) as credentials:
        if validation_device not in (None, "cpu", "cuda", "rocm"):
            raise ValueError("Choose cpu, cuda, or rocm for Validation evaluation.")
        receipt = publish(source_run, bundle, group_verification, cohort, out, best_epoch=best_epoch,
                          completed_epoch=completed_epoch, validation_device=cast(Device | None, validation_device),
                          credentials=credentials, progress=_finalization_progress())
        typer.echo(str(receipt))


def _finalization_progress() -> Progress:
    active = ""
    report = console_progress(active, file=sys.stderr)

    def update(stage: str, done: int, total: int) -> None:
        nonlocal active, report
        if stage != active:
            active = stage
            report = console_progress(stage, file=sys.stderr)
        report(done, total)

    return update


@integrity_app.command()
def analyze(
    root: SourceRoot,
    manifest: Manifest,
    path_base: PathBase = None,
    out: Annotated[Path | None, typer.Option(help="Fresh output directory for leakage.json. Omit for a read-only summary.")] = None,
    image_hamming: Annotated[int, typer.Option(min=0, max=8)] = 4,
    image_mae: Annotated[float, typer.Option(min=0, max=255)] = 8,
    sensor_scale: Annotated[float, typer.Option(help="Multiplier for all five sensor similarity tolerances.")] = 1,
    verify_xlsx: Annotated[bool, typer.Option(help="Cross-check embedded readings against XLSX and allow XLSX-only readings.")] = False,
    checkpoint: Annotated[Path | None, typer.Option(help="Save resumable observations to a local SQLite file.")] = None,
) -> None:
    """Scan the original assignments for cross-split evidence."""
    with _command(root, manifest, path_base, out, checkpoint) as credentials:
        if out is not None and (out.exists() or out.is_symlink()):
            raise ValueError("Output already exists. Choose a fresh directory.")
        if out is not None and checkpoint is not None and checkpoint.resolve().is_relative_to(out.resolve()):
            raise ValueError("Choose a checkpoint path outside the report directory.")
        typer.echo("Scanning source images, annotations, and sensor evidence.")
        report = analyze_leakage(
            root, manifest, path_base or root, AnalysisConfig(image_hamming, image_mae, sensor_scale, verify_xlsx),
            console_progress("Inspect source samples"), similarity_progress=console_progress, checkpoint=checkpoint,
        )
        cross = [group for group in report["groups"] if len(group["by_original_split"]) > 1]
        summary = {
            **{key: report[key] for key in ("original_split_counts", "candidate_split_counts", "unknown_provenance")},
            "cross_split_groups_by_method": dict(Counter(group["method"] for group in cross)),
            "candidate_cross_split_groups_by_method": dict(Counter(
                group["method"] for group in cross if len(group["candidates_by_original_split"]) > 1
            )),
        }
        typer.echo(json.dumps(summary, indent=2))
        if out is not None:
            credentials.protect_path(json.dumps(report, ensure_ascii=False), "integrity report")
            with fresh_output(out, root) as temporary:
                write_json(temporary / "leakage.json", {"fingerprint": digest_json(report), "report": report})
            typer.echo(f"Leakage report written to {out / 'leakage.json'}")


@integrity_app.command()
def split(
    root: SourceRoot,
    manifest: Manifest,
    analysis: Annotated[Path, typer.Option(help="leakage.json produced by integrity analyze.")],
    path_base: PathBase = None,
    strategy: Strategy = "group-aware",
    grouping: Grouping = "capture",
    val_fraction: float = 0.15,
    test_fraction: float = 0.10,
    seed: int = 42,
    group_near_images: bool = False,
    group_near_sensors: bool = False,
    out: Annotated[Path | None, typer.Option(help="Fresh output directory for a published revision.")] = None,
    publish: Annotated[bool, typer.Option(help="Publish the previewed assignments to --out.")] = False,
    dry_run: Annotated[bool, typer.Option(help="Preview without writing files. This is the default.")] = False,
    export_detection: Annotated[bool, typer.Option(help="Also copy and validate YOLO/COCO files for training and evaluation.")] = False,
    allow_missing_classes: Annotated[bool, typer.Option(help="Acknowledge incomplete class coverage before publishing.")] = False,
) -> None:
    """Preview a seeded revision or preserve the AI Hub assignment as a baseline."""
    with _command(root, manifest, analysis, path_base, out):
        if (dry_run and publish) or (publish and out is None):
            raise ValueError("Use --dry-run for preview, or --publish with a fresh --out.")
        payload = json.loads(analysis.read_text(encoding="utf-8-sig"))
        report = payload["report"]
        if digest_json(report) != payload["fingerprint"]:
            raise ValueError("The leakage report fingerprint does not match its contents. Rerun analysis.")
        plan = plan_split(report, SplitConfig(
            strategy, grouping, val_fraction, test_fraction, seed, group_near_images, group_near_sensors,
        ))
        typer.echo(json.dumps({key: value for key, value in plan.items() if key != "assignments"}, indent=2))
        if publish and out is not None:
            typer.echo(f"Publishing split revision to {out}")
            publish_split(root, manifest, path_base or root, report, plan, out, export_detection=export_detection,
                          allow_missing_classes=allow_missing_classes, progress=console_progress)
        else:
            typer.echo("Preview complete.")
