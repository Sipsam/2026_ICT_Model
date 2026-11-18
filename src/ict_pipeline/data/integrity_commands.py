"""Commands for the dataset integrity and split revision workflow."""

import json
import os
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from ..config.settings import Credentials, SetupError
from ..config.storage import SetupFiles
from ..pipeline.store import digest_json
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
        typer.echo("Leakage analysis requires the installed preprocessing extra (Pillow and openpyxl).", err=True)
        raise typer.Exit(2) from None
    except (KeyError, TypeError, IndexError):
        typer.echo("Invalid manifest, source metadata, or leakage report schema.", err=True)
        raise typer.Exit(2) from None
    except (ValueError, OSError, SetupError) as exc:
        typer.echo(credentials.redact_text(str(exc)), err=True)
        raise typer.Exit(2) from None


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
