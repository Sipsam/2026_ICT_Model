"""Read-only multimodal audit entry point."""

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import typer

from ..config.settings import Credentials, SetupError
from ..config.storage import SetupFiles
from ..data.preprocessing.common import require_separate_output, write_json
from ..data.preprocessing.console import console_progress, fresh_output, protect_source_data
from .audit import audit_cohort, audit_report

multimodal_app = typer.Typer(no_args_is_help=True, help="Inspect and prepare SMT image and sensor data.")


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
