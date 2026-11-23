"""CLI commands for the external domain-generalization benchmark."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from .domain_generalization import prepare_external, prepare_internal, evaluate_domain


domain_app = typer.Typer(
    name="domain-benchmark",
    help="Prepare and run the independent Faster R-CNN domain benchmark.",
    no_args_is_help=True,
)


@domain_app.command("prepare")
def prepare(
    source_dataset: Annotated[Path, typer.Option(help="Published detection data root.")] = Path("data/processed/detection"),
    internal_output: Annotated[Path, typer.Option(help="Derived internal diagnostic COCO root.")] = Path("data/processed/domain-generalization/internal"),
    external_output: Annotated[Path, typer.Option(help="External image and COCO annotation root.")] = Path("data/external/domain-generalization"),
) -> None:
    try:
        internal = prepare_internal(source_dataset, internal_output)
        external = prepare_external(external_output)
        typer.echo(f"Prepared internal: {internal_output}")
        typer.echo(f"Internal images: {internal['selected_image_count']}")
        typer.echo(f"Prepared external directory: {external_output}")
        typer.echo(f"External annotation template: {external['annotation_template']}")
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        typer.echo(f"Preparation failed. {exc}", err=True)
        raise typer.Exit(code=1) from None


@domain_app.command("evaluate")
def evaluate(
    checkpoint: Annotated[Path, typer.Option(help="Finalized Faster R-CNN .pt checkpoint or run directory.")],
    internal_dataset: Annotated[Path, typer.Option(help="Labeled internal diagnostic COCO root.")] = Path("data/processed/domain-generalization/internal"),
    external_dataset: Annotated[Path, typer.Option(help="External COCO root.")] = Path("data/external/domain-generalization"),
    output_dir: Annotated[Path, typer.Option(help="Derived report directory.")] = Path("outputs/domain-generalization"),
    image_size: Annotated[int, typer.Option(help="Faster R-CNN input resize bound.")] = 640,
    confidence: Annotated[float, typer.Option(help="Fixed confidence cutoff.")] = 0.25,
    match_iou: Annotated[float, typer.Option(help="Fixed match IoU.")] = 0.5,
    nms_iou: Annotated[float, typer.Option(help="TorchVision class-aware NMS IoU.")] = 0.7,
) -> None:
    try:
        report = evaluate_domain(
            checkpoint,
            internal_dataset,
            external_dataset,
            output_dir,
            image_size=image_size,
            confidence=confidence,
            match_iou=match_iou,
            nms_iou=nms_iou,
        )
        typer.echo(str(output_dir / "domain_generalization.json"))
        typer.echo(f"External status: {report['external']['status']}")
    except (OSError, RuntimeError, KeyError, ValueError) as exc:
        typer.echo(f"Benchmark failed. {exc}", err=True)
        raise typer.Exit(code=1) from None
