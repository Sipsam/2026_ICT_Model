"""CLI for single-image and batch inspection."""

import os
from pathlib import Path
from time import monotonic
from typing import Annotated
from uuid import uuid4

import typer
from pydantic import TypeAdapter, ValidationError

from ..config.settings import SetupError
from ..config.storage import SetupFiles
from ..data.preprocessing.common import require_separate_output
from ..data.preprocessing.console import fresh_output, protect_source_data
from ..multimodal.inference_contracts import ProductInspectionInput, ProductModelSpec
from ..pipeline.model import PipelineError
from .contracts import CheckpointKind, Device, InspectionError, InspectionRequest, InspectionThresholds, MockScenario
from .records import publish_batch, read_manifest, write_records
from .service import InspectionService, local_path


def inspect_command(
    image: Annotated[Path | None, typer.Argument(help="Local JPEG or PNG image.")] = None,
    batch: Annotated[Path | None, typer.Option(help="UTF-8 JSON Lines of version 1 inspection requests.")] = None,
    request_id: Annotated[str | None, typer.Option(help="Optional single-image request identifier.")] = None,
    source_id: Annotated[str | None, typer.Option(help="Optional single-image source identifier.")] = None,
    mock: Annotated[MockScenario | None, typer.Option(help="Simulate an empty or defect prediction.")] = None,
    run: Annotated[Path | None, typer.Option(help="Completed local ICT model run with run.json.")] = None,
    trust_run: Annotated[bool, typer.Option(help="Confirm the entire model run comes from a trusted source.")] = False,
    checkpoint: Annotated[CheckpointKind, typer.Option(help="Completed run checkpoint selection.")] = "best",
    expected_fingerprint: Annotated[str | None, typer.Option(help="Optional independently verified ICT fingerprint.")] = None,
    confidence: Annotated[float, typer.Option(help="Inclusive minimum detection confidence.")] = 0.25,
    nms_iou: Annotated[float, typer.Option(help="Class-aware suppression IoU threshold.")] = 0.7,
    device: Annotated[Device, typer.Option(help="Detector device.")] = "cpu",
    image_size: Annotated[int | None, typer.Option(help="Override saved model resize size with a multiple of 32.")] = None,
    output_dir: Annotated[Path, typer.Option(help="Parent for new inspection attempts.")] = Path("outputs/inspection"),
    resume_from: Annotated[Path | None, typer.Option(help="Resume an unpublished batch attempt.")] = None,
    product_model: Annotated[Path | None, typer.Option(help="Local product model settings JSON.")] = None,
    product_request: Annotated[Path | None, typer.Option(help="Local product request JSON with optional sensor evidence.")] = None,
) -> None:
    """Inspect images or product evidence with a local model or explicit simulation."""
    if product_model is not None or product_request is not None:
        if (product_model is None or product_request is None or batch is not None or mock is not None
                or request_id is not None or source_id is not None):
            raise typer.BadParameter("Use --product-model and --product-request together with an optional image and detector run.")
        if run is None and (trust_run or expected_fingerprint is not None or checkpoint != "best"
                            or device != "cpu" or image_size is not None):
            raise typer.BadParameter("Configure the product device in --product-model. Detector options require --run.")
    elif (image is None) == (batch is None) or (mock is None) == (run is None):
        raise typer.BadParameter("Choose IMAGE or --batch, and choose --mock or --run.")
    if batch is not None and (request_id is not None or source_id is not None):
        raise typer.BadParameter("Supply request and source identifiers in the batch manifest.")
    if resume_from is not None and batch is None:
        raise typer.BadParameter("--resume-from requires --batch.")
    if run is not None and not trust_run:
        raise typer.BadParameter("--run requires --trust-run.")
    if image_size is not None and (image_size < 64 or image_size % 32 != 0):
        raise typer.BadParameter("--image-size must be at least 64 and a multiple of 32.")
    if mock is not None and (trust_run or expected_fingerprint is not None or checkpoint != "best"
                             or device != "cpu" or image_size is not None):
        raise typer.BadParameter("Checkpoint and device options are for real inference.")

    started = last = monotonic()

    def progress(stage: str, done: int, total: int) -> None:
        nonlocal started, last
        now = monotonic()
        if not total:
            typer.echo(stage, err=True)
        elif done in (0, total) or now - last >= 1:
            started = now if done == 0 else started
            elapsed = now - started
            eta = f"{elapsed * (total - done) / done:.1f}s" if done else "unknown"
            typer.echo(f"{stage} {done}/{total}   elapsed {elapsed:.1f}s   ETA {eta}", err=True)
            last = now

    try:
        workspace = Path.cwd()
        credentials = SetupFiles.read(workspace, None).credentials(os.environ)
        output = local_path(output_dir, workspace, credentials)
        resume = local_path(resume_from, workspace, credentials) if resume_from is not None else None
        destination = resume if resume is not None else output
        protect_source_data(destination, workspace)
        thresholds = InspectionThresholds(confidence=confidence, nms_iou=nms_iou)
        if product_model is not None and product_request is not None:
            product_config_path = local_path(product_model, workspace, credentials)
            product_input_path = local_path(product_request, workspace, credentials)
            spec = TypeAdapter(ProductModelSpec).validate_json(product_config_path.read_bytes())
            product_input = ProductInspectionInput.model_validate_json(product_input_path.read_bytes())
        elif batch is not None:
            manifest = read_manifest(local_path(batch, workspace, credentials))
        else:
            assert image is not None
            default_id = image.name if image.name else "image"
            request = InspectionRequest(
                request_id=default_id if request_id is None else request_id,
                source_id=default_id if source_id is None else source_id, image_path=image,
            )

        if run is not None:
            run = local_path(run, workspace, credentials)
            require_separate_output(run, destination)
            typer.echo("initialize local detector", err=True)
            service = InspectionService.from_run(
                run, trust_run=trust_run, thresholds=thresholds, checkpoint_kind=checkpoint,
                expected_fingerprint=expected_fingerprint, device=device, image_size=image_size, credentials=credentials,
            )
        elif product_model is None:
            assert mock is not None
            service = InspectionService.mock(mock, thresholds=thresholds, credentials=credentials)

        if product_model is not None:
            typer.echo("initialize local product inference", err=True)
            service = InspectionService.from_product(
                spec, product_config_path.parent, credentials=credentials,
                detector=service if run is not None else None,
            )
            progress("inspect product", 0, 1)
            result = service.inspect_product(product_input, image)
            reports = output / uuid4().hex
            with fresh_output(reports, product_input_path) as working:
                complete = write_records(working, (result,))
            progress("inspect product", 1, 1)
        elif batch is not None:
            reports, complete = publish_batch(service, manifest, output, progress, resume_from=resume)
        else:
            reports = output / uuid4().hex
            with fresh_output(reports, local_path(request.image_path, workspace, credentials)) as working:
                complete = write_records(working, (service.inspect(request, base=workspace),))
        typer.echo(str(reports))
        if not complete:
            raise typer.Exit(code=1)
    except ValidationError:
        typer.echo("Invalid inspection request or thresholds. See ict inspect --help.", err=True)
        raise typer.Exit(code=2) from None
    except (InspectionError, SetupError, PipelineError) as exc:
        typer.echo(f"Inspection failed. {exc}", err=True)
        raise typer.Exit(code=1) from None
    except (OSError, ValueError, RuntimeError):
        typer.echo("Inspection could not complete. Check the local inputs, output directory and permissions.", err=True)
        raise typer.Exit(code=1) from None
    except KeyboardInterrupt:
        typer.echo("Inspection interrupted.", err=True)
        raise typer.Exit(code=130) from None
