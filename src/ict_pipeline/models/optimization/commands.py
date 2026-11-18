"""Offline CLI entry points for SMT training data and pending ablations."""

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import monotonic
from typing import Annotated, Any, TypeVar

import typer
import yaml
from pydantic import BaseModel, ValidationError

from ...config.settings import Credentials, SetupError
from ...config.storage import SetupFiles
from ...data.preprocessing.common import require_separate_output, write_json
from ...data.preprocessing.console import fresh_output
from ...data.preprocessing.revision import read_dataset_seal
from ...evaluation.contracts import EvaluationConfig
from ...pipeline.model import PipelineError
from ...pipeline.store import digest_json
from ..contracts import ModelName, Progress, TrainingError
from ..dataset import DetectionData, inspect_data, read_coco_split, verify_shared_labels
from ..training import read_training_profile
from .config import Optimization
from .experiments import DEFAULT_ABLATION, experiment_manifest
from .reports import imbalance_report, preview_sample

training_data_app = typer.Typer(no_args_is_help=True, help="Inspect SMT training data and plan controlled ablations.")
DatasetPath = Annotated[Path, typer.Option(help="Path to generated detection conversion containing coco/.")]
ProfilesPath = Annotated[Path, typer.Option(help="Directory containing per-detector training YAML files.")]
Model = TypeVar("Model", bound=BaseModel)


@contextmanager
def _command(*paths: Path | None) -> Iterator[Credentials]:
    credentials = Credentials()
    try:
        credentials = SetupFiles.read(Path.cwd(), None).credentials(os.environ)
        for path in paths:
            if path is not None:
                credentials.protect_path(str(path.resolve()), "training-data path")
        yield credentials
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except ImportError:
        typer.echo("Image previews require Pillow in the current environment.", err=True)
        raise typer.Exit(2) from None
    except (TrainingError, SetupError, PipelineError) as exc:
        typer.echo(credentials.redact_text(str(exc)), err=True)
        raise typer.Exit(2) from None
    except (ValueError, OSError, TypeError, KeyError, RuntimeError):
        typer.echo("Cannot inspect or publish training data artifacts. Check inputs and use a fresh output directory.",
                   err=True)
        raise typer.Exit(2) from None


def _read_config(path: Path, model: type[Model]) -> Model:
    try:
        return model.model_validate(yaml.safe_load(path.read_text(encoding="utf-8-sig")))
    except ValidationError as exc:
        fields = ", ".join(str(error["loc"][0]) if error["loc"] and error["loc"][0] in model.model_fields
                           else "configuration" for error in exc.errors(include_input=False))
        raise TrainingError(f"Invalid {model.__name__} settings. Check {fields}.") from None
    except (yaml.YAMLError, RecursionError):
        raise TrainingError("Invalid training-data YAML syntax.") from None


def _progress() -> Progress:
    times: dict[str, tuple[float, float]] = {}

    def report(stage: str, done: int, total: int) -> None:
        now = monotonic()
        started, last = times.get(stage, (now, float("-inf")))
        if done == total or now - last >= 1:
            elapsed = now - started
            eta = f"{elapsed * (total - done) / done:.1f}s" if done else "unknown"
            typer.echo(f"{stage}   {done}/{total}   elapsed {elapsed:.1f}s   ETA {eta}", err=True)
            times[stage] = started, now

    return report


def _inspect(dataset: Path, credentials: Credentials, progress: Progress) -> tuple[DetectionData, dict[str, Any] | None]:
    try:
        seal = read_dataset_seal(dataset)
    except ValueError as exc:
        raise TrainingError(str(exc)) from None
    if not (dataset / "coco").resolve().is_relative_to(dataset):
        raise TrainingError("The COCO conversion must resolve inside the selected dataset.")
    data = inspect_data(dataset, "coco", credentials, progress)
    for split in data.splits.values():
        if len({sample.image.stem.casefold() for sample in split.samples}) != len(split.samples):
            raise TrainingError("Each split must use unique portable sample IDs.")
    return data, seal


@training_data_app.command()
def imbalance(
    dataset: DatasetPath,
    manifest: Annotated[Path | None, typer.Option(help="Candidate manifest for manufacturer and group metadata.")] = None,
    include_test: Annotated[bool, typer.Option(help="Include Test annotation counts for a final dataset audit.")] = False,
    out: Annotated[Path | None, typer.Option(help="Fresh output directory. Omit to print JSON only.")] = None,
) -> None:
    """Report image and object support, including multi-label and background images."""
    with _command(dataset, manifest, out) as credentials:
        dataset = dataset.resolve()
        data, seal = _inspect(dataset, credentials, _progress())
        bundle_manifest = dataset.parent / "manifest.csv"
        if manifest is None and seal is not None and bundle_manifest.is_file():
            manifest = bundle_manifest
            credentials.protect_path(str(manifest.resolve()), "split manifest")
        test = None
        if include_test:
            path = dataset / "coco" / "annotations" / "instances_test.json"
            if not path.is_file() or not path.resolve().is_relative_to(dataset):
                raise TrainingError("No Test annotation file is available in this conversion.")
            test = read_coco_split(dataset / "coco", "test")
        report = imbalance_report(dataset, data, seal, manifest, test)
        document = json.dumps(report, ensure_ascii=False, indent=2)
        credentials.protect_path(document, "imbalance report")
        if out is None:
            typer.echo(document)
        else:
            with fresh_output(out, dataset) as temporary:
                write_json(temporary / "imbalance.json", report)
            typer.echo(str(out / "imbalance.json"))


@training_data_app.command()
def preview(
    dataset: DatasetPath,
    config_dir: ProfilesPath,
    model: Annotated[ModelName, typer.Option(help="Detector YAML supplying the augmentation settings and seed.")],
    out: Annotated[Path, typer.Option(help="Fresh directory for overlays and preview.json.")],
    sample_id: Annotated[list[str] | None, typer.Option(help="Train sample ID. Repeat to preview multiple samples.")] = None,
    limit: Annotated[int, typer.Option(min=1, max=30, help="Maximum examples when no sample IDs are specified.")] = 5,
) -> None:
    """Render originals and transformed Train examples with both sets of boxes."""
    with _command(dataset, config_dir, out) as credentials:
        dataset = dataset.resolve()
        config, profile_hash = read_training_profile(model, config_dir)
        policy = config.optimization.augmentation if config.optimization else None
        if policy is None:
            raise TrainingError("The selected profile must specify optimization.augmentation.")
        progress = _progress()
        data, seal = _inspect(dataset, credentials, progress)
        train = {sample.image.stem: sample for sample in data.splits["train"].samples}
        ids = list(dict.fromkeys(sample_id)) if sample_id else sorted(train)[:limit]
        if len(ids) > 30 or any(name not in train for name in ids):
            raise TrainingError("Select up to 30 sample IDs from Train.")
        with fresh_output(out, dataset) as temporary:
            examples = []
            progress("render Train previews", 0, len(ids))
            for index, name in enumerate(ids, 1):
                examples.append(preview_sample(train[name], policy, config.seed, temporary / f"{index:03d}.png"))
                progress("render Train previews", index, len(ids))
            report = {
                "schema_version": 1, "split": "train", "seed": config.seed,
                "profile_sha256": profile_hash, "augmentation": policy.model_dump(),
                "split_revision": digest_json(seal) if seal else None,
                "image_fingerprint": data.split_fingerprints["train"],
                "annotation_fingerprint": data.split_annotation_fingerprints["train"], "examples": examples,
            }
            credentials.protect_path(json.dumps(report, ensure_ascii=False), "augmentation preview report")
            write_json(temporary / "preview.json", report)
        typer.echo(str(out / "preview.json"))


@training_data_app.command()
def ablation(
    dataset: DatasetPath,
    config_dir: ProfilesPath,
    model: Annotated[list[ModelName] | None, typer.Option(help="Detector to plan. Repeat, or omit for all three.")] = None,
    policy: Annotated[Path | None, typer.Option(help="Optional Optimization YAML for the combined variant.")] = None,
    evaluation_config: Annotated[Path | None, typer.Option(help="Optional common EvaluationConfig YAML.")] = None,
    out: Annotated[Path | None, typer.Option(help="Fresh directory for experiment.json and variant YAMLs.")] = None,
    dry_run: Annotated[bool, typer.Option(help="Validate and print the pending plan without writing files.")] = False,
) -> None:
    """Validate pinned inputs and preview or publish pending pretrained experiments."""
    with _command(dataset, config_dir, policy, evaluation_config, out) as credentials:
        dataset = dataset.resolve()
        names = tuple(dict.fromkeys(model)) if model else tuple(ModelName)
        settings = _read_config(policy, Optimization) if policy else DEFAULT_ABLATION
        evaluation = _read_config(evaluation_config, EvaluationConfig) if evaluation_config else EvaluationConfig()
        if evaluation.precision != "fp32" and evaluation.device in ("cpu", "mps"):
            raise TrainingError("FP16 evaluation requires CUDA or ROCm.")
        progress = _progress()
        data, seal = _inspect(dataset, credentials, progress)
        yolo = None
        if ModelName.YOLO in names or (dataset / "yolo").is_dir():
            if not (dataset / "yolo").resolve().is_relative_to(dataset):
                raise TrainingError("The YOLO conversion must resolve inside the selected dataset.")
            yolo = inspect_data(dataset, "yolo", credentials, progress)
            verify_shared_labels(yolo, data)
        report = experiment_manifest(dataset, data, seal, names, config_dir, evaluation, settings, credentials, yolo)
        document = json.dumps(report, ensure_ascii=False, indent=2)
        credentials.protect_path(document, "ablation manifest")
        if out is None or dry_run:
            typer.echo(document)
        else:
            require_separate_output(config_dir, out)
            with fresh_output(out, dataset) as temporary:
                for variant, configs in report["variants"].items():
                    directory = temporary / variant
                    directory.mkdir()
                    for name, record in configs.items():
                        (directory / f"{name}.yaml").write_text(
                            yaml.safe_dump(record["configuration"], sort_keys=False), encoding="utf-8",
                        )
                write_json(temporary / "experiment.json", report)
            typer.echo(str(out / "experiment.json"))
