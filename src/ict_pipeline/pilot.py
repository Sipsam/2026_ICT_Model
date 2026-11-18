"""Prepare and execute bounded detector pilots from a sealed split bundle."""

import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from typing import Any

import yaml

from .adaptive import Overrides, Recommendation, recommend
from .config.storage import SetupFiles
from .data.preprocessing.common import CsvRow, detection_candidates, read_csv, require_separate_output, write_csv, write_json
from .data.preprocessing.console import fresh_output, protect_source_data
from .data.preprocessing.policy import CLASS_MAP, NAMES, SPLIT_MAP, label_policy
from .data.preprocessing.revision import SEAL, read_dataset_seal, seal_dataset
from .data.preprocessing.validate import validate_detection
from .disk import require_space, volume_at
from .environment.doctor import collect_environment
from .environment.model import EnvironmentReport
from .evaluation.contracts import EvaluationConfig, EvaluationError, EvaluationSplit
from .evaluation.runner import evaluate_runs, read_report
from .models.contracts import ModelName, Progress, TrainConfig, TrainingError
from .models.dataset import inspect_data, read_coco_split, verify_shared_labels
from .models.registry import REGISTRY
from .models.training import TrainingPlan, execute_training, plan_training
from .models.weights import verify_weights
from .pipeline.model import PipelineError
from .pipeline.store import RunStore, _reject_link, digest_json, fingerprint_path, timestamp
from .training_storage import StorageEstimate, check_training_storage, storage_estimates

MAX_IMAGES_PER_SPLIT = 256
STRATA = (*NAMES, "background")


@dataclass(frozen=True)
class PilotSample:
    row: CsvRow
    filename: str
    strata: frozenset[str]


def select_groups(
    samples: list[PilotSample], limit: int, minimum: int, seed: int, split: str,
) -> list[PilotSample]:
    groups: dict[str, list[PilotSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.row["group_id"]].append(sample)
    order = sorted(groups, key=lambda group: digest_json([seed, split, group]))
    coverage = {group: Counter(label for sample in groups[group] for label in sample.strata) for group in order}
    remaining = dict.fromkeys(STRATA, minimum)
    selected: list[PilotSample] = []
    available = [group for group in order if len(groups[group]) <= limit]
    support = {label: sum(coverage[group][label] for group in available) for label in STRATA}
    while any(remaining.values()):
        focus = min((label for label in STRATA if remaining[label]), key=support.__getitem__)
        eligible = [group for group in available
                    if coverage[group][focus] and len(selected) + len(groups[group]) <= limit]
        if not eligible:
            raise TrainingError(f"Cannot fit the requested {split} coverage within the group budget. "
                                "Increase the image limit, reduce the minimum, or try another seed.")
        group = max(eligible, key=lambda candidate: sum(
            min(remaining[label], coverage[candidate][label]) for label in STRATA
        ) / len(groups[candidate]))
        selected.extend(groups[group])
        available.remove(group)
        for label in STRATA:
            remaining[label] = max(0, remaining[label] - coverage[group][label])
    for group in available:
        if len(selected) + len(groups[group]) <= limit:
            selected.extend(groups[group])
    return sorted(selected, key=lambda sample: sample.filename)


def _input_files(dataset: Path, split: str, sample: PilotSample) -> tuple[Path, ...]:
    return (
        dataset / "yolo" / "images" / split / sample.filename,
        dataset / "yolo" / "labels" / split / f"{Path(sample.filename).stem}.txt",
        dataset / "coco" / "images" / split / sample.filename,
    )


def prepare_pilot(
    workspace: Path, bundle: Path, output: Path, train_limit: int, val_limit: int, minimum: int,
    seed: int, progress: Progress,
) -> dict[str, object]:
    preparation_started = perf_counter()
    if not (1 <= minimum <= min(train_limit, val_limit) <= max(train_limit, val_limit) <= MAX_IMAGES_PER_SPLIT
            and 0 <= seed < 2**32):
        raise TrainingError("Use image limits of 1–256 per split, a positive coverage minimum, and a valid seed.")
    credentials = SetupFiles.read(workspace, None).credentials(os.environ)
    _reject_link(workspace / output)
    bundle, output = (workspace / bundle).resolve(), (workspace / output).resolve()
    for path in (bundle, output):
        credentials.protect_path(str(path), "Pilot path")
    protect_source_data(output, workspace)
    require_separate_output(workspace / "runs", output)
    require_separate_output(output, bundle)
    dataset = bundle / "detection"
    progress("verify parent split bundle", 0, 1)
    seal = read_dataset_seal(dataset)
    if seal is None or seal["split"].get("configuration", {}).get("strategy") != "group-aware":
        raise TrainingError("Use a sealed group-aware split bundle with its published manifest.")
    manifest = bundle / "manifest.csv"
    rows = read_csv(manifest)
    manifest_fingerprint = digest_json(rows)
    if manifest_fingerprint != seal["split"].get("selected_manifest_fingerprint"):
        raise TrainingError("Parent manifest does not match the verified split revision.")
    by_split: dict[str, dict[str, CsvRow]] = defaultdict(dict)
    group_splits: dict[str, str] = {}
    for row in rows:
        if row.get("decision") != "candidate" or row.get("use_for_detection") != "candidate":
            continue
        split, group, sample_id = row["split"], row["group_id"], row["sample_id"]
        if not group or group_splits.setdefault(group, split) != split or sample_id in by_split[split]:
            raise TrainingError("Parent manifest contains missing group IDs, cross-split groups or duplicate samples.")
        by_split[split][sample_id] = row
    progress("verify parent split bundle", 1, 1)

    limits = {"train": train_limit, "val": val_limit}
    selected: dict[str, list[PilotSample]] = {}
    parent_counts = {split: len(by_split[original]) for original, split in SPLIT_MAP.items()}
    if train_limit >= parent_counts["train"]:
        raise TrainingError("Pilot Train limit must be strictly smaller than the parent Train split.")
    for original, split in SPLIT_MAP.items():
        samples = read_coco_split(dataset / "coco", split).samples
        if {item.image.stem for item in samples} != by_split[original].keys():
            raise TrainingError("Parent COCO membership differs from its manifest.")
        selected[split] = select_groups([
            PilotSample(by_split[original][item.image.stem], item.image.name,
                        frozenset(NAMES[obj.category - 1] for obj in item.objects) or frozenset({"background"}))
            for item in samples
        ], limits[split], minimum, seed, original)

    fields = ["sample_id", "split", "original_split", "group_id", "decision", "use_for_detection", "target_box_count"]
    picked = [{field: sample.row[field] for field in fields} for items in selected.values() for sample in items]
    for row in picked:
        for value in row.values():
            credentials.protect_path(value, "Pilot provenance")
    revision = {
        "kind": "ict_detection_pilot", "schema_version": 1, "label_policy": label_policy(), "test_origin": None,
        "parent_revision": seal["split"]["revision_id"], "parent_detection_seal": digest_json(seal),
        "parent_manifest_fingerprint": manifest_fingerprint,
        "parent_counts": parent_counts, "grouping_methods": seal["split"]["grouping_methods"],
        "selected_manifest_fingerprint": digest_json(picked), "seed": seed, "limits": limits,
        "minimum_per_stratum": minimum, "counts": {split: len(items) for split, items in selected.items()},
        "coverage": {split: dict(Counter(label for item in items for label in sorted(item.strata)))
                     for split, items in selected.items()},
    }
    revision["revision_id"] = digest_json(revision)
    total = len(picked)
    copy_bytes = sum(path.stat().st_size for split, items in selected.items()
                     for item in items for path in _input_files(dataset, split, item))
    require_space(output, copy_bytes + 1024**2, label="pilot cohort")
    started = perf_counter()
    with fresh_output(output, bundle) as working:
        derived = working / "detection"
        (derived / "coco" / "annotations").mkdir(parents=True)
        progress("copy pilot inputs", 0, total)
        done = 0
        for split, items in selected.items():
            for item in items:
                for source in _input_files(dataset, split, item):
                    target = derived / source.relative_to(dataset)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    require_space(target, source.stat().st_size, label="pilot cohort")
                    copyfile(source, target)
                done += 1
                progress("copy pilot inputs", done, total)
            relative = Path("coco", "annotations", f"instances_{split}.json")
            payload = json.loads((dataset / relative).read_text(encoding="utf-8-sig"))
            names = {item.filename for item in items}
            images = [row for row in payload["images"] if row["file_name"] in names]
            ids = {row["id"] for row in images}
            write_json(derived / relative, {
                "images": images, "categories": payload["categories"],
                "annotations": [row for row in payload["annotations"] if row["image_id"] in ids],
            })
        (derived / "yolo" / "data.yaml").write_text(yaml.safe_dump({
            "train": "images/train", "val": "images/val", "names": dict(enumerate(NAMES)),
        }), encoding="utf-8")
        write_json(derived / "conversion_summary.json", {
            "selected_manifest_rows": total,
            "class_mapping_yolo_zero_based": CLASS_MAP,
            "class_mapping_coco_one_based": {key: index + 1 for key, index in CLASS_MAP.items()},
        })
        write_csv(working / "manifest.csv", picked, fields)
        validation = validate_detection(working / "manifest.csv", derived, working / "validation",
                                        progress=lambda done, count: progress("validate pilot labels", done, count or total))
        if not validation["passed"]:
            raise TrainingError("Pilot cohort failed detection validation.")
        verify_shared_labels(*(inspect_data(derived, fmt, credentials) for fmt in ("yolo", "coco")))
        progress("recheck parent split bundle", 0, 1)
        if read_dataset_seal(dataset) != seal:
            raise TrainingError("The parent split changed during pilot preparation.")
        for split, items in selected.items():
            for item in items:
                for source in _input_files(dataset, split, item):
                    if fingerprint_path(source) != fingerprint_path(derived / source.relative_to(dataset)):
                        raise TrainingError("A copied pilot file differs from its sealed parent input.")
        progress("recheck parent split bundle", 1, 1)
        write_json(derived / SEAL, seal_dataset(derived, revision))
        write_json(working / "cohort.json", {
            "revision": revision, "measured_input_copy_bytes": copy_bytes,
            "materialization_and_verification_seconds": perf_counter() - started,
            "preparation_wall_seconds": perf_counter() - preparation_started,
        })
    return revision


@dataclass(frozen=True)
class PilotPlan:
    training: TrainingPlan
    recommendations: dict[ModelName, Recommendation]
    storage: tuple[StorageEstimate, ...]
    cohort_record: dict[str, Any]
    summary_directory: Path
    environment: EnvironmentReport


def plan_pilot(
    workspace: Path, cohort: Path, names: tuple[ModelName, ...], config_dir: Path, epochs: int,
    overrides: Overrides, resume_from: Path | None = None,
) -> PilotPlan:
    if not 1 <= epochs <= 3:
        raise TrainingError("A pilot allows one to three epochs.")
    cohort = (workspace / cohort).resolve()
    credentials = SetupFiles.read(workspace, None).credentials(os.environ)
    credentials.protect_path(str(cohort), "Pilot cohort")
    require_separate_output(workspace / "runs", cohort)
    seal = read_dataset_seal(cohort / "detection")
    if seal is None or seal["split"].get("kind") != "ict_detection_pilot":
        raise TrainingError("Prepare a bounded cohort with ict pilot prepare before planning a run.")
    revision = seal["split"]
    record = json.loads((cohort / "cohort.json").read_text(encoding="utf-8"))
    rows = detection_candidates(cohort / "manifest.csv")
    if record["revision"] != revision or digest_json(rows) != revision["selected_manifest_fingerprint"]:
        raise TrainingError("Pilot cohort metadata changed after preparation.")
    for original, split in SPLIT_MAP.items():
        count = revision["counts"][split]
        if not (0 < count <= MAX_IMAGES_PER_SPLIT and count == sum(row["split"] == original for row in rows)):
            raise TrainingError("The prepared cohort exceeds the pilot bounds or has inconsistent membership.")
        for fmt in ("coco", "yolo"):
            images = cohort / "detection" / fmt / "images"
            if {path.name for path in images.iterdir()} != {"train", "val"} or len(list((images / split).iterdir())) != count:
                raise TrainingError("Pilot inputs must contain only the bounded Train and Validation images.")
    if revision["counts"]["train"] >= revision["parent_counts"]["train"]:
        raise TrainingError("Pilot Train must be a strict subset of parent Train.")

    attempts = cohort / "attempts"
    _reject_link(attempts)
    if resume_from is not None:
        resume_from = (workspace / resume_from).resolve()
        if resume_from.parent.parent.parent != attempts or resume_from.parent.name != "training":
            raise TrainingError("Resume from an unfinished model inside this cohort's attempts directory.")
    base = plan_training(workspace, names, cohort / "detection", attempts, config_dir, resume_from)
    summary = base.destination
    environment = collect_environment(volume_at(attempts).path)
    previous = None
    if resume_from is not None:
        run = json.loads((resume_from / "run.json").read_text(encoding="utf-8"))
        if run["status"] not in ("running", "failed", "interrupted"):
            raise TrainingError("Choose an unfinished pilot model run.")
        previous = TrainConfig.model_validate(run["configuration"])
        inherited = {key: getattr(previous, key) for key, value in asdict(overrides).items() if value is None}
        if names == (ModelName.YOLO,):
            inherited.pop("gradient_accumulation", None)
        overrides = replace(overrides, **inherited)
    recommendations = {name: recommend(name, environment, overrides) for name in names}
    models = {}
    for name, config in base.models.items():
        settings = recommendations[name]
        if config.initialization != "pretrained":
            raise TrainingError(f"{name} pilot requires a pretrained profile and staged local checkpoint.")
        batch = settings.batch_size
        if overrides.batch_size is None:
            batch = min(batch, revision["counts"]["train"])
            if batch != settings.batch_size:
                settings = recommend(name, environment, replace(overrides, batch_size=batch))
                settings = replace(settings, reasoning=(*settings.reasoning, "Batch size capped at cohort Train count."))
                recommendations[name] = settings
        if batch > min(32, revision["counts"]["train"]) or settings.image_size > 1280:
            raise TrainingError("A pilot permits batches up to 32 and Train count, and image size up to 1280.")
        models[name] = TrainConfig.model_validate({
            **config.model_dump(), "epochs": epochs, "save_period": 0, "seed": revision["seed"],
            "device": settings.device, "precision": settings.precision, "image_size": settings.image_size,
            "batch_size": batch, "workers": settings.workers, "gradient_accumulation": settings.gradient_accumulation or 1,
        })
        if previous is not None and models[name] != previous:
            raise TrainingError("Pilot resume requires the original profile, epochs, and training settings.")
        try:
            verify_weights(name, models[name], base.weight_sources.get(name), REGISTRY[name].versions())
        except TrainingError as exc:
            settings = recommendations[name]
            recommendations[name] = replace(settings, checks_passed=False,
                                            issues=tuple(dict.fromkeys((*settings.issues, str(exc)))))
    # Separate sibling directories for evaluation and training.
    training = replace(base, destination=summary / "training", models=models, monitor_resources=True,
                       dataset_revision={"selected_split": seal})
    return PilotPlan(training, recommendations, storage_estimates(training), record, summary, environment)


def _next_settings(config: TrainConfig, recommendation: Recommendation, record: dict[str, Any]) -> dict[str, object] | None:
    resources = record.get("resources", {})
    if not resources.get("adapter_wall_seconds"):
        return None
    device_peak = resources.get("peak_torch_device_reserved_bytes")
    device_free = recommendation.measured_free_device_bytes
    ram_peak = resources.get("sampled_peak_process_rss_bytes")
    ram_free = recommendation.measured_available_ram_bytes
    pressure = ((device_peak is not None and device_free and device_peak >= 0.8 * device_free)
                or (ram_peak is not None and ram_free and ram_peak >= 0.8 * ram_free))
    return {
        "device": config.device, "precision": config.precision, "image_size": config.image_size,
        "batch_size": max(1, config.batch_size // 2) if pressure else config.batch_size,
        "workers": config.workers, "gradient_accumulation": recommendation.gradient_accumulation,
        "basis": "Pilot peak approached measured free memory; reduce batch and verify again." if pressure else
                 "Start at the completed pilot settings; measure with representative data before increasing workload.",
    }


def pilot_scope(plan: PilotPlan) -> dict[str, object]:
    training = plan.training
    return training.credentials.redact({
        "output": str(plan.summary_directory), "cohort": plan.cohort_record,
        "resume_from": str(training.resume_from) if training.resume_from else None,
        "checkpoint_sources": {name: str(path) for name, path in training.weight_sources.items()},
        "storage_ready": all(item.measured_free_bytes >= item.required_bytes for item in plan.storage),
        "configuration": {name: config.model_dump(mode="json") for name, config in training.models.items()},
        "storage": [asdict(item) for item in plan.storage], "hardware": plan.environment.to_dict(),
        "recommendations": {name: asdict(settings) for name, settings in plan.recommendations.items()},
    })


def run_pilot(plan: PilotPlan, *, approved: bool, progress: Progress) -> Path:
    if not approved:
        raise TrainingError("Pilot training requires --approve-train after reviewing the scope.")
    training = plan.training
    check_training_storage(storage_estimates(training))
    for name, cfg in training.models.items():
        settings = plan.recommendations[name]
        if not settings.checks_passed:
            raise TrainingError(f"{name}. " + " ".join(settings.issues))
        verify_weights(name, cfg, training.weight_sources[name], REGISTRY[name].versions())

    directory = plan.summary_directory
    directory.mkdir(parents=True, exist_ok=False)
    store = RunStore(directory.parent, directory.name, training.credentials)
    summary: dict[str, Any] = {
        "schema_version": 1, "status": "running", "started_at": timestamp(),
        **pilot_scope(plan), "models": {}, "evaluation_split": "val",
        "code_fingerprints": {
            relative: fingerprint_path(Path(__file__).parent / relative)
            for relative in ("pilot.py", "adaptive.py", "environment/monitor.py")
        },
    }
    started = perf_counter()
    with store.lock():
        store.write_json("pilot.json", summary)
        try:
            for name, _ in execute_training(training, progress):
                _record_model(plan, summary, name)
                store.write_json("pilot.json", summary)
            for name, cfg in training.models.items():
                started_evaluation = perf_counter()
                evaluation = EvaluationConfig(image_size=cfg.image_size, device=cfg.device, precision="fp32", warmup=1)
                (report,) = evaluate_runs(
                    training.workspace, (training.destination / name,), training.dataset, EvaluationSplit.VALIDATION,
                    evaluation, directory / "validation", "best", False, None, None, progress,
                )
                summary["models"][name].update(validation_report=report.relative_to(directory).as_posix(),
                                               validation_wall_seconds=perf_counter() - started_evaluation,
                                               validation_metrics=read_report(report)["metrics"])
                store.write_json("pilot.json", summary)
            summary["status"] = "completed"
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            raise
        except (TrainingError, EvaluationError, PipelineError) as exc:
            summary.update(status="failed", error=training.credentials.redact_text(str(exc)))
            raise
        except (OSError, ValueError, RuntimeError, KeyError):
            summary.update(status="failed", error="Cannot read or publish pilot artifacts. Check storage and inputs.")
            raise TrainingError(summary["error"]) from None
        finally:
            for name in training.models:
                _record_model(plan, summary, name)
            summary.update(finished_at=timestamp(), elapsed_seconds=perf_counter() - started)
            store.write_json("pilot.json", summary)
    return directory / "pilot.json"


def _record_model(plan: PilotPlan, summary: dict[str, Any], name: ModelName) -> None:
    path = plan.training.destination / name / "run.json"
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
        summary["models"].setdefault(name, {}).update({
            key: record.get(key) for key in ("status", "configuration", "metrics", "epoch_metrics", "checkpoints",
                                             "elapsed_seconds", "resources", "overhead", "error", "run_fingerprint")
        })
        summary["models"][name]["run"] = path.relative_to(plan.summary_directory).as_posix()
        if record["status"] == "completed":
            summary["models"][name]["next_training_settings"] = _next_settings(
                plan.training.models[name], plan.recommendations[name], record,
            )
