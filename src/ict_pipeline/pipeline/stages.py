"""Production DAG declarations for dataset and model stages."""

from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from ..classification import stages as classification_stages
from ..classification.contracts import LABEL_POLICY, SCHEMA_VERSION, ClassificationConfig
from ..data.preprocessing import stages as preprocessing
from ..data.preprocessing.policy import POLICY_VERSION
from ..data.stages import DataStages
from ..config.settings import Credentials
from ..evaluation.contracts import EvaluationConfig
from .detection import DetectionStages, _package_versions
from .model import Stage, StageContext, StageInput, StageResult
from .store import digest_json, fingerprint_path

CLASSIFICATION_STAGES = ("classification_prepare", "classification_train", "classification_evaluate")


def pipeline_stages(
    data: DataStages | None = None, *, dataset_root: Path | None = None,
    classification: ClassificationConfig | None = None, approve_train: bool = False,
    detectors: DetectionStages | None = None, split_bundle: Path | None = None,
) -> tuple[Stage, ...]:
    revision = version("ict-pipeline")

    def artifact(producer: str, path: str, *, name: str | None = None) -> StageInput:
        return StageInput(name or path, Path(path), producer)

    def stage(
        name: str,
        inputs: tuple[StageInput, ...],
        outputs: tuple[str, ...],
        config_keys: tuple[str, ...] = (),
        execute: Callable[[StageContext], StageResult] | None = None,
        stage_version: str = revision,
    ) -> Stage:
        dependencies = tuple(dict.fromkeys(item.producer for item in inputs if item.producer is not None))
        return Stage(name, dependencies, inputs, tuple(map(Path, outputs)), ("execution_mode", *config_keys),
                     stage_version, execute)

    dataset = (
        artifact("extract", "dataset") if dataset_root is None else StageInput("dataset", dataset_root)
    )
    manifest = artifact("manifest", "manifest.csv", name="manifest")
    manifest_summary = artifact("manifest", "manifest_summary.json", name="manifest_summary")
    detection = artifact("convert", "detection")
    if split_bundle is not None:
        manifest = StageInput("manifest", split_bundle / "manifest.csv")
        manifest_summary = StageInput("manifest_summary", split_bundle / "split.json")
        detection = StageInput("detection", split_bundle / "detection")
    validation = artifact("validate_detection", "receipt.json", name="detection_validation")
    checkpoints = artifact("train", "checkpoints")

    existing = (StageInput("existing_dataset", data.existing),) if data and data.existing else ()
    data_config = ("dataset_source", "data_path", "raw_path")
    data_version = f"{revision}:ingestion-v1:{data.acquisition.selection.fingerprint}" if data else revision
    preprocessing_version = f"{revision}:preprocessing-v1:{POLICY_VERSION}"
    detection_version = f"{revision}:detection-v1:{fingerprint_path(Path(__file__).with_name('detection.py'))}"
    training_version = detection_version
    evaluation_version = detection_version
    if detectors:
        from ..evaluation.runner import evaluation_code
        from ..models.registry import REGISTRY

        training_version += ":" + digest_json([
            [name, detectors.model_revision(name)] for name in detectors.profiles
        ])
        evaluation_version += ":" + digest_json({
            "config": detectors.config.model_dump(mode="json"), "code": evaluation_code(),
            "packages": _package_versions(tuple({
                package for name in detectors.profiles for package in REGISTRY[name].libraries
            })),
        })
    else:
        detectors = DetectionStages(Path.cwd(), Credentials(), {}, EvaluationConfig())

    detection_stages = (
        stage("download", existing, ("archives",), data_config, data.download if data else None, data_version),
        stage("extract", (artifact("download", "archives"), *existing), ("dataset",), data_config,
              data.extract if data else None, data_version),
        stage("audit", (dataset,), ("samples.csv", "issues.csv", "summary.json"),
              execute=preprocessing.audit, stage_version=preprocessing_version),
        stage("manifest", (
            dataset,
            artifact("audit", "samples.csv", name="samples"),
            artifact("audit", "issues.csv", name="issues"),
            artifact("audit", "summary.json", name="audit_summary"),
        ), ("manifest.csv", "manifest_summary.json"),
              execute=preprocessing.manifest, stage_version=preprocessing_version),
        stage("convert", (dataset, manifest), (
            "detection", "detection/yolo/data.yaml", "detection/coco/annotations/instances_train.json",
            "detection/coco/annotations/instances_val.json", "detection/conversion_summary.json",
        ), execute=preprocessing.convert, stage_version=preprocessing_version),
        stage("validate_detection", (manifest, manifest_summary, detection), ("issues.csv", "summary.json", "receipt.json"),
              ("detection_split_bundle",) if split_bundle is not None else (),
              execute=preprocessing.validate_revision if split_bundle is not None else preprocessing.validate,
              stage_version=f"{preprocessing_version}:receipt-{'split-v1' if split_bundle is not None else 'v1'}"),
        stage("train", (detection, manifest, manifest_summary, validation), ("checkpoints",),
              ("detection_training",), detectors.train, training_version),
        stage("validate_models", (detection, checkpoints), ("predictions",),
              ("detection_evaluation",), detectors.validate_models, evaluation_version),
        stage("evaluate", (detection, checkpoints, artifact("validate_models", "predictions")),
              ("metrics.json", "model_reports"), execute=detectors.evaluate, stage_version=evaluation_version),
        stage("report", (artifact("evaluate", "metrics.json", name="metrics"),), ("report.html", "report.json"),
              execute=detectors.report,
              stage_version=f"{detection_version}:{fingerprint_path(Path(__file__).parents[1] / 'reporting/export.py')}"),
    )
    if split_bundle is not None:
        return detection_stages[5:]
    if classification is None:
        return detection_stages

    prepare, train, evaluate = CLASSIFICATION_STAGES
    classification_root = StageInput("dataset", data.existing) if data and data.existing is not None else dataset
    prepared = (artifact(prepare, "samples.jsonl", name="samples"), artifact(prepare, "alignment.json", name="alignment"))
    config_keys = ("classification",)
    code = Path(classification_stages.__file__).parent
    code_fingerprint = digest_json({file.name: fingerprint_path(file) for file in sorted(code.glob("*.py"))})
    adapters = classification_stages.ClassificationStages(classification, code_fingerprint, approved=approve_train)
    classification_version = f"{revision}:{SCHEMA_VERSION}:{LABEL_POLICY}:{POLICY_VERSION}:{code_fingerprint}"
    alignment_options = classification.model_dump(include={"sensor_value_atol", "group_policy"})
    prepare_version = f"{classification_version}:{digest_json(alignment_options)}"
    fitting_version = f"{classification_version}:{digest_json(classification.model_dump())}"
    return detection_stages + (
        stage(prepare, (classification_root,), ("samples.jsonl", "alignment.json", "alignment_issues.csv"),
              execute=adapters.prepare, stage_version=prepare_version),
        stage(train, prepared, ("checkpoint.json",), config_keys, adapters.train, fitting_version),
        stage(evaluate, (*prepared, artifact(train, "checkpoint.json", name="checkpoint")),
              ("comparison.json", "comparison.csv", "predictions.csv"), config_keys, adapters.evaluate, fitting_version),
    )
