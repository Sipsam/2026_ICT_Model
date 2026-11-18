"""Production DAG declarations for dataset and model stages."""

from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path

from ..classification import stages as classification_stages
from ..classification.contracts import LABEL_POLICY, SCHEMA_VERSION, ClassificationConfig
from ..data.preprocessing import stages as preprocessing
from ..data.preprocessing.policy import POLICY_VERSION
from ..data.stages import DataStages
from .model import Stage, StageContext, StageInput, StageResult
from .store import digest_json, fingerprint_path

CLASSIFICATION_STAGES = ("classification_prepare", "classification_train", "classification_evaluate")


def pipeline_stages(
    data: DataStages | None = None, *, dataset_root: Path | None = None,
    classification: ClassificationConfig | None = None, approve_train: bool = False,
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
    detection = artifact("convert", "detection")
    validation = artifact("validate_detection", "summary.json", name="detection_validation")
    checkpoints = artifact("train", "checkpoints")

    existing = (StageInput("existing_dataset", data.existing),) if data and data.existing else ()
    data_config = ("dataset_source", "data_path", "raw_path")
    data_version = f"{revision}:ingestion-v1:{data.acquisition.selection.fingerprint}" if data else revision
    preprocessing_version = f"{revision}:preprocessing-v1:{POLICY_VERSION}"

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
        stage("validate_detection", (manifest, detection), ("issues.csv", "summary.json"),
              execute=preprocessing.validate, stage_version=preprocessing_version),
        stage("train", (detection, validation), ("checkpoints",), ("model", "device", "training_profile", "tracking")),
        stage("validate_models", (detection, checkpoints), ("predictions",), ("model", "device")),
        stage("evaluate", (detection, artifact("validate_models", "predictions")), ("metrics.json",)),
        stage("report", (artifact("evaluate", "metrics.json", name="metrics"),), ("report.html",)),
    )
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
