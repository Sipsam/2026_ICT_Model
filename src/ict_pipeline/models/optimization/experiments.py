"""Plan controlled pretrained ablations without invoking a model backend."""

from pathlib import Path
from typing import Any

from ...config.settings import Credentials
from ...data.preprocessing.policy import label_policy
from ...evaluation.contracts import EvaluationConfig
from ...evaluation.runner import evaluation_code, evaluation_protocol
from ...pipeline.store import digest_json
from ..contracts import ModelName, TrainConfig, TrainingError
from ..dataset import DetectionData
from ..registry import REGISTRY
from ..training import read_training_profile, training_code, training_configuration
from ..weights import _verify_hash, checkpoint_source
from .config import Augmentation, Balancing, Geometric, Optimization, Photometric
from .sampling import cohort_record, select_train

DEFAULT_ABLATION = Optimization(
    augmentation=Augmentation(
        photometric=Photometric(brightness=0.08, contrast=0.08, noise=3),
        geometric=Geometric(rotation_degrees=3, translation_fraction=0.02, scale_delta=0.03, crop_fraction=0.02),
    ),
    balancing=Balancing(method="weighted"),
)


def experiment_manifest(
    dataset: Path, data: DetectionData, seal: dict[str, Any] | None, names: tuple[ModelName, ...],
    profiles: Path, evaluation: EvaluationConfig, policy: Optimization, credentials: Credentials,
    yolo: DetectionData | None = None,
) -> dict[str, Any]:
    photo = policy.augmentation.photometric if policy.augmentation else None
    geo = policy.augmentation.geometric if policy.augmentation else None
    if photo is None or geo is None or policy.balancing is None:
        raise TrainingError("The ablation policy requires photometric, geometric, and balancing settings.")
    if not any((photo.brightness, photo.contrast, photo.noise, photo.saturation)) or not any((
        geo.rotation_degrees, geo.translation_fraction, geo.scale_delta, geo.crop_fraction,
        geo.horizontal_flip_probability,
    )):
        raise TrainingError("The ablation policy requires at least one active setting in each augmentation section.")
    variants = {
        "pretrained_baseline": Optimization(),
        "photometric": Optimization(augmentation=Augmentation(photometric=photo)),
        "geometric": Optimization(augmentation=Augmentation(geometric=geo)),
        "class_balancing": Optimization(balancing=policy.balancing),
        "combined": policy,
    }
    train = data.splits["train"].samples
    matrix: dict[str, dict[str, object]] = {name: {} for name in variants}
    model_protocol = {}
    for name in names:
        backend = REGISTRY[name]
        if evaluation.device not in backend.devices:
            raise TrainingError(f"The evaluation device is unsupported for {name}.")
        config, profile_fingerprint = read_training_profile(name, profiles)
        if config.checkpoint is None or config.optimization is not None:
            raise TrainingError("Select pretrained base profiles without an optimization section.")
        source = checkpoint_source(name, config, profiles, credentials)
        assert source is not None
        _verify_hash(source, config.checkpoint.sha256)
        base = training_configuration(config)
        base["checkpoint"] = {**config.checkpoint.model_dump(mode="json"), "source": str(source)}
        model_protocol[name] = {
            "base_profile_sha256": profile_fingerprint, "seed": config.seed,
            "initialization": base["checkpoint"], "best_checkpoint_rule": backend.best_criterion,
            "dataset_format": backend.data_format,
        }
        for variant, settings in variants.items():
            resolved = TrainConfig.model_validate({**base, "optimization": settings})
            selected = select_train(train, settings.balancing, resolved.seed)
            matrix[variant][name] = {
                "status": "pending", "results": None, "configuration": training_configuration(resolved),
                "train_cohort": cohort_record(train, selected), "training_code": training_code(resolved),
            }
    manifest = {
        "schema_version": 1,
        "status": "pending",
        "dataset": {
            "path": str(dataset), "selected_split": seal,
            "split_fingerprints": data.split_fingerprints,
            "annotation_fingerprints": {"coco": data.split_annotation_fingerprints} | (
                {"yolo": yolo.split_annotation_fingerprints} if yolo is not None else {}
            ),
            "label_policy": label_policy(),
            "test_role": "reserved_for_final_evaluation",
            "test_origin": seal["split"].get("test_origin") if seal else None,
        },
        "evaluation": {
            "split": "val", "checkpoint": "best", "protocol": evaluation_protocol(evaluation),
            "code_fingerprints": evaluation_code(),
        },
        "models": model_protocol,
        "variants": matrix,
    }
    return {**manifest, "fingerprint": digest_json(manifest)}
