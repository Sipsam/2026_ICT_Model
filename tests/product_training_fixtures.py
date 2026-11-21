"""Shared product-training profile with a synthetic reserved Test sample."""

from pathlib import Path

import pytest

from ict_pipeline.data.preprocessing.common import read_csv, write_csv, write_json
from ict_pipeline.data.preprocessing.leakage import source_hashes
from ict_pipeline.data.preprocessing.policy import label_policy
from ict_pipeline.data.preprocessing.revision import SEAL, seal_dataset
from ict_pipeline.evaluation.product import write_artifact
from ict_pipeline.evaluation.product_protocol import FrozenValidation
from ict_pipeline.multimodal.audit import audit_cohort
from ict_pipeline.multimodal.model_config import ArchitectureConfig
from ict_pipeline.multimodal.preprocessing import PreprocessingConfig, fit_preprocessor
from ict_pipeline.multimodal.training_plan import (
    ProductTrainingConfig,
    ValidationConfig,
)
from ict_pipeline.pipeline.store import digest_json

from multimodal_fixtures import make_sample


@pytest.fixture(name="profile")
def product_training_profile(
    tmp_path: Path, request: pytest.FixtureRequest
) -> ProductTrainingConfig:
    source = tmp_path / "source"
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    rows = [
        make_sample(source, index, source=group, split=provider, normal=index % 2 == 0)
        for index, group, provider in (
            (1, "A", "train"),
            (2, "A", "train"),
            (3, "B", "validation"),
            (4, "B", "validation"),
            (5, "C", "validation"),
        )
    ]
    for index, row in enumerate(rows):
        row["original_split"] = row["split"]
        row["group_id"] = f"group-{index}"
    rows[-1]["split"] = "test"
    if getattr(request, "param", False):
        extra = make_sample(source, 6, source="D", split="validation", normal=True)
        rows.append({**extra, "original_split": "validation", "group_id": "group-5"})
    manifest = bundle / "manifest.csv"
    write_csv(manifest, rows, list(rows[0]))
    revision = {
        "schema_version": 1,
        "label_policy": label_policy(),
        "selected_manifest_fingerprint": digest_json(read_csv(manifest)),
        "dataset_fingerprint": digest_json(
            [
                [
                    {
                        key: row[key]
                        for key in ("image_path", "sensor_path", "annotation_path")
                    },
                    source_hashes(source, row),
                ]
                for row in sorted(
                    rows,
                    key=lambda row: (
                        row["original_split"],
                        row["sample_id"],
                        row["annotation_path"],
                    ),
                )
            ]
        ),
        "assignments": [],
    }
    revision["revision_id"] = digest_json(revision)
    revision_path = bundle / "split.json"
    write_json(revision_path, revision)
    detection = bundle / "detection"
    for fmt in ("yolo", "coco"):
        (detection / fmt).mkdir(parents=True)
    write_json(detection / SEAL, seal_dataset(detection, revision))

    audited = audit_cohort(source, manifest, revision_path=revision_path)
    write_json(
        tmp_path / "custody-index.json",
        {
            "cohort": audited.model_dump(mode="json"),
            "fingerprint": audited.fingerprint,
        },
    )
    cohort = audited.for_product_training()
    cohort_path = tmp_path / "cohort.json"
    write_json(
        cohort_path,
        {"cohort": cohort.model_dump(mode="json"), "fingerprint": cohort.fingerprint},
    )
    state = fit_preprocessor(cohort, PreprocessingConfig())
    state_path = tmp_path / "sensor-state.json"
    write_json(state_path, state.payload())
    frozen = FrozenValidation.from_audit(
        cohort, tuple(row["sample_id"] for row in rows[2:4])
    )
    frozen_path = tmp_path / "frozen.json"
    write_artifact(frozen_path, frozen)
    return ProductTrainingConfig(
        source=source,
        bundle=bundle,
        revision_id=revision["revision_id"],
        cohort=cohort_path,
        cohort_fingerprint=cohort.fingerprint,
        preprocessing=state_path,
        architecture=ArchitectureConfig(preprocessing_fingerprint=state.fingerprint),
        branch="sensor",
        epochs=3,
        validation=ValidationConfig(
            cohort=frozen_path,
            fingerprint=frozen.identity.cohort_fingerprint,
        ),
    )
