"""Regression tests for the production-contract Test Release Gate."""

import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image
from openpyxl import Workbook

from ict_pipeline.data.preprocessing.common import write_csv
from ict_pipeline.data.test_release_gate import (
    _prior_reservation_conflicts,
    read_public_index,
    verify_test_payload_seal,
)
from ict_pipeline.evaluation.product import read_artifact, write_artifact
from ict_pipeline.evaluation.product_protocol import FrozenValidation
from ict_pipeline.pipeline.store import digest_json, fingerprint_path
from ict_pipeline.evaluation.limited_holdout import (
    _subset_metrics,
    LIMITATION_FIELDS,
    evaluate_limited_holdout,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _public_index(tmp_path: Path, root: Path, revision: str = "a" * 64) -> tuple[Path, dict]:
    image = root / "test-001.jpg"
    annotation = root / "test-001.json"
    sensor = root / "test-001.xlsx"
    Image.new("RGB", (8, 8), "red").save(image)
    annotation.write_text(json.dumps({
        "images": [{"id": 1, "file_name": "test-001.jpg"}],
        "annotations": [],
    }), encoding="utf-8")
    workbook = Workbook()
    workbook.active.append(["timestamp", "temperature"])
    workbook.active.append([1, 20])
    workbook.save(sensor)
    workbook.close()
    samples = [{
        "sample_id": "test-001",
        "image_sha256": _sha(image),
        "annotation_sha256": _sha(annotation),
        "sensor_sha256": _sha(sensor),
    }]
    identity = {
        "schema_version": 1,
        "revision_id": revision,
        "publication_fingerprint": "b" * 64,
        "manifest_fingerprint": "c" * 64,
        "samples": samples,
    }
    payload = {**identity, "fingerprint": digest_json(identity)}
    path = tmp_path / "test-public-index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _seal_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    root = tmp_path / "source"
    root.mkdir()
    index, index_payload = _public_index(tmp_path, root)
    bundle = tmp_path / "bundle"
    coco = bundle / "detection" / "coco" / "annotations"
    coco.mkdir(parents=True)
    row = {
        "split": "test", "sample_id": "test-001",
        "image_path": "test-001.jpg", "annotation_path": "test-001.json",
        "sensor_path": "test-001.xlsx", "decision": "candidate",
    }
    write_csv(bundle / "manifest.csv", [row],
              ["split", "sample_id", "image_path", "annotation_path", "sensor_path", "decision"])
    revision = {
        "schema_version": 1, "revision_id": "a" * 64,
        "selected_manifest_fingerprint": "c" * 64,
    }
    (bundle / "split.json").write_text(json.dumps(revision), encoding="utf-8")
    (coco / "instances_test.json").write_text(json.dumps({
        "images": [{"file_name": "test-001.jpg"}], "annotations": [],
    }), encoding="utf-8")
    preflight = {
        "schema_version": 1, "stage": "preflight", "passed": True,
        "seal_verification_allowed": True, "final_evaluation_allowed": False,
        "test_sample_ids": ["test-001"],
        "historical_reservation_status": "unverified",
        "group_split_revision": "a" * 64,
        "publication_fingerprint": "b" * 64,
        "manifest_fingerprint": "c" * 64,
        "cohort_fingerprint": "d" * 64,
        "frozen_validation_fingerprint": "e" * 64,
        "preprocessing_fingerprint": "f" * 64,
        "group_verification_fingerprint": "1" * 64,
        "prior_test_manifest_fingerprints": [],
        "evaluation_plan": {"threshold": 0.5, "test_sample_fingerprint": digest_json(["test-001"])},
        "public_test_hash_index": {
            "status": "pinned_verified_metadata_only",
            "fingerprint": fingerprint_path(index),
        },
    }
    preflight["fingerprint"] = digest_json({
        key: value for key, value in preflight.items() if key != "fingerprint"
    })
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    return root, bundle, index, preflight


def test_public_index_requires_all_raw_sha256_hashes(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    path, payload = _public_index(tmp_path, root)
    rows = read_public_index(path)
    assert rows["test-001"]["image_sha256"] == _sha(root / "test-001.jpg")

    payload["samples"][0].pop("sensor_sha256")
    payload["fingerprint"] = digest_json({
        key: payload[key] for key in ("schema_version", "revision_id",
                                      "publication_fingerprint", "manifest_fingerprint", "samples")
    })
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="sensor_sha256"):
        read_public_index(path)


def test_frozen_validation_uses_official_artifact_loader(tmp_path: Path) -> None:
    from ict_pipeline.evaluation.product_protocol import ValidationSample
    frozen = FrozenValidation(
        dataset_fingerprint="a" * 64, split_fingerprint="b" * 64,
        samples=(ValidationSample(
            sample_id="test-001", image_sha256="c" * 64, image_size=(8, 8),
            sensor_fingerprint="d" * 64, capture_timestamp="2026-01-01T00:00:00+00:00",
            annotation_sha256="e" * 64, targets=None,
        ),),
    )
    path = tmp_path / "frozen.json"
    write_artifact(path, frozen)
    assert read_artifact(path, FrozenValidation) == frozen


def test_historical_reservation_missing_and_conflict_are_distinct(tmp_path: Path) -> None:
    prior = tmp_path / "prior.csv"
    write_csv(prior, [{
        "split": "test", "sample_id": "test-001",
        "image_path": "test-001.jpg", "annotation_path": "test-001.json",
        "sensor_path": "", "decision": "candidate",
    }], ["split", "sample_id", "image_path", "annotation_path", "sensor_path", "decision"])
    development = [{
        "split": "train", "sample_id": "test-001",
        "image_path": "test-001.jpg", "annotation_path": "test-001.json",
        "sensor_path": "",
    }]
    _, conflicts = _prior_reservation_conflicts(development, (prior,))
    assert conflicts and "custody conflicts" in conflicts[0]
    _, conflicts = _prior_reservation_conflicts(development, (tmp_path / "missing.csv",))
    assert conflicts and "missing" in conflicts[0]


def test_seal_requires_preflight_revalidation_inputs(tmp_path: Path) -> None:
    _, bundle, index, preflight = _seal_fixture(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    with pytest.raises(ValueError, match="revalidation inputs"):
        verify_test_payload_seal(
            root=tmp_path / "source", bundle=bundle,
            preflight_receipt=preflight, preflight_path=preflight_path,
            public_index=index,
        )


def test_limited_holdout_requires_explicit_test_access(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="approve-test-access"):
        evaluate_limited_holdout(
            root=tmp_path / "source", bundle=tmp_path / "bundle",
            revision_id="a" * 64, preflight=tmp_path / "preflight.json",
            group_verification=tmp_path / "verification.json",
            cohort=tmp_path / "cohort.json", frozen_validation=tmp_path / "frozen.json",
            preprocessing=tmp_path / "sensor-state.json",
            image_run=tmp_path / "image", concat_run=tmp_path / "concat",
            out=tmp_path / "holdout", approve_test_access=False,
        )


def test_limited_holdout_subgroup_metrics_and_limitations() -> None:
    ids = ["test-a", "test-b", "test-c", "test-d"]
    scores = {
        "test-a": (0.9, 0.1, 0.1, 0.1, 0.1),
        "test-b": (0.1, 0.9, 0.1, 0.1, 0.1),
        "test-c": (0.1, 0.1, 0.1, 0.1, 0.1),
        "test-d": (0.9, 0.1, 0.9, 0.1, 0.1),
    }
    records = {
        sample_id: {"labels": tuple(index == (0 if sample_id != "test-b" else 1)
                                    for index in range(5))}
        for sample_id in ids
    }
    metrics = _subset_metrics(ids, scores, records, ids)
    assert metrics["sample_count"] == 4
    assert set(metrics) >= {"macro_auroc", "macro_f1", "micro_f1",
                            "positive_supported_macro_f1", "classes"}
    assert all("tp" in row and "fp" in row and "fn" in row and "tn" in row
               and "fpr" in row for row in metrics["classes"])
    assert LIMITATION_FIELDS == {
        "evaluation_origin": "internal_same_source",
        "evaluation_protocol": "limited_internal_holdout",
        "historical_reservation_status": "unverified",
        "historical_payload_seal_verified": False,
        "external_independence_verified": False,
        "production_session_independence_verified": False,
    }


def test_limited_holdout_cli_exposes_device_batch_seed_and_checkpoint(tmp_path: Path) -> None:
    from typer.testing import CliRunner
    from ict_pipeline.cli import app

    result = CliRunner().invoke(app, ["integrity", "limited-holdout-evaluate", "--help"])
    assert result.exit_code == 0
    assert "--device" in result.output
    assert "--batch-size" in result.output
    assert "--seed" in result.output
    assert "--checkpoint-dir" in result.output
