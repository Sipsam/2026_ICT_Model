"""Synthetic regressions for the separate Test Release Gate phases."""

import json
from pathlib import Path

import pytest
from PIL import Image

from ict_pipeline.data.preprocessing.common import write_csv
from ict_pipeline.data.test_release_gate import (
    _prior_reservation_conflicts,
    verify_test_payload_seal,
)
from ict_pipeline.pipeline.store import digest_json, fingerprint_path


def _seal_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict, list[dict[str, str]]]:
    root = tmp_path / "source"
    root.mkdir()
    Image.new("RGB", (8, 8), "red").save(root / "test-001.jpg")
    (root / "test-001.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "test-001.jpg"}],
        "annotations": [],
    }), encoding="utf-8")
    (root / "test-001.sensor.json").write_text(json.dumps({"values": [1, 2]}), encoding="utf-8")

    bundle = tmp_path / "bundle"
    coco = bundle / "detection" / "coco" / "annotations"
    coco.mkdir(parents=True)
    write_csv(bundle / "manifest.csv", [{
        "split": "test", "sample_id": "test-001",
        "image_path": "test-001.jpg", "annotation_path": "test-001.json",
        "sensor_path": "test-001.sensor.json", "decision": "candidate",
    }], ["split", "sample_id", "image_path", "annotation_path", "sensor_path", "decision"])
    (bundle / "split.json").write_text(json.dumps({"revision_id": "a" * 64}), encoding="utf-8")
    (coco / "instances_test.json").write_text(json.dumps({
        "images": [{"file_name": "test-001.jpg"}], "annotations": [],
    }), encoding="utf-8")

    index = tmp_path / "test-public-index.json"
    samples = [{
        "sample_id": "test-001",
        "image_sha256": fingerprint_path(root / "test-001.jpg"),
        "annotation_sha256": fingerprint_path(root / "test-001.json"),
        "sensor_sha256": fingerprint_path(root / "test-001.sensor.json"),
    }]
    index.write_text(json.dumps({"samples": samples}), encoding="utf-8")

    preflight = {
        "schema_version": 1, "stage": "preflight", "passed": True,
        "seal_verification_allowed": True, "test_sample_ids": ["test-001"],
        "historical_reservation_status": "unverified",
        "evaluation_plan": {"threshold": 0.5},
    }
    preflight["fingerprint"] = digest_json({key: value for key, value in preflight.items() if key != "fingerprint"})
    return root, bundle, index, preflight, [{
        "split": "test", "sample_id": "test-001",
        "image_path": "test-001.jpg", "annotation_path": "test-001.json",
        "sensor_path": "test-001.sensor.json",
    }]


def test_seal_pass_tamper_and_receipt_mismatch(tmp_path: Path) -> None:
    root, bundle, index, preflight, _ = _seal_fixture(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    receipt = verify_test_payload_seal(
        root=root, bundle=bundle, preflight_receipt=preflight,
        preflight_path=preflight_path, public_index=index,
    )
    assert receipt["passed"] is True
    assert receipt["payloads_opened"] is True
    assert receipt["payload_seal_reverified"] is True
    assert receipt["model_inference_performed"] is False
    assert receipt["performance_computed"] is False

    (root / "test-001.jpg").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="payload hash mismatch"):
        verify_test_payload_seal(
            root=root, bundle=bundle, preflight_receipt=preflight,
            preflight_path=preflight_path, public_index=index,
        )

    changed = tmp_path / "changed-preflight.json"
    changed.write_text(json.dumps({**preflight, "passed": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="Preflight passed=false"):
        verify_test_payload_seal(
            root=root, bundle=bundle, preflight_receipt={**preflight, "passed": False},
            preflight_path=changed, public_index=index,
        )


def test_historical_reservation_conflict_is_not_resolved(tmp_path: Path) -> None:
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
    assert conflicts
    assert "custody conflicts" in conflicts[0]
