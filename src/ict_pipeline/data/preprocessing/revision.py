"""Bind verified conversion bytes and label policy to a split revision."""

import json
from pathlib import Path
from typing import Any

from ...pipeline.store import _reject_link, digest_json, fingerprint_path
from .common import CsvRow, read_csv
from .policy import label_policy

SEAL = "split_revision.json"
FORMATS = ("yolo", "coco")


def read_publication(bundle: Path, revision_id: str) -> tuple[dict, str, list[CsvRow]]:
    """Check publication metadata without reading either conversion tree."""
    revision = json.loads((bundle / "split.json").read_bytes())
    seal_path = bundle / "detection" / SEAL
    _reject_link(seal_path.parent)
    _reject_link(seal_path)
    seal = json.loads(seal_path.read_bytes())
    receipt = seal["record"]
    rows = read_csv(bundle / "manifest.csv")
    if (
        revision["schema_version"] != 1 or revision["revision_id"] != revision_id
        or digest_json({k: v for k, v in revision.items() if k != "revision_id"}) != revision_id
        or digest_json(rows) != revision["selected_manifest_fingerprint"]
        or digest_json(revision["label_policy"]) != digest_json(label_policy())
        or receipt["schema_version"] != 1 or digest_json(receipt) != seal["fingerprint"]
        or receipt["split"] != {k: v for k, v in revision.items() if k != "assignments"}
        or set(receipt["formats"]) != set(FORMATS)
    ):
        raise ValueError("Use the explicitly pinned sealed split bundle and its original manifest.")
    return revision, seal["fingerprint"], rows


def _fingerprints(dataset: Path) -> dict[str, str]:
    return {name: fingerprint_path(dataset / name) for name in FORMATS}


def seal_dataset(dataset: Path, revision: dict) -> dict[str, object]:
    provenance = {key: value for key, value in revision.items() if key != "assignments"}
    record = {"schema_version": 1, "split": provenance, "formats": _fingerprints(dataset)}
    return {"fingerprint": digest_json(record), "record": record}


def read_dataset_seal(dataset: Path) -> dict[str, Any] | None:
    path = dataset / SEAL
    if not path.exists() and not path.is_symlink():
        return None
    try:
        if path.is_symlink():
            raise ValueError
        payload = json.loads(path.read_text(encoding="utf-8"))
        record = payload["record"]
        if (
            record["schema_version"] != 1 or digest_json(record) != payload["fingerprint"]
            or digest_json(record["split"]["label_policy"]) != digest_json(label_policy())
            or record["formats"] != _fingerprints(dataset)
        ):
            raise ValueError
        return record
    except (OSError, ValueError, KeyError, TypeError):
        raise ValueError("The selected split revision or its verified YOLO/COCO files have changed.") from None
