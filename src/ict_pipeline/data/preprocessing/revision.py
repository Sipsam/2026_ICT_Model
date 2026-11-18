"""Bind verified conversion bytes and label policy to a split revision."""

import json
from pathlib import Path
from typing import Any

from ...pipeline.store import digest_json, fingerprint_path
from .policy import label_policy

SEAL = "split_revision.json"
FORMATS = ("yolo", "coco")


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
