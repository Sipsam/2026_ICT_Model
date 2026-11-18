"""The local selection and progress contracts for dataset ingestion."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

import yaml

from ..pipeline.store import digest_json
from .naming import DEFECTS


class DataError(Exception):
    """A sanitized operator-facing ingestion error."""


@dataclass(frozen=True)
class Split:
    name: str
    file_keys: tuple[int, ...]
    max_download_bytes: int


@dataclass(frozen=True)
class Selection:
    dataset_key: int
    splits: tuple[Split, ...]
    folders: dict[str, tuple[str, ...]]
    fingerprint: str
    raw_root: Path
    extracted_root: Path


def load_selection() -> Selection:
    config = yaml.safe_load(files(__package__).joinpath("aihub_smt.yaml").read_text(encoding="utf-8"))
    assert config["preserve_source_archives"] is True
    assert set(config["defects"]) == set(DEFECTS.values())
    folders = {}
    splits = []
    for split, spec in config["splits"].items():
        keys = []
        for kind, extensions in config["extensions"].items():
            keys.extend(spec[kind])
            for state in config["states"]:
                for defect in config["defects"]:
                    folders[f"{split}_{kind}_soldering_{state}_{defect}"] = tuple(extensions)
        if len(keys) != len(set(keys)) or any(type(key) is not int or key <= 0 for key in keys):
            raise DataError("The packaged AI Hub file selection is invalid.")
        splits.append(Split(split, tuple(keys), spec["max_download_bytes"]))
    return Selection(config["dataset_key"], tuple(splits), folders, digest_json(config),
                     Path(config["raw_root"]), Path(config["extracted_root"]))


Progress = Callable[[int, int | None], None]
