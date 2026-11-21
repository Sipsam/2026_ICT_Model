"""Shared file and row contracts for SMT preprocessing."""

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path, PureWindowsPath

from ...pipeline.store import digest_json

Row = dict[str, str | int | bool]
CsvRow = dict[str, str]


def raw_file_sha256(path: Path) -> str:
    """Hash file bytes without incorporating the file name or tree metadata."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path: Path) -> list[CsvRow]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[Row], fields: list[str]) -> None:
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, report: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)


def detection_candidates(manifest: Path) -> list[CsvRow]:
    return [
        row for row in read_csv(manifest)
        if row.get("decision") == "candidate" and row.get("use_for_detection") == "candidate"
    ]


def split_fingerprint(rows: Iterable[Mapping[str, object]]) -> str:
    return digest_json(sorted((row["split"], row["folder"], row["sample"]) for row in rows))


def require_separate_output(source: Path, output: Path) -> None:
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError("Choose a separate output directory outside the input data.")


def safe_component(name: str) -> bool:
    return (
        bool(name) and name.isprintable() and not name.endswith((".", " "))
        and not any(char in name for char in '/\\:<>"|?*')
        and not PureWindowsPath(name).is_reserved()
    )
