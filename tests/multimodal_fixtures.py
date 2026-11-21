"""Synthetic image, annotation and workbook records for multimodal tests."""

import json
from datetime import datetime, timedelta
from pathlib import Path

from openpyxl import Workbook
from PIL import Image

from ict_pipeline.data.preprocessing.audit import XLSX_FIELDS
from ict_pipeline.data.preprocessing.common import CsvRow
from ict_pipeline.data.preprocessing.policy import SENSOR_FIELDS

START = datetime(2025, 8, 7, 9, 3, 40)


def write_workbook(path: Path, steps: list[dict]) -> None:
    workbook = Workbook()
    workbook.active.append(("timestamp", *XLSX_FIELDS))
    for step in steps:
        workbook.active.append(
            (step["timestamp"], *(step[key] for key in SENSOR_FIELDS))
        )
    workbook.save(path)
    workbook.close()


def make_sample(
    root: Path,
    index: int,
    *,
    split: str = "train",
    source: str = "A",
    capture: int | None = None,
    values: tuple[float | None, ...] = (0, 4, 8),
    image: bool = True,
    workbook: bool = True,
    session: str | None = None,
    normal: bool = False,
) -> CsvRow:
    instant = START + timedelta(minutes=index if capture is None else capture)
    status = "normal" if normal else "defect"
    stem = f"SD_{'NOR' if normal else 'DEF'}_CC_{source}_{instant:%Y%m%d-%H%M%S}_{index:05}"
    row = {
        "sample_id": stem,
        "split": split,
        "source_class": "CC",
        "manufacturer": source,
        "decision": "candidate",
        "use_for_detection": "candidate",
        **{
            f"{key}_path": f"{split}_{kind}_soldering_{status}_cold_solder/{stem}{ext}"
            for key, kind, ext in (
                ("annotation", "annotations", ".json"),
                ("image", "images", ".JPG"),
                ("sensor", "sensors", ".xlsx"),
            )
        },
    }
    for key in ("annotation_path", "image_path", "sensor_path"):
        (root / row[key]).parent.mkdir(parents=True, exist_ok=True)
    steps = [
        {
            "timestamp": (instant + timedelta(seconds=time)).isoformat(sep=" "),
            **dict(zip(SENSOR_FIELDS, (value, 70, 0, 0.01, 61), strict=True)),
            "defect_detected": int(not normal),
        }
        for time, value in enumerate(values)
    ]
    document = {
        "images": [
            {"id": index, "file_name": stem + ".JPG", "width": 24, "height": 18}
        ],
        "annotations": []
        if normal
        else [{"category_id": 23, "image_id": index, "bbox": [3, 4, 9, 8]}],
        "sensor_data": [{"image_id": index + 600, "sensor_sequence": steps}],
    }
    if session is not None:
        document["production_session_id"] = session
    (root / row["annotation_path"]).write_text(json.dumps(document), encoding="utf-8")
    if image:
        Image.new("RGB", (24, 18), (index * 16 % 256, 20, 60)).save(
            root / row["image_path"]
        )
    if workbook:
        write_workbook(root / row["sensor_path"], steps)
    return row
