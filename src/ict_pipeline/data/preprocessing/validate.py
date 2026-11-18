"""Check exported YOLO and COCO datasets against the candidate manifest."""

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ..model import Progress
from .common import CsvRow, Row, detection_candidates, require_separate_output, safe_component, write_csv, write_json
from .policy import CLASS_MAP, NAMES, SPLIT_MAP

def add_error(errors: list[Row], split: str, sample: str, kind: str, detail: object) -> None:
    errors.append({"split": split, "sample": sample, "kind": kind, "detail": str(detail)})


def check(
    split: str, base: Path, manifest_rows: list[CsvRow], errors: list[Row],
    counters: Counter[str], tol: float, progress: Progress | None, offset: int, total: int,
) -> None:
    image_dir = base / "yolo" / "images" / split
    coco_img_dir = base / "coco" / "images" / split
    label_dir = base / "yolo" / "labels" / split
    ann_path = base / "coco" / "annotations" / f"instances_{split}.json"
    with ann_path.open(encoding="utf-8") as f:
        coco = json.load(f)
    cats = sorted((c["id"], c["name"]) for c in coco["categories"])
    if cats != [(i + 1, name) for i, name in enumerate(NAMES)]:
        add_error(errors, split, "-", "category_mapping", cats)

    by_id = {}
    by_stem = {}
    for record in coco["images"]:
        iid = record["id"]
        stem = Path(record["file_name"]).stem
        if iid in by_id:
            add_error(errors, split, stem, "duplicate_coco_image_id", iid)
        if stem in by_stem:
            add_error(errors, split, stem, "duplicate_coco_stem", stem)
        by_id[iid] = record
        by_stem[stem] = record
    anns = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("image_id") not in by_id:
            add_error(errors, split, "-", "orphan_coco_annotation", ann.get("id"))
            continue
        anns[ann["image_id"]].append(ann)

    stems = {r["sample_id"] for r in manifest_rows}
    if stems != set(by_stem):
        missing = stems - set(by_stem)
        extra = set(by_stem) - stems
        for stem in sorted(missing):
            add_error(errors, split, stem, "missing_coco_image", "not in instances JSON")
        for stem in sorted(extra):
            add_error(errors, split, stem, "unexpected_coco_image", "not a manifest candidate")

    for done, rec in enumerate(manifest_rows, offset + 1):
        if progress:
            progress(done, total)
        stem = rec["sample_id"]
        record = by_stem.get(stem)
        if record is None:
            continue
        name = record["file_name"]
        if not safe_component(name) or not safe_component(stem):
            add_error(errors, split, stem, "invalid_image_name", name)
            continue
        width, height = record.get("width"), record.get("height")
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)) or width <= 0 or height <= 0:
            add_error(errors, split, stem, "invalid_image_dimensions", f"{width}x{height}")
            continue
        for path, typ in ((image_dir / name, "yolo"), (coco_img_dir / name, "coco")):
            if not path.is_file():
                add_error(errors, split, stem, "missing_image", f"{typ}: {path}")
        label = label_dir / f"{stem}.txt"
        if not label.is_file():
            add_error(errors, split, stem, "missing_yolo_label", str(label))
            continue
        parsed = []
        for n, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            try:
                if len(parts) != 5:
                    raise ValueError("expected 5 columns")
                cls = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                if cls not in range(5) or not all(math.isfinite(v) for v in vals):
                    raise ValueError("invalid class or non-finite box")
                cx, cy, w, h = vals
                if w <= 0 or h <= 0 or cx-w/2 < -tol or cy-h/2 < -tol or cx+w/2 > 1+tol or cy+h/2 > 1+tol:
                    raise ValueError("invalid normalized bounding box")
                parsed.append((cls, cx, cy, w, h))
            except (ValueError, TypeError) as exc:
                add_error(errors, split, stem, "invalid_yolo_line", f"line {n}: {exc}")
        corresponding = anns.get(record["id"], [])
        if len(parsed) != len(corresponding):
            add_error(errors, split, stem, "box_count_mismatch", f"yolo={len(parsed)} coco={len(corresponding)}")
        target_count = rec.get("target_box_count")
        if target_count and target_count != str(len(corresponding)):
            add_error(errors, split, stem, "manifest_box_count_mismatch", f"manifest={target_count} coco={len(corresponding)}")
        expected = []
        for ann in corresponding:
            try:
                cls = int(ann["category_id"]) - 1
                x, y, w, h = map(float, ann["bbox"])
                if cls not in range(5) or not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
                    raise ValueError("invalid COCO bbox/class")
                if x < -tol or y < -tol or x+w > width+tol or y+h > height+tol:
                    raise ValueError("COCO box outside image")
                expected.append((cls, (x+w/2)/width, (y+h/2)/height, w/width, h/height))
            except (KeyError, ValueError, TypeError) as exc:
                add_error(errors, split, stem, "invalid_coco_annotation", repr(exc))
        remaining = expected.copy()
        for item in parsed:
            best_idx = next((i for i, exp in enumerate(remaining)
                             if item[0] == exp[0] and all(abs(a-b) <= tol for a, b in zip(item[1:], exp[1:]))), None)
            if best_idx is None:
                add_error(errors, split, stem, "box_geometry_mismatch", f"YOLO {item} not matched in COCO")
            else:
                remaining.pop(best_idx)
        counters[f"images_{split}"] += 1
        counters[f"objects_{split}"] += len(parsed)
        counters[f"background_images_{split}"] += not parsed
        for item in parsed:
            counters[f"objects_{NAMES[item[0]]}"] += 1

    for path in label_dir.glob("*.txt"):
        if path.stem not in stems:
            add_error(errors, split, path.stem, "unexpected_yolo_label", path.name)
    for path in image_dir.iterdir():
        if path.is_file() and path.stem not in stems:
            add_error(errors, split, path.stem, "unexpected_yolo_image", path.name)
    for path in coco_img_dir.iterdir():
        if path.is_file() and path.stem not in stems:
            add_error(errors, split, path.stem, "unexpected_coco_image", path.name)



def validate_detection(
    manifest: Path, dataset: Path, out: Path, *, tolerance: float = 1e-6,
    progress: Progress | None = None,
) -> dict[str, object]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Detection tolerance must be finite and non-negative.")
    require_separate_output(dataset, out)
    selected = detection_candidates(manifest)
    if not selected:
        raise ValueError("No candidate rows; check manifest flags.")
    grouped: dict[str, list[CsvRow]] = {"train": [], "val": []}
    for r in selected:
        mapped = SPLIT_MAP.get(r["split"])
        if not mapped:
            raise ValueError(f"Unknown manifest split: {r['split']}")
        grouped[mapped].append(r)
    errors: list[Row] = []
    counters: Counter[str] = Counter()
    if progress:
        progress(0, len(selected))
    offset = 0
    for split, rows in grouped.items():
        check(split, dataset, rows, errors, counters, tolerance, progress, offset, len(selected))
        offset += len(rows)
    with (dataset / "conversion_summary.json").open(encoding="utf-8") as f:
        source_summary = json.load(f)
    for key, value in source_summary.get("counts", {}).items():
        if counters[key] != value:
            add_error(errors, "both", "-", "summary_count_mismatch", f"{key}: actual={counters[key]}, recorded={value}")
    if len(selected) != source_summary.get("selected_manifest_rows"):
        add_error(errors, "both", "-", "selected_rows_mismatch", f"actual={len(selected)}")
    for kind, offset in (("class_mapping_yolo_zero_based", 0), ("class_mapping_coco_one_based", 1)):
        if source_summary.get(kind) != {str(source): index + offset for source, index in CLASS_MAP.items()}:
            add_error(errors, "both", "-", "source_mapping", kind)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "issues.csv", errors, ["split", "sample", "kind", "detail"])
    report: dict[str, object] = {"selected_images": len(selected), "counts": dict(counters),
              "issue_count": len(errors), "issue_types": dict(Counter(e["kind"] for e in errors)),
              "passed": len(errors) == 0}
    write_json(out / "summary.json", report)
    return report
