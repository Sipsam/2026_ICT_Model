#!/usr/bin/env python3
"""Validate exported YOLO/COCO datasets for one-to-one sample and box consistency.
Run from project root: python scripts/validate_detection.py
Read-only: writes reports only to outputs/detection_validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

EXPECTED = ["cold_solder", "misalignment", "short", "missing_component", "tombstone"]
EXPECTED_SOURCE = {"23": 0, "24": 1, "25": 2, "27": 3, "30": 4}


def add_error(errors, split, sample, kind, detail):
    errors.append({"split": split, "sample": sample, "kind": kind, "detail": str(detail)})


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def check(split, base, manifest_rows, errors, counters, tol):
    image_dir = base / "yolo" / "images" / split
    coco_img_dir = base / "coco" / "images" / split
    label_dir = base / "yolo" / "labels" / split
    ann_path = base / "coco" / "annotations" / f"instances_{split}.json"
    with ann_path.open(encoding="utf-8") as f:
        coco = json.load(f)
    cats = sorted((c["id"], c["name"]) for c in coco["categories"])
    if cats != [(i + 1, name) for i, name in enumerate(EXPECTED)]:
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

    for rec in tqdm(manifest_rows, desc=f"검증 {split}", unit="image", dynamic_ncols=True):
        stem = rec["sample_id"]
        record = by_stem.get(stem)
        if record is None:
            continue
        name = record["file_name"]
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
            counters[f"objects_{EXPECTED[item[0]]}"] += 1

    for path in label_dir.glob("*.txt"):
        if path.stem not in stems:
            add_error(errors, split, path.stem, "unexpected_yolo_label", path.name)
    for path in image_dir.iterdir():
        if path.is_file() and path.stem not in stems:
            add_error(errors, split, path.stem, "unexpected_yolo_image", path.name)
    for path in coco_img_dir.iterdir():
        if path.is_file() and path.stem not in stems:
            add_error(errors, split, path.stem, "unexpected_coco_image", path.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    p.add_argument("--dataset", type=Path, default=Path("data/processed/detection"))
    p.add_argument("--out", type=Path, default=Path("outputs/detection_validation"))
    p.add_argument("--tolerance", type=float, default=0.000001)
    args = p.parse_args()
    rows = load_csv(args.manifest)
    selected = [r for r in rows if r.get("decision") == "candidate" and r.get("use_for_detection") == "candidate"]
    grouped = {"train": [], "val": []}
    for r in selected:
        mapped = {"train": "train", "validation": "val"}.get(r["split"])
        if not mapped:
            raise ValueError(f"Unknown manifest split: {r['split']}")
        grouped[mapped].append(r)
    errors = []
    counters = Counter()
    for split in ("train", "val"):
        check(split, args.dataset, grouped[split], errors, counters, args.tolerance)
    with (args.dataset / "conversion_summary.json").open(encoding="utf-8") as f:
        source_summary = json.load(f)
    for key, value in source_summary.get("counts", {}).items():
        if counters[key] != value:
            add_error(errors, "both", "-", "summary_count_mismatch", f"{key}: actual={counters[key]}, recorded={value}")
    if len(selected) != source_summary.get("selected_manifest_rows"):
        add_error(errors, "both", "-", "selected_rows_mismatch", f"actual={len(selected)}")
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "issues.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["split", "sample", "kind", "detail"])
        w.writeheader()
        w.writerows(errors)
    report = {"selected_images": len(selected), "counts": dict(counters),
              "issue_count": len(errors), "issue_types": dict(Counter(e["kind"] for e in errors)),
              "passed": len(errors) == 0}
    (args.out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 데이터셋 변환 검증 완료 ===")
    print(f"검증 이미지: {len(selected):,} | 발견 문제: {len(errors):,} | {'PASS' if not errors else 'REVIEW REQUIRED'}")
    print("결과:", args.out)
    if errors:
        print("문제별 건수:", report["issue_types"])
        raise SystemExit(1)


if __name__ == "__main__":
    main()
