#!/usr/bin/env python3
"""Convert the vetted SMT manifest to equivalent YOLO and COCO detection datasets.

Run from the repository root:
    python scripts/convert_detection.py --dry-run
    python scripts/convert_detection.py

Only manifest rows with decision=candidate AND use_for_detection=candidate are used.
Source files are never edited. Normal-class boxes are intentionally excluded from
5-defect detection; images with unknown defect classes are rejected defensively.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path
from tqdm import tqdm

CLASS_MAP = {23: 0, 24: 1, 25: 2, 27: 3, 30: 4}
NAMES = ["cold_solder", "misalignment", "short", "missing_component", "tombstone"]
NORMAL_IDS = {7, 8, 9, 11, 14}
SPLIT_MAP = {"train": "train", "validation": "val"}


def link_or_copy(src: Path, dst: Path, mode: str, fallback: Counter) -> None:
    if mode == "copy":
        shutil.copy2(src, dst)
        fallback["copied"] += 1
        return
    try:
        os.link(src, dst)
        fallback["linked"] += 1
    except OSError:
        shutil.copy2(src, dst)
        fallback["copied_on_link_failure"] += 1


def check_source(json_path: Path, image_path: Path, sample_id: str):
    with json_path.open(encoding="utf-8-sig") as f:
        data = json.load(f)
    images = data.get("images")
    if not isinstance(images, list) or len(images) != 1:
        raise ValueError(f"{sample_id}: exactly one images[] record required")
    meta = images[0]
    if Path(str(meta.get("file_name", ""))).stem != sample_id:
        raise ValueError(f"{sample_id}: inconsistent images.file_name")
    width, height = meta["width"], meta["height"]
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)) or width <= 0 or height <= 0:
        raise ValueError(f"{sample_id}: invalid dimensions")
    boxes = []
    for ann in data.get("annotations", []):
        cid = ann.get("category_id")
        if cid not in CLASS_MAP and cid not in NORMAL_IDS:
            raise ValueError(f"{sample_id}: unexpected category_id={cid}; do not silently label it as background")
        if ann.get("image_id") != meta.get("id"):
            raise ValueError(f"{sample_id}: annotation image_id mismatch")
        if cid not in CLASS_MAP:
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{sample_id}: malformed bbox {bbox}")
        x, y, w, h = map(float, bbox)
        if not (0 <= x < width and 0 <= y < height and w > 0 and h > 0 and x + w <= width + 0.01 and y + h <= height + 0.01):
            raise ValueError(f"{sample_id}: invalid bbox {bbox}")
        # Allow only sub-pixel floating-point overflow covered by the source audit tolerance.
        x2, y2 = min(width, x + w), min(height, y + h)
        boxes.append((cid, [x, y, x2 - x, y2 - y]))
    return int(width), int(height), boxes


def convert(args):
    root = args.root.resolve()
    manifest = (root / args.manifest).resolve()
    out = (root / args.out).resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    with manifest.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r.get("decision") == "candidate" and r.get("use_for_detection") == "candidate"]
    if args.limit:
        rows = rows[: args.limit]
    print(f"변환 대상: {len(rows):,}장 (candidate만 포함)", flush=True)
    if not rows:
        raise RuntimeError("No candidate rows; check manifest flags")
    seen = set()
    split_counts = Counter()
    for r in rows:
        split = r.get("split")
        if split not in SPLIT_MAP:
            raise ValueError(f"Unknown split: {split}")
        key = (split, r["sample_id"])
        if key in seen:
            raise ValueError(f"Duplicate sample ID in split: {key}")
        seen.add(key)
        split_counts[split] += 1
    print("분할:", dict(split_counts), flush=True)
    if args.dry_run:
        print("Dry-run 완료. 실제 파일은 생성하지 않았습니다.")
        return
    if out.exists():
        raise FileExistsError(f"출력 폴더가 이미 존재합니다: {out}\n--out으로 새 폴더를 지정하세요. 기존 데이터 보호를 위해 덮어쓰지 않습니다.")
    for typ in ("yolo", "coco"):
        for split in ("train", "val"):
            (out / typ / "images" / split).mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (out / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "coco" / "annotations").mkdir(parents=True, exist_ok=True)

    coco = {split: {"images": [], "annotations": [], "categories": [
        {"id": i + 1, "name": name, "supercategory": "smt_defect"} for i, name in enumerate(NAMES)
    ]} for split in ("train", "val")}
    image_ids = Counter()
    annotation_ids = Counter()
    stats = Counter()
    transfer = Counter()
    try:
        for r in tqdm(rows, desc="YOLO/COCO 변환", unit="image", dynamic_ncols=True):
            split = SPLIT_MAP[r["split"]]
            stem = r["sample_id"]
            src_image = (root / r["image_path"]).resolve()
            src_json = (root / r["annotation_path"]).resolve()
            if not src_image.is_file() or not src_json.is_file():
                raise FileNotFoundError(f"{stem}: JPG/JSON missing")
            width, height, boxes = check_source(src_json, src_image, stem)
            # Keep source extension (.JPG/.jpg) to avoid changing actual encoded bytes.
            image_name = f"{stem}{src_image.suffix}"
            for typ in ("yolo", "coco"):
                link_or_copy(src_image, out / typ / "images" / split / image_name, args.image_mode, transfer)
            label_path = out / "yolo" / "labels" / split / f"{stem}.txt"
            yolo_lines = []
            image_ids[split] += 1
            iid = image_ids[split]
            coco[split]["images"].append({"id": iid, "file_name": image_name, "width": width, "height": height})
            for source_id, (x, y, w, h) in boxes:
                class_id = CLASS_MAP[source_id]
                cx = (x + w / 2) / width
                cy = (y + h / 2) / height
                nw, nh = w / width, h / height
                yolo_lines.append(f"{class_id} {cx:.9f} {cy:.9f} {nw:.9f} {nh:.9f}")
                annotation_ids[split] += 1
                coco[split]["annotations"].append({
                    "id": annotation_ids[split], "image_id": iid,
                    "category_id": class_id + 1, "bbox": [x, y, w, h],
                    "area": w * h, "iscrowd": 0,
                })
                stats[f"objects_{NAMES[class_id]}"] += 1
            label_path.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            stats[f"images_{split}"] += 1
            if not boxes:
                stats[f"background_images_{split}"] += 1
        for split in ("train", "val"):
            filename = "instances_train.json" if split == "train" else "instances_val.json"
            with (out / "coco" / "annotations" / filename).open("w", encoding="utf-8") as f:
                json.dump(coco[split], f, ensure_ascii=False)
        # Output absolute path to avoid CWD-dependent Ultralytics resolution.
        yolo_dir = (out / "yolo").resolve()
        (yolo_dir / "data.yaml").write_text(
            "path: " + json.dumps(yolo_dir.as_posix(), ensure_ascii=False) + "\n"
            "train: images/train\nval: images/val\n"
            "names:\n" + "".join(f"  {i}: {name}\n" for i, name in enumerate(NAMES)),
            encoding="utf-8",
        )
        summary = {
            "selected_manifest_rows": len(rows), "counts": dict(stats),
            "image_transfer": dict(transfer), "class_mapping_yolo_zero_based": {str(k): v for k, v in CLASS_MAP.items()},
            "class_mapping_coco_one_based": {str(k): v + 1 for k, v in CLASS_MAP.items()},
            "notes": ["Only manifest candidate rows included", "All source files preserved",
                      "Normal boxes omitted; empty YOLO labels for zero-defect images",
                      "Image-level sensor labels are not created", "Non-target defect classes are not silently mapped to background"],
        }
        (out / "conversion_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    except Exception:
        print(f"\n변환 실패: 출력 폴더에 부분 결과가 남아 있을 수 있습니다: {out}")
        print("원인을 해결한 후 별도 --out 경로를 사용하세요.")
        raise
    print("\n=== 변환 완료 ===")
    print("통계:", dict(stats))
    print("이미지 연결:", dict(transfer))
    print("YOLO 설정:", out / "yolo" / "data.yaml")
    print("COCO 어노테이션:", out / "coco" / "annotations")
    print("요약:", out / "conversion_summary.json")


def main():
    parser = argparse.ArgumentParser(description="Convert SMT manifest to matching YOLO and COCO 5-class datasets")
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/detection"))
    parser.add_argument("--image-mode", choices=("link", "copy"), default="link", help="hardlink images, fall back to copy when needed")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="first N candidates, for testing")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be >= 0")
    convert(args)


if __name__ == "__main__":
    main()
