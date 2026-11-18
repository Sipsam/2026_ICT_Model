"""Export candidate manifest rows as equivalent YOLO and COCO datasets."""

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Literal
from ..model import Progress
from .common import detection_candidates, require_separate_output, safe_component
from .policy import CLASS_MAP, NAMES, NORMAL_IDS, SPLIT_MAP

def link_or_copy(src: Path, dst: Path, mode: Literal["link", "copy"], fallback: Counter[str]) -> None:
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


def check_source(json_path: Path, sample_id: str) -> tuple[int, int, list[tuple[int, list[float]]]]:
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



def convert_detection(
    root: Path, manifest: Path, out: Path, *, image_mode: Literal["link", "copy"] = "copy",
    limit: int = 0, dry_run: bool = False, progress: Progress | None = None,
    published_out: Path | None = None, restrict_sources: bool = True,
) -> dict[str, object]:
    root, manifest, out = root.resolve(), manifest.resolve(), out.resolve()
    if limit < 0:
        raise ValueError("Candidate limit must be non-negative.")
    rows = detection_candidates(manifest)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No candidate rows; check manifest flags")
    seen = set()
    split_counts = Counter()
    for r in rows:
        split = r.get("split")
        if split not in SPLIT_MAP:
            raise ValueError(f"Unknown split: {split}")
        stem = r["sample_id"]
        if not safe_component(stem):
            raise ValueError("Manifest contains an invalid sample ID.")
        key = (split, stem)
        if key in seen:
            raise ValueError(f"Duplicate sample ID in split: {key}")
        seen.add(key)
        split_counts[split] += 1
    if dry_run:
        return {"selected_manifest_rows": len(rows), "by_split": dict(split_counts)}
    if restrict_sources:
        require_separate_output(root, out)
    out.mkdir(parents=True, exist_ok=False)
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
    if progress:
        progress(0, len(rows))
    for done, r in enumerate(rows, 1):
        split = SPLIT_MAP[r["split"]]
        stem = r["sample_id"]
        src_image = (root / r["image_path"]).resolve()
        src_json = (root / r["annotation_path"]).resolve()
        if restrict_sources and any(not path.is_relative_to(root) for path in (src_image, src_json)):
            raise ValueError("Manifest paths must stay within the dataset root.")
        if not src_image.is_file() or not src_json.is_file():
            raise FileNotFoundError(f"{stem}: JPG/JSON missing")
        width, height, boxes = check_source(src_json, stem)
        # Keep source extension (.JPG/.jpg) to avoid changing actual encoded bytes.
        image_name = f"{stem}{src_image.suffix}"
        for typ in ("yolo", "coco"):
            link_or_copy(src_image, out / typ / "images" / split / image_name, image_mode, transfer)
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
        if progress:
            progress(done, len(rows))
    for split in ("train", "val"):
        filename = "instances_train.json" if split == "train" else "instances_val.json"
        with (out / "coco" / "annotations" / filename).open("w", encoding="utf-8") as f:
            json.dump(coco[split], f, ensure_ascii=False)
    # Output absolute path to avoid CWD-dependent Ultralytics resolution.
    yolo_dir = ((published_out or out) / "yolo").resolve()
    (out / "yolo" / "data.yaml").write_text(
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
    return summary
