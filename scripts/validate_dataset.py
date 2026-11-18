#!/usr/bin/env python3
"""Audit AI Hub SMT soldering dataset without modifying source files.

Usage (from repository root):
  python scripts/validate_dataset.py
  python scripts/validate_dataset.py --deep
  python scripts/validate_dataset.py --deep --hash-images
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

DEFECT_BY_CODE = {"CC": "cold_solder", "ML": "misalignment", "SH": "short", "MS": "missing_component", "ST": "tombstone"}
CLASS_BY_CODE = {"CC": (7, 23), "ML": (8, 24), "SH": (9, 25), "MS": (11, 27), "ST": (14, 30)}
CLASS_NAMES = {
    7: "정상_냉납", 8: "정상_밀림", 9: "정상_쇼트", 11: "정상_미삽", 14: "정상_일어섬",
    17: "불량_미납", 20: "불량_납볼", 23: "불량_냉납", 24: "불량_밀림",
    25: "불량_쇼트", 26: "불량_오삽", 27: "불량_미삽", 29: "불량_뒤집힘", 30: "불량_일어섬",
}
TARGET_DEFECT_IDS = {23, 24, 25, 27, 30}
SENSOR_FIELDS = ("temperature", "humidity", "vibration", "acceleration", "noise")
XLSX_FIELDS = ("temperature(°C)", "humidity(%)", "vibration(m/s²)", "acceleration(g)", "noise(dB)")
FATAL = {"json_parse_error", "missing_image", "missing_sensor", "invalid_filename", "invalid_bbox", "invalid_category_id", "invalid_image_metadata"}
REVIEW = {
    "expected_category_missing", "folder_status_conflict", "mixed_normal_defect_boxes",
    "other_defect_class", "image_metadata_filename_mismatch", "json_sensor_status_conflict",
    "sensor_value_mismatch", "sensor_timestamp_mismatch", "sensor_length_mismatch",
    "sensor_read_error", "image_read_error", "duplicate_across_splits", "image_size_mismatch",
    "filename_status_sensor_conflict", "sensor_sequence_missing",
}


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def issue(issues: list[dict], sample: str, split: str, kind: str, detail: str) -> None:
    issues.append({"sample": sample, "split": split, "issue": kind, "detail": str(detail)})


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare_sensor(path: Path, sequence: list, issues: list[dict], stem: str, split: str, tolerance: float) -> None:
    try:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        try:
            ws = wb.active
            iterator = ws.iter_rows(values_only=True)
            head = next(iterator, None)
            if head is None:
                issue(issues, stem, split, "sensor_read_error", "empty xlsx")
                return
            indexes = {str(name).strip(): i for i, name in enumerate(head) if name is not None}
            needed = ("timestamp",) + XLSX_FIELDS
            absent = [k for k in needed if k not in indexes]
            if absent:
                issue(issues, stem, split, "sensor_read_error", f"missing xlsx columns: {absent}")
                return
            rows = list(iterator)
        finally:
            wb.close()
    except Exception as e:
        issue(issues, stem, split, "sensor_read_error", repr(e))
        return

    if len(rows) != len(sequence):
        issue(issues, stem, split, "sensor_length_mismatch", f"xlsx={len(rows)} json={len(sequence)}")
    value_diffs = 0
    timestamp_diffs = 0
    for row, sample in zip(rows, sequence):
        if not isinstance(sample, dict):
            issue(issues, stem, split, "sensor_read_error", "invalid JSON sensor step")
            break
        x_time = row[indexes["timestamp"]]
        x_text = x_time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(x_time, "strftime") else str(x_time).strip()
        j_text = str(sample.get("timestamp", "")).strip()
        if x_text != j_text:
            timestamp_diffs += 1
        for j_key, x_key in zip(SENSOR_FIELDS, XLSX_FIELDS):
            try:
                a = float(sample[j_key])
                b = float(row[indexes[x_key]])
                if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > tolerance:
                    value_diffs += 1
            except (ValueError, TypeError, KeyError, IndexError):
                value_diffs += 1
    if timestamp_diffs:
        issue(issues, stem, split, "sensor_timestamp_mismatch", f"{timestamp_diffs} different timestamps")
    if value_diffs:
        issue(issues, stem, split, "sensor_value_mismatch", f"{value_diffs} different values")


def analyze(path: Path, root: Path, deep: bool, hash_images: bool, tolerance: float,
            issues: list[dict], counts: Counter, hashes: dict, stem_splits: dict) -> dict:
    stem = path.stem
    folder = path.parent.name
    split = "train" if folder.startswith("train_annotations_") else "validation" if folder.startswith("validation_annotations_") else "unknown"
    sample = {"sample": stem, "split": split, "folder": folder, "status": "ok", "categories": "", "boxes": 0,
              "image_exists": False, "sensor_exists": False, "issue_count": 0}
    before = len(issues)
    counts[f"samples_{split}"] += 1
    stem_splits[stem].add(split)
    fields = stem.split("_")
    if len(fields) < 6 or fields[0] != "SD" or fields[1] not in ("NOR", "DEF") or fields[2] not in CLASS_BY_CODE:
        issue(issues, stem, split, "invalid_filename", stem)
        sample["status"] = "invalid"
        sample["issue_count"] = len(issues) - before
        return sample
    _, status, code = fields[:3]
    expected_id = CLASS_BY_CODE[code][0 if status == "NOR" else 1]

    state_name = "normal" if status == "NOR" else "defect"
    defect_name = DEFECT_BY_CODE[code]
    expected_folder = f"{split}_annotations_soldering_{state_name}_{defect_name}"
    if folder != expected_folder:
        issue(issues, stem, split, "invalid_filename", f"folder={folder}, expected={expected_folder}")
    image_dir = root / f"{split}_images_soldering_{state_name}_{defect_name}"
    sensor_dir = root / f"{split}_sensors_soldering_{state_name}_{defect_name}"
    image = next((image_dir / f"{stem}{ext}" for ext in (".JPG", ".jpg", ".JPEG", ".jpeg") if (image_dir / f"{stem}{ext}").is_file()), None)
    sensor = next((sensor_dir / f"{stem}{ext}" for ext in (".xlsx", ".csv") if (sensor_dir / f"{stem}{ext}").is_file()), None)
    sample["image_exists"] = image is not None
    sample["sensor_exists"] = sensor is not None
    if image is None:
        issue(issues, stem, split, "missing_image", str(image_dir))
    if sensor is None:
        issue(issues, stem, split, "missing_sensor", str(sensor_dir))
    try:
        with path.open(encoding="utf-8-sig") as f:
            data = json.load(f)
    except Exception as e:
        issue(issues, stem, split, "json_parse_error", repr(e))
        sample["status"] = "invalid"
        sample["issue_count"] = len(issues) - before
        return sample

    images = data.get("images", [])
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        issue(issues, stem, split, "invalid_image_metadata", "missing images[0]")
        width = height = 0
        image_id = None
    else:
        meta = images[0]
        width, height, image_id = meta.get("width", 0), meta.get("height", 0), meta.get("id")
        if not isinstance(width, (float, int)) or not isinstance(height, (float, int)) or width <= 0 or height <= 0:
            issue(issues, stem, split, "invalid_image_metadata", f"size={width}x{height}")
            width = height = 0
        if Path(str(meta.get("file_name", ""))).stem != stem:
            issue(issues, stem, split, "image_metadata_filename_mismatch", str(meta.get("file_name")))
    anns = data.get("annotations", [])
    if not isinstance(anns, list):
        issue(issues, stem, split, "invalid_image_metadata", "annotations not a list")
        anns = []
    cats = set()
    for ann in anns:
        if not isinstance(ann, dict):
            issue(issues, stem, split, "invalid_category_id", str(ann))
            continue
        cid = ann.get("category_id")
        if not isinstance(cid, int) or not 1 <= cid <= 32:
            issue(issues, stem, split, "invalid_category_id", str(cid))
        else:
            cats.add(cid)
            counts[f"objects_{cid}"] += 1
        if image_id is not None and ann.get("image_id") != image_id:
            issue(issues, stem, split, "invalid_image_metadata", "annotation image_id mismatch")
        box = ann.get("bbox")
        try:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("bbox must be [x,y,w,h]")
            x, y, w, h = map(float, box)
            if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 0.01 or y + h > height + 0.01:
                raise ValueError(f"bbox={box}, image={width}x{height}")
        except (ValueError, TypeError) as e:
            issue(issues, stem, split, "invalid_bbox", str(e))
    sample["boxes"] = len(anns)
    sample["categories"] = ",".join(map(str, sorted(cats)))
    counts["objects_total"] += len(anns)
    if expected_id not in cats:
        issue(issues, stem, split, "expected_category_missing", f"expected={expected_id}, actual={sorted(cats)}")
    has_normal = any(1 <= c <= 16 for c in cats)
    has_defect = any(17 <= c <= 32 for c in cats)
    if status == "NOR" and has_defect or status == "DEF" and has_normal and not has_defect:
        issue(issues, stem, split, "folder_status_conflict", f"filename={status}, categories={sorted(cats)}")
    if has_normal and has_defect:
        issue(issues, stem, split, "mixed_normal_defect_boxes", str(sorted(cats)))
    other = (cats - TARGET_DEFECT_IDS) & set(range(17, 33))
    if other:
        issue(issues, stem, split, "other_defect_class", f"outside five selected types: {sorted(other)}")
    sensor_groups = data.get("sensor_data", [])
    if not isinstance(sensor_groups, list) or not sensor_groups or not isinstance(sensor_groups[0], dict):
        issue(issues, stem, split, "sensor_sequence_missing", "no sensor_data[0]")
        seq = []
    else:
        seq = sensor_groups[0].get("sensor_sequence", [])
        if not isinstance(seq, list) or not seq:
            issue(issues, stem, split, "sensor_sequence_missing", "empty/invalid sequence")
            seq = []
        sensor_image_id = sensor_groups[0].get("image_id")
        if isinstance(image_id, int) and isinstance(sensor_image_id, int):
            counts[f"sensor_id_offset_{sensor_image_id - image_id}"] += 1
        else:
            issue(issues, stem, split, "json_sensor_status_conflict", f"invalid IDs: image={image_id}, sensor={sensor_image_id}")
    targets = {v.get("defect_detected") for v in seq if isinstance(v, dict)}
    expected_target = 0 if status == "NOR" else 1
    if targets and targets != {expected_target}:
        issue(issues, stem, split, "filename_status_sensor_conflict", f"file={status}, sensor_labels={sorted(map(str,targets))}")
    if deep and image is not None:
        try:
            from PIL import Image
            with Image.open(image) as im:
                real_size = im.size
                im.verify()
            if real_size != (width, height):
                issue(issues, stem, split, "image_size_mismatch", f"real={real_size}, json={(width,height)}")
        except Exception as e:
            issue(issues, stem, split, "image_read_error", repr(e))
    if deep and sensor is not None:
        if sensor.suffix.lower() == ".xlsx":
            compare_sensor(sensor, seq, issues, stem, split, tolerance)
        else:
            issue(issues, stem, split, "sensor_read_error", "CSV deep comparison not implemented; XLSX expected")
    if hash_images and image is not None:
        try:
            image_hash = sha256(image)
            prior = hashes.get(image_hash)
            if prior and prior[1] != split:
                issue(issues, stem, split, "duplicate_across_splits", f"same bytes as {prior[0]} ({prior[1]})")
            else:
                hashes.setdefault(image_hash, (stem, split))
        except Exception as e:
            issue(issues, stem, split, "image_read_error", f"hash: {e!r}")
    problems = [row["issue"] for row in issues[before:]]
    sample["issue_count"] = len(problems)
    sample["status"] = "invalid" if any(k in FATAL for k in problems) else "review_required" if problems else "ok"
    counts[f"status_{sample['status']}"] += 1
    return sample


def main():
    parser = argparse.ArgumentParser(description="SMT dataset audit (read-only)")
    parser.add_argument("--root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out", type=Path, default=Path("outputs/data_audit_english"))
    parser.add_argument("--deep", action="store_true", help="verify image pixels and compare XLSX to JSON sensor data; slow")
    parser.add_argument("--hash-images", action="store_true", help="SHA256 image duplicate check across train/validation; slow")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="sensor numerical comparison tolerance")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.out.resolve()
    if not root.exists():
        parser.error(f"Dataset folder not found: {root}")
    labels = sorted(p for p in root.rglob("*.json") if p.parent.name.startswith(("train_annotations_", "validation_annotations_")))
    if not labels:
        parser.error("No TL_/VL_ JSON labels found")
    out.mkdir(parents=True, exist_ok=True)
    issues: list[dict] = []
    samples: list[dict] = []
    counts: Counter = Counter()
    hashes: dict = {}
    stem_splits: defaultdict = defaultdict(set)
    print(f"JSON {len(labels):,}개 검증 시작 | deep={args.deep}, hash_images={args.hash_images}", flush=True)
    for path in tqdm(labels, desc="데이터 검증", unit="file", dynamic_ncols=True):
        samples.append(analyze(path, root, args.deep, args.hash_images, args.tolerance, issues, counts, hashes, stem_splits))
    overlaps = {stem: splits for stem, splits in stem_splits.items() if len(splits) > 1}
    for stem, splits in overlaps.items():
        issue(issues, stem, "both", "same_filename_across_splits", ",".join(sorted(splits)))
    flags = defaultdict(set)
    for entry in issues:
        flags[(entry["sample"], entry["split"])].add(entry["issue"])
    for sample in samples:
        if sample["sample"] in overlaps and sample["status"] == "ok":
            sample["status"] = "review_required"
        elif sample["status"] == "ok" and flags[(sample["sample"], sample["split"])]:
            sample["status"] = "review_required"
    dispositions = Counter(s["status"] for s in samples)
    issue_counts = Counter(i["issue"] for i in issues)
    csv_write(out / "issues.csv", issues, ["sample", "split", "issue", "detail"])
    csv_write(out / "samples.csv", samples, ["sample", "split", "folder", "status", "categories", "boxes", "image_exists", "sensor_exists", "issue_count"])
    summary = {"total_samples": len(samples), "by_split": {k: counts[f"samples_{k}"] for k in ("train", "validation", "unknown")},
               "status_counts": dict(dispositions), "issue_counts": dict(issue_counts),
               "total_annotations": counts["objects_total"],
               "annotation_category_counts": {str(k): counts[f"objects_{k}"] for k in range(1, 33) if counts[f"objects_{k}"]},
               "same_filename_across_splits": len(overlaps),
               "sensor_id_offsets": {str(i): n for key, n in counts.items() if key.startswith("sensor_id_offset_") for i in [int(key.removeprefix("sensor_id_offset_"))]},
               "checks": {"deep": args.deep, "hash_images": args.hash_images},
               "notes": ["Issues are investigation flags, not confirmed mislabels.",
                         "Status=ok means checks passed, not a guarantee of manufacturing label correctness.",
                         "Class mapping is from the uploaded SMT data documentation."]}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 완료 ===")
    print(f"샘플 {len(samples):,} | OK {dispositions['ok']:,} | 검토필요 {dispositions['review_required']:,} | 구조오류 {dispositions['invalid']:,}")
    print("검토항목:", dict(issue_counts))
    print("결과:", out)
    print("주의: review_required 샘플은 원본 수정 없이 목록으로만 표시했습니다.")


if __name__ == "__main__":
    main()
