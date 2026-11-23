"""Offline diagnostics for the finalized Faster R-CNN detector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from ict_pipeline.evaluation.contracts import CLASSES, EvaluationConfig, Preprocessing, Prediction
from ict_pipeline.evaluation.metrics import box_iou, detection_metrics
from ict_pipeline.models.dataset import Object, read_coco_split


DEFECT_NAMES = {name for name in CLASSES.values()}
CONFIDENCE_THRESHOLDS = (0.05, 0.10, 0.25, 0.50)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash(path: Path) -> str:
    with Image.open(path) as source:
        pixels = source.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        get_pixels = getattr(pixels, "get_flattened_data", pixels.getdata)
        rows = list(get_pixels())
    bits = "".join(
        "1" if rows[row * 9 + column] > rows[row * 9 + column + 1] else "0"
        for row in range(8)
        for column in range(8)
    )
    return f"{int(bits, 2):016x}"


def hamming(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def aggregate_manifest(manifest: Path) -> dict[str, object]:
    import csv as csv_module

    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_images: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("decision") != "candidate":
            continue
        split = row["split"]
        split_images[split].add(row["sample_id"])
        names = [name for name in row["target_class_names"].split(";") if name]
        for name in names:
            split_counts[split][name] += int(row["target_box_count"]) // max(1, len(names))
    return {
        "rows": len(rows),
        "candidate_rows": sum(len(value) for value in split_images.values()),
        "split_images": {split: len(values) for split, values in split_images.items()},
        "split_defects": {split: dict(counts) for split, counts in split_counts.items()},
    }


def leakage_report(dataset: Path, manifest: Path, output: Path) -> dict[str, object]:
    coco = dataset / "coco"
    split_images: dict[str, dict[str, Path]] = {}
    for split in ("train", "val", "test"):
        directory = coco / "images" / split
        if directory.is_dir():
            split_images[split] = {path.stem: path for path in directory.iterdir() if path.is_file()}

    hash_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, files in split_images.items():
        for stem, path in files.items():
            hash_groups[sha256(path)].append((split, stem))
    exact = [
        {"sha256": digest, "members": members}
        for digest, members in hash_groups.items()
        if len({split for split, _ in members}) > 1
    ]

    hashes: dict[str, str] = {}
    similar: list[dict[str, object]] = []
    for split, files in split_images.items():
        for stem, path in files.items():
            hashes[stem] = dhash(path)
    stems = list(hashes)
    for index, left in enumerate(stems):
        for right in stems[index + 1:]:
            left_split = next(split for split, files in split_images.items() if left in files)
            right_split = next(split for split, files in split_images.items() if right in files)
            if left_split != right_split:
                distance = hamming(hashes[left], hashes[right])
                if distance <= 5:
                    similar.append({"left": left, "right": right, "distance": distance,
                                    "left_split": left_split, "right_split": right_split})

    import csv as csv_module
    group_crossings: list[dict[str, str]] = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv_module.DictReader(handle))
    groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row.get("decision") == "candidate" and row.get("group_id"):
            groups[row["group_id"]].add(row["split"])
    for group, split_names in sorted(groups.items()):
        if len(split_names) > 1:
            group_crossings.append({"group_id": group, "splits": ";".join(sorted(split_names))})

    result = {
        "exact_cross_split_hashes": exact,
        "near_duplicate_cross_split_count": len(similar),
        "near_duplicate_examples": similar[:100],
        "group_id_cross_split_count": len(group_crossings),
        "group_id_cross_split_examples": group_crossings[:100],
    }
    (output / "data_split_leakage.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def load_predictions(dataset: Path, checkpoint: Path, nms_iou: float) -> tuple[tuple, dict[str, Prediction], object]:
    import torch
    from ict_pipeline.models.coco import build_faster_rcnn
    from ict_pipeline.models.contracts import TrainConfig

    samples = read_coco_split(dataset / "coco", "val").samples
    extracted = Path("data") / "extracted"
    image_index = {path.name: path for path in extracted.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    resolved_samples = tuple(
        replace(sample, image=image_index[sample.image.name])
        for sample in samples
        if sample.image.name in image_index
    )
    if len(resolved_samples) != len(samples):
        missing = [sample.image.name for sample in samples if sample.image.name not in image_index]
        raise RuntimeError(f"Missing extracted Validation images: {len(missing)}")
    samples = resolved_samples
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    training = TrainConfig(
        epochs=50, batch_size=4, image_size=640, optimizer="SGD", lr=0.005,
        seed=42, device="cpu", precision="fp32", save_period=0,
    )
    model = build_faster_rcnn(640, pretrained_layout=False)
    if state.get("format") != "ict-torch-state-v1" or state.get("architecture") != "fasterrcnn_mobilenet_v3_large_320_fpn":
        raise RuntimeError("Checkpoint format or architecture mismatch")
    model.load_state_dict(state["model"])
    model.roi_heads.score_thresh = 0.0
    # Apply each NMS setting inside the real TorchVision ROI postprocessing
    # stage before reusing the resulting predictions for confidence sweeps.
    model.roi_heads.nms_thresh = nms_iou
    model.roi_heads.detections_per_img = 1000
    model.to("cpu").eval()

    predictions: dict[str, Prediction] = {}
    for sample in samples:
        import torchvision
        with Image.open(sample.image) as source:
            image = source.convert("RGB")
            pixels = torchvision.transforms.functional.pil_to_tensor(image).float() / 255
        with torch.inference_mode():
            result = model([pixels])[0]
        boxes = tuple(tuple(float(value) for value in row) for row in result["boxes"].cpu().tolist())
        labels = tuple(int(value) for value in result["labels"].cpu().tolist())
        scores = tuple(float(value) for value in result["scores"].cpu().tolist())
        predictions[sample.image.stem] = Prediction(
            sample.image.stem, boxes, labels, scores,
            Preprocessing((sample.width, sample.height), (640, 640), "aspect_ratio"),
        )
    return samples, predictions, model


def write_threshold_metrics(output: Path, samples: tuple, predictions: dict[str, Prediction], nms_iou: float) -> None:
    rows = []
    summary_rows = []
    for confidence in CONFIDENCE_THRESHOLDS:
        config = EvaluationConfig(
            image_size=640, score_floor=0.001, confidence=confidence,
            match_iou=0.5, nms_iou=nms_iou, warmup=0, device="cpu", precision="fp32",
        )
        metrics = detection_metrics(samples, predictions, config)
        class_rows = list(metrics["per_class"])
        macro_precision = sum(row["precision"] for row in class_rows) / len(class_rows)
        macro_recall = sum(row["recall"] for row in class_rows) / len(class_rows)
        macro_f1 = sum(row["f1"] for row in class_rows) / len(class_rows)
        for row in metrics["per_class"]:
            rows.append({
                "confidence": confidence, "nms_iou": nms_iou, **row,
            })
        rows.append({
            "confidence": confidence, "nms_iou": nms_iou, "class_id": "micro",
            "name": "micro", "ap50": metrics["map50"], "ap50_95": metrics["map50_95"],
            "tp": metrics["tp"], "fp": metrics["fp"], "fn": metrics["fn"],
            "precision": metrics["precision"], "recall": metrics["recall"], "f1": metrics["f1"],
        })
        summary_rows.append({
            "confidence": confidence,
            "nms_iou": nms_iou,
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "micro_precision": metrics["precision"],
            "micro_recall": metrics["recall"],
            "micro_f1": metrics["f1"],
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "map50": metrics["map50"],
            "map50_95": metrics["map50_95"],
        })
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary_rows for key in row}))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output / "class_performance.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def write_visualizations(output: Path, samples: tuple, predictions: dict[str, Prediction]) -> None:
    visual_dir = output / "bbox_comparisons"
    visual_dir.mkdir(parents=True, exist_ok=True)
    rendered = 0
    for sample in samples:
        pred = predictions[sample.image.stem]
        truth = [(obj.category, obj.box) for obj in sample.objects]
        if not truth:
            continue
        matched_truth: set[int] = set()
        has_error = False
        for index, (category, box) in enumerate(truth):
            xyxy = (box[0], box[1], box[0] + box[2], box[1] + box[3])
            best = max(
                ((box_iou(xyxy, predicted), position) for position, predicted in enumerate(pred.boxes)),
                default=(0.0, -1),
            )
            if best[0] >= 0.5:
                matched_truth.add(index)
            else:
                has_error = True
        if not has_error and rendered >= 12:
            continue
        if rendered >= 20:
            break
        with Image.open(sample.image).convert("RGB") as source:
            canvas = source.copy()
        draw = ImageDraw.Draw(canvas)
        for category, box in truth:
            x, y, w, h = box
            draw.rectangle((x, y, x + w, y + h), outline="lime", width=3)
            draw.text((x, y), f"GT {CLASSES[category]}", fill="lime")
        for box, category, score in zip(pred.boxes, pred.class_ids, pred.confidence_scores, strict=True):
            if score < 0.25:
                continue
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1] + 12), f"PR {CLASSES[category]} {score:.3f}", fill="red")
        canvas.save(visual_dir / f"{sample.image.stem}.jpg", quality=92)
        rendered += 1


def write_error_diagnostics(output: Path, samples: tuple, predictions: dict[str, Prediction]) -> None:
    config = EvaluationConfig(
        image_size=640, score_floor=0.001, confidence=0.25,
        match_iou=0.5, nms_iou=0.7, warmup=0, device="cpu", precision="fp32",
    )
    metrics = detection_metrics(samples, predictions, config)
    (output / "confusion_matrix.json").write_text(
        json.dumps(metrics["confusion_matrix"], ensure_ascii=False, indent=2), encoding="utf-8"
    )

    iou_rows: list[dict[str, object]] = []
    error_counts: Counter[str] = Counter()
    for sample in samples:
        pred = predictions[sample.image.stem]
        truths = [(obj.category, (obj.box[0], obj.box[1], obj.box[0] + obj.box[2], obj.box[1] + obj.box[3]))
                  for obj in sample.objects]
        operating = [
            (category, box, score)
            for box, category, score in zip(pred.boxes, pred.class_ids, pred.confidence_scores, strict=True)
            if score >= 0.25
        ]
        unmatched_truth = set(range(len(truths)))
        unmatched_pred = set(range(len(operating)))
        pairs: list[tuple[int, int, float]] = []
        for predicted_index, (predicted_category, predicted_box, _score) in enumerate(operating):
            for truth_index, (truth_category, truth_box) in enumerate(truths):
                iou = box_iou(predicted_box, truth_box)
                pairs.append((predicted_index, truth_index, iou))
        for predicted_index, truth_index, iou in sorted(pairs, key=lambda row: -row[2]):
            if iou < 0.5 or predicted_index not in unmatched_pred or truth_index not in unmatched_truth:
                continue
            predicted_category = operating[predicted_index][0]
            truth_category = truths[truth_index][0]
            predicted_box = operating[predicted_index][1]
            truth_box = truths[truth_index][1]
            truth_width = truth_box[2] - truth_box[0]
            truth_height = truth_box[3] - truth_box[1]
            predicted_width = predicted_box[2] - predicted_box[0]
            predicted_height = predicted_box[3] - predicted_box[1]
            iou_rows.append({
                "sample_id": sample.image.stem,
                "truth_class": CLASSES[truth_category],
                "predicted_class": CLASSES[predicted_category],
                "iou": iou,
                "predicted_area_ratio": (predicted_width * predicted_height) / max(truth_width * truth_height, 1),
                "confidence": operating[predicted_index][2],
            })
            if predicted_category == truth_category:
                error_counts["exact_detection"] += 1
            else:
                error_counts["class_misclassification"] += 1
            unmatched_truth.remove(truth_index)
            unmatched_pred.remove(predicted_index)
        for _truth_index in unmatched_truth:
            error_counts["missed_defect"] += 1
        for _predicted_index in unmatched_pred:
            error_counts["duplicate_or_normal_false_positive"] += 1

    with (output / "iou_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "sample_id", "truth_class", "predicted_class", "iou",
            "predicted_area_ratio", "confidence",
        ])
        writer.writeheader()
        writer.writerows(iou_rows)
    with (output / "error_statistics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["error_type", "count"])
        writer.writeheader()
        writer.writerows({"error_type": name, "count": count} for name, count in sorted(error_counts.items()))


def write_postprocessing_outputs(output: Path, nms_values: Iterable[float]) -> None:
    postprocessing = output / "postprocessing"
    postprocessing.mkdir(parents=True, exist_ok=True)
    visualizations = postprocessing / "visualizations"
    visualizations.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for nms_iou in nms_values:
        directory = output / f"nms-{str(nms_iou).replace('.', '_')}"
        with (directory / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
            summary_rows.extend(csv.DictReader(handle))
        with (directory / "class_performance.csv").open("r", encoding="utf-8", newline="") as handle:
            class_rows.extend(csv.DictReader(handle))

    for name, rows in (("summary.csv", summary_rows), ("class_performance.csv", class_rows)):
        fieldnames = sorted({key for row in rows for key in row})
        with (postprocessing / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    candidate_analysis = {
        "scope": "published Validation COCO split",
        "rpn_roi_stage": "Available through the external diagnostic CLI with TorchVision pinned hooks.",
        "nms_sweeps": [
            {
                "nms_iou": float(row["nms_iou"]),
                "confidence": float(row["confidence"]),
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "micro_precision": float(row["micro_precision"]),
                "micro_recall": float(row["micro_recall"]),
                "micro_f1": float(row["micro_f1"]),
                "macro_precision": float(row["macro_precision"]),
                "macro_recall": float(row["macro_recall"]),
                "macro_f1": float(row["macro_f1"]),
                "map50": float(row["map50"]),
                "map50_95": float(row["map50_95"]),
            }
            for row in summary_rows
        ],
        "metric_definitions": {
            "tp_fp_fn": "Matched at IoU >= 0.5 and canonical class; unmatched detections are FP and unmatched truths are FN.",
            "micro": "TP/FP/FN summed over all five classes.",
            "macro": "Arithmetic mean of class-level TP-based precision, recall, and F1.",
            "ap": "COCO-style interpolated AP, independent of the fixed confidence threshold; confidence may change fixed TP/FP/FN only.",
        },
    }
    (postprocessing / "candidate_analysis.json").write_text(
        json.dumps(candidate_analysis, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = [
        "# Faster R-CNN 후처리 진단 보고서",
        "",
        "## 기존 지표 불일치",
        "",
        "- 기존 보고서의 2,208장 수치는 현재 로컬 COCO Validation의 1,656장, 1,877개 정답 객체와 일치하지 않는다.",
        "- 본 재검증은 `data/processed/detection/coco/annotations/instances_val.json`과 같은 파일의 1,656장으로 수행했다.",
        "- 보고서의 TP/FP/FN을 단순히 덮어쓰지 않고 공통 IoU 0.5 매칭과 현재 체크포인트로 재계산했다.",
        "- 클래스별 행은 각 클래스의 TP/FP/FN이다. Micro는 전체 합계 기반, Macro는 다섯 클래스 평균이다.",
        "",
        "## Confidence 및 NMS",
        "",
        "Confidence 0.05/0.10/0.25/0.50 및 NMS 0.50/0.70/0.90을 실제 TorchVision `roi_heads.nms_thresh`와 confidence 필터에 적용했다.",
        "각 NMS 조합은 별도의 모델 추론으로 생성했고 기존 최종 출력을 임의 재NMS하지 않았다.",
        "",
        "상세 수치는 `summary.csv`와 `class_performance.csv`를 참조한다.",
        "",
        "## 후보 단계",
        "",
        "TorchVision 0.21.0의 RPN은 proposal 생성 후 150개 post-NMS proposal을 ROI head에 전달한다.",
        "ROI head는 softmax 분류 점수와 box regression을 계산하고, confidence 필터와 class-aware batched NMS를 순서대로 적용한다.",
        "외부 이미지 진단 CLI는 RPN hook, ROI 분류/회귀 hook 및 최종 결과를 별도 JSON과 이미지로 저장한다.",
        "",
        "## 원인 해석",
        "",
        "- 낮은 confidence 후보는 FP를 크게 늘리면서 Recall을 약간만 개선한다.",
        "- NMS 0.90은 중복 박스 억제가 약해 FP가 크게 증가할 수 있고, 0.50은 중복 억제를 강화한다.",
        "- 높은 confidence는 FP를 줄이지만 FN과 박스 누락 가능성을 높인다.",
        "- 외부 라벨이 없으므로 외부 도메인 미탐을 별도 AP/Precision/Recall로 판정할 수 없다.",
        "",
        "## 개선 우선순위",
        "",
        "1. 외부 정답이 있는 독립 PCB validation 세트를 확보한다.",
        "2. 대시보드 좌표와 CLI 최종 박스를 원본 픽셀 좌표로 비교한다.",
        "3. 현재 환경에서 가장 낮은 FP와 요구 Recall을 만족하는 confidence/NMS를 Validation에서 고정한다.",
        "4. 냉납/밀림의 낮은 후보와 ROI 분류 점수를 실제 이미지에 대해 검토한다.",
        "5. 충분한 외부 데이터와 제품별 일반화 검증 전에는 재학습을 권고하지 않는다.",
        "",
    ]
    (postprocessing / "diagnostic_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/faster-rcnn"))
    parser.add_argument("--nms-iou", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = aggregate_manifest(args.manifest)
    leakage = leakage_report(args.dataset, args.manifest, args.output)
    (args.output / "data_split_summary.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    for nms_iou in args.nms_iou:
        samples, predictions, _ = load_predictions(args.dataset, args.checkpoint, nms_iou)
        nms_output = args.output / f"nms-{str(nms_iou).replace('.', '_')}"
        nms_output.mkdir(parents=True, exist_ok=True)
        write_threshold_metrics(nms_output, samples, predictions, nms_iou)
        if nms_iou == 0.7:
            write_visualizations(nms_output, samples, predictions)
            write_error_diagnostics(nms_output, samples, predictions)
    write_postprocessing_outputs(args.output, args.nms_iou)
    (args.output / "training_inference_settings.json").write_text(
        json.dumps({
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256(args.checkpoint),
            "training": {
                "initialization": "scratch",
                "image_size": 640,
                "optimizer": "SGD",
                "lr": 0.005,
                "epochs": 50,
                "seed": 42,
                "device": "cuda",
                "precision": "fp32",
            },
            "inference_diagnostic": {
                "image_size": 640,
                "normalization": "RGB pil_to_tensor / 255",
                "resize": "TorchVision Faster R-CNN aspect-ratio transform",
                "score_floor": 0.001,
                "confidence_sweep": list(CONFIDENCE_THRESHOLDS),
                "nms_iou_sweep_requested": [0.5, 0.7, 0.9],
                "nms_iou_executed": list(args.nms_iou),
                "device": "cpu",
                "runtime": "torch 2.6.0+cu126; torchvision 0.21.0+cu126; CUDA unavailable in this host",
            },
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
