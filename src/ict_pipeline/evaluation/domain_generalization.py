"""Independent internal/external domain-generalization benchmark preparation."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from ..models.contracts import ModelName, TrainConfig
from ..models.dataset import Object, Sample, read_coco_split
from .adapters import load_predictor
from .contracts import CLASSES, EvaluationConfig, Prediction
from .metrics import box_iou, detection_metrics


CLASSES_BY_NAME = {name: category for category, name in CLASSES.items()}
SELECTED_CLASSES = ("cold_solder", "misalignment")
EXTERNAL_DATASET_NAME = "instances_external.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_internal(
    source_dataset: Path,
    output: Path,
    *,
    per_class: int = 10,
) -> dict[str, Any]:
    coco = (source_dataset / "coco").resolve()
    source_annotation = coco / "annotations" / "instances_val.json"
    if not source_annotation.is_file():
        raise RuntimeError("The published Validation COCO annotation is missing.")
    source = json.loads(source_annotation.read_text(encoding="utf-8-sig"))

    categories = {row["id"]: row["name"] for row in source["categories"]}
    if set(categories) != set(CLASSES) or categories != CLASSES:
        raise RuntimeError("The public Validation split is not in the canonical five-class mapping.")

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in source["annotations"]:
        annotations_by_image[row["image_id"]].append(row)
    images_by_id = {row["id"]: row for row in source["images"]}

    selected_by_class: dict[str, list[int]] = {}
    for class_name in SELECTED_CLASSES:
        category = CLASSES_BY_NAME[class_name]
        candidates = sorted(
            image_id
            for image_id, rows in annotations_by_image.items()
            if any(row["category_id"] == category for row in rows)
        )
        if len(candidates) < per_class:
            raise RuntimeError(
                f"Public Validation contains only {len(candidates)} {class_name} images; {per_class} are required."
            )
        selected_by_class[class_name] = candidates[:per_class]

    selected_ids = sorted(set(
        image_id
        for image_ids in selected_by_class.values()
        for image_id in image_ids
    ))
    old_to_new = {old_id: index for index, old_id in enumerate(selected_ids, 1)}
    target_images_dir = output / "coco" / "images" / "val"
    target_annotation_dir = output / "coco" / "annotations"
    target_images_dir.mkdir(parents=True, exist_ok=True)
    target_annotation_dir.mkdir(parents=True, exist_ok=True)

    target_images: list[dict[str, Any]] = []
    target_annotations: list[dict[str, Any]] = []
    copied: list[dict[str, Any]] = []
    next_annotation_id = 1
    for old_id in selected_ids:
        source_row = images_by_id[old_id]
        source_image = coco / "images" / "val" / source_row["file_name"]
        if not source_image.is_file():
            raise RuntimeError(f"Missing published Validation image: {source_row['file_name']}")
        target_name = source_row["file_name"]
        target_image = target_images_dir / target_name
        shutil.copy2(source_image, target_image)
        with Image.open(target_image) as image:
            width, height = image.size
        target_row = dict(source_row)
        target_row["id"] = old_to_new[old_id]
        target_row["width"] = width
        target_row["height"] = height
        target_images.append(target_row)
        classes = sorted({
            CLASSES[row["category_id"]] for row in annotations_by_image[old_id]
            if row["category_id"] in CLASSES
        })
        copied.append({
            "file_name": target_name,
            "sha256": sha256(target_image),
            "classes": classes,
        })
        for row in annotations_by_image[old_id]:
            if row["category_id"] not in CLASSES:
                continue
            target_annotation = dict(row)
            target_annotation["id"] = next_annotation_id
            target_annotation["image_id"] = old_to_new[old_id]
            target_annotations.append(target_annotation)
            next_annotation_id += 1

    target_annotation = {
        "info": {
            "description": "Internal 20-image cold_solder/misalignment diagnostic subset of published Validation",
            "source": str(source_annotation.resolve()),
        },
        "licenses": source.get("licenses", []),
        "images": target_images,
        "annotations": target_annotations,
        "categories": source["categories"],
    }
    target_path = target_annotation_dir / "instances_val.json"
    target_path.write_text(json.dumps(target_annotation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "source_split": "published_val",
        "selection": selected_by_class,
        "selected_image_count": len(selected_ids),
        "per_class": per_class,
        "images": copied,
        "annotation_fingerprint": sha256(target_path),
    }
    (output / "selection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_external(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "coco" / "images" / "external"
    annotation_dir = output / "coco" / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    template = {
        "info": {"description": "External PCB domain-generalization images"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {"id": category, "name": name, "supercategory": "pcb_defect"}
            for category, name in CLASSES.items()
        ],
    }
    template_path = annotation_dir / EXTERNAL_DATASET_NAME
    template_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / ".gitignore").write_text(
        "*\n!.gitignore\n!ANNOTATION_GUIDE.md\n!coco/\n!coco/annotations/\n"
        "!coco/annotations/instances_external.json\n",
        encoding="utf-8",
    )
    guide = """# External PCB annotation guide

Place external PCB image files in `coco/images/external/` and update
`coco/annotations/instances_external.json`.

## Canonical classes

| COCO category_id | Name |
|---|---|
| 1 | cold_solder |
| 2 | misalignment |
| 3 | short |
| 4 | missing_component |
| 5 | tombstone |

## Bounding boxes

- Use `bbox: [x, y, width, height]` in absolute original-pixel coordinates.
- Coordinates must not leave the image; width and height must be positive.
- Add one annotation for every visible defect instance.
- Use `iscrowd: 0`.
- Assign stable integer `image_id` values and unique integer `annotation_id` values.
- `file_name` must match the exact filename in `coco/images/external/`.
- Keep the canonical `categories` array unchanged.

Do not fabricate external images or labels. If `annotations` is empty, the external
dataset is reported as unlabeled and AP/Precision/Recall are intentionally omitted.
"""
    (output / "ANNOTATION_GUIDE.md").write_text(guide, encoding="utf-8")
    return {
        "directory": str(output),
        "images": str(images_dir),
        "annotation_template": str(template_path),
        "guide": str(output / "ANNOTATION_GUIDE.md"),
    }


def _resolve_checkpoint(checkpoint: Path) -> tuple[Path, dict[str, Any]]:
    checkpoint = checkpoint.resolve()
    if checkpoint.is_dir():
        run = checkpoint
        checkpoint = run / "checkpoints" / "faster_rcnn" / "best.pt"
        if not checkpoint.is_file():
            checkpoint = run / "best.pt"
    if not checkpoint.is_file():
        raise RuntimeError("Faster R-CNN checkpoint was not found.")
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "ict-torch-state-v1":
        raise RuntimeError("The checkpoint is not an ICT Torch state checkpoint.")
    if state.get("architecture") != "fasterrcnn_mobilenet_v3_large_320_fpn":
        raise RuntimeError("The checkpoint is not the finalized Faster R-CNN architecture.")
    return checkpoint, state


def _read_samples(dataset: Path, split: str) -> tuple:
    coco = dataset / "coco"
    if split == "external":
        path = coco / "annotations" / EXTERNAL_DATASET_NAME
    else:
        path = coco / "annotations" / f"instances_{split}.json"
    if not path.is_file():
        raise RuntimeError(f"Missing COCO annotation: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    categories = {row["id"]: row["name"] for row in data["categories"]}
    if categories != CLASSES:
        raise RuntimeError("COCO categories do not match the canonical five-class mapping.")
    by_id = {row["id"]: row for row in data["images"]}
    objects: dict[int, list[Object]] = defaultdict(list)
    for row in data["annotations"]:
        x, y, width, height = map(float, row["bbox"])
        objects[row["image_id"]].append(Object(row["category_id"], (x, y, width, height)))
    samples = []
    images_dir = coco / "images" / ("external" if split == "external" else split)
    for image_id, row in by_id.items():
        image = images_dir / row["file_name"]
        if not image.is_file():
            raise RuntimeError(f"Missing image: {image}")
        samples.append(Sample(image, row["width"], row["height"], tuple(objects[image_id])))
    return tuple(sorted(samples, key=lambda sample: sample.image.stem))


def evaluate_domain(
    checkpoint: Path,
    internal_dataset: Path,
    external_dataset: Path,
    output: Path,
    *,
    image_size: int = 640,
    confidence: float = 0.25,
    match_iou: float = 0.5,
    nms_iou: float = 0.7,
    score_floor: float = 0.001,
    device: str = "cpu",
    precision: str = "fp32",
) -> dict[str, Any]:
    checkpoint, _ = _resolve_checkpoint(checkpoint)
    config = EvaluationConfig(
        image_size=image_size,
        score_floor=score_floor,
        confidence=confidence,
        match_iou=match_iou,
        nms_iou=nms_iou,
        warmup=0,
        device=device,
        precision=precision,
    )
    training = TrainConfig(
        epochs=50,
        batch_size=4,
        image_size=image_size,
        optimizer="SGD",
        lr=0.005,
        seed=42,
        device=device,
        precision=precision,
        save_period=0,
    )
    output.mkdir(parents=True, exist_ok=True)
    predictor = load_predictor(
        ModelName.FASTER_RCNN,
        checkpoint,
        training,
        config,
        output,
    )

    internal_samples = _read_samples(internal_dataset, "val")
    external_samples = _read_samples(external_dataset, "external")
    internal_predictions = _predict(predictor, internal_samples)
    external_predictions = _predict(predictor, external_samples)

    internal_metrics = detection_metrics(internal_samples, internal_predictions, config)
    external_labeled = bool(external_samples and any(sample.objects for sample in external_samples))
    external_metrics = (
        detection_metrics(external_samples, external_predictions, config)
        if external_labeled
        else None
    )
    comparison = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "configuration": config.model_dump(mode="json"),
        "internal": {
            "status": "labeled",
            "image_count": len(internal_samples),
            "metrics": internal_metrics,
            "iou_and_errors": _iou_and_errors(internal_samples, internal_predictions, confidence),
        },
        "external": {
            "status": "labeled" if external_labeled else "unlabeled",
            "image_count": len(external_samples),
            "metrics": external_metrics,
            "iou_and_errors": (
                _iou_and_errors(external_samples, external_predictions, confidence)
                if external_labeled else None
            ),
            "detections": {
                sample.image.stem: [
                    {
                        "box": list(box),
                        "class_id": category,
                        "class_name": CLASSES[category],
                        "confidence": score,
                    }
                    for box, category, score in zip(
                        external_predictions[sample.image.stem].boxes,
                        external_predictions[sample.image.stem].class_ids,
                        external_predictions[sample.image.stem].confidence_scores,
                        strict=True,
                    )
                    if score >= confidence
                ]
                for sample in external_samples
            },
        },
    }
    result_path = output / "domain_generalization.json"
    result_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "domain_generalization.md").write_text(
        _comparison_markdown(comparison),
        encoding="utf-8",
    )
    return comparison


def _predict(predictor: Any, samples: tuple[Sample, ...]) -> dict[str, Prediction]:
    predictions = {}
    for sample in samples:
        prediction = predictor.predict(sample)
        predictor.synchronize()
        predictions[sample.image.stem] = prediction
    return predictions


def _iou_and_errors(
    samples: tuple[Sample, ...],
    predictions: dict[str, Prediction],
    confidence: float,
) -> dict[str, Any]:
    ious: list[float] = []
    errors: Counter[str] = Counter()
    class_ious: dict[str, list[float]] = defaultdict(list)
    for sample in sorted(samples, key=lambda item: item.image.stem):
        prediction = predictions[sample.image.stem]
        truths = [
            (obj.category, (obj.box[0], obj.box[1], obj.box[0] + obj.box[2], obj.box[1] + obj.box[3]))
            for obj in sample.objects
        ]
        operating = [
            (category, box)
            for box, category, score in zip(
                prediction.boxes,
                prediction.class_ids,
                prediction.confidence_scores,
                strict=True,
            )
            if score >= confidence
        ]
        unmatched_truth = set(range(len(truths)))
        unmatched_prediction = set(range(len(operating)))
        pairs = [
            (predicted_index, truth_index, box_iou(predicted_box, truth_box))
            for predicted_index, (_category, predicted_box) in enumerate(operating)
            for truth_index, (_truth_category, truth_box) in enumerate(truths)
        ]
        for predicted_index, truth_index, iou in sorted(pairs, key=lambda row: -row[2]):
            if iou < 0.5 or predicted_index not in unmatched_prediction or truth_index not in unmatched_truth:
                continue
            predicted_category = operating[predicted_index][0]
            truth_category = truths[truth_index][0]
            ious.append(iou)
            class_ious[CLASSES[truth_category]].append(iou)
            if predicted_category == truth_category:
                errors["exact_detection"] += 1
            else:
                errors["class_misclassification"] += 1
            unmatched_truth.remove(truth_index)
            unmatched_prediction.remove(predicted_index)
        errors["missed_defect"] += len(unmatched_truth)
        errors["false_positive"] += len(unmatched_prediction)
    return {
        "mean_matched_iou": sum(ious) / len(ious) if ious else None,
        "matched_iou_by_class": {
            name: sum(values) / len(values) if values else None
            for name, values in ((name, class_ious[name]) for name in CLASSES.values())
        },
        "error_counts": dict(sorted(errors.items())),
    }


def _comparison_markdown(report: dict[str, Any]) -> str:
    internal = report["internal"]
    external = report["external"]
    lines = [
        "# Faster R-CNN domain-generalization benchmark",
        "",
        f"- Checkpoint: `{report['checkpoint']}`",
        f"- Checkpoint SHA-256: `{report['checkpoint_sha256']}`",
        "- Model, preprocessing, confidence, and NMS settings are identical for both domains.",
        "",
        "## Internal labeled diagnostic",
        "",
        f"- Status: `{internal['status']}`",
        f"- Images: {internal['image_count']}",
    ]
    metrics = internal["metrics"]
    lines.extend([
        f"- mAP@50: {metrics['map50']:.6f}",
        f"- mAP@50:95: {metrics['map50_95']:.6f}",
        f"- Precision: {metrics['precision']:.6f}",
        f"- Recall: {metrics['recall']:.6f}",
        "",
        "| Class | GT objects | AP@50 | AP@50:95 | TP | FP | FN | Precision | Recall | Mean matched IoU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    iou_by_class = internal["iou_and_errors"]["matched_iou_by_class"]
    for row in metrics["per_class"]:
        ap50 = "N/A" if row["ap50"] is None else f"{row['ap50']:.6f}"
        ap5095 = "N/A" if row["ap50_95"] is None else f"{row['ap50_95']:.6f}"
        mean_iou = iou_by_class[row["name"]]
        mean_iou_text = "N/A" if mean_iou is None else f"{mean_iou:.6f}"
        lines.append(
            f"| {row['name']} | {row['ground_truth_objects']} | {ap50} | {ap5095} | "
            f"{row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {mean_iou_text} |"
        )
    lines.extend([
        "",
        "- Mean matched IoU: "
        + (
            "N/A"
            if internal["iou_and_errors"]["mean_matched_iou"] is None
            else f"{internal['iou_and_errors']['mean_matched_iou']:.6f}"
        ),
        "- Error counts: "
        + ", ".join(f"{name}={count}" for name, count in internal["iou_and_errors"]["error_counts"].items()),
    ])
    lines.extend(["", "## External domain", "", f"- Status: `{external['status']}`", f"- Images: {external['image_count']}"])
    if external["metrics"] is None:
        lines.extend([
            "",
            "External AP, Precision, and Recall are **not available** because ground-truth boxes are missing.",
            "No results have been fabricated.",
        ])
    else:
        external_metrics = external["metrics"]
        lines.extend([
            f"- mAP@50: {external_metrics['map50']:.6f}",
            f"- mAP@50:95: {external_metrics['map50_95']:.6f}",
            f"- Precision: {external_metrics['precision']:.6f}",
            f"- Recall: {external_metrics['recall']:.6f}",
        ])
    return "\n".join(lines) + "\n"
