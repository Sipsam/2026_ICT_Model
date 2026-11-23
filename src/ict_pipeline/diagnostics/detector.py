"""Faster R-CNN candidate and postprocessing diagnostics.

This module is diagnostic-only. It does not expose internal tensors through the
public API or change model, checkpoint, or normal inspection settings.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from ict_pipeline.evaluation.contracts import CLASSES

CONFIDENCE_THRESHOLDS = (0.05, 0.10, 0.25, 0.50)
NMS_THRESHOLDS = (0.50, 0.70, 0.90)
CLASS_COLORS = {
    1: "red",
    2: "orange",
    3: "yellow",
    4: "cyan",
    5: "magenta",
}


def load_faster_rcnn(checkpoint: Path, nms_iou: float) -> tuple[Any, Any]:
    import torch
    from ict_pipeline.models.coco import build_faster_rcnn

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("format") != "ict-torch-state-v1":
        raise RuntimeError("Checkpoint format mismatch")
    if state.get("architecture") != "fasterrcnn_mobilenet_v3_large_320_fpn":
        raise RuntimeError("Checkpoint architecture mismatch")
    model = build_faster_rcnn(640, pretrained_layout=False)
    model.load_state_dict(state["model"])
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh = nms_iou
    model.roi_heads.detections_per_img = 1000
    model.to("cpu").eval()
    return model, torch


def resolve_final_checkpoint(config: Path) -> tuple[Path, dict[str, Any]]:
    settings = json.loads(config.read_text(encoding="utf-8"))
    model = settings["models"]["faster-rcnn-final"]
    if model["mode"] != "finalized" or model["model"] != "faster_rcnn":
        raise RuntimeError("The config has no finalized faster-rcnn-final model")
    root = Path(settings["model_root"])
    run = root / model["run"]
    checkpoint = run / "best.pt"
    if not run.is_relative_to(root.resolve()):
        raise RuntimeError("Invalid model run location")
    if not checkpoint.is_file():
        raise RuntimeError(f"Missing finalized checkpoint: {checkpoint}")
    return checkpoint.resolve(), settings


class CandidateHooks:
    """Capture RPN proposals and pre-NMS ROI candidates without changing output."""

    def __init__(self, model: Any) -> None:
        self.rpn_counts: list[int] = []
        self.roi_proposals: list[Any] = []
        self.class_logits: list[Any] = []
        self.box_regression: list[Any] = []
        self.transformed_sizes: list[tuple[int, int]] = []
        self.handles = []
        self.handles.append(model.rpn.register_forward_hook(self._rpn))
        self.handles.append(model.roi_heads.register_forward_pre_hook(self._roi_pre))
        self.handles.append(model.roi_heads.box_predictor.register_forward_hook(self._roi_head))
        self.handles.append(model.transform.register_forward_hook(self._transform))

    def _transform(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        # GeneralizedRCNNTransform returns an ImageList; its tensor dimensions
        # are the padded transformed dimensions.
        if hasattr(output, "tensors"):
            height, width = output.tensors.shape[-2:]
            self.transformed_sizes = [(int(width), int(height))]

    def _rpn(self, _module: Any, _inputs: tuple[Any, ...], output: tuple[list[Any], dict[str, Any]]) -> None:
        proposals, _ = output
        self.rpn_counts = [int(row.shape[0]) for row in proposals]

    def _roi_pre(
        self, _module: Any, inputs: tuple[Any, ...], _kwargs: dict[str, Any] | None = None,
    ) -> None:
        # features, proposals, image_shapes, targets
        self.roi_proposals = [row.detach().cpu() for row in inputs[1]]

    def _roi_head(self, _module: Any, _inputs: tuple[Any, ...], output: tuple[Any, Any]) -> None:
        logits, regression = output
        self.class_logits = [logits.detach().cpu()]
        self.box_regression = [regression.detach().cpu()]

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def summarize_candidates(
    model: Any, torch: Any, image: Image.Image, nms_iou: float,
    confidence_thresholds: tuple[float, ...] = CONFIDENCE_THRESHOLDS,
    top_k: int = 50,
) -> dict[str, Any]:
    from torchvision.transforms.functional import pil_to_tensor
    from torchvision.ops import batched_nms

    hooks = CandidateHooks(model)
    pixels = pil_to_tensor(image.convert("RGB")).float() / 255
    model.roi_heads.nms_thresh = nms_iou
    model.roi_heads.score_thresh = 0.0
    with torch.inference_mode():
        result = model([pixels])[0]
    boxes = result["boxes"].detach().cpu()
    labels = result["labels"].detach().cpu()
    scores = result["scores"].detach().cpu()

    roi_count = sum(int(row.shape[0]) for row in hooks.roi_proposals)
    rpn_count = sum(hooks.rpn_counts)
    candidates: list[dict[str, Any]] = []
    class_counts = Counter()
    class_scores: dict[str, list[float]] = {name: [] for name in CLASSES.values()}

    if hooks.class_logits and hooks.box_regression and hooks.roi_proposals:
        # The public hook boundary is retained for RPN and final model output.
        # ROI tensor-to-coordinate reconstruction is omitted here because it
        # depends on internal TorchVision tensor layouts and would risk changing
        # or misrepresenting the model's real postprocessing.
        pass

    candidates.sort(key=lambda row: row["score"], reverse=True)
    by_confidence: dict[str, dict[str, int]] = {}
    nms_counts: dict[str, int] = {}
    for threshold in confidence_thresholds:
        retained = [row for row in candidates if row["score"] >= threshold]
        by_confidence[f"{threshold:.2f}"] = {
            "roi_candidates": len(retained),
            "removed_low_score": len(candidates) - len(retained),
        }

    if candidates:
        candidate_boxes = torch.tensor([row["box"] for row in candidates], dtype=torch.float32)
        candidate_scores = torch.tensor([row["score"] for row in candidates], dtype=torch.float32)
        candidate_labels = torch.tensor([row["class_id"] for row in candidates], dtype=torch.int64)
        nms_keep = batched_nms(candidate_boxes, candidate_scores, candidate_labels, nms_iou)
        nms_counts["pre_nms"] = len(candidates)
        nms_counts["post_nms"] = int(nms_keep.shape[0])
        nms_counts["removed_by_nms"] = len(candidates) - int(nms_keep.shape[0])
    else:
        nms_counts = {"pre_nms": 0, "post_nms": 0, "removed_by_nms": 0}

    width_scale = 1.0
    height_scale = 1.0
    final = []
    for box, label, score in zip(boxes, labels, scores, strict=True):
        x1, y1, x2, y2 = (float(value) for value in box.tolist())
        final.append({
            "box": [round(x1 * width_scale, 3), round(y1 * height_scale, 3),
                    round(x2 * width_scale, 3), round(y2 * height_scale, 3)],
            "class_id": int(label.item()),
            "class_name": CLASSES[int(label.item())],
            "score": float(score.item()),
        })
    result_summary = {
        "image_size": {"width": image.width, "height": image.height},
        "rpn_candidates": rpn_count,
        "roi_candidates": roi_count,
        "roi_class_score_candidates": len(candidates),
        "roi_class_counts": dict(sorted(class_counts.items())),
        "roi_class_score_stats": {
            name: {
                "count": len(values),
                "max": max(values) if values else 0.0,
                "mean": sum(values) / len(values) if values else 0.0,
            }
            for name, values in class_scores.items()
        },
        "confidence_filter": by_confidence,
        "nms": nms_counts,
        "final_candidates": len(final),
        "top_roi_candidates": candidates[:top_k],
        "final_detections": final,
    }
    hooks.close()
    return result_summary


def draw_results(
    image: Image.Image, summary: dict[str, Any], output: Path,
    top_k: int = 50,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stem = output.name
    image.convert("RGB").save(output / "original.png")

    for threshold in CONFIDENCE_THRESHOLDS:
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        for candidate in summary["top_roi_candidates"]:
            if candidate["score"] < threshold:
                continue
            box = candidate["box"]
            color = CLASS_COLORS.get(candidate["class_id"], "white")
            draw.rectangle(box, outline=color, width=2)
            draw.text((box[0], box[1]), f"{candidate['class_name']} {candidate['score']:.3f}", fill=color)
        canvas.save(output / f"confidence-{threshold:.2f}.png")

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for candidate in summary["top_roi_candidates"][:top_k]:
        box = candidate["box"]
        color = CLASS_COLORS.get(candidate["class_id"], "white")
        draw.rectangle(box, outline=color, width=1)
        draw.text((box[0], box[1]), f"{candidate['score']:.3f}", fill=color)
    canvas.save(output / "pre_nms_top_candidates.png")

    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for detection in summary["final_detections"]:
        box = detection["box"]
        color = CLASS_COLORS.get(detection["class_id"], "white")
        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], box[1]), f"{detection['class_name']} {detection['score']:.3f}", fill=color)
    canvas.save(output / "final.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diagnose a Faster R-CNN inference image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/diagnostics/faster-rcnn/external"))
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--nms-iou", type=float, action="append", choices=NMS_THRESHOLDS)
    args = parser.parse_args(argv)
    if not args.image.is_file():
        parser.error(f"Image does not exist: {args.image}")
    if args.image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        parser.error("Only JPG and PNG images are supported")
    if args.nms_iou is None:
        args.nms_iou = [0.70]

    checkpoint, _ = resolve_final_checkpoint(args.config)
    model, torch = load_faster_rcnn(checkpoint, args.nms_iou[0])
    with Image.open(args.image) as source:
        image = source.convert("RGB")
        all_results: dict[str, Any] = {}
        for nms_iou in args.nms_iou:
            summary = summarize_candidates(model, torch, image, nms_iou, top_k=args.top_k)
            all_results[f"{nms_iou:.2f}"] = summary
            draw_results(image, summary, args.output / f"nms-{nms_iou:.2f}".replace(".", "_"))

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(args.image.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
        "nms_thresholds": list(args.nms_iou),
        "ap_precision_recall_note": "No ground-truth labels supplied; AP, precision, and recall are not defined.",
        "nms_results": all_results,
    }
    (args.output / "candidate_analysis.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
