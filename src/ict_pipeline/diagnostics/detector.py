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


def validate_roi_tensor_shapes(
    logits: Any,
    regression: Any,
    proposals: Any,
    image_shapes: list[tuple[int, int]],
) -> int:
    if logits.ndim != 2:
        raise ValueError(f"class_logits ndim is {logits.ndim}, expected 2")
    if regression.ndim != 2:
        raise ValueError(f"box_regression ndim is {regression.ndim}, expected 2")
    if proposals.ndim != 2 or proposals.shape[-1] != 4:
        raise ValueError(
            f"ROI proposals shape is {tuple(proposals.shape)}, expected (N, 4)"
        )
    if logits.shape[0] != proposals.shape[0]:
        raise ValueError(
            f"class_logits rows={logits.shape[0]} differ from proposals={proposals.shape[0]}"
        )
    if regression.shape[0] != proposals.shape[0]:
        raise ValueError(
            f"box_regression rows={regression.shape[0]} differ from proposals={proposals.shape[0]}"
        )
    num_classes = logits.shape[-1]
    if num_classes < 2:
        raise ValueError(f"class_logits has only {num_classes} class columns")
    if regression.shape[-1] != num_classes * 4:
        raise ValueError(
            "box_regression columns do not match "
            f"{num_classes} classes x 4 coordinates "
            f"(got {tuple(regression.shape)})"
        )
    if len(image_shapes) != 1:
        raise ValueError(
            f"ROI image_shapes unavailable or ambiguous: {image_shapes}"
        )
    return int(num_classes)


class CandidateHooks:
    """Capture RPN proposals and pre-NMS ROI candidates without changing output."""

    def __init__(self, model: Any) -> None:
        self.rpn_counts: list[int] = []
        self.roi_proposals: list[Any] = []
        self.class_logits: list[Any] = []
        self.box_regression: list[Any] = []
        self.transformed_sizes: list[tuple[int, int]] = []
        self.original_sizes: list[tuple[int, int]] = []
        self.roi_image_shapes: list[tuple[int, int]] = []
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
        if _inputs and isinstance(_inputs[0], (list, tuple)) and _inputs[0]:
            original = _inputs[0][0]
            if hasattr(original, "shape"):
                self.original_sizes = [(int(original.shape[-2]), int(original.shape[-1]))]

    def _rpn(self, _module: Any, _inputs: tuple[Any, ...], output: tuple[list[Any], dict[str, Any]]) -> None:
        proposals, _ = output
        self.rpn_counts = [int(row.shape[0]) for row in proposals]

    def _roi_pre(
        self, _module: Any, inputs: tuple[Any, ...], _kwargs: dict[str, Any] | None = None,
    ) -> None:
        # features, proposals, image_shapes, targets
        self.roi_proposals = [row.detach().cpu() for row in inputs[1]]
        if len(inputs) >= 3:
            self.roi_image_shapes = [(int(height), int(width)) for height, width in inputs[2]]

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
    confidence_threshold: float = 0.25,
) -> dict[str, Any]:
    from torchvision.transforms.functional import pil_to_tensor
    from torchvision.ops import batched_nms, clip_boxes_to_image, remove_small_boxes

    hooks = CandidateHooks(model)
    pixels = pil_to_tensor(image.convert("RGB")).float() / 255
    model.roi_heads.nms_thresh = nms_iou
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.detections_per_img = 1000
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
    unavailable: dict[str, Any] | None = None

    if hooks.class_logits and hooks.box_regression and hooks.roi_proposals:
        logits = hooks.class_logits[0].detach().cpu()
        regression = hooks.box_regression[0].detach().cpu()
        proposals = hooks.roi_proposals[0].detach().cpu()

        try:
            num_classes = validate_roi_tensor_shapes(
                logits,
                regression,
                proposals,
                hooks.roi_image_shapes,
            )

            softmax_scores = torch.softmax(logits, dim=-1)
            decoded = model.roi_heads.box_coder.decode(regression, [proposals])
            if decoded.shape == (
                proposals.shape[0],
                num_classes,
                4,
            ):
                decoded = decoded.reshape(proposals.shape[0], num_classes, 4)
            elif decoded.shape == (proposals.shape[0] * num_classes, 4):
                decoded = decoded.reshape(proposals.shape[0], num_classes, 4)
            else:
                raise ValueError(
                    f"BoxCoder.decode returned {tuple(decoded.shape)}, expected "
                    f"({proposals.shape[0]}, {num_classes}, 4) or "
                    f"({proposals.shape[0] * num_classes}, 4)"
                )

            transformed_shape = hooks.roi_image_shapes[0]
            decoded = clip_boxes_to_image(decoded.reshape(-1, 4), transformed_shape)
            decoded = decoded.reshape(proposals.shape[0], num_classes, 4)
            decoded = decoded[:, 1:, :].reshape(-1, 4)
            softmax_scores = softmax_scores[:, 1:].reshape(-1)
            class_ids = torch.arange(1, num_classes, device=logits.device)
            class_ids = class_ids.view(1, -1).expand(proposals.shape[0], -1).reshape(-1)

            keep = remove_small_boxes(decoded, min_size=1e-2)
            decoded = decoded[keep]
            softmax_scores = softmax_scores[keep]
            class_ids = class_ids[keep]

            if len(hooks.original_sizes) != 1:
                raise ValueError(
                    f"Original image size unavailable or ambiguous: {hooks.original_sizes}"
                )
            original_size = hooks.original_sizes[0]
            if decoded.shape[0]:
                original_boxes = model.transform.postprocess(
                    [{
                        "boxes": decoded,
                        "labels": class_ids,
                        "scores": softmax_scores,
                    }],
                    [transformed_shape],
                    [original_size],
                )[0]["boxes"].detach().cpu()
            else:
                original_boxes = decoded.detach().cpu()

            for box, class_id, score in zip(
                original_boxes.tolist(),
                class_ids.tolist(),
                softmax_scores.tolist(),
                strict=True,
            ):
                class_name = CLASSES.get(int(class_id), f"class-{int(class_id)}")
                candidate = {
                    "box": [round(float(value), 3) for value in box],
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "score": float(score),
                }
                candidates.append(candidate)
                class_counts[class_name] += 1
                class_scores[class_name].append(float(score))
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as exc:
            unavailable = {
                "stage": "roi_candidate_reconstruction",
                "reason": str(exc),
            }
            candidates = []
    else:
        unavailable = {
            "stage": "roi_candidate_reconstruction",
            "reason": "ROI proposals, class_logits, or box_regression were not captured",
        }

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
        confidence_keep = candidate_scores >= confidence_threshold
        confident_candidates = [
            candidate
            for candidate, keep in zip(candidates, confidence_keep.tolist(), strict=True)
            if keep
        ]
        nms_keep = batched_nms(
            candidate_boxes[confidence_keep],
            candidate_scores[confidence_keep],
            candidate_labels[confidence_keep],
            nms_iou,
        )
        diagnostic_final = [
            confident_candidates[int(index)]
            for index in nms_keep.tolist()
        ]
        nms_counts["pre_nms"] = len(candidates)
        nms_counts["confidence_threshold"] = confidence_threshold
        nms_counts["pre_confidence"] = len(candidates)
        nms_counts["post_confidence"] = int(confidence_keep.sum().item())
        nms_counts["removed_by_confidence"] = len(candidates) - int(confidence_keep.sum().item())
        nms_counts["pre_nms"] = int(confidence_keep.sum().item())
        nms_counts["post_nms"] = int(nms_keep.shape[0])
        nms_counts["removed_by_nms"] = int(confidence_keep.sum().item()) - int(nms_keep.shape[0])
    else:
        diagnostic_final = []
        nms_counts = {
            "confidence_threshold": confidence_threshold,
            "pre_nms": 0,
            "post_confidence": 0,
            "removed_by_confidence": 0,
            "post_nms": 0,
            "removed_by_nms": 0,
        }

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

    # Compare against the public inference behavior with identical confidence
    # and NMS settings. The diagnostic call itself never changes checkpoint bytes
    # or public model configuration outside this invocation.
    model.roi_heads.score_thresh = confidence_threshold
    model.roi_heads.nms_thresh = nms_iou
    model.roi_heads.detections_per_img = 1000
    with torch.inference_mode():
        public_result = model([pixels])[0]
    public_final = [
        {
            "box": [round(float(value), 3) for value in box.tolist()],
            "class_id": int(label.item()),
            "class_name": CLASSES[int(label.item())],
            "score": float(score.item()),
        }
        for box, label, score in zip(
            public_result["boxes"].detach().cpu(),
            public_result["labels"].detach().cpu(),
            public_result["scores"].detach().cpu(),
            strict=True,
        )
    ]
    final = public_final
    max_coordinate_delta = 0.0
    if len(public_final) == len(diagnostic_final):
        for left, right in zip(public_final, diagnostic_final, strict=True):
            if left["class_id"] != right["class_id"]:
                max_coordinate_delta = float("inf")
                break
            if abs(left["score"] - right["score"]) > 1e-6:
                max_coordinate_delta = float("inf")
                break
            for a, b in zip(left["box"], right["box"], strict=True):
                max_coordinate_delta = max(max_coordinate_delta, abs(a - b))
    else:
        max_coordinate_delta = float("inf")

    invariant = (
        len(public_final) == len(diagnostic_final)
        and max_coordinate_delta != float("inf")
        and max_coordinate_delta <= 1e-3
    )
    result_summary = {
        "image_size": {"width": image.width, "height": image.height},
        "rpn_candidates": rpn_count,
        "roi_candidates": roi_count,
        "candidate_reconstruction": {
            "status": "unavailable" if unavailable else "available",
            **({"failure": unavailable} if unavailable else {}),
        },
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
        "model_output_invariance": {
            "confidence_threshold": confidence_threshold,
            "nms_iou": nms_iou,
            "public_path_detections": len(public_final),
            "diagnostic_detections": len(diagnostic_final),
            "same_class_score_coordinates": invariant,
            "max_coordinate_delta": (
                None if not invariant and max_coordinate_delta == float("inf")
                else max_coordinate_delta
            ),
            "public_path_detections_detail": public_final,
        },
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
    parser.add_argument(
        "--nms-iou",
        type=float,
        action="append",
        help="Class-aware NMS IoU threshold; repeat for multiple sweeps.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Public inspection confidence threshold used for the invariance check.",
    )
    args = parser.parse_args(argv)
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("--confidence must be between 0 and 1")
    if any(not 0.0 <= value <= 1.0 for value in args.nms_iou or []):
        parser.error("--nms-iou must be between 0 and 1")
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
            summary = summarize_candidates(
                model,
                torch,
                image,
                nms_iou,
                top_k=args.top_k,
                confidence_threshold=args.confidence,
            )
            all_results[f"{nms_iou:.2f}"] = summary
            draw_results(image, summary, args.output / f"nms-{nms_iou:.2f}".replace(".", "_"))

    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "image": str(args.image.resolve()),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": __import__("hashlib").sha256(checkpoint.read_bytes()).hexdigest(),
        "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
        "public_confidence_threshold": args.confidence,
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
