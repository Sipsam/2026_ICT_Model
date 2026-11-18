"""COCO bbox AP (all areas, non-crowd) and fixed-threshold detection metrics."""

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Mapping
from itertools import accumulate
from statistics import mean

from ..models.contracts import Progress
from ..models.dataset import Box, Sample
from .contracts import CLASSES, MAX_DETECTIONS, EvaluationConfig, EvaluationError, Prediction, xywh_to_xyxy

IOU_THRESHOLDS = tuple(i / 100 for i in range(50, 100, 5))
RECALL_THRESHOLDS = tuple(i / 100 for i in range(101))


def box_iou(a: Box, b: Box) -> float:
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return overlap / (area - overlap)


def _match(overlaps: Iterable[tuple[float, ...]], threshold: float) -> Iterable[int | None]:
    used: set[int] = set()
    for row in overlaps:
        best = max(
            ((iou, index) for index, iou in enumerate(row) if iou >= threshold and index not in used),
            default=None,
        )
        index = best[1] if best is not None else None
        if index is not None:
            used.add(index)
        yield index


def _interpolated_ap(matches: Iterable[bool], positives: int) -> float:
    tp = list(accumulate(matches))
    recall = [count / positives for count in tp]
    precision = [count / rank for rank, count in enumerate(tp, 1)]
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    # Unreached recall points contribute zero.
    return mean(
        precision[index] if (index := bisect_left(recall, point)) < len(precision) else 0.0
        for point in RECALL_THRESHOLDS
    )


def _prf(tp: int, fp: int, fn: int) -> dict[str, int | float]:
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
    }


def detection_metrics(
    samples: tuple[Sample, ...], predictions: Mapping[str, Prediction], config: EvaluationConfig,
    progress: Progress | None = None,
) -> dict[str, object]:
    ids = {sample.image.stem for sample in samples}
    if len(ids) != len(samples) or ids != predictions.keys():
        raise EvaluationError("Predictions must cover every labeled sample exactly once.")
    if not any(sample.objects for sample in samples):
        raise EvaluationError("AP requires ground-truth objects in the selected split.")

    thresholds = (*IOU_THRESHOLDS, config.match_iou)
    ranked: dict[int, list[tuple[float, int]]] = {category: [] for category in CLASSES}
    positives = dict.fromkeys(CLASSES, 0)
    labels = [*CLASSES.values(), "background"]
    background = len(CLASSES)
    matrix = [[0] * len(labels) for _ in labels]

    if progress:
        progress("match detections", 0, len(samples))
    for index, sample in enumerate(sorted(samples, key=lambda sample: sample.image.stem), 1):
        pred = predictions[sample.image.stem]
        if pred.sample_id != sample.image.stem or pred.preprocessing.original_size != (sample.width, sample.height):
            raise EvaluationError("Prediction metadata does not match its ground-truth sample.")
        truth = [(obj.category, xywh_to_xyxy(obj.box)) for obj in sample.objects]
        regions: dict[int, list[Box]] = defaultdict(list)
        for category, box in truth:
            regions[category].append(box)
            positives[category] += 1
        candidates = sorted(
            zip(pred.class_ids, pred.boxes, pred.confidence_scores, strict=True), key=lambda row: -row[2],
        )
        counts: dict[int, int] = defaultdict(int)
        detections: list[tuple[int, Box, float]] = []
        for category, box, score in candidates:
            if score >= config.score_floor and counts[category] < MAX_DETECTIONS:
                detections.append((category, box, score))
                counts[category] += 1

        for category in CLASSES:
            rows = [(score, tuple(box_iou(box, gt) for gt in regions[category]))
                    for predicted, box, score in detections if predicted == category]
            flags = [0] * len(rows)
            if regions[category]:
                # Record the matching results in one bit mask per prediction instead
                # of keeping a copy of every prediction at every IoU threshold.
                for curve, threshold in enumerate(thresholds):
                    for rank, match in enumerate(_match((overlaps for _, overlaps in rows), threshold)):
                        if match is not None:
                            flags[rank] |= 1 << curve
            ranked[category].extend((score, bits) for (score, _), bits in zip(rows, flags, strict=True))
        operating = [(category, box) for category, box, score in detections if score >= config.confidence]
        unclaimed = set(range(len(truth)))
        matches = _match((tuple(box_iou(box, gt) for _, gt in truth) for _, box in operating), config.match_iou)
        for (category, _), match in zip(operating, matches, strict=True):
            if match is None:
                matrix[background][category - 1] += 1
            else:
                unclaimed.remove(match)
                matrix[truth[match][0] - 1][category - 1] += 1
        for unmatched in unclaimed:
            matrix[truth[unmatched][0] - 1][background] += 1
        if progress:
            progress("match detections", index, len(samples))

    ap50, ap5095 = [], []
    per_class = []
    total_tp = total_fp = total_fn = 0
    for category, name in CLASSES.items():
        scored = sorted(ranked[category], key=lambda row: -row[0])
        count = positives[category]
        ap = [
            _interpolated_ap((bool(bits & (1 << curve)) for _, bits in scored), count)
            for curve in range(len(IOU_THRESHOLDS))
        ] if count else []
        operating = [bits for score, bits in scored if score >= config.confidence]
        tp = sum(bool(bits & (1 << len(IOU_THRESHOLDS))) for bits in operating)
        fp, fn = len(operating) - tp, count - tp
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
        if ap:
            ap50.append(ap[0])
            ap5095.append(mean(ap))
        per_class.append({
            "class_id": category, "name": name, "ground_truth_objects": count,
            "ap50": ap[0] if ap else None, "ap50_95": mean(ap) if ap else None, **_prf(tp, fp, fn),
        })
    return {
        "map50": mean(ap50), "map50_95": mean(ap5095), "per_class": per_class,
        **_prf(total_tp, total_fp, total_fn),
        "confusion_matrix": {"truth_rows": labels, "prediction_columns": labels, "counts": matrix},
    }
