from PIL import Image
import json
from pathlib import Path
from PIL import Image

from scripts.detector_generalization_diagnostics import dhash, hamming
from ict_pipeline.diagnostics.detector import resolve_final_checkpoint, validate_roi_tensor_shapes


def test_dhash_is_stable_across_identical_images(tmp_path):
    path = tmp_path / "same.png"
    Image.new("RGB", (32, 32), "white").save(path)

    assert hamming(dhash(path), dhash(path)) == 0


def test_dhash_detects_different_images(tmp_path):
    left = tmp_path / "left.png"
    right = tmp_path / "right.png"
    Image.new("RGB", (32, 32), "white").save(left)
    right_image = Image.new("RGB", (32, 32), "white")
    for y in range(32):
        for x in range(32):
            if x == y:
                right_image.putpixel((x, y), (0, 0, 0))
    right_image.save(right)

    assert hamming(dhash(left), dhash(right)) > 0


def test_resolve_final_checkpoint_rejects_non_finalized_config(tmp_path: Path):
    config = tmp_path / "api.json"
    config.write_text(json.dumps({
        "model_root": str(tmp_path),
        "models": {
            "faster-rcnn-final": {
                "mode": "real",
                "model": "faster_rcnn",
            }
        },
    }), encoding="utf-8")

    try:
        resolve_final_checkpoint(config)
    except RuntimeError as exc:
        assert "finalized" in str(exc)
    else:
        raise AssertionError("non-finalized config was accepted")


def test_validate_roi_tensor_shapes_and_class_dimension():
    import torch

    logits = torch.zeros((25, 6))
    regression = torch.zeros((25, 24))
    proposals = torch.zeros((25, 4))

    assert validate_roi_tensor_shapes(logits, regression, proposals, [(480, 640)]) == 6


def test_validate_roi_tensor_shapes_rejects_mismatched_rows():
    import torch

    try:
        validate_roi_tensor_shapes(
            torch.zeros((24, 6)),
            torch.zeros((25, 24)),
            torch.zeros((25, 4)),
            [(480, 640)],
        )
    except ValueError as exc:
        assert "differ" in str(exc)
    else:
        raise AssertionError("mismatched ROI tensor rows were accepted")


def test_box_coder_identity_and_softmax_runtime_shapes():
    import torch
    import torch.nn.functional as functional
    from torchvision.models.detection._utils import BoxCoder

    proposals = torch.tensor([[10.0, 20.0, 110.0, 220.0]])
    regression = torch.zeros((1, 6, 4))
    coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
    decoded = coder.decode(regression, [proposals])

    assert decoded.shape == (1, 6, 4)
    assert torch.allclose(decoded[0, 0], proposals[0])
    scores = functional.softmax(torch.tensor([[1.0, 2.0, 1.0, 0.5, 0.25, 0.1]]), dim=-1)
    assert scores.shape == (1, 6)
    assert 0.0 <= float(scores.sum()) <= 1.000001


def test_confidence_filter_and_batched_nms_counts():
    import torch
    from torchvision.ops import batched_nms

    boxes = torch.tensor([
        [0.0, 0.0, 100.0, 100.0],
        [5.0, 5.0, 95.0, 95.0],
        [200.0, 200.0, 300.0, 300.0],
    ])
    scores = torch.tensor([0.9, 0.8, 0.4])
    labels = torch.tensor([1, 1, 2])

    retained = scores >= 0.5
    keep = batched_nms(boxes[retained], scores[retained], labels[retained], 0.7)

    assert int(retained.sum()) == 2
    assert int(keep.shape[0]) == 1


def test_original_coordinate_restore():
    import torch
    from torchvision.models.detection.transform import resize_boxes

    transformed = torch.tensor([[0.0, 0.0, 320.0, 240.0]])
    original = resize_boxes(transformed, (240, 320), (480, 640))
    assert torch.allclose(original, torch.tensor([[0.0, 0.0, 640.0, 480.0]]))


def test_prepare_internal_selects_ten_per_public_validation_class(tmp_path: Path):
    from ict_pipeline.evaluation.domain_generalization import prepare_internal

    coco = tmp_path / "source" / "coco"
    image_dir = coco / "images" / "val"
    annotation_dir = coco / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir(parents=True)
    categories = [
        {"id": category, "name": name, "supercategory": "pcb_defect"}
        for category, name in {
            1: "cold_solder",
            2: "misalignment",
            3: "short",
            4: "missing_component",
            5: "tombstone",
        }.items()
    ]
    images = []
    annotations = []
    annotation_id = 1
    for category, stem in ((1, "cold"), (2, "mis")):
        for index in range(12):
            image_id = len(images) + 1
            name = f"{stem}-{index:02d}.jpg"
            Image.new("RGB", (64, 64), "white").save(image_dir / name)
            images.append({"id": image_id, "file_name": name, "width": 64, "height": 64})
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category,
                "bbox": [5, 5, 20, 20],
                "area": 400,
                "iscrowd": 0,
            })
            annotation_id += 1
    (annotation_dir / "instances_val.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )

    output = tmp_path / "diagnostic"
    manifest = prepare_internal(tmp_path / "source", output)

    assert len(manifest["images"]) == 20
    assert len(manifest["selection"]["cold_solder"]) == 10
    assert len(manifest["selection"]["misalignment"]) == 10
    annotation = json.loads((output / "coco" / "annotations" / "instances_val.json").read_text())
    assert len(annotation["images"]) == 20
    assert len(annotation["annotations"]) == 20


def test_prepare_external_creates_unlabeled_template_and_guide(tmp_path: Path):
    from ict_pipeline.evaluation.domain_generalization import prepare_external

    result = prepare_external(tmp_path / "external")
    template = json.loads((tmp_path / "external" / "coco" / "annotations" / "instances_external.json").read_text())

    assert template["images"] == []
    assert template["annotations"] == []
    assert [row["name"] for row in template["categories"]] == [
        "cold_solder", "misalignment", "short", "missing_component", "tombstone",
    ]
    assert (tmp_path / "external" / "ANNOTATION_GUIDE.md").is_file()
    assert result["images"].endswith("coco\\images\\external") or result["images"].endswith("coco/images/external")


def test_unlabeled_comparison_report_omits_external_metrics():
    from ict_pipeline.evaluation.domain_generalization import _comparison_markdown

    report = {
        "checkpoint": "best.pt",
        "checkpoint_sha256": "sha",
        "internal": {
            "status": "labeled",
            "image_count": 20,
            "metrics": {
                "map50": 0.1, "map50_95": 0.05, "precision": 0.2, "recall": 0.3,
                "per_class": [],
            },
            "iou_and_errors": {
                "mean_matched_iou": None,
                "matched_iou_by_class": {name: None for name in (
                    "cold_solder", "misalignment", "short", "missing_component", "tombstone",
                )},
                "error_counts": {},
            },
        },
        "external": {"status": "unlabeled", "image_count": 3, "metrics": None},
    }
    markdown = _comparison_markdown(report)

    assert "AP, Precision, and Recall are **not available**" in markdown
    assert "not available" in markdown
