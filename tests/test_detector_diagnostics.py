from PIL import Image
import json
from pathlib import Path

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
