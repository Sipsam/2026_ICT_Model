from PIL import Image
import json
from pathlib import Path

from scripts.detector_generalization_diagnostics import dhash, hamming
from ict_pipeline.diagnostics.detector import resolve_final_checkpoint


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
