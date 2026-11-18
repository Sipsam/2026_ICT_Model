"""SMT optimization contracts exercised without running a detector."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image
from pydantic import ValidationError
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.evaluation.contracts import EvaluationConfig
from ict_pipeline.models import training
from ict_pipeline.models.coco import coco_training_datasets
from ict_pipeline.models.contracts import Checkpoints, ModelName, TrainConfig, TrainingError, TrainingRequest
from ict_pipeline.models.dataset import Object, Sample, inspect_data
from ict_pipeline.models.optimization.augmentation import augment, read_rgb, transform_objects
from ict_pipeline.models.optimization.config import Augmentation, Balancing, Geometric, Optimization, Photometric
from ict_pipeline.models.optimization.sampling import frequencies, select_train
from ict_pipeline.models.registry import Backend
from ict_pipeline.models.weights import LAYOUTS
from ict_pipeline.models.yolo import YoloAdapter
from ict_pipeline.pipeline.detection import DetectionStages
from ict_pipeline.pipeline.store import fingerprint_path
from test_pretrained import pretrained_profile
from test_training import config_directory, fake_torch
from test_training import detection as detection

PHOTO = Augmentation(photometric=Photometric(noise=6))
FLIP = Augmentation(geometric=Geometric(horizontal_flip_probability=1, flip_policy="verified_mirror_invariant"))


def test_geometry_clip_reject_and_image_alignment() -> None:
    source = Object(1, (1, 2, 20, 8))
    move = (1, 0, -2, 0, 1, 0)
    boxes, changes = transform_objects((source,), move, (32, 24), 0.95)
    assert boxes == (Object(1, (0, 2, 19, 8)),)
    assert changes.clipped == (0,) and not changes.geometry_rejected
    boxes, changes = transform_objects((source,), move, (32, 24), 1.0)
    assert boxes == (source,) and changes.insufficient_visibility == (0,) and changes.geometry_rejected
    removed = Object(2, (30, 10, 2, 4))
    boxes, changes = transform_objects((source, removed), (1, 0, 2, 0, 1, 0), (32, 24), 0.95)
    assert boxes == (source, removed) and changes.removed == (1,) and changes.geometry_rejected
    boxes, _ = transform_objects((Object(2, (10, 8, 4, 2)),), (0, -1, 20, 1, 0, 0), (32, 24), 1.0)
    assert boxes == (Object(2, (10, 10, 2, 4)),)

    image = Image.new("RGB", (32, 24))
    image.paste("white", (1, 2, 21, 10))
    result, target, changes = augment(image, Sample(Path("s.png"), 32, 24, (source,)), FLIP, 1)
    assert not changes.geometry_rejected
    assert target.objects == (Object(1, (11, 2, 20, 8)),)
    assert result.getbbox() == (11, 2, 31, 10)
    assert image.getbbox() == (1, 2, 21, 10)


def test_deterministic_replay_and_annotation_validation() -> None:
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    sample = Sample(Path("a.jpg"), 64, 64, ())
    output, normal, changes = augment(image, sample, PHOTO, 42)
    assert output.tobytes() == augment(image, replace(sample, image=Path("other/a.jpg")), PHOTO, 42)[0].tobytes()
    assert output.tobytes() != augment(image, sample, PHOTO, 43)[0].tobytes()
    assert output.getextrema() == ((122, 134),) * 3
    assert normal == sample and not changes.geometry_rejected
    dark = Image.new("RGB", image.size, (1, 1, 1))
    defect = replace(sample, objects=(Object(3, (12, 12, 20, 20)),))
    guarded, _, diagnostics = augment(dark, defect, PHOTO, 42)
    assert guarded is dark and diagnostics.photometric_rejected == (0,)
    crop = Augmentation(geometric=Geometric(crop_fraction=0.05))
    cropped, target, _ = augment(image, sample, crop, 42)
    assert cropped.size == (target.width, target.height) and target.objects == ()
    assert cropped.width < 64 or cropped.height < 64
    original_boxes = (Object(1, (0, 0, 64, 64)), Object(2, (20, 20, 2, 2)))
    original = replace(sample, objects=original_boxes)
    rejected_image, rejected_sample, diagnostics = augment(image, original, crop, 42)
    assert rejected_image is image and rejected_sample == original
    assert diagnostics.clipped == (0,) and diagnostics.geometry_rejected
    for invalid in (Object(1, (0, 0, 0, 2)), Object(1, (0, 0, float("nan"), 2)), Object(6, (0, 0, 2, 2))):
        with pytest.raises(TrainingError, match=r"invalid annotation indices \[0\]"):
            augment(image, replace(sample, objects=(invalid,)), PHOTO, 42)


@pytest.mark.parametrize("method", ["weighted", "class_aware"])
def test_balancing_distributions_and_multilabel(method: str) -> None:
    typical = Object(1, (1, 1, 4, 4))
    rare = Object(2, (10, 10, 4, 4))
    train = tuple(
        Sample(Path(f"{index:04d}.png"), 16, 16,
               (typical, rare, rare) if index < 20 else (typical,) if index < 800 else ())
        for index in range(1000)
    )
    baseline = frequencies(train)
    assert baseline["image_counts"]["misalignment"] == 20
    assert baseline["object_counts"]["misalignment"] == 40
    assert baseline["multi_label_images"] == 20 and baseline["background_images"] == 200
    cfg = Balancing(method=method)
    selected = select_train(train, cfg, 42)
    assert select_train(train[::-1], cfg, 42) == selected
    assert select_train(train, cfg, 43) != selected
    assert len(selected) == len(train) and set(selected) <= set(train)
    assert frequencies(selected)["image_counts"]["misalignment"] > 30


@pytest.mark.parametrize("policy", [
    {"augmentation": {"geometric": {"horizontal_flip_probability": 1}}},
    {"augmentation": {"geometric": {"vertical_flip_probability": 0.5}}},
    {"augmentation": {"photometric": {"noise": 7}}},
    {"augmentation": {"geometric": {"min_visibility": 0.9}}},
    {"balancing": {"method": "validation_weighted"}},
])
def test_invalid_policy(policy: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        TrainConfig(optimization=policy)


def test_train_only_adapter_inputs(detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = inspect_data(detection, "coco", Credentials())
    yolo = inspect_data(detection, "yolo", Credentials())
    original = fingerprint_path(detection)
    train, val = (data.splits[split].samples for split in ("train", "val"))
    control = TrainingRequest(TrainConfig(), data, tmp_path, "cpu", lambda *_: None)
    legacy, legacy_val = coco_training_datasets(control)
    assert legacy[0][1] == train[0] and legacy_val[0][1] == val[0]
    config = TrainConfig(seed=1, optimization=Optimization(augmentation=FLIP))
    cohort = (train[0], train[1], train[0])
    request = replace(control, config=config, train_cohort=cohort)
    optimized, untouched = coco_training_datasets(request)
    pixels, transformed = optimized[0]
    assert transformed.objects[0].box == (32, 8, 24, 24)
    assert optimized[1][1].objects == () and optimized[2][1] == transformed
    assert untouched[0][1] == val[0] and untouched[0][0].tobytes() == read_rgb(val[0]).tobytes()

    def inspect_staging(_adapter: YoloAdapter, _request: TrainingRequest, staged: Path) -> Checkpoints:
        images = sorted((staged / "images/train").iterdir())
        assert len(images) == len(cohort)
        with Image.open(images[0]) as staged_image:
            assert staged_image.tobytes() == pixels.tobytes()
        assert images[0].read_bytes() == images[2].read_bytes()
        assert (staged / "labels/train/00000001.txt").read_text() == "4 0.6875 0.3125 0.375 0.375\n"
        assert (staged / "labels/train/00000002.txt").read_bytes() == b""
        for path in (*yolo.splits["val"].images, *yolo.splits["val"].annotations):
            assert (staged / path.relative_to(detection / "yolo")).read_bytes() == path.read_bytes()
        assert not (staged / "images/test").exists()
        return Checkpoints(staged / "best.pt", staged / "last.pt")

    monkeypatch.setattr(YoloAdapter, "_train", inspect_staging)
    output = tmp_path / "model"
    output.mkdir()
    YoloAdapter().train(replace(request, data=yolo, output=output))
    assert fingerprint_path(detection) == original


def test_fingerprints_and_resume_invalidate_policy_changes(
    detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = ModelName.FASTER_RCNN
    profiles = config_directory(tmp_path, epochs=2)
    base = (profiles / f"{name}.yaml").read_text()
    assert training.read_training_profile(name, profiles)[0].optimization is None
    assert "optimization" not in training.training_configuration(TrainConfig())

    def revision() -> str:
        return DetectionStages.configure(
            tmp_path, Credentials(), (name,), profiles, EvaluationConfig(), False, lambda _: None,
        ).model_revision(name)

    default_revision = revision()
    profile = profiles / f"{name}.yaml"
    photo = "optimization:\n  augmentation:\n    photometric: {noise: 2}\n"
    profile.write_text(base + photo, encoding="utf-8")
    optimized_revision = revision()
    assert optimized_revision != default_revision
    seen: list[TrainingRequest] = []

    def train(request: TrainingRequest) -> Checkpoints:
        seen.append(request)
        checkpoints = training.checkpoint_paths(name, request.output)
        for path in (checkpoints.last, checkpoints.best):
            path.write_bytes(b"fake-checkpoint")
        request.record_metrics({"epoch": 2 if request.resume else 1})
        if request.resume is None:
            raise KeyboardInterrupt
        return checkpoints

    monkeypatch.setattr(training, "import_module", lambda _: fake_torch())
    monkeypatch.setattr(Backend, "versions", lambda _: {"Pillow": "fixture"})
    monkeypatch.setattr(Backend, "load", lambda _: SimpleNamespace(architecture=LAYOUTS[name].architecture, train=train))
    first = training.plan_training(tmp_path, (name,), detection, Path("runs"), profiles, None)
    with pytest.raises(KeyboardInterrupt):
        list(training.execute_training(first, lambda *_: None))
    prior = first.destination / name
    initial = json.loads((prior / "run.json").read_text())
    assert initial["checkpoint_epoch"] == 1 and initial["configuration"]["optimization"]
    profile.write_text(base + photo.replace("2", "3"), encoding="utf-8")
    assert revision() != optimized_revision
    changed = training.plan_training(tmp_path, (name,), detection, Path("runs"), profiles, prior)
    with pytest.raises(TrainingError, match="original dataset"):
        list(training.execute_training(changed, lambda *_: None))
    assert len(seen) == 1
    profile.write_text(base + photo, encoding="utf-8")
    same = training.plan_training(tmp_path, (name,), detection, Path("runs"), profiles, prior)
    list(training.execute_training(same, lambda *_: None))
    resumed = json.loads((same.destination / name / "run.json").read_text())
    assert resumed["run_fingerprint"] == initial["run_fingerprint"]
    assert resumed["train_cohort"] == initial["train_cohort"]
    assert seen[0].train_cohort == seen[1].train_cohort


def test_preview_imbalance_and_ablation_cli(detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    profiles = config_directory(tmp_path)
    weight = profiles / "fixture.pth"
    weight.write_bytes(b"pinned synthetic bytes")
    for name in ModelName:
        pretrained_profile(profiles, name, weight)
    runner = CliRunner()
    common = ["--dataset", str(detection)]
    report = runner.invoke(app, ["training-data", "imbalance", *common, "--manifest", "manifest.csv"])
    assert report.exit_code == 0, report.output
    statistics = json.loads(report.stdout)
    assert statistics["splits"]["train"]["image_counts"]["tombstone"] == 1
    assert statistics["splits"]["val"]["by_group"]["(unknown)"]["background_images"] == 1
    assert "test" not in statistics["splits"]

    command = ["training-data", "ablation", *common, "--config-dir", str(profiles), "--out", "ablation"]
    preview = runner.invoke(app, [*command, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert not Path("ablation").exists()
    planned = runner.invoke(app, command)
    assert planned.exit_code == 0, planned.output
    manifest = json.loads(Path("ablation/experiment.json").read_text())
    assert json.loads(preview.stdout) == manifest
    assert len(manifest["variants"]) == 5
    assert manifest["evaluation"]["split"] == "val" and not Path("runs").exists()
    for name in ModelName:
        base = manifest["variants"]["pretrained_baseline"][name]["configuration"]
        assert base["optimization"] == {"augmentation": None, "balancing": None}
        for variant, settings in manifest["variants"].items():
            record = settings[name]
            assert record["status"] == "pending" and record["results"] is None
            config = yaml.safe_load(Path("ablation", variant, f"{name}.yaml").read_text())
            assert {k: v for k, v in config.items() if k != "optimization"} == {
                k: v for k, v in base.items() if k != "optimization"
            }
    examples = ["training-data", "preview", *common, "--config-dir", "ablation/combined", "--model", "yolo",
                "--out", "examples"]
    invalid = runner.invoke(app, [*examples, "--sample-id", "sample-3"])
    assert invalid.exit_code == 2 and "from Train" in invalid.output
    result = runner.invoke(app, [*examples, "--sample-id", "sample-1", "--sample-id", "sample-2"])
    assert result.exit_code == 0, result.output
    saved = json.loads(Path("examples/preview.json").read_text())
    assert saved["split"] == "train" and len(saved["examples"]) == 2
    assert saved["examples"][0]["original_boxes"] and not saved["examples"][1]["transformed_boxes"]
    assert Path("examples/001.png").is_file()
    assert runner.invoke(app, examples).exit_code == 2
    assert runner.invoke(app, [*examples[:-1], str(detection / "inside")]).exit_code == 2
