"""CPU-only recommendation and sealed pilot workflow regression tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline import pilot, training_storage
from ict_pipeline.adaptive import Device, Overrides, recommend
from ict_pipeline.cli import app
from ict_pipeline.data.preprocessing.common import read_csv, write_csv, write_json
from ict_pipeline.data.preprocessing.convert import convert_detection
from ict_pipeline.data.preprocessing.policy import CLASS_MAP, label_policy
from ict_pipeline.data.preprocessing.revision import SEAL, seal_dataset
from ict_pipeline.disk import Volume
from ict_pipeline.environment.model import Backend, EnvironmentReport, Graphics, Host, Python, Resources, Torch
from ict_pipeline.evaluation.contracts import EvaluationSplit
from ict_pipeline.models import training
from ict_pipeline.models.coco import CocoAdapter
from ict_pipeline.models.contracts import Checkpoints, ModelName, TrainConfig, TrainingError, TrainingRequest
from ict_pipeline.models.dataset import DetectionData, Sample, Split
from ict_pipeline.models.registry import REGISTRY, Backend as ModelBackend
from ict_pipeline.pipeline.store import digest_json, fingerprint_path

GIB = 1024**3


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> EnvironmentReport:
    monkeypatch.setattr(ModelBackend, "versions", lambda self: {"torchvision": "fixture"})
    return EnvironmentReport(
        Host("Linux", None, None, False, False), Python("3.11.9", "CPython", True),
        Resources(None, 8, 32 * GIB, 40 * GIB, 16 * GIB), Graphics("available", (), "unknown", None),
        Torch("available", cpu=Backend("available", "available", precisions_tested=("fp32",),
                                       operators_tested=("nms", "roi_align_backward"))),
        {}, {}, (),
    )


@pytest.mark.parametrize("device", ("cuda", "rocm", "mps"))
def test_tested_hardware_and_overrides(environment: EnvironmentReport, device: Device) -> None:
    observed = Backend("available", "available", free_device_bytes=6 * GIB,
                       precisions_tested=("fp32", "fp16", "bf16"),
                       operators_tested=("nms", "roi_align_backward"))
    env = replace(environment, torch=replace(environment.torch, **{device: observed}))
    model = ModelName.YOLO if device == "mps" else ModelName.FASTER_RCNN
    assert recommend(model, env).device == device
    explicit = recommend(model, env, Overrides(device, "fp32" if device == "mps" else "bf16",
                                               image_size=160, batch_size=2, workers=0))
    assert explicit.checks_passed and explicit.batch_size == 2 and explicit.image_size == 160
    assert recommend(ModelName.RT_DETR, env, Overrides(device="cpu")).device == "cpu"
    if device == "mps":
        assert not recommend(ModelName.RT_DETR, env, Overrides(device="mps")).checks_passed
        assert not recommend(model, env, Overrides(device="mps", precision="fp16")).checks_passed
    else:
        assert not recommend(ModelName.YOLO, env, Overrides(device=device, precision="bf16")).checks_passed


def test_detection_without_execution_and_low_resources(environment: EnvironmentReport) -> None:
    env = replace(environment, torch=replace(environment.torch, cuda=Backend("available", "unavailable")))
    assert recommend(ModelName.YOLO, env).device == "cpu"
    assert not recommend(ModelName.YOLO, env, Overrides(device="cuda")).checks_passed
    assert not recommend(ModelName.RT_DETR, env, Overrides(precision="fp16")).checks_passed
    low = replace(env, resources=replace(env.resources, available_ram_bytes=1))
    settings = recommend(ModelName.FASTER_RCNN, low, free_disk_bytes=0)
    assert settings.batch_size == 1
    assert any("Estimated memory" in issue for issue in settings.issues)
    assert any("Low disk" in issue for issue in settings.issues)
    assert recommend(ModelName.RT_DETR, env, Overrides(gradient_accumulation=3)).gradient_accumulation == 3
    with pytest.raises(TrainingError, match="native accumulation"):
        recommend(ModelName.YOLO, env, Overrides(gradient_accumulation=3))


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    source = tmp_path / "source"
    root.mkdir()
    source.mkdir()
    rows = []
    index = 0
    for split in ("train", "validation", "test"):
        for label, category in enumerate((*CLASS_MAP, None)):
            for copy in range(2):
                index += 1
                sample = f"sample-{index:02}"
                filename = f"{sample}.jpg"
                Image.new("RGB", (64, 64), (index * 5, label * 20, index)).save(source / filename)
                annotation = f"{sample}.json"
                write_json(source / annotation, {
                    "images": [{"id": index, "file_name": filename, "width": 64, "height": 64}],
                    "annotations": [] if category is None else
                        [{"image_id": index, "category_id": category, "bbox": [8, 8, 20, 20]}],
                })
                rows.append({
                    "sample_id": sample, "split": split, "original_split": "train", "decision": "candidate",
                    "use_for_detection": "candidate", "target_box_count": "0" if category is None else "1",
                    "group_id": f"{split}-{label}-{0 if label == 0 else copy}",
                    "image_path": filename, "annotation_path": annotation,
                })
    manifest = root / "manifest.csv"
    write_csv(manifest, rows, list(rows[0]))
    dataset = root / "detection"
    convert_detection(source, manifest, dataset, include_test=True, portable_yaml=True)
    revision = {
        "revision_id": "fixture", "configuration": {"strategy": "group-aware"}, "label_policy": label_policy(),
        "grouping_methods": ["capture_group"], "selected_manifest_fingerprint": digest_json(rows),
    }
    write_json(dataset / SEAL, seal_dataset(dataset, revision))
    return root


def test_prepare_preserves_groups_and_excludes_test(bundle: Path, tmp_path: Path) -> None:
    cohort, second = tmp_path / "pilot", tmp_path / "repeat"
    before = fingerprint_path(bundle)
    args = (8, 8, 1, 42, lambda *_: None)
    first = pilot.prepare_pilot(tmp_path, bundle, cohort, *args)
    assert first == pilot.prepare_pilot(tmp_path, bundle, second, *args)
    assert first["counts"] == {"train": 8, "val": 8}
    assert all(set(coverage) == set(pilot.STRATA) and min(coverage.values()) >= 1
               for coverage in first["coverage"].values())
    rows = read_csv(cohort / "manifest.csv")
    assert all(row["split"] in {"train", "validation"} and row["original_split"] == "train" for row in rows)
    assert sum(row["group_id"] == "train-0-0" for row in rows) == 2
    assert not list((cohort / "detection").rglob("*test*"))
    assert fingerprint_path(bundle) == before
    with pytest.raises(ValueError, match="separate output"):
        pilot.prepare_pilot(tmp_path, bundle, tmp_path / "data/extracted/pilot", *args)
    with pytest.raises(TrainingError, match="strictly smaller"):
        pilot.prepare_pilot(tmp_path, bundle, tmp_path / "full", 12, 8, 1, 42, lambda *_: None)
    with pytest.raises(TrainingError, match="Cannot fit"):
        pilot.prepare_pilot(tmp_path, bundle, tmp_path / "small", 6, 6, 1, 42, lambda *_: None)


def test_cpu_pilot_approval_interruption_resume_and_summary(
    bundle: Path, tmp_path: Path, environment: EnvironmentReport, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pilot, "collect_environment", lambda _: environment)
    monkeypatch.setattr(training, "import_module", lambda name: object())
    monkeypatch.setattr(training, "resolve_device", lambda *_: "cpu")
    cohort = tmp_path / "pilot"
    pilot.prepare_pilot(tmp_path, bundle, cohort, 8, 8, 1, 42, lambda *_: None)
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    weights = profiles / "weights.pt"
    weights.write_bytes(b"fixture")
    (profiles / "faster_rcnn.yaml").write_text(yaml.safe_dump({
        "initialization": "pretrained", "checkpoint": {
            "source": weights.name, "origin": "fixture", "revision": "fixture",
            "sha256": hashlib.sha256(b"fixture").hexdigest(), "license": "BSD-3-Clause",
            "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn", "library_version": "fixture",
            "format": "state_dict",
        },
    }), encoding="utf-8")
    name = ModelName.FASTER_RCNN
    overrides = Overrides(device="cpu", image_size=64, batch_size=2, workers=0, gradient_accumulation=2)
    plan = pilot.plan_pilot(tmp_path, cohort, (name,), profiles, 1, overrides)
    assert plan.training.models[name].gradient_accumulation == 2
    arguments = ["pilot", "run", "--cohort", str(cohort), "--config-dir", str(profiles),
                 "--model", name, "--device", "cpu", "--image-size", "64"]
    preview = CliRunner().invoke(app, arguments)
    assert preview.exit_code == 0, preview.output
    assert not (cohort / "attempts").exists()
    with pytest.raises(TrainingError, match="requires --approve-train"):
        pilot.run_pilot(plan, approved=False, progress=lambda *_: None)
    with monkeypatch.context() as patch:
        patch.setattr(training_storage, "volume_at", lambda path: Volume(1, path, 1))
        with pytest.raises(TrainingError, match="Low disk"):
            pilot.run_pilot(plan, approved=True, progress=lambda *_: None)
        assert not plan.summary_directory.exists()

    class Adapter:
        architecture = "fasterrcnn_mobilenet_v3_large_320_fpn"

        def train(self, request: TrainingRequest) -> Checkpoints:
            assert set(request.data.splits) == {"train", "val"}
            assert request.pretrained.path == weights
            assert request.config.gradient_accumulation == 2
            for kind in ("best", "last"):
                (request.output / f"{kind}.pt").write_bytes(b"fixture checkpoint")
            request.record_overhead("loader_wait_seconds", 0.5)
            request.record_metrics({"epoch": 1, "train_loss": 2.0, "val_loss": 3.0})
            raise KeyboardInterrupt

    monkeypatch.setattr(ModelBackend, "load", lambda self: Adapter())
    with pytest.raises(KeyboardInterrupt):
        pilot.run_pilot(plan, approved=True, progress=lambda *_: None)
    interrupted = json.loads((plan.summary_directory / "pilot.json").read_text())
    assert interrupted["status"] == "interrupted"
    assert interrupted["models"][name]["resources"]["checkpoint_logical_bytes"] > 0
    assert interrupted["models"][name]["overhead"]["loader_wait_seconds"] == 0.5

    def evaluate(*args: Any) -> list[Path]:
        assert args[3] == EvaluationSplit.VALIDATION
        assert args[6:9] == ("best", False, None)
        path = args[5] / "result.json"
        path.parent.mkdir()
        return [path]

    monkeypatch.setattr(pilot, "evaluate_runs", evaluate)
    monkeypatch.setattr(pilot, "read_report", lambda _: {"metrics": {"map": 0.0}})
    resumed = pilot.plan_pilot(tmp_path, cohort, (name,), profiles, 1, overrides, plan.training.destination / name)
    completed = json.loads(pilot.run_pilot(resumed, approved=True, progress=lambda *_: None).read_text())
    assert completed["status"] == "completed" and completed["evaluation_split"] == "val"
    assert completed["models"][name]["metrics"]["val_loss"] == 3.0
    assert completed["models"][name]["validation_metrics"] == {"map": 0.0}
    assert completed["cohort"]["revision"] == interrupted["cohort"]["revision"]


@pytest.mark.cpu_backend
def test_coco_accumulation_flushes_partial_group(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("numpy")
    image = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8)).save(image)
    sample = Sample(image, 8, 8, ())
    data = DetectionData({
        "train": Split((image,), (), (sample,) * 5),
        "val": Split((image,), (), (sample,)),
    }, {}, "")

    class ScalarAdapter(CocoAdapter):
        architecture = "scalar fixture"

        def build_model(self, config: TrainConfig) -> Any:
            return torch.nn.Linear(1, 1, bias=False)

        def batch_loss(self, model: Any, batch: Any, request: TrainingRequest) -> Any:
            return model.weight.sum()

    config = TrainConfig(epochs=1, batch_size=2, gradient_accumulation=2, optimizer="SGD", lr=0.1)
    torch.manual_seed(config.seed)
    initial = ScalarAdapter().build_model(config).weight.detach().clone()
    result = ScalarAdapter().train(TrainingRequest(config, data, tmp_path, "cpu", lambda *_: None))
    state = torch.load(result.last, weights_only=True)
    torch.testing.assert_close(state["model"]["weight"], initial - 0.2)


@pytest.mark.cpu_backend
def test_native_yolo_cpu_pilot(bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from test_pretrained import pretrained_profile
    from test_training import config_directory, optional_runtime

    torch = optional_runtime(ModelName.YOLO)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils import ROOT

    profiles = config_directory(tmp_path)
    source = profiles / "fixture.pt"
    torch.save({"model": DetectionModel(str(ROOT / "cfg/models/11/yolo11n.yaml"), nc=80, verbose=False)}, source)
    pretrained_profile(profiles, ModelName.YOLO, source, file_format="ultralytics_pickle",
                       library_version=REGISTRY[ModelName.YOLO].versions()["ultralytics"])
    # Exercise the actual CPU model without making fixture assertions depend on host load.
    environment = EnvironmentReport(
        Host("Windows", None, None, False, False), Python("3.11.9", "CPython", True),
        Resources(None, 8, 8 * GIB, 40 * GIB, 4 * GIB), Graphics("unknown", (), "unknown", None),
        Torch("available", cpu=Backend("available", "available", precisions_tested=("fp32",),
                                       operators_tested=("nms",))),
        {}, {}, (),
    )
    monkeypatch.setattr(pilot, "collect_environment", lambda _: environment)
    cohort = tmp_path / "cohort"
    pilot.prepare_pilot(tmp_path, bundle, cohort, 8, 8, 1, 42, lambda *_: None)
    before = fingerprint_path(cohort / "detection")
    plan = pilot.plan_pilot(tmp_path, cohort, (ModelName.YOLO,), profiles, 1,
                           Overrides(device="cpu", image_size=64, batch_size=2))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        summary = json.loads(pilot.run_pilot(plan, approved=True, progress=lambda *_: None).read_text())
    finally:
        torch.set_num_threads(threads)
    assert summary["status"] == "completed"
    result = summary["models"]["yolo"]
    assert result["overhead"]["yolo_cache_logical_bytes"] > 0
    assert result["resources"]["adapter_wall_seconds"] > 0
    assert result["validation_metrics"]["map50_95"] is not None
    assert result["next_training_settings"]["batch_size"] == 2
    assert fingerprint_path(cohort / "detection") == before
