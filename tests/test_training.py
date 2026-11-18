"""Training contract, hardware diagnostics and tiny CPU smoke tests."""

import csv
import importlib.util
import json
from pathlib import Path
from shutil import copyfile
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.data.preprocessing.convert import convert_detection
from ict_pipeline.data.preprocessing.policy import NAMES
from ict_pipeline.models.contracts import Checkpoints, ModelName, TrainConfig, TrainingError, TrainingRequest
from ict_pipeline.models.dataset import inspect_data
from ict_pipeline.models.registry import Backend, REGISTRY, resolve_device
from ict_pipeline.models.training import execute_training, plan_training
from ict_pipeline.pipeline.store import fingerprint_path


@pytest.fixture
def detection(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    source.mkdir()
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        fields = ["split", "sample_id", "decision", "use_for_detection", "image_path", "annotation_path"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for index, split in enumerate(("train", "train", "validation", "validation"), 1):
            stem = f"sample-{index}"
            image, annotation = f"{stem}.jpg", f"{stem}.json"
            Image.new("RGB", (64, 64), (index * 37, 0, 0)).save(source / image)
            (source / annotation).write_text(json.dumps({
                "images": [{"id": 1, "file_name": image, "width": 64, "height": 64}],
                "annotations": [{"image_id": 1, "category_id": 30, "bbox": [8, 8, 24, 24]}] if index % 2 else [],
                "sensor_data": {"defect_detected": 1},
            }), encoding="utf-8")
            writer.writerow(dict(zip(fields, (split, stem, "candidate", "candidate", image, annotation))))
    result = tmp_path / "detection"
    convert_detection(source, manifest, result, image_mode="link")
    return result


def config_directory(tmp_path: Path, epochs: int = 1) -> Path:
    folder = tmp_path / "models"
    folder.mkdir()
    for model in REGISTRY:
        (folder / f"{model}.yaml").write_text(
            f"epochs: {epochs}\nbatch_size: 2\nimage_size: 64\nlr: 0.0001\nsave_period: 1\n", encoding="utf-8",
        )
    return folder


def test_preview_and_approval(detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    args = ["train", "--all", "--dataset", str(detection)]
    preview = runner.invoke(app, [*args, "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert all(model in preview.output for model in REGISTRY)
    denied = runner.invoke(app, args)
    assert denied.exit_code == 1
    assert "requires explicit approval" in denied.stderr
    assert not (tmp_path / "runs").exists()
    assert runner.invoke(app, [*args, "--model", "yolo"]).exit_code == 2


def test_generated_formats_preserve_class_and_split(detection: Path) -> None:
    coco = inspect_data(detection, "coco", Credentials())
    yolo = inspect_data(detection, "yolo", Credentials())
    assert coco.split_fingerprints == yolo.split_fingerprints
    assert [sample.image.name for sample in coco.splits["train"].samples] == ["sample-1.jpg", "sample-2.jpg"]
    assert [sample.image.name for sample in coco.splits["val"].samples] == ["sample-3.jpg", "sample-4.jpg"]
    assert coco.splits["train"].samples[0].objects[0].category == len(NAMES)
    assert not coco.splits["train"].samples[1].objects
    assert yolo.splits["train"].annotations[0].read_text().split()[0] == str(len(NAMES) - 1)
    assert yolo.splits["train"].annotations[1].read_bytes() == b""


def test_duplicate_image_rejected(detection: Path) -> None:
    folder = detection / "coco" / "images"
    # Replace the converted hard link itself, leaving source bytes untouched.
    destination = folder / "val" / "sample-4.jpg"
    destination.unlink()
    copyfile(folder / "train" / "sample-1.jpg", destination)
    with pytest.raises(TrainingError, match="duplicate"):
        inspect_data(detection, "coco", Credentials())


def fake_torch(*, cuda: bool = False, hip: bool = False, mps: bool = False, bf16: bool = False) -> ModuleType:
    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(
        is_available=lambda: cuda, device_count=lambda: int(cuda), is_bf16_supported=lambda **_: bf16,
    )
    torch.version = SimpleNamespace(hip="mock" if hip else None)
    torch.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
    return torch


@pytest.mark.parametrize(("name", "config", "hardware", "expected"), [
    (ModelName.YOLO, {"device": "auto"}, {"mps": True}, "mps"),
    (ModelName.FASTER_RCNN, {"device": "auto"}, {"mps": True}, "cpu"),
    (ModelName.RT_DETR, {"device": "rocm"}, {"hip": True, "cuda": True}, "cuda:0"),
    (ModelName.RT_DETR, {"device": "cuda", "precision": "bf16"}, {"cuda": True, "bf16": True}, "cuda:0"),
    (ModelName.YOLO, {"device": "cuda"}, {}, "CUDA is unavailable"),
    (ModelName.FASTER_RCNN, {"device": "mps"}, {"mps": True}, "does not support"),
    (ModelName.RT_DETR, {"device": "cpu", "precision": "fp16"}, {}, "Mixed precision requires"),
    (ModelName.RT_DETR, {"device": "cuda", "precision": "bf16"}, {"cuda": True}, "lacks native BF16"),
])
def test_hardware(
    name: ModelName, config: dict[str, str], hardware: dict[str, bool], expected: str,
) -> None:
    arguments = (fake_torch(**hardware), TrainConfig(**config), REGISTRY[name])
    if expected in {"cpu", "mps", "cuda:0"}:
        assert resolve_device(*arguments) == expected
    else:
        with pytest.raises(TrainingError, match=expected):
            resolve_device(*arguments)


def test_all_and_oom_are_reported(
    detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Adapter:
        architecture = "synthetic"

        def __init__(self, backend: Backend) -> None:
            self.backend = backend

        def train(self, request: TrainingRequest) -> Checkpoints:
            if self.backend is REGISTRY[ModelName.RT_DETR]:
                raise RuntimeError("CUDA out of memory, private backend detail")
            root = request.output / "weights" if self.backend.data_format == "yolo" else request.output
            root.mkdir(exist_ok=True)
            outputs = Checkpoints(root / "best.pt", root / "last.pt")
            for path in (outputs.best, outputs.last):
                path.write_bytes(b"mock weights")
            return outputs

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ict_pipeline.models.training.import_module", lambda _name: fake_torch())
    monkeypatch.setattr(Backend, "versions", lambda self: {})
    monkeypatch.setattr(Backend, "load", lambda backend: Adapter(backend))
    result = CliRunner().invoke(app, ["train", "--all", "--dataset", str(detection), "--approve-train"])
    assert result.exit_code == 1, result.output
    assert "yolo completed" in result.stdout and "faster_rcnn completed" in result.stdout, result.output
    assert "Reduce batch_size or image_size" in result.stderr
    assert "private backend detail" not in result.output
    by_model = {
        path.parent.name: json.loads(path.read_text())
        for path in (tmp_path / "runs").rglob("run.json")
    }
    assert {name: data["status"] for name, data in by_model.items()} == {
        "yolo": "completed", "faster_rcnn": "completed", "rt_detr": "failed",
    }
    assert by_model["yolo"]["split_fingerprints"] == by_model["faster_rcnn"]["split_fingerprints"]
    assert by_model["yolo"]["checkpoints"]["best"]["path"] == "weights/best.pt"


def optional_runtime(name: ModelName) -> ModuleType:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    optional = {ModelName.YOLO: ("ultralytics",), ModelName.RT_DETR: ("transformers", "scipy")}
    for module in optional.get(name, ()):
        if importlib.util.find_spec(module) is None:
            pytest.skip(f"{module} is not installed")
    return torch


@pytest.mark.parametrize("name", tuple(REGISTRY))
def test_tiny_cpu_training(
    name: ModelName, detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = optional_runtime(name)
    monkeypatch.chdir(tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    before = fingerprint_path(detection)
    try:
        result = CliRunner().invoke(app, [
            "train", "--model", name, "--dataset", str(detection),
            "--config-dir", str(config_directory(tmp_path)), "--approve-train",
        ])
        assert result.exit_code == 0, result.output
        metadata_path, = (tmp_path / "runs").rglob("run.json")
        metadata = json.loads(metadata_path.read_text())
        assert metadata["status"] == "completed"
        for item in metadata["checkpoints"].values():
            assert fingerprint_path(metadata_path.parent / item["path"]) == item["fingerprint"]
        assert next(metadata_path.parent.rglob("epoch-1.pt")).is_file()
        last = metadata_path.parent / metadata["checkpoints"]["last"]["path"]
        if name == ModelName.YOLO:
            from ultralytics.nn.tasks import load_checkpoint

            model, state = load_checkpoint(last)
            assert state["epoch"] == 0 and model.names == dict(enumerate(NAMES))
            with torch.inference_mode():
                prediction, _ = model.eval()(torch.zeros(1, 3, 64, 64))
            assert prediction.shape[1] == 4 + len(NAMES)
        else:
            state = torch.load(last, weights_only=True)
            adapter = REGISTRY[name].load()
            model = adapter.build_model(TrainConfig(image_size=64)).eval()
            model.load_state_dict(state["model"])
            with torch.inference_mode():
                pixels = torch.zeros(1, 3, 64, 64)
                prediction = model(pixels) if name == ModelName.FASTER_RCNN else model(pixel_values=pixels)
            if name == ModelName.FASTER_RCNN:
                assert "boxes" in prediction[0]
            else:
                assert prediction.logits.shape[-1] == len(NAMES)
        assert fingerprint_path(detection) == before
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("name", tuple(REGISTRY))
def test_resume_after_epoch_checkpoint(
    name: ModelName, detection: Path, tmp_path: Path,
) -> None:
    torch = optional_runtime(name)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cfg = config_directory(tmp_path, epochs=2)

    def interrupt(stage: str, done: int, _total: int) -> None:
        if stage.endswith("epoch checkpoint") and done == 1:
            raise KeyboardInterrupt

    try:
        plan = plan_training(tmp_path, (name,), detection, Path("runs"), cfg, None)
        with pytest.raises(KeyboardInterrupt):
            list(execute_training(plan, interrupt))
        previous = plan.destination / name
        assert json.loads((previous / "run.json").read_text())["status"] == "interrupted"
        before = fingerprint_path(previous)
        resumed = plan_training(tmp_path, (name,), detection, Path("runs"), cfg, previous)
        [(model, result)] = execute_training(resumed, lambda *_: None)
        assert model == name
        assert result.last.is_file() and result.best.is_file()
        assert fingerprint_path(previous) == before
        assert json.loads((resumed.destination / name / "run.json").read_text())["status"] == "completed"
    finally:
        torch.set_num_threads(threads)
