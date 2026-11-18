"""Offline checkpoint verification, strict transfer and detector CPU smoke tests."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.config.settings import Credentials
from ict_pipeline.data.preprocessing.policy import NAMES
from ict_pipeline.models import training
from ict_pipeline.models.contracts import ModelName, TrainConfig, TrainingError, TrainingRequest
from ict_pipeline.models.registry import Backend, REGISTRY
from ict_pipeline.models.training import execute_training, plan_training
from ict_pipeline.models.weights import LAYOUTS, checkpoint_source, load_weights, transfer_state, verify_weights
from ict_pipeline.pipeline.store import digest_json
from test_training import config_directory, detection, fake_torch, optional_runtime  # noqa: F401


def pretrained_profile(directory: Path, name: ModelName, path: Path, *,
                       file_format: str = "state_dict", library_version: str = "fixture") -> dict[str, Any]:
    profile = directory / f"{name}.yaml"
    previous = yaml.safe_load(profile.read_text(encoding="utf-8"))
    layout = LAYOUTS[name]
    values = {
        **previous, "initialization": "pretrained", "freeze_backbone": True,
        "fine_tuning": {"optimizer": "SGD", "lr": 0.0002, "momentum": 0.8, "weight_decay": 0.0},
        "checkpoint": {
            "source": path.name, "origin": "https://example.org/weights", "revision": "fixture-v1",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "license": "Apache-2.0",
            "architecture": layout.architecture, "library_version": library_version, "format": file_format,
        },
    }
    if file_format == "ultralytics_pickle":
        values["checkpoint"]["trusted_pickle"] = True
    profile.write_text(yaml.safe_dump(values), encoding="utf-8")
    return values


def test_local_scope_missing_cache_and_integrity(detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    name = ModelName.FASTER_RCNN
    directory = config_directory(tmp_path)
    local = directory / "weights.pt"
    local.write_bytes(b"pinned fixture")
    values = pretrained_profile(directory, name, local)
    config = TrainConfig.model_validate(values)
    assert TrainConfig().initialization == "scratch"
    assert config.optimizer_name == "SGD" and config.learning_rate == 0.0002
    source = checkpoint_source(name, config, directory, Credentials())
    assert source == local
    with pytest.raises(TrainingError, match="requires torchvision fixture"):
        verify_weights(name, config, source, {"torchvision": "wrong"})
    with pytest.raises(TrainingError, match="architecture"):
        checkpoint_source(ModelName.YOLO, config, directory, Credentials())
    preview = CliRunner().invoke(app, [
        "train", "--model", name, "--config-dir", str(directory), "--dataset", str(detection), "--dry-run",
    ])
    assert preview.exit_code == 0, preview.output
    assert all(value in preview.output for value in ("pretrained", "checkpoint_bytes", "fp32_rgb_batch_bytes"))
    assert not (tmp_path / "runs").exists()
    libraries = {"torchvision": "fixture"}
    first = verify_weights(name, config, source, libraries)
    assert first is not None and verify_weights(name, config, source, libraries) == first
    local.write_bytes(b"tampered")
    with pytest.raises(TrainingError, match="SHA-256 mismatch"):
        verify_weights(name, config, source, libraries)
    local.unlink()
    with pytest.raises(TrainingError, match="missing from the local cache"):
        verify_weights(name, config, source, libraries)


def test_checkpoint_configuration_rejects_unsafe_origins(tmp_path: Path) -> None:
    directory = config_directory(tmp_path)
    name = ModelName.YOLO
    local = directory / "weights.pt"
    local.touch()
    values = pretrained_profile(directory, name, local, file_format="ultralytics_pickle")
    values["checkpoint"]["trusted_pickle"] = False
    with pytest.raises(ValueError, match="explicit trusted_pickle"):
        TrainConfig.model_validate(values)
    values["checkpoint"].update(trusted_pickle=True, source="https://example.org/file.pt")
    with pytest.raises(TrainingError, match="local checkpoint file path"):
        checkpoint_source(name, TrainConfig.model_validate(values), directory, Credentials())
    values["checkpoint"].update(source=local.name, origin="https://username@example.org/file.pt")
    with pytest.raises(TrainingError, match="public HTTPS URL"):
        checkpoint_source(name, TrainConfig.model_validate(values), directory, Credentials())


def test_deserialization_uses_verified_copy_and_never_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    directory = config_directory(tmp_path)
    name = ModelName.YOLO
    local = directory / "weights.pt"
    local.write_bytes(b"pinned")
    values = pretrained_profile(directory, name, local)
    config = TrainConfig.model_validate(values)
    verified = verify_weights(name, config, local, {"ultralytics": "fixture"})
    request = TrainingRequest(config, SimpleNamespace(), tmp_path, "cpu", lambda *_: None, pretrained=verified)
    calls = []

    def reject(path: Path, *, map_location: str, weights_only: bool) -> None:
        assert path.parent.parent == tmp_path and path.read_bytes() == b"pinned"
        calls.append((map_location, weights_only))
        raise ValueError("private deserialization detail")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=reject))
    with pytest.raises(TrainingError, match="declared format"):
        load_weights(request)
    assert calls == [("cpu", True)]
    local.write_bytes(b"changed during startup")
    with pytest.raises(TrainingError, match="SHA-256 mismatch"):
        load_weights(request)
    assert len(calls) == 1 and not list(tmp_path.glob("init-*"))


def test_partial_transfer_rejects_unapproved_keys_before_loading(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 5)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    reports = []
    request = TrainingRequest(TrainConfig(), SimpleNamespace(), tmp_path, "cpu", lambda *_: None,
                              record_transfer=reports.append)
    with pytest.raises(TrainingError, match="incompatible"):
        transfer_state(
            model, {"weight": torch.zeros(4, 3), "other": torch.zeros(1)},
            {"weight": (5, 3), "bias": (5,)}, {"bias"}, request,
        )
    assert reports[0] == {
        "missing_keys": ["bias"], "unexpected_keys": ["other"], "incompatible_keys": ["weight"],
        "reset_keys": ["bias"], "restored_aliases": {},
    }
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_transfer_checks_explicit_legacy_tensor_renames(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Linear(3, 5)
    reports = []
    request = TrainingRequest(TrainConfig(), SimpleNamespace(), tmp_path, "cpu", lambda *_: None,
                              record_transfer=reports.append)
    source = {"legacy.weight": torch.ones(5, 3), "bias": torch.zeros(5)}
    shapes = {"weight": (5, 3), "bias": (5,)}
    renames = {"legacy.weight": "weight"}
    transfer_state(model, source, shapes, set(), request, renames=renames)
    assert reports[0]["missing_keys"] == ["weight"]
    assert reports[0]["unexpected_keys"] == ["legacy.weight"]
    assert reports[0]["renamed_keys"] == renames
    assert torch.equal(model.weight, source["legacy.weight"])
    with pytest.raises(TrainingError, match="incompatible"):
        transfer_state(model, {**source, "weight": torch.zeros(5, 3)}, shapes, set(), request, renames=renames)
    with pytest.raises(TrainingError, match="incompatible"):
        transfer_state(model, {**source, "legacy.weight": torch.ones(4, 3)}, shapes, set(), request, renames=renames)
    assert torch.equal(model.weight, source["legacy.weight"])


@pytest.mark.parametrize("pretrained", [False, True])
def test_revision_bound_resume_with_legacy_scratch_compatibility(
    detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pretrained: bool,
) -> None:
    name = ModelName.FASTER_RCNN
    folder = config_directory(tmp_path, epochs=2)
    if pretrained:
        path = folder / "weight.pth"
        path.write_bytes(b"pinned")
        pretrained_profile(folder, name, path)

    def train(request: TrainingRequest) -> training.Checkpoints:
        checkpoints = training.checkpoint_paths(name, request.output)
        for path in (checkpoints.last, checkpoints.best):
            path.write_bytes(b"fixture")
        request.record_metrics({"epoch": 2 if request.resume else 1})
        if request.resume is None:
            raise KeyboardInterrupt
        return checkpoints

    monkeypatch.setattr(training, "import_module", lambda _: fake_torch())
    monkeypatch.setattr(Backend, "versions", lambda _: {"torchvision": "fixture"})
    monkeypatch.setattr(Backend, "load", lambda _: SimpleNamespace(architecture=LAYOUTS[name].architecture, train=train))
    first = plan_training(tmp_path, (name,), detection, Path("runs"), folder, None)
    first = replace(first, dataset_revision={"fixture-revision": "one"})
    with pytest.raises(KeyboardInterrupt):
        list(execute_training(first, lambda *_: None))
    previous = first.destination / name
    if not pretrained:
        path = previous / "run.json"
        metadata = json.loads(path.read_text())
        # Reconstruct the earlier schema and its repository LF code fingerprint.
        for key in ("initialization", "initialization_mode", "run_fingerprint"):
            metadata.pop(key)
        for key in ("initialization", "checkpoint", "fine_tuning", "freeze_backbone"):
            del metadata["configuration"][key]
        del metadata["code_fingerprints"]["weights.py"]
        metadata["code_fingerprints"].update(training._LEGACY_SCRATCH_CHANGES[1])
        contract_keys = (
            "schema_version", "model", "architecture", "configuration", "device", "dataset_format",
            "split_fingerprints", "annotation_fingerprint", "label_policy", "split_annotation_fingerprints",
            "libraries", "ict_pipeline", "code_fingerprints", "python", "system", "machine", "checkpoint_format",
            "library_license", "best_criterion", "dataset_revision", "profile_fingerprint",
        )
        metadata["run_fingerprint"] = digest_json({key: metadata[key] for key in contract_keys})
        path.write_text(json.dumps(metadata), encoding="utf-8")
    retry = plan_training(tmp_path, (name,), detection, Path("runs"), folder, previous)
    with pytest.raises(TrainingError, match="original dataset"):
        list(execute_training(replace(retry, dataset_revision={"fixture-revision": "two"}), lambda *_: None))
    retry = plan_training(tmp_path, (name,), detection, Path("runs"), folder, previous)
    [(model, checkpoint)] = execute_training(replace(retry, dataset_revision=first.dataset_revision), lambda *_: None)
    assert model == name and checkpoint.last.is_file()


@pytest.mark.parametrize(("name", "file_format"), [
    (ModelName.YOLO, "ultralytics_pickle"), (ModelName.FASTER_RCNN, "state_dict"),
    (ModelName.RT_DETR, "safetensors"),
])
def test_real_cpu_fine_tuning_and_resume(
    name: ModelName, file_format: str, detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = optional_runtime(name)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        _fine_tuning_smoke(name, file_format, detection, tmp_path, monkeypatch, torch)
    finally:
        torch.set_num_threads(threads)


def _fine_tuning_smoke(
    name: ModelName, file_format: str, detection: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, torch: Any,
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = config_directory(tmp_path, epochs=2)
    local = folder / ("model.safetensors" if file_format == "safetensors" else "weights.pt")
    local.touch()
    adapter = REGISTRY[name].load()
    values = pretrained_profile(folder, name, local, file_format=file_format,
                                library_version=REGISTRY[name].versions()[LAYOUTS[name].library])
    cfg = TrainConfig.model_validate(values)
    with torch.no_grad():
        if name == ModelName.YOLO:
            with monkeypatch.context() as environment:
                environment.setenv("YOLO_AUTOINSTALL", "false")
                environment.setenv("YOLO_OFFLINE", "true")
                environment.setenv("YOLO_CONFIG_DIR", str(tmp_path))
                from ultralytics.nn.tasks import DetectionModel
                from ultralytics.utils import ROOT

                source_model = DetectionModel(str(ROOT / "cfg/models/11/yolo11n.yaml"), nc=80, verbose=False)
                key = "model.0.conv.weight"
                source_model.state_dict()[key].fill_(0.125)
                torch.save({"model": source_model}, local)
        else:
            if name == ModelName.FASTER_RCNN:
                from torchvision.models.detection import (
                    FasterRCNN, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, fasterrcnn_mobilenet_v3_large_320_fpn,
                )

                # Use the official builder's pretrained layout without any weight transfer or network call.
                with monkeypatch.context() as offline:
                    offline.setattr(FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, "get_state_dict", lambda *_a, **_k: {})
                    offline.setattr(FasterRCNN, "load_state_dict", lambda *_a, **_k: None)
                    source_model = fasterrcnn_mobilenet_v3_large_320_fpn(
                        weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1,
                    )
                state = source_model.state_dict()
                key = next(key for key in state if key.startswith("backbone.body.") and state[key].ndim == 4)
                state[key].fill_(0.125)
                torch.save(state, local)
            else:
                from safetensors.torch import save_file

                state = adapter.build_model(cfg).state_dict()
                key = next(key for key in state if key.startswith("model.backbone.") and state[key].ndim == 4)
                state[key].fill_(0.125)
                for head in list(state):
                    if head.startswith(("class_embed.", "bbox_embed.")):
                        del state[head]
                    elif head.startswith(("model.decoder.class_embed.", "model.enc_score_head.", "model.denoising_class_embed.")):
                        rows = 81 if "denoising" in head else 80
                        state[head] = torch.zeros((rows, *state[head].shape[1:]))
                save_file(state, local)
    values["checkpoint"]["sha256"] = hashlib.sha256(local.read_bytes()).hexdigest()
    profile = folder / f"{name}.yaml"
    profile.write_text(yaml.safe_dump(values), encoding="utf-8")

    def interrupt(stage: str, done: int, _total: int) -> None:
        if stage.endswith("epoch checkpoint") and done == 1:
            raise KeyboardInterrupt

    plan = plan_training(tmp_path, (name,), detection, Path("runs"), folder, None)
    with pytest.raises(KeyboardInterrupt):
        list(execute_training(plan, interrupt))
    previous = plan.destination / name
    stopped = json.loads((previous / "run.json").read_text())
    assert stopped["checkpoint_epoch"] == 1 and stopped["status"] == "interrupted"
    assert stopped["initialization_mode"] == "pretrained"
    assert stopped["pretrained_weights"]["sha256"] == values["checkpoint"]["sha256"]
    transfer = stopped["weight_transfer"]
    assert not transfer["unexpected_keys"] and not transfer["incompatible_keys"]
    assert transfer["reset_keys"]
    if name == ModelName.RT_DETR:
        assert transfer["missing_keys"] and set(transfer["missing_keys"]) == transfer["restored_aliases"].keys()
    changed = {**values, "freeze_backbone": False}
    profile.write_text(yaml.safe_dump(changed), encoding="utf-8")
    with pytest.raises(TrainingError, match="original dataset, initialization"):
        list(execute_training(plan_training(tmp_path, (name,), detection, Path("runs"), folder, previous), lambda *_: None))
    profile.write_text(yaml.safe_dump(values), encoding="utf-8")
    resumed = plan_training(tmp_path, (name,), detection, Path("runs"), folder, previous)
    [(_, result)] = execute_training(resumed, lambda *_: None)
    final = json.loads((resumed.destination / name / "run.json").read_text())
    assert final["status"] == "completed" and final["initialization"] == "resume"
    assert final["weight_transfer"] == transfer
    if name == ModelName.YOLO:
        raw = torch.load(result.last, weights_only=False)
        model = raw["ema"]
        assert model.names == dict(enumerate(NAMES))
        assert raw["optimizer"]["param_groups"][0]["momentum"] == 0.8
        saved_state = model.state_dict()
    else:
        raw = torch.load(result.last, weights_only=True)
        assert raw["optimizer"]["param_groups"][0]["momentum"] == 0.8
        saved_state = raw["model"]
        if name == ModelName.FASTER_RCNN:
            assert saved_state["roi_heads.box_predictor.cls_score.weight"].shape[0] == len(NAMES) + 1
        else:
            assert saved_state["class_embed.0.weight"].shape[0] == len(NAMES)
    assert torch.all(saved_state[key] == 0.125)
    evaluated = CliRunner().invoke(app, [
        "evaluate", "--run", str(resumed.destination / name), "--dataset", str(detection),
        "--device", "cpu", "--image-size", "64", "--warmup", "0",
    ])
    assert evaluated.exit_code == 0, evaluated.output
