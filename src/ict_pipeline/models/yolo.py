"""Offline Ultralytics training on a private copy of generated YOLO inputs."""

import math
import os
from numbers import Real
from pathlib import Path
from shutil import copyfile
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from ..data.preprocessing.policy import NAMES
from .contracts import Checkpoints, TrainingError, TrainingRequest


class YoloAdapter:
    architecture = "yolo11n.yaml"

    def train(self, request: TrainingRequest) -> Checkpoints:
        directory = request.output / "dataset"
        total = sum(len(split.images) for split in request.data.splits.values())
        copied = 0
        for name, split in request.data.splits.items():
            images = directory / "images" / name
            labels = directory / "labels" / name
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            for image, label in zip(split.images, split.annotations, strict=True):
                # Ultralytics can rewrite corrupt JPEGs while scanning. Never give it
                # links or the original detection files.
                copyfile(image, images / image.name)
                copyfile(label, labels / label.name)
                copied += 1
                request.progress("copy YOLO inputs", copied, total)
        config_dir = request.output / "ultralytics-config"
        # Ultralytics falls back to a global or CWD directory if this parent is missing.
        config_dir.mkdir(exist_ok=True)
        with patch.dict(os.environ, {
            "YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "true",
            "YOLO_CONFIG_DIR": str(config_dir),
        }):
            return self._train(request, directory)

    def _train(self, request: TrainingRequest, directory: Path) -> Checkpoints:
        import torch
        from ultralytics.engine import trainer as engine
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.nn.tasks import load_checkpoint
        from ultralytics.utils import ROOT
        from ultralytics.utils import checks

        cfg = request.config
        if engine.RANK != -1:
            raise TrainingError("Use ict train as a single process outside a distributed launcher.")
        checkpoint_dir = request.output / "weights"
        device = torch.device(request.device)
        model_directory = ROOT / "cfg" / "models" / "11"
        if not (model_directory / "yolo11.yaml").is_file():
            raise TrainingError("The installed Ultralytics package is missing its YOLO11 architecture definition.")

        class OfflineTrainer(DetectionTrainer):
            def get_dataset(self) -> dict[str, Any]:
                return {
                    "path": directory, "train": str(directory / "images" / "train"),
                    "val": str(directory / "images" / "val"),
                    "names": dict(enumerate(NAMES)), "nc": len(NAMES), "channels": 3,
                }

            def check_resume(self, overrides: dict[str, Any]) -> None:
                # Keep the new paths and the verified original configuration. Native
                # setup_model and resume_training restore the actual training state.
                self.resume = request.resume is not None
                if request.resume:
                    self.args.model = self.args.resume = str(request.resume.last)

            def resume_training(self, checkpoint: dict[str, Any] | None) -> None:
                if request.resume:
                    if (
                        not checkpoint or checkpoint.get("optimizer") is None
                        or not 0 <= checkpoint.get("epoch", -1) < cfg.epochs - 1
                    ):
                        raise TrainingError("YOLO requires a native last checkpoint with remaining epochs and optimizer state.")
                    best = request.resume.best
                    metrics = checkpoint.get("train_metrics", {})
                    if metrics.get("fitness") == checkpoint["best_fitness"]:
                        best = request.resume.last
                    if not best.is_file():
                        raise TrainingError("The resume run is missing its best YOLO checkpoint.")
                    copyfile(best, self.best)
                super().resume_training(checkpoint)

            def save_model(self) -> None:
                if not self.loss.isfinite().all().item() or not math.isfinite(self.fitness):
                    raise TrainingError("YOLO produced non-finite training results. Try fp32 or lower lr.")
                last, best = self.last, self.best
                self.last, self.best = last.with_suffix(".tmp"), best.with_suffix(".tmp")
                try:
                    super().save_model()
                    self.last.replace(last)
                    if self.best.is_file():
                        self.best.replace(best)
                finally:
                    self.last, self.best = last, best
                epoch = self.epoch + 1
                if cfg.save_period and epoch % cfg.save_period == 0:
                    temporary = self.wdir / "periodic.tmp"
                    copyfile(last, temporary)
                    temporary.replace(self.wdir / f"epoch-{epoch}.pt")
                request.record_metrics({
                    **{key: float(value) for key, value in self.metrics.items() if isinstance(value, Real)},
                    "epoch": epoch, "fitness": float(self.fitness),
                })
                request.progress("epoch checkpoint", epoch, cfg.epochs)

            def final_eval(self) -> None:
                # Validation fitness was measured each epoch. Keep full native
                # checkpoints instead of the optimizer-stripped inference export.
                pass

        # This Ultralytics release probes AMP with a pretrained network. The
        # device and precision have already been checked, so skip that probe.
        with (
            patch.object(engine, "check_amp", return_value=True),
            patch.object(engine, "select_device", return_value=device),
            patch.object(engine.callbacks, "add_integration_callbacks", return_value=None),
            patch.object(engine, "GIT", SimpleNamespace(root=None, branch=None, commit=None, origin=None)),
            patch.object(checks, "AUTOINSTALL", False),
        ):
            trainer = OfflineTrainer(overrides={
                "model": str(model_directory / self.architecture), "data": str(directory), "task": "detect",
                "epochs": cfg.epochs, "batch": cfg.batch_size, "imgsz": cfg.image_size,
                "optimizer": cfg.optimizer, "lr0": cfg.lr, "seed": cfg.seed, "device": str(device),
                "amp": cfg.precision == "fp16", "pretrained": False, "save": True,
                "save_period": -1, "project": str(request.output.parent), "name": request.output.name,
                "exist_ok": True, "workers": 0, "plots": False, "patience": 0, "val": True,
            })
            trainer.world_size = int(device.type == "cuda")
            trainer.ddp = False
            trainer.train()
        checkpoints = Checkpoints(checkpoint_dir / "best.pt", checkpoint_dir / "last.pt")
        if checkpoints.last.is_file():
            _, state = load_checkpoint(checkpoints.last)
            if state["epoch"] + 1 != cfg.epochs:
                raise TrainingError("YOLO stopped before reaching the configured epoch count.")
        return checkpoints
