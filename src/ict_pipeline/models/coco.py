"""TorchVision and Transformers training over the same generated COCO splits."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from shutil import copyfile
from typing import TYPE_CHECKING, Any

from ..data.preprocessing.policy import NAMES
from .contracts import Checkpoints, TrainConfig, TrainingError, TrainingRequest
from .dataset import Sample

if TYPE_CHECKING:
    import torch

Batch = tuple[list[Any], list[Sample]]


class CocoImages:
    def __init__(self, samples: tuple[Sample, ...]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Sample]:
        from PIL import Image

        sample = self.samples[index]
        with Image.open(sample.image) as image:
            if image.size != (sample.width, sample.height):
                raise TrainingError("COCO image dimensions differ from the encoded image. Rerun detection validation.")
            return image.convert("RGB"), sample


def collate(items: list[tuple[Any, Sample]]) -> Batch:
    images, samples = zip(*items)
    return list(images), list(samples)


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_suffix(".tmp")
    copyfile(source, temporary)
    temporary.replace(target)


class CocoAdapter:
    architecture: str

    def build_model(self, config: TrainConfig) -> Any:
        raise NotImplementedError

    def batch_loss(self, model: Any, batch: Batch, request: TrainingRequest) -> torch.Tensor:
        raise NotImplementedError

    def validation_mode(self, model: Any) -> None:
        model.eval()

    def train(self, request: TrainingRequest) -> Checkpoints:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        cfg = request.config
        torch.manual_seed(cfg.seed)
        model = self.build_model(cfg).to(request.device)
        make_optimizer: Callable[..., Any] = getattr(torch.optim, cfg.optimizer)
        optimizer = make_optimizer(model.parameters(), lr=cfg.lr)
        dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
        mixed = cfg.precision != "fp32"
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
        device_type = torch.device(request.device).type
        generator = torch.Generator()
        train, val = (
            DataLoader(
                CocoImages(request.data.splits[split].samples), batch_size=cfg.batch_size,
                shuffle=split == "train", generator=generator, collate_fn=collate, num_workers=0,
            )
            for split in ("train", "val")
        )
        checkpoints = Checkpoints(request.output / "best.pt", request.output / "last.pt")
        start, best_epoch, best_loss = 0, 0, float("inf")
        if request.resume:
            state = torch.load(request.resume.last, map_location="cpu", weights_only=True)
            if state.get("format") != "ict-torch-state-v1" or state.get("architecture") != self.architecture:
                raise TrainingError("The last checkpoint does not use this adapter's training format.")
            start, best_epoch, best_loss = state["epoch"], state["best_epoch"], state["best_loss"]
            if not (1 <= best_epoch <= start < cfg.epochs):
                raise TrainingError("The checkpoint has no remaining epochs under the original configuration.")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler"])
            best_source = request.resume.last if best_epoch == start else request.resume.best
            best_state = torch.load(best_source, map_location="cpu", weights_only=True)
            if best_state["epoch"] != best_epoch:
                raise TrainingError("The resume run is missing the checkpoint for its recorded best epoch.")
            _atomic_copy(best_source, checkpoints.best)

        for epoch in range(start + 1, cfg.epochs + 1):
            # Seed each epoch independently so resume reconstructs data order and sampling.
            seed = (cfg.seed + epoch) % 2**32
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            generator.manual_seed(seed)
            model.train()
            request.progress(f"epoch {epoch}/{cfg.epochs} train batch", 0, len(train))
            for step, batch in enumerate(train, 1):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type, dtype=dtype, enabled=mixed):
                    loss = self.batch_loss(model, batch, request)
                if not loss.isfinite().item():
                    raise TrainingError("Training produced a non-finite loss. Try fp32 or lower lr.")
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                request.progress(f"epoch {epoch}/{cfg.epochs} train batch", step, len(train))

            self.validation_mode(model)
            val_loss = 0.0
            # Match checkpoint selection at each epoch without carrying over Train sampler state.
            torch.manual_seed(cfg.seed)
            request.progress(f"epoch {epoch}/{cfg.epochs} validation batch", 0, len(val))
            with torch.no_grad():
                for step, batch in enumerate(val, 1):
                    with torch.autocast(device_type, dtype=dtype, enabled=mixed):
                        loss = self.batch_loss(model, batch, request)
                    if not loss.isfinite().item():
                        raise TrainingError("Validation produced a non-finite loss. Try fp32 or lower lr.")
                    val_loss += loss.item() * len(batch[1])
                    request.progress(f"epoch {epoch}/{cfg.epochs} validation batch", step, len(val))
            val_loss /= len(val.dataset)
            if val_loss < best_loss:
                best_epoch, best_loss = epoch, val_loss
            state = {
                "format": "ict-torch-state-v1", "architecture": self.architecture,
                "epoch": epoch, "best_epoch": best_epoch, "best_loss": best_loss, "val_loss": val_loss,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
            }
            temporary = checkpoints.last.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoints.last)
            if best_epoch == epoch:
                _atomic_copy(checkpoints.last, checkpoints.best)
            if cfg.save_period and epoch % cfg.save_period == 0:
                _atomic_copy(checkpoints.last, request.output / f"epoch-{epoch}.pt")
            request.progress("epoch checkpoint", epoch, cfg.epochs)
        return checkpoints


class FasterRCNNAdapter(CocoAdapter):
    architecture = "fasterrcnn_mobilenet_v3_large_320_fpn"

    def build_model(self, config: TrainConfig) -> Any:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, num_classes=len(NAMES) + 1,
            min_size=config.image_size, max_size=config.image_size,
        )

    def batch_loss(self, model: Any, batch: Batch, request: TrainingRequest) -> torch.Tensor:
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        images, samples = batch
        pixels = [pil_to_tensor(image).to(request.device, dtype=torch.float32) / 255 for image in images]
        targets = []
        for sample in samples:
            boxes = [(x, y, x + w, y + h) for x, y, w, h in (obj.box for obj in sample.objects)]
            targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32, device=request.device).reshape(-1, 4),
                "labels": torch.tensor([obj.category for obj in sample.objects], dtype=torch.int64, device=request.device),
            })
        return sum(model(pixels, targets).values())

    def validation_mode(self, model: Any) -> None:
        # Request losses from the detection heads while keeping normalization in eval mode.
        model.eval()
        model.training = model.rpn.training = model.roi_heads.training = True

    def train(self, request: TrainingRequest) -> Checkpoints:
        import torch
        from torchvision.ops import nms

        with torch.no_grad():
            nms(torch.empty((0, 4), device=request.device), torch.empty(0, device=request.device), 0.5)
        return super().train(request)


class RTDETRAdapter(CocoAdapter):
    architecture = "rt-detr-resnet18"

    def build_model(self, config: TrainConfig) -> Any:
        from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrResNetConfig

        size = config.image_size
        names = dict(enumerate(NAMES))
        backbone = RTDetrResNetConfig(
            layer_type="basic", depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 256, 512],
            out_indices=[2, 3, 4],
        )
        return RTDetrForObjectDetection(RTDetrConfig(
            backbone_config=backbone, use_pretrained_backbone=False, disable_custom_kernels=True,
            encoder_in_channels=[128, 256, 512], decoder_layers=3,
            num_queries=min(300, sum((size // stride) ** 2 for stride in (8, 16, 32))),
            id2label=names, label2id={name: index for index, name in names.items()},
        ))

    def batch_loss(self, model: Any, batch: Batch, request: TrainingRequest) -> torch.Tensor:
        import torch
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        images, samples = batch
        size = request.config.image_size
        pixels = torch.stack([
            pil_to_tensor(image.resize((size, size), Image.Resampling.BILINEAR)) for image in images
        ]).to(request.device, dtype=torch.float32) / 255
        labels = []
        for sample in samples:
            boxes = [
                ((x + w / 2) / sample.width, (y + h / 2) / sample.height, w / sample.width, h / sample.height)
                for x, y, w, h in (obj.box for obj in sample.objects)
            ]
            labels.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32, device=request.device).reshape(-1, 4),
                "class_labels": torch.tensor(
                    [obj.category - 1 for obj in sample.objects], dtype=torch.int64, device=request.device,
                ),
            })
        return model(pixel_values=pixels, labels=labels).loss
