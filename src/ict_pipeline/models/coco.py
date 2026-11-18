"""TorchVision and Transformers training over the same generated COCO splits."""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator
from pathlib import Path
from shutil import copyfile
from time import perf_counter
from typing import TYPE_CHECKING, Any

from ..data.preprocessing.policy import NAMES
from ..training_storage import commit_checkpoint, guard_training_write
from .contracts import Checkpoints, TrainConfig, TrainingError, TrainingRequest
from .dataset import Sample
from .weights import LAYOUTS, load_weights, transfer_state

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


def timed_batches(loader: Any, request: TrainingRequest) -> Iterator[Batch]:
    started = perf_counter()
    for batch in loader:
        if request.record_overhead:
            request.record_overhead("loader_wait_seconds", perf_counter() - started)
        yield batch
        started = perf_counter()


def _atomic_copy(source: Path, target: Path) -> None:
    temporary = target.with_suffix(".tmp")
    guard_training_write(target.parent, source.stat().st_size)
    try:
        copyfile(source, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


class CocoAdapter:
    architecture: str

    def build_model(self, config: TrainConfig) -> Any:
        raise NotImplementedError

    def batch_loss(self, model: Any, batch: Batch, request: TrainingRequest) -> torch.Tensor:
        raise NotImplementedError

    def validation_mode(self, model: Any) -> None:
        model.eval()

    def backbone(self, model: Any) -> Any:
        raise NotImplementedError

    def initialize(self, model: Any, request: TrainingRequest) -> None:
        raise NotImplementedError

    def train(self, request: TrainingRequest) -> Checkpoints:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader

        cfg = request.config
        torch.manual_seed(cfg.seed)
        model = self.build_model(cfg)
        if cfg.initialization == "pretrained" and request.resume is None:
            self.initialize(model, request)
        frozen_backbone = self.backbone(model) if cfg.freeze_backbone else None
        if frozen_backbone is not None:
            frozen_backbone.requires_grad_(False)
        model.to(request.device)
        make_optimizer: Callable[..., Any] = getattr(torch.optim, cfg.optimizer_name)
        fine_tuning = cfg.fine_tuning
        options = {}
        if fine_tuning and fine_tuning.weight_decay is not None:
            options["weight_decay"] = fine_tuning.weight_decay
        if fine_tuning and fine_tuning.momentum is not None:
            if cfg.optimizer_name == "SGD":
                options["momentum"] = fine_tuning.momentum
            else:
                options["betas"] = (fine_tuning.momentum, 0.999)
        parameters = model.parameters()
        if frozen_backbone is not None:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        optimizer = make_optimizer(parameters, lr=cfg.learning_rate, **options)
        dtype = torch.bfloat16 if cfg.precision == "bf16" else torch.float16
        mixed = cfg.precision != "fp32"
        scaler = torch.amp.GradScaler("cuda", enabled=cfg.precision == "fp16")
        device_type = torch.device(request.device).type
        generator = torch.Generator()
        train, val = (
            DataLoader(
                CocoImages(request.data.splits[split].samples), batch_size=cfg.batch_size,
                shuffle=split == "train", generator=generator, collate_fn=collate, num_workers=cfg.workers,
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
            if frozen_backbone is not None:
                frozen_backbone.eval()
            request.progress(f"epoch {epoch}/{cfg.epochs} train batch", 0, len(train))
            train_loss = 0.0
            accumulation = cfg.gradient_accumulation
            for step, batch in enumerate(timed_batches(train, request), 1):
                group_start = ((step - 1) // accumulation) * accumulation
                group_images = min(accumulation * cfg.batch_size, len(train.dataset) - group_start * cfg.batch_size)
                if step == group_start + 1:
                    optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type, dtype=dtype, enabled=mixed):
                    loss = self.batch_loss(model, batch, request)
                if not loss.isfinite().item():
                    raise TrainingError("Training produced a non-finite loss. Try fp32 or lower lr.")
                scaled_loss = loss if accumulation == 1 else loss * (len(batch[1]) / group_images)
                scaler.scale(scaled_loss).backward()
                if step % accumulation == 0 or step == len(train):
                    scaler.step(optimizer)
                    scaler.update()
                train_loss += loss.detach().item() * len(batch[1])
                request.progress(f"epoch {epoch}/{cfg.epochs} train batch", step, len(train))

            self.validation_mode(model)
            val_loss = 0.0
            # Match checkpoint selection at each epoch without carrying over Train sampler state.
            torch.manual_seed(cfg.seed)
            request.progress(f"epoch {epoch}/{cfg.epochs} validation batch", 0, len(val))
            with torch.no_grad():
                for step, batch in enumerate(timed_batches(val, request), 1):
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
            request.check_checkpoint_space()
            snapshots = []
            if best_epoch == epoch:
                snapshots.append(checkpoints.best)
            if cfg.save_period and epoch % cfg.save_period == 0:
                snapshots.append(request.output / f"epoch-{epoch}.pt")
            temporary = checkpoints.last.with_suffix(".tmp")
            try:
                torch.save(state, temporary)
                commit_checkpoint(temporary, checkpoints.last, tuple(snapshots))
            finally:
                temporary.unlink(missing_ok=True)
            request.record_metrics({
                "epoch": epoch, "train_loss": train_loss / len(train.dataset),
                "val_loss": val_loss, "best_val_loss": best_loss,
            })
            request.progress("epoch checkpoint", epoch, cfg.epochs)
        return checkpoints


class FasterRCNNAdapter(CocoAdapter):
    architecture = LAYOUTS["faster_rcnn"].architecture

    def build_model(self, config: TrainConfig) -> Any:
        from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

        if config.initialization == "pretrained":
            from torchvision.models.detection import FasterRCNN
            from torchvision.models.detection.anchor_utils import AnchorGenerator
            from torchvision.models.detection.backbone_utils import mobilenet_backbone

            backbone = mobilenet_backbone(
                backbone_name="mobilenet_v3_large", weights=None, fpn=True, trainable_layers=6,
            )
            anchors = AnchorGenerator(((32, 64, 128, 256, 512),) * 3, ((0.5, 1.0, 2.0),) * 3)
            return FasterRCNN(
                backbone, num_classes=len(NAMES) + 1, rpn_anchor_generator=anchors,
                min_size=config.image_size, max_size=config.image_size,
                rpn_pre_nms_top_n_test=150, rpn_post_nms_top_n_test=150, rpn_score_thresh=0.05,
            )
        return fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=None, weights_backbone=None, num_classes=len(NAMES) + 1,
            min_size=config.image_size, max_size=config.image_size,
        )

    def backbone(self, model: Any) -> Any:
        return model.backbone.body

    def initialize(self, model: Any, request: TrainingRequest) -> None:
        state = load_weights(request)
        assert request.pretrained is not None
        classes = request.pretrained.source_classes
        shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        reset = {key for key in shapes if key.startswith("roi_heads.box_predictor.")}
        for key in reset:
            rows = classes * 4 if ".bbox_pred." in key else classes
            shapes[key] = (rows, *shapes[key][1:])
        # The published COCO_V1 state predates TorchVision's convolution wrappers.
        # Match its FPN and RPN migrations before ICT checks the source tensor layout.
        renames = {
            f"backbone.fpn.{block}.{index}.{parameter}": f"backbone.fpn.{block}.{index}.0.{parameter}"
            for block in ("inner_blocks", "layer_blocks") for index in range(2) for parameter in ("bias", "weight")
        }
        renames.update({f"rpn.head.conv.{key}": f"rpn.head.conv.0.0.{key}" for key in ("bias", "weight")})
        transfer_state(model, state, shapes, reset, request, renames=renames)

    def batch_loss(self, model: Any, batch: Batch, request: TrainingRequest) -> torch.Tensor:
        import torch
        from torchvision.transforms.functional import pil_to_tensor

        started = perf_counter()
        images, samples = batch
        pixels = [pil_to_tensor(image).to(request.device, dtype=torch.float32) / 255 for image in images]
        targets = []
        for sample in samples:
            boxes = [(x, y, x + w, y + h) for x, y, w, h in (obj.box for obj in sample.objects)]
            targets.append({
                "boxes": torch.tensor(boxes, dtype=torch.float32, device=request.device).reshape(-1, 4),
                "labels": torch.tensor([obj.category for obj in sample.objects], dtype=torch.int64, device=request.device),
            })
        if request.record_overhead:
            request.record_overhead("batch_preparation_host_seconds", perf_counter() - started)
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
    architecture = LAYOUTS["rt_detr"].architecture

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
            hidden_expansion=0.5 if config.initialization == "pretrained" else 1.0,
            num_queries=min(300, sum((size // stride) ** 2 for stride in (8, 16, 32))),
            id2label=names, label2id={name: index for index, name in names.items()},
        ))

    def backbone(self, model: Any) -> Any:
        return model.model.backbone

    def initialize(self, model: Any, request: TrainingRequest) -> None:
        state = load_weights(request)
        assert request.pretrained is not None
        classes = request.pretrained.source_classes
        shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        reset = {
            key for key in shapes if key.startswith((
                "class_embed.", "model.decoder.class_embed.", "model.enc_score_head.", "model.denoising_class_embed.",
            ))
        }
        aliases = {
            key: "model.decoder." + key for key in shapes if key.startswith(("bbox_embed.", "class_embed."))
        }
        for key in reset:
            rows = classes + int(key == "model.denoising_class_embed.weight")
            shapes[key] = (rows, *shapes[key][1:])
        transfer_state(model, state, shapes, reset, request, aliases=aliases)

    def batch_loss(self, model: Any, batch: Batch, request: TrainingRequest) -> torch.Tensor:
        import torch
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        started = perf_counter()
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
        if request.record_overhead:
            request.record_overhead("batch_preparation_host_seconds", perf_counter() - started)
        return model(pixel_values=pixels, labels=labels).loss
