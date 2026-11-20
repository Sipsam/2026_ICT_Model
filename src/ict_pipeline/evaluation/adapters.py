"""Inference with the same local detector architectures used for training."""

from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import os
import platform
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch

from ..environment.probes import probe_hardware, probe_host
from ..models.contracts import ModelName, TrainConfig, TrainingError
from ..models.dataset import Sample
from ..models.registry import REGISTRY, resolve_device
from .contracts import (
    CLASSES, EvaluationConfig, EvaluationError, Prediction, Predictor, Preprocessing,
    normalize_prediction, relative_cxcywh_to_xyxy,
)


class TorchPredictor:
    def __init__(
        self, name: ModelName, checkpoint: Path, training: TrainConfig, config: EvaluationConfig,
        scratch: Path, *, require_finite: bool = False,
    ) -> None:
        import torch

        backend = REGISTRY[name]
        self.libraries = backend.versions()
        self.device = resolve_device(torch, training.model_copy(update={
            "device": config.device, "precision": config.precision,
        }), backend)
        self.name = name
        self.config = config
        self.require_finite = require_finite
        self.torch = torch
        resources, graphics = probe_hardware(probe_host(), scratch)
        accelerator = None
        if self.device.startswith("cuda"):
            accelerator = torch.cuda.get_device_name(self.device)
        elif self.device == "mps":
            accelerator = next((gpu.model for gpu in graphics.devices if gpu.vendor == "Apple"), None)
        self.hardware = {
            "system": platform.system(), "system_release": platform.release(), "architecture": platform.machine(),
            "cpu": resources.cpu_model, "ram_bytes": resources.ram_bytes,
            "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "device": self.device, "accelerator": accelerator,
            "cuda": torch.version.cuda, "rocm": torch.version.hip, "nvidia_driver": graphics.nvidia_driver,
            "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32 if torch.version.cuda else None,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32 if torch.version.cuda else None,
            "synchronization": "torch.cuda.synchronize" if self.device.startswith("cuda") else
            "torch.mps.synchronize" if self.device == "mps" else "synchronous CPU",
        }
        self._faster_shape: list[int] = []
        if name == ModelName.YOLO:
            self.model = self._load_yolo(checkpoint, scratch)
            self.postprocessing = {"resize": "letterbox", "nms": "class-aware", "nms_iou": config.nms_iou}
        else:
            adapter = backend.load()
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state.get("format") != backend.checkpoint_format or state.get("architecture") != adapter.architecture:
                raise EvaluationError("Checkpoint format or architecture does not match the training adapter.")
            model_config = training
            if name == ModelName.FASTER_RCNN:
                model_config = training.model_copy(update={"image_size": config.image_size})
            self.model = adapter.build_model(model_config)
            self.model.load_state_dict(state["model"])
            self.model.to(self.device).eval()
            if name == ModelName.FASTER_RCNN:
                self.model.roi_heads.score_thresh = 0.0
                self.model.roi_heads.nms_thresh = config.nms_iou
                self.model.roi_heads.detections_per_img = 1000
                shape = self._faster_shape

                def capture_size(module: Any, inputs: Any, output: Any) -> None:
                    height, width = output[0].tensors.shape[-2:]
                    shape[:] = [width, height]

                self.model.transform.register_forward_hook(capture_size)
                self.postprocessing = {
                    "resize": "aspect_ratio", "nms": "class-aware", "nms_iou": config.nms_iou,
                    "rpn_nms_iou": self.model.rpn.nms_thresh, "output_limit": 1000,
                }
            else:
                queries = self.model.config.num_queries
                if queries > sum((config.image_size // stride) ** 2 for stride in (8, 16, 32)):
                    raise EvaluationError("RT-DETR image size is too small for the checkpoint's query count.")
                self.postprocessing = {"resize": "stretch", "nms": "none", "top_k": queries}

    def _load_yolo(self, checkpoint: Path, scratch: Path) -> Any:
        # Import the pinned package for pickle class resolution. Read the exact local
        # checkpoint directly to avoid asset resolution and dependency installation.
        with TemporaryDirectory(dir=scratch) as directory, patch.dict(os.environ, {
            "YOLO_AUTOINSTALL": "false", "YOLO_OFFLINE": "true", "YOLO_CONFIG_DIR": directory,
        }):
            from ultralytics.nn.tasks import DetectionModel

            state = self.torch.load(checkpoint, map_location="cpu", weights_only=False)
            model = state.get("ema") or state.get("model")
            if not isinstance(model, DetectionModel) or model.names != dict(enumerate(CLASSES.values())):
                raise EvaluationError("YOLO checkpoint is missing the canonical detection head or class mapping.")
        model.to(self.device).eval()
        return model.half() if self.config.precision == "fp16" else model.float()

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            self.torch.mps.synchronize()

    def predict(self, sample: Sample) -> Prediction:
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        original = (sample.width, sample.height)
        size = self.config.image_size
        with Image.open(sample.image) as source:
            if source.size != original:
                raise EvaluationError("Encoded image dimensions differ from the COCO labels.")
            image = source.convert("RGB")

        scale_x = scale_y = 1.0
        left = top = 0
        if self.name == ModelName.YOLO:
            ratio = min(size / sample.width, size / sample.height)
            width, height = (max(1, round(value * ratio)) for value in original)
            left, top = (size - width) // 2, (size - height) // 2
            canvas = Image.new("RGB", (size, size), (114, 114, 114))
            canvas.paste(image.resize((width, height), Image.Resampling.BILINEAR), (left, top))
            image = canvas
            scale_x, scale_y = width / sample.width, height / sample.height
        elif self.name == ModelName.RT_DETR:
            image = image.resize((size, size), Image.Resampling.BILINEAR)

        torch = self.torch
        half = self.config.precision == "fp16"
        yolo = self.name == ModelName.YOLO
        pixels = pil_to_tensor(image).to(self.device, dtype=torch.float16 if yolo and half else torch.float32) / 255
        with torch.inference_mode(), torch.autocast("cuda", enabled=half and not yolo):
            if yolo:
                boxes, labels, scores = self._yolo_detections(pixels)
            elif self.name == ModelName.FASTER_RCNN:
                result = self.model([pixels])[0]
                boxes, labels, scores = result["boxes"], result["labels"], result["scores"]
            else:
                result = self.model(pixel_values=pixels.unsqueeze(0))
                scores, indices = result.logits[0].float().sigmoid().flatten().topk(result.logits.shape[1])
                labels = indices % len(CLASSES)
                boxes = result.pred_boxes[0][indices // len(CLASSES)]
            if self.require_finite and any(not value.isfinite().all().item() for value in (boxes, labels, scores)):
                raise EvaluationError("Detector returned non-finite predictions before score filtering.")
            keep = scores >= self.config.score_floor
            rows = boxes[keep].float().cpu().tolist()
            class_ids = labels[keep].cpu().tolist()
            confidence = scores[keep].float().cpu().tolist()

        if yolo:
            converted = [
                ((x1 - left) / scale_x, (y1 - top) / scale_y, (x2 - left) / scale_x, (y2 - top) / scale_y)
                for x1, y1, x2, y2 in rows
            ]
            preprocessing = Preprocessing(original, (size, size), "letterbox")
        elif self.name == ModelName.FASTER_RCNN:
            converted = [tuple(row) for row in rows]
            width, height = self._faster_shape
            preprocessing = Preprocessing(original, (width, height), "aspect_ratio")
        else:
            converted = [relative_cxcywh_to_xyxy(tuple(row), *original) for row in rows]
            preprocessing = Preprocessing(original, (size, size), "stretch")

        return normalize_prediction(self.name, sample, converted, class_ids, confidence, preprocessing)

    def _yolo_detections(self, pixels: Any) -> tuple[Any, Any, Any]:
        from torchvision.ops import batched_nms, box_convert

        raw = self.model(pixels.unsqueeze(0))
        predictions = (raw[0] if isinstance(raw, tuple) else raw)[0].T.float()
        scores, labels = predictions[:, 4:].max(dim=1)
        selected = scores >= self.config.score_floor
        scores, labels = scores[selected], labels[selected]
        boxes = box_convert(predictions[selected, :4], "cxcywh", "xyxy")
        indices = batched_nms(boxes, scores, labels, self.config.nms_iou)
        return boxes[indices], labels[indices], scores[indices]


def load_predictor(
    name: ModelName, checkpoint: Path, training: TrainConfig, config: EvaluationConfig, scratch: Path,
    *, require_finite: bool = False,
) -> TorchPredictor:
    try:
        # Backends can print checkpoint paths and arbitrary configuration during load.
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return TorchPredictor(name, checkpoint, training, config, scratch, require_finite=require_finite)
    except (TrainingError, EvaluationError) as exc:
        raise EvaluationError(str(exc)) from None
    except Exception:
        raise EvaluationError(
            f"Cannot load {name}. Check the checkpoint and compatible ict-pipeline[{REGISTRY[name].extra}] dependencies."
        ) from None


PredictorFactory = Callable[[ModelName, Path, TrainConfig, EvaluationConfig, Path], Predictor]
