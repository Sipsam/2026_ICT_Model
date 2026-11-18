"""PyTorch backend for the approval-gated product epoch runner."""

import os
import random
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import torch

from ..data.model import Progress
from ..evaluation.product import classification_metrics, pending_experiment
from ..evaluation.product_protocol import ExperimentReport
from ..inspection.contracts import ProductModelIdentity
from ..models.registry import resolve_torch_device
from ..pipeline.store import digest_json
from .checkpoint import FORMAT, load_checkpoint, save_checkpoint
from .dataset import collate
from .model import MultimodalModel, ProductOutput, product_loss, product_targets
from .model_config import Branch, WeightStatus
from .training import EpochReceipt
from .training_plan import ProductTrainingPlan

RESUME_FORMAT = "ict-product-training-resume-v1"


def configure_optimizer(
    model: MultimodalModel, plan: ProductTrainingPlan,
) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.StepLR]:
    config = plan.config
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.optimizer.learning_rate, weight_decay=config.optimizer.weight_decay,
    )
    settings = config.scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=settings.step_size if settings.kind == "step" else 1,
        gamma=settings.gamma if settings.kind == "step" else 1.0,
    )
    return optimizer, scheduler


def _branch_logits(output: ProductOutput, branch: Branch, count: int) -> torch.Tensor:
    logits, usable = output.for_branch(branch)
    if (usable.dtype != torch.bool or usable.shape != (count,) or usable.device != logits.device
            or logits.shape != (count, len(output.output_names)) or not usable.all().item()):
        raise ValueError("The requested product branch requires valid logits and a usable modality mask for every row.")
    return logits


def _capture_rng(device: str) -> dict:
    return {
        "python": random.getstate(), "cpu": torch.get_rng_state(),
        "device": (torch.cuda.get_rng_state(device) if device.startswith("cuda") else
                   torch.mps.get_rng_state() if device == "mps" else None),
    }


def _restore_rng(device: str, saved: dict) -> None:
    if set(saved) != {"python", "cpu", "device"} or (saved["device"] is None) != (device == "cpu"):
        raise ValueError("Checkpoint RNG state does not match the current training device.")
    random.setstate(saved["python"])
    torch.set_rng_state(saved["cpu"])
    if device.startswith("cuda"):
        torch.cuda.set_rng_state(saved["device"], device)
    elif device == "mps":
        torch.mps.set_rng_state(saved["device"])


class TorchEpochEngine:
    def __init__(self, plan: ProductTrainingPlan, device: str) -> None:
        self.plan = plan
        self.device = device
        self.model = MultimodalModel(plan.config.architecture, plan.state).to(device)
        self.optimizer, self.scheduler = configure_optimizer(self.model, plan)

    def train_epoch(self, epoch: int, progress: Progress) -> float:
        config = self.plan.config
        dataset = self.plan.train
        self.model.train()
        order = list(range(len(dataset)))
        random.shuffle(order)
        processed = 0
        total_loss = 0.0
        progress(0, len(order))
        group_size = config.batch_size * config.gradient_accumulation
        for start in range(0, len(order), group_size):
            group = order[start:start + group_size]
            self.optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(group), config.batch_size):
                indices = group[offset:offset + config.batch_size]
                batch = collate([dataset[index] for index in indices])
                targets = product_targets(batch, config.architecture)
                if targets.split != "train":
                    raise ValueError("Parameter updates require exclusively Train targets.")
                output = self.model(batch)
                _branch_logits(output, config.branch, len(indices))
                loss = product_loss(output, targets, branch=config.branch)
                (loss * (len(indices) / len(group))).backward()
                total_loss += loss.detach().item() * len(indices)
                processed += len(indices)
                progress(processed, len(order))
            if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all().item()
                   for parameter in self.model.parameters()):
                raise ValueError("Product training produced non-finite gradients.")
            self.optimizer.step()
        self.scheduler.step()
        branches = {"image", "sensor", "paired"} if (
            config.branch == "paired" and config.architecture.fusion.kind == "late"
        ) else {config.branch}
        self.model.weight_status = WeightStatus(**{branch: "trained" for branch in branches})
        return total_loss / len(order)

    def save(self, directory: Path, epoch: int) -> None:
        save_checkpoint(directory / "model.pt", self.model)
        with (directory / "resume.pt").open("xb") as stream:
            torch.save({
                "format": RESUME_FORMAT, "compatibility": self.plan.compatibility, "epoch": epoch,
                "architecture": self.model.architecture_fingerprint,
                "preprocessing": self.plan.state.fingerprint,
                "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
                "rng": _capture_rng(self.device),
            }, stream)
            stream.flush()
            os.fsync(stream.fileno())

    def restore(self, directory: Path, receipt: EpochReceipt) -> None:
        model = load_checkpoint(directory / "model.pt", self.plan.state, expected_config=self.plan.config.architecture)
        payload = torch.load(directory / "resume.pt", weights_only=True, map_location="cpu")
        if (not isinstance(payload, dict)
                or payload.keys() != {"format", "compatibility", "epoch", "architecture", "preprocessing",
                                      "optimizer", "scheduler", "rng"}
                or payload["format"] != RESUME_FORMAT or payload["compatibility"] != self.plan.compatibility
                or type(payload["epoch"]) is not int or payload["epoch"] != receipt.epoch
                or payload["architecture"] != model.architecture_fingerprint
                or payload["preprocessing"] != self.plan.state.fingerprint
                or model.branch_status(self.plan.config.branch) != "trained"):
            raise ValueError("Resume tensor metadata, architecture, or preprocessing is incompatible.")
        optimizer, scheduler = payload["optimizer"], payload["scheduler"]
        expected = self.optimizer.state_dict()["param_groups"][0]
        if (set(optimizer) != {"param_groups", "state"} or len(optimizer["param_groups"]) != 1
                or optimizer["param_groups"][0].keys() != expected.keys()
                or any(optimizer["param_groups"][0][key] != value for key, value in expected.items() if key != "lr")
                or scheduler.keys() != self.scheduler.state_dict().keys()
                or scheduler["last_epoch"] != receipt.epoch or scheduler["_step_count"] != receipt.epoch + 1
                or scheduler["step_size"] != self.scheduler.step_size or scheduler["gamma"] != self.scheduler.gamma
                or scheduler["base_lrs"] != self.scheduler.base_lrs
                or scheduler["_last_lr"] != [optimizer["param_groups"][0]["lr"]]
                or not 0 <= scheduler["_last_lr"][0] <= self.plan.config.optimizer.learning_rate):
            raise ValueError("Resume optimizer or scheduler configuration is incompatible.")
        parameters = tuple(self.model.parameters())
        for index, state in optimizer["state"].items():
            if (type(index) is not int or not 0 <= index < len(parameters)
                    or state.keys() != {"step", "exp_avg", "exp_avg_sq"}
                    or any(type(value) is not torch.Tensor or not torch.isfinite(value).all().item()
                           for value in state.values())
                    or state["step"].ndim != 0 or state["step"].item() <= 0
                    or any(state[key].shape != parameters[index].shape or state[key].dtype != parameters[index].dtype
                           for key in ("exp_avg", "exp_avg_sq"))):
                raise ValueError("Resume optimizer tensor state is incompatible.")
        if not optimizer["state"]:
            raise ValueError("Resume optimizer state is empty.")
        self.model.load_state_dict(model.state_dict(), strict=True)
        self.model.weight_status = model.weight_status
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)
        _restore_rng(self.device, payload["rng"])

    @torch.inference_mode()
    def evaluate(self, checkpoint_hash: str, progress: Progress) -> ExperimentReport:
        dataset, frozen = self.plan.validation, self.plan.frozen
        if dataset is None or frozen is None:
            raise ValueError("Configure a frozen Validation cohort before scoring.")
        self.model.eval()
        config = self.plan.config
        if self.model.branch_status(config.branch) != "trained":
            raise ValueError("Validation reporting requires trained weights for the requested branch.")
        scores = {}
        progress(0, len(dataset))
        for start in range(0, len(dataset), config.batch_size):
            end = min(start + config.batch_size, len(dataset))
            batch = collate([dataset[index] for index in range(start, end)])
            output = self.model(batch)
            if output.split != "validation":
                raise ValueError("The frozen evaluation only accepts Validation predictions.")
            logits = _branch_logits(output, config.branch, end - start)
            scores.update(zip(output.sample_ids, map(tuple, logits.sigmoid().tolist()), strict=True))
            progress(end, len(dataset))
        architecture = config.architecture
        protocol = self.plan.validation_protocol()
        identity = ProductModelIdentity(
            checkpoint_format=FORMAT, checkpoint_sha256=checkpoint_hash,
            architecture_version=architecture.schema_version,
            architecture_fingerprint=self.model.architecture_fingerprint,
            image_encoder=architecture.image.kind, image_encoder_version=architecture.image.schema_version,
            image_preprocessing_fingerprint=protocol.image_preprocessing_fingerprint,
            sensor_encoder=architecture.sensor.kind, sensor_encoder_version=architecture.sensor.schema_version,
            preprocessing_version=self.plan.state.schema_version, preprocessing_fingerprint=self.plan.state.fingerprint,
            window_policy=self.plan.state.config.window_policy, fusion=architecture.fusion.kind,
            fusion_version=architecture.fusion.schema_version,
            fusion_fingerprint=digest_json(architecture.fusion.model_dump(mode="json")), device=config.device,
            seed=architecture.seed, libraries={name: version(name) for name in ("torch", "Pillow")},
        )
        pending = pending_experiment(self.plan.output.name, protocol, branch=config.branch)
        return ExperimentReport(
            experiment_id=pending.experiment_id, protocol=protocol, branch=config.branch, status="measured",
            product_model=identity, fusion=architecture.fusion, prediction_fingerprint=digest_json(scores),
            classification_metrics=classification_metrics(frozen, protocol, scores),
            code_fingerprint=pending.code_fingerprint, runtime=pending.runtime,
        )


@contextmanager
def training_engine(plan: ProductTrainingPlan) -> Iterator[TorchEpochEngine]:
    config = plan.config
    if config.device == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before starting deterministic CUDA product training.")
    device = resolve_torch_device(torch, config.device, ("cpu", "cuda", "rocm", "mps"))
    rng = _capture_rng(device)
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config.cpu_threads)
        torch.use_deterministic_algorithms(True)
        seed = config.architecture.seed
        random.seed(seed)
        torch.random.default_generator.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed(seed)
        elif device == "mps":
            torch.mps.manual_seed(seed)
        yield TorchEpochEngine(plan, device)
    finally:
        _restore_rng(device, rng)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
