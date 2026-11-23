"""Publish a verified best checkpoint after a group Detector validation failure."""

import hashlib
import json
from io import BytesIO
from pathlib import Path
from pickle import UnpicklingError
from typing import Any, Literal

from ..config.settings import Credentials
from ..evaluation.contracts import EvaluationError
from ..models.contracts import ModelName, Progress, TrainConfig
from ..models.dataset import inspect_data, read_coco_split
from ..models.registry import REGISTRY
from ..models.weights import LAYOUTS
from ..multimodal.audit import Cohort
from ..multimodal.model_config import FPNDetectorConfig
from ..pipeline.store import _reject_link, digest_json, fingerprint_path
from .group_retraining import detector_scope, read_verification_receipt
from .preprocessing.common import require_separate_output, write_json
from .preprocessing.console import fresh_output
from .preprocessing.policy import label_policy
from .preprocessing.revision import read_publication

RECEIPT = "finalization.json"
STATUS = "finalized_early"
REASON = "Select the minimum-loss completed Validation checkpoint after non-finite Validation loss."
FAILURE = "Validation produced a non-finite loss. Try fp32 or lower lr."
ARCHITECTURE = LAYOUTS["faster_rcnn"].architecture
BACKEND = REGISTRY[ModelName.FASTER_RCNN]
Device = Literal["cpu", "cuda", "rocm"]


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _file_fingerprint(sha256: str) -> str:
    # Match fingerprint_path's encoding of a regular file at the root.
    return _sha256(json.dumps(["file", ".", sha256]).encode())


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _loss(value: object) -> bool:
    return type(value) in (float, int) and 0 <= value < float("inf")


def _revision(result: dict, receipt: str) -> dict:
    return {"group_split_revision": result["revision_id"], "group_verification": receipt,
            "detector_scope": result["detector_scope"]}


def _checkpoint(content: bytes, name: str, epoch: int) -> dict[str, Any]:
    import torch

    try:
        state = torch.load(BytesIO(content), map_location="cpu", weights_only=True)
    except (UnpicklingError, RuntimeError, EOFError, ValueError) as exc:
        raise ValueError(f"Cannot read the tensor-only {name} checkpoint.") from exc
    if (not isinstance(state, dict) or state.get("format") != BACKEND.checkpoint_format
            or state.get("architecture") != ARCHITECTURE
            or type(state.get("epoch")) is not int or state["epoch"] != epoch
            or type(state.get("best_epoch")) is not int
            or not _loss(state.get("best_loss")) or not _loss(state.get("val_loss"))):
        raise ValueError(f"{name} checkpoint format, architecture, epoch or loss is invalid.")
    return state


def _evaluate(dataset: Path, best: Path, training: TrainConfig, device: Device, progress: Progress) -> dict:
    from ..evaluation.adapters import load_predictor
    from ..evaluation.contracts import EvaluationConfig, validate_predictions
    from ..evaluation.metrics import detection_metrics

    settings = EvaluationConfig(image_size=training.image_size, device=device)
    samples = read_coco_split(dataset / "coco", "val").samples
    predictor = load_predictor(ModelName.FASTER_RCNN, best, training, settings, best.parent, require_finite=True)
    predictions = {}
    stage = "Evaluate best checkpoint on Validation"
    progress(stage, 0, len(samples))
    for done, sample in enumerate(samples, 1):
        predictions[sample.image.stem] = predictor.predict(sample)
        progress(stage, done, len(samples))
    predictor.synchronize()
    validate_predictions(samples, predictions)
    metrics = detection_metrics(samples, predictions, settings, progress)
    return {"status": "passed", "split": "val", "samples": len(samples),
            "finite_predictions": True, "configuration": settings.model_dump(mode="json"),
            "device": predictor.device, "libraries": predictor.libraries, "postprocessing": predictor.postprocessing,
            "map50": metrics["map50"], "map50_95": metrics["map50_95"]}


def finalize_detector(
    source_run: Path, bundle: Path, verification: Path, cohort_path: Path, out: Path, *,
    best_epoch: int, completed_epoch: int, validation_device: Device | None,
    credentials: Credentials, progress: Progress,
) -> Path:
    """Validate read-only inputs and publish an independent FPN initialization artifact."""
    try:
        from ..multimodal.fpn import detector_from_checkpoint
    except (ImportError, RuntimeError, OSError):
        raise ValueError("Install compatible PyTorch and TorchVision from the Faster R-CNN extra before finalizing.") from None

    _reject_link(out)
    for path in (source_run, bundle, verification, cohort_path):
        require_separate_output(path, out)
        require_separate_output(out, path)
    for path in (source_run, verification, cohort_path, *(source_run / name for name in ("run.json", "best.pt", "last.pt"))):
        _reject_link(path)
    with fresh_output(out, source_run) as working:
        progress("Verify group publication and Detector scope", 0, 1)
        result, receipt = read_verification_receipt(verification)
        revision, seal, _ = read_publication(bundle, result["revision_id"])
        cohort = Cohort.load(cohort_path)
        dataset = bundle / "detection"
        if (seal != result["publication_fingerprint"] or revision["configuration"]["strategy"] != "group-aware"
                or revision["selected_manifest_fingerprint"] != result["manifest_fingerprint"]
                or cohort.revision_id != result["revision_id"] or cohort.fingerprint != result["cohort_fingerprint"]
                or cohort.split_fingerprint != result["split_fingerprint"]
                or any(record.split == "test" and (record.targets is not None or record.sequence is not None)
                       for record in cohort.records)):
            raise ValueError("Group verification receipt differs from the published revision or sanitized Cohort.")
        raw = (source_run / "run.json").read_bytes()
        run = json.loads(raw)
        if (run.get("schema_version") != 1 or run.get("status") != "failed"
                or run.get("model") != ModelName.FASTER_RCNN or run.get("architecture") != ARCHITECTURE
                or run.get("dataset_format") != BACKEND.data_format
                or run.get("checkpoint_format") != BACKEND.checkpoint_format
                or run.get("best_criterion") != BACKEND.best_criterion or run.get("error") != FAILURE
                or not _is_sha256(run.get("run_fingerprint"))
                or digest_json(run.get("label_policy")) != digest_json(label_policy())):
            raise ValueError("Expected a failed ICT Faster R-CNN run with non-finite Validation loss.")
        if run.get("dataset_revision") != _revision(result, receipt) or detector_scope(dataset) != result["detector_scope"]:
            raise ValueError("Failed Detector inputs differ from the group revision, verification receipt or scoped COCO files.")
        progress("Verify group publication and Detector scope", 1, 1)
        data = inspect_data(dataset, BACKEND.data_format, credentials, progress)
        if (run.get("split_fingerprints") != data.split_fingerprints
                or run.get("annotation_fingerprint") != data.annotation_fingerprint
                or run.get("split_annotation_fingerprints") != data.split_annotation_fingerprints):
            raise ValueError("Detector dataset fingerprints differ from the failed run.")

        training = TrainConfig.model_validate_json(json.dumps(run["configuration"]))
        if (not 1 <= best_epoch <= completed_epoch < training.epochs
                or run.get("checkpoint_epoch") != completed_epoch
                or run.get("initialization_mode") != training.initialization):
            raise ValueError("Detector completed epoch or initialization differs from the failed run.")
        stage = "Verify best and last checkpoints"
        progress(stage, 0, 2)
        records = {}
        for role, epoch in (("last", completed_epoch), ("best", best_epoch)):
            name = f"{role}.pt"
            content = (source_run / name).read_bytes()
            sha256 = _sha256(content)
            record = run["checkpoints"][role]
            if record != {"path": name, "fingerprint": _file_fingerprint(sha256)}:
                raise ValueError(f"{name} fingerprint differs from the failed run record.")
            state = _checkpoint(content, name, epoch)
            records[role] = {**record, "sha256": sha256,
                             **{key: state[key] for key in ("epoch", "best_epoch", "best_loss", "val_loss")}}
            if role == "best":
                try:
                    detector_from_checkpoint(
                        state, FPNDetectorConfig(image_size=training.image_size, layout=training.initialization),
                    )
                except RuntimeError:
                    raise ValueError("Could not construct the configured Detector and load its best weights.") from None
                (working / name).write_bytes(content)
            del state, content
            progress(stage, len(records), 2)
        best, last = records["best"], records["last"]
        metrics = run["metrics"]
        if (best["best_epoch"] != best_epoch or last["best_epoch"] != best_epoch
                or best["best_loss"] != best["val_loss"] or last["best_loss"] != best["best_loss"]
                or last["val_loss"] < best["best_loss"] or metrics.get("epoch") != completed_epoch
                or metrics.get("best_val_loss") != best["best_loss"] or metrics.get("val_loss") != last["val_loss"]):
            raise ValueError("Checkpoint best epoch or loss disagrees with the failed run and its last checkpoint.")

        validation: dict = {"status": "not_run", "split": "val"}
        if validation_device is not None:
            try:
                validation = _evaluate(dataset, working / "best.pt", training, validation_device, progress)
            except EvaluationError as exc:
                raise ValueError(f"Best checkpoint Validation evaluation failed. {exc}") from None
        if ((source_run / "run.json").read_bytes() != raw
                or any(fingerprint_path(source_run / record["path"]) != record["fingerprint"] for record in records.values())
                or detector_scope(dataset) != result["detector_scope"]):
            raise ValueError("Failed run or scoped Detector inputs changed during finalization.")

        finalized = {
            "schema_version": 1, "status": STATUS, "model": ModelName.FASTER_RCNN, "architecture": ARCHITECTURE,
            "reason": REASON, "source": {"run_id": source_run.parent.name, "run_fingerprint": run["run_fingerprint"],
                                        "run_json_sha256": _sha256(raw), "status": "failed", "error": run["error"]},
            "planned_epochs": training.epochs, "completed_epoch": completed_epoch, "best_epoch": best_epoch,
            "best_val_loss": best["best_loss"], "checkpoints": records, "dataset_revision": run["dataset_revision"],
            "publication_fingerprint": seal, "cohort_fingerprint": cohort.fingerprint,
            "split_fingerprint": cohort.split_fingerprint, "detector_layout": training.initialization,
            "image_size": training.image_size, "checks": {
                "group_inputs": "passed", "checkpoint_fingerprints": "passed", "checkpoint_epochs_and_losses": "passed",
                "best_model_layout_and_finiteness": "passed", "validation_inference": validation, "test_payloads_opened": False,
            },
        }
        payload = {"finalization": finalized, "fingerprint": digest_json(finalized)}
        credentials.protect_path(json.dumps(payload), "Detector finalization receipt")
        write_json(working / RECEIPT, payload)
    return out / RECEIPT


def verify_finalized_detector(
    best: Path, sha256: str, layout: FPNDetectorConfig, group: dict, group_receipt: str, bundle: Path,
) -> str:
    """Check a finalized Detector receipt against the current group FPN inputs."""
    try:
        path = best.parent / RECEIPT
        _reject_link(path)
        payload = json.loads(path.read_bytes())
        record = payload["finalization"]
        checkpoint = record["checkpoints"]["best"]
        last = record["checkpoints"]["last"]
        checks = record["checks"]
        if (
            payload["fingerprint"] != digest_json(record) or record["schema_version"] != 1
            or record["status"] != STATUS or record["model"] != ModelName.FASTER_RCNN
            or record["architecture"] != ARCHITECTURE or record["reason"] != REASON
            or record["source"]["status"] != "failed" or record["source"]["error"] != FAILURE
            or not all(_is_sha256(record["source"][key]) for key in ("run_fingerprint", "run_json_sha256"))
            or record["dataset_revision"] != _revision(group, group_receipt)
            or record["publication_fingerprint"] != group["publication_fingerprint"]
            or record["cohort_fingerprint"] != group["cohort_fingerprint"]
            or record["split_fingerprint"] != group["split_fingerprint"]
            or record["detector_layout"] != layout.layout or record["image_size"] != layout.image_size
            or best.name != "best.pt" or checkpoint["path"] != best.name or checkpoint["sha256"] != sha256
            or checkpoint["fingerprint"] != _file_fingerprint(sha256)
            or checkpoint["fingerprint"] != fingerprint_path(best)
            or not 1 <= record["best_epoch"] <= record["completed_epoch"] < record["planned_epochs"]
            or checkpoint["epoch"] != record["best_epoch"] or last["epoch"] != record["completed_epoch"]
            or checkpoint["best_epoch"] != record["best_epoch"] or last["best_epoch"] != record["best_epoch"]
            or not _loss(record["best_val_loss"]) or checkpoint["val_loss"] != record["best_val_loss"]
            or checkpoint["best_loss"] != record["best_val_loss"] or last["best_loss"] != record["best_val_loss"]
            or any(checks[key] != "passed" for key in ("group_inputs", "checkpoint_fingerprints",
                     "checkpoint_epochs_and_losses", "best_model_layout_and_finiteness"))
            or checks["test_payloads_opened"] is not False
            or checks["validation_inference"]["status"] not in ("not_run", "passed")
            or checks["validation_inference"]["split"] != "val"
            or detector_scope(bundle / "detection") != group["detector_scope"]
        ):
            raise ValueError
    except (KeyError, OSError, TypeError, ValueError):
        raise ValueError("Invalid finalized Detector receipt, checkpoint or group provenance.") from None
    return payload["fingerprint"]


def inspect_finalized_detector(directory: Path, credentials: Credentials) -> dict[str, Any]:
    """Inspect a finalized Detector receipt and checkpoint without relying on run.json."""
    try:
        _reject_link(directory)
        receipt = directory / RECEIPT
        best = directory / "best.pt"
        _reject_link(receipt)
        _reject_link(best)
        if not directory.is_dir() or not receipt.is_file() or not best.is_file():
            raise ValueError("Missing finalized Detector receipt or best checkpoint.")
        if not receipt.resolve().is_relative_to(directory.resolve()) or not best.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Finalized Detector receipt or checkpoint is outside the model directory.")

        payload = json.loads(receipt.read_bytes())
        record = payload["finalization"]
        checkpoint = record["checkpoints"]["best"]
        last = record["checkpoints"]["last"]
        checks = record["checks"]
        content = best.read_bytes()
        sha256 = _sha256(content)
        file_fingerprint = fingerprint_path(best)
        if (
            payload["fingerprint"] != digest_json(record) or record["schema_version"] != 1
            or record["status"] != STATUS or record["model"] != ModelName.FASTER_RCNN
            or record["architecture"] != ARCHITECTURE or record["reason"] != REASON
            or record["source"]["status"] != "failed" or record["source"]["error"] != FAILURE
            or not all(_is_sha256(record["source"][key]) for key in ("run_fingerprint", "run_json_sha256"))
            or checkpoint["path"] != "best.pt" or checkpoint["sha256"] != sha256
            or checkpoint["fingerprint"] != _file_fingerprint(sha256)
            or checkpoint["fingerprint"] != file_fingerprint
            or not 1 <= record["best_epoch"] <= record["completed_epoch"] < record["planned_epochs"]
            or checkpoint["epoch"] != record["best_epoch"] or last["epoch"] != record["completed_epoch"]
            or checkpoint["best_epoch"] != record["best_epoch"] or last["best_epoch"] != record["best_epoch"]
            or not _loss(record["best_val_loss"]) or checkpoint["val_loss"] != record["best_val_loss"]
            or checkpoint["best_loss"] != record["best_val_loss"] or last["best_loss"] != record["best_val_loss"]
            or any(checks[key] != "passed" for key in ("group_inputs", "checkpoint_fingerprints",
                     "checkpoint_epochs_and_losses", "best_model_layout_and_finiteness"))
            or checks["test_payloads_opened"] is not False
        ):
            raise ValueError("Invalid finalized Detector receipt or checkpoint provenance.")

        state = _checkpoint(content, "best.pt", record["best_epoch"])
        layout = FPNDetectorConfig(image_size=record["image_size"], layout=record["detector_layout"])
        try:
            from ..multimodal.fpn import detector_from_checkpoint
            detector_from_checkpoint(state, layout)
        except (ImportError, RuntimeError, OSError):
            raise ValueError("Cannot construct the finalized Faster R-CNN checkpoint.") from None
        if _sha256(best.read_bytes()) != sha256 or fingerprint_path(best) != file_fingerprint:
            raise ValueError("Finalized Detector checkpoint changed during verification.")

        training = TrainConfig(image_size=layout.image_size, initialization=layout.layout)
        credentials.protect_path(str(receipt.resolve()), "finalization receipt")
        credentials.protect_path(str(best.resolve()), "finalized checkpoint")
        return {
            "name": ModelName.FASTER_RCNN,
            "path": best,
            "sha256": sha256,
            "fingerprint": file_fingerprint,
            "receipt_fingerprint": fingerprint_path(receipt),
            "metadata_fingerprint": payload["fingerprint"],
            "training": training,
            "image_size": layout.image_size,
            "layout": layout.layout,
        }
    except FileNotFoundError:
        raise ValueError("Finalized Detector metadata or checkpoint is missing.") from None
    except (KeyError, json.JSONDecodeError, OSError, TypeError, ValueError):
        raise ValueError("Invalid finalized Detector receipt, checkpoint or metadata.") from None
