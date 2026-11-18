"""Opt-in tests for official checkpoints and the prepared real SMT cohort."""

import json
import math
import os
import socket
import sys
from pathlib import Path
from time import monotonic
from typing import Any

import pytest

from ict_pipeline.data.preprocessing.policy import NAMES
from ict_pipeline.data.preprocessing.revision import read_dataset_seal
from ict_pipeline.evaluation.contracts import EvaluationConfig, EvaluationSplit
from ict_pipeline.evaluation.runner import evaluate_runs, read_report, write_result
from ict_pipeline.models.contracts import ModelName, TrainingError
from ict_pipeline.models.registry import REGISTRY
from ict_pipeline.models.training import execute_training, plan_training
from ict_pipeline.models.weights import verify_weights
from ict_pipeline.smoke import IMAGE_SIZE, PER_STRATUM, STRATA, select_samples, smoke_profiles


def test_balanced_smoke_selection_keeps_split_and_excludes_partial_groups() -> None:
    samples: list[dict[str, Any]] = []
    assignments: list[dict[str, Any]] = []
    for split in ("train", "validation", "test"):
        for label in range(len(STRATA)):
            for _ in range(PER_STRATUM):
                index = len(samples)
                samples.append({"classes": [label] if label < len(NAMES) else []})
                assignments.append({"sample_index": index, "group_id": str(index), "split": split})
    report, plan = {"samples": samples}, {"assignments": assignments}
    assert select_samples(report, plan) == assignments
    # Join two eligible members. Neither can be drawn as a singleton now.
    assignments[1]["group_id"] = assignments[0]["group_id"]
    with pytest.raises(ValueError, match="Insufficient singleton groups for train"):
        select_samples(report, plan)


@pytest.mark.real_data_smoke
@pytest.mark.parametrize("name", list(ModelName))
def test_official_checkpoint_real_smt(name: ModelName, monkeypatch: pytest.MonkeyPatch) -> None:
    if os.getenv("ICT_REAL_SMOKE_APPROVE_TRAIN") != "1":
        pytest.skip("Set ICT_REAL_SMOKE_APPROVE_TRAIN=1 only after authorizing one CPU epoch per model.")
    cohort = Path(os.environ["ICT_REAL_SMOKE_COHORT"]).resolve(strict=True)
    output = Path(os.environ["ICT_REAL_SMOKE_OUTPUT"]).resolve()
    dataset = cohort / "detection"
    profiles = cohort / "profiles"
    seal = read_dataset_seal(dataset)
    assert seal is not None
    revision = seal["split"]
    assert revision["counts"] == dict.fromkeys(("train", "validation", "test"), 12)
    assert revision["test_origin"] == "internal_same_source"
    for distribution in revision["class_samples"].values():
        assert distribution == dict.fromkeys(STRATA, PER_STRATUM)
    protocol = json.loads((cohort / "protocol.json").read_text(encoding="utf-8"))
    evaluation = EvaluationConfig(image_size=IMAGE_SIZE, warmup=1)
    assert protocol == {
        "dataset_revision": revision["revision_id"], "split": "val", "checkpoint": "best",
        "evaluation": evaluation.model_dump(mode="json"),
    }
    plan = plan_training(Path.cwd(), (name,), dataset, output / "training", profiles, None)
    config = plan.models[name]
    assert (config.initialization, config.epochs, config.batch_size, config.image_size, config.device,
            config.precision, config.freeze_backbone) == ("pretrained", 1, 2, IMAGE_SIZE, "cpu", "fp32", True)
    assert config.checkpoint is not None
    assert config == smoke_profiles(Path(config.checkpoint.source).parent)[name]

    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("Unexpected network access in the offline SMT smoke test.")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_OFFLINE", "true")
    monkeypatch.setattr(socket.socket, "connect", reject_network)
    monkeypatch.setattr(socket.socket, "connect_ex", reject_network)
    monkeypatch.setattr(socket, "getaddrinfo", reject_network)
    wrong_digest = config.checkpoint.model_copy(update={"sha256": "0" * 64})
    with pytest.raises(TrainingError, match="SHA-256 mismatch"):
        verify_weights(name, config.model_copy(update={"checkpoint": wrong_digest}),
                       plan.weight_sources[name], REGISTRY[name].versions())
    import torch

    torch.set_num_threads(2)

    def progress(stage: str, done: int, total: int) -> None:
        print(f"{stage} {done}/{total}   elapsed {monotonic() - started:.1f}s", flush=True)

    def interrupt(stage: str, done: int, total: int) -> None:
        progress(stage, done, total)
        if stage.endswith("epoch checkpoint") and done == total == 1:
            raise KeyboardInterrupt

    started = monotonic()
    with pytest.raises(KeyboardInterrupt):
        list(execute_training(plan, interrupt))
    train_elapsed = monotonic() - started
    previous = plan.destination / name
    interrupted = json.loads((previous / "run.json").read_text(encoding="utf-8"))
    assert interrupted["status"] == "interrupted" and interrupted["checkpoint_epoch"] == 1
    transfer = interrupted["weight_transfer"]
    assert transfer["reset_keys"] and not transfer["incompatible_keys"]
    renamed = transfer.get("renamed_keys", {})
    assert set(transfer["unexpected_keys"]) == renamed.keys()
    assert set(transfer["missing_keys"]) == transfer["restored_aliases"].keys() | set(renamed.values())

    resumed = plan_training(Path.cwd(), (name,), dataset, output / "training", profiles, previous)
    started = monotonic()
    [(_, checkpoints)] = execute_training(resumed, progress)
    resume_elapsed = monotonic() - started
    run = resumed.destination / name
    final = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert final["status"] == "completed" and final["initialization"] == "resume"
    assert final["weight_transfer"] == transfer and final["checkpoints"] == interrupted["checkpoints"]
    assert final["checkpoint_epoch"] == 1
    train_losses = {key: loss for key, loss in final["metrics"].items() if key.startswith(("train/", "train_loss"))}
    assert train_losses and all(math.isfinite(loss) for loss in train_losses.values())

    state = torch.load(checkpoints.last, map_location="cpu", weights_only=name != ModelName.YOLO)
    assert state["optimizer"]["state"]
    if name == ModelName.YOLO:
        head = state["ema"].model[-1]
        dimensions = {"classes": head.nc, "classification_rows": head.cv3[0][-1].out_channels}
        assert dimensions == {"classes": 5, "classification_rows": 5}
    elif name == ModelName.FASTER_RCNN:
        dimensions = {head: state["model"][f"roi_heads.box_predictor.{head}.weight"].shape[0]
                      for head in ("cls_score", "bbox_pred")}
        assert dimensions == {"cls_score": 6, "bbox_pred": 24}
    else:
        dimensions = {key: state["model"][f"{key}.weight"].shape[0]
                      for key in ("class_embed.0", "model.enc_score_head", "model.denoising_class_embed")}
        assert dimensions == {"class_embed.0": 5, "model.enc_score_head": 5, "model.denoising_class_embed": 6}
    del state

    started = monotonic()
    [result] = evaluate_runs(Path.cwd(), (run,), dataset, EvaluationSplit.VALIDATION, evaluation,
                             output / "evaluation", "best", False, None, None, progress)
    evaluation_elapsed = monotonic() - started
    metrics = read_report(result)
    assert metrics["status"] == "completed"
    assert metrics["dataset"]["split"] == "val" and metrics["dataset"]["image_count"] == 12
    assert all(row["ground_truth_objects"] > 0 for row in metrics["metrics"]["per_class"])
    if sys.platform == "win32":
        import psutil

        peak_rss = psutil.Process().memory_info().peak_wset
    else:
        import resource

        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    write_result(run / "smoke.json", {
        "schema_version": 1, "model": name.value, "status": "completed", "dataset_revision": revision["revision_id"],
        "training_run": str(run / "run.json"), "validation_report": str(result), "head_dimensions": dimensions,
        "resume": "Interrupted after the committed final epoch, then recovered both byte-identical checkpoints",
        "offline": True, "incorrect_digest_rejected": True,
        "elapsed_seconds": {"train": train_elapsed, "resume": resume_elapsed,
                                            "evaluation": evaluation_elapsed},
        "process_peak_rss_bytes": peak_rss,
    })
