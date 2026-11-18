"""Approval-gated product training orchestration and atomic epoch receipts."""

import json
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Literal, Protocol

from pydantic import Field

from ..config.settings import Credentials
from ..data.model import Progress
from ..data.preprocessing.console import fresh_output
from ..data.preprocessing.leakage import file_sha
from ..evaluation.product import read_artifact, write_artifact
from ..evaluation.product_protocol import ExperimentReport
from ..inspection.contracts import Contract, SchemaVersion
from ..models.contracts import TrainingError
from ..pipeline.store import Fingerprint, RunStore, _reject_link, digest_json, timestamp
from ..training_storage import guard_training_write
from .preprocessing import PreprocessingState
from .training_plan import ProductTrainingPlan, preflight

ProgressFactory = Callable[[str], Progress]


class EpochReceipt(Contract):
    schema_version: SchemaVersion = 1
    compatibility: Fingerprint
    execution: Fingerprint
    epoch: int = Field(gt=0)
    previous: Fingerprint | None
    training_loss: float = Field(ge=0)
    elapsed_seconds: float = Field(ge=0)
    model_sha256: Fingerprint
    resume_sha256: Fingerprint
    validation_sha256: Fingerprint | None
    validation_metric: float | None = Field(ge=0, le=1)

    @property
    def fingerprint(self) -> str:
        return digest_json(self.model_dump(mode="json"))


class EpochEngine(Protocol):
    def restore(self, directory: Path, receipt: EpochReceipt) -> None: ...
    def train_epoch(self, epoch: int, progress: Progress) -> float: ...
    def save(self, directory: Path, epoch: int) -> None: ...
    def evaluate(self, checkpoint_hash: str, progress: Progress) -> ExperimentReport: ...


def epoch_directory(output: Path, epoch: int) -> Path:
    return output / f"epoch-{epoch:06d}"


def should_evaluate(plan: ProductTrainingPlan, epoch: int) -> bool:
    validation = plan.config.validation
    return validation is not None and (validation.frequency == "each_epoch" or epoch == plan.config.epochs)


def selection_metric(plan: ProductTrainingPlan, report: ExperimentReport) -> float | None:
    field = "f1" if plan.config.selection_metric == "macro_f1" else "auroc"
    values = [getattr(metric, field) for metric in report.classification_metrics or ()]
    defined = [value for value in values if value is not None]
    return sum(defined) / len(defined) if defined else None


def selected_epoch(plan: ProductTrainingPlan, receipts: tuple[EpochReceipt, ...]) -> EpochReceipt | None:
    if not receipts:
        return None
    if plan.config.checkpoint_policy == "final_epoch":
        return receipts[-1] if receipts[-1].epoch == plan.config.epochs else None
    eligible = [receipt for receipt in receipts if receipt.validation_metric is not None]
    return max(eligible, key=lambda entry: (entry.validation_metric, -entry.epoch), default=None)


def _check_report(plan: ProductTrainingPlan, report: ExperimentReport, checkpoint: str) -> None:
    architecture = plan.config.architecture
    if (report.status != "measured" or report.branch != plan.config.branch
            or plan.frozen is None or report.protocol != plan.validation_protocol() or report.product_model is None
            or report.product_model.checkpoint_sha256 != checkpoint
            or report.product_model.device != plan.config.device
            or report.product_model.architecture_fingerprint != plan.manifest["architecture_fingerprint"]
            or report.fusion != architecture.fusion or report.classification_metrics is None
            or tuple(metric.name for metric in report.classification_metrics) != architecture.output_names):
        raise ValueError("Validation report does not match the frozen protocol and epoch checkpoint.")


def _run_metadata(output: Path) -> dict:
    path = output / "run.json"
    _reject_link(path)
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("Product run metadata is invalid.")
    return value


def inspect_recovery(plan: ProductTrainingPlan, *, resume: bool) -> tuple[EpochReceipt, ...]:
    """Verify the committed epoch sequence without loading tensor state."""
    if not resume:
        if plan.output.exists():
            raise FileExistsError("Choose a fresh output directory or explicitly request --resume.")
        return ()
    try:
        metadata = _run_metadata(plan.output)
        if (
            metadata["schema_version"] != 1 or metadata["compatibility"] != plan.compatibility
            or metadata["status"] not in ("running", "failed", "interrupted", "completed")
            or type(metadata["completed_epochs"]) is not int
        ):
            raise ValueError("Run metadata or resolved training configuration is incompatible.")
        comparable = set(plan.manifest) - {"configuration", "runtime"}
        if digest_json({key: metadata[key] for key in comparable}) != digest_json(
            {key: plan.manifest[key] for key in comparable}
        ):
            raise ValueError("Saved experiment provenance is incompatible.")
        state = plan.output / "sensor-state.json"
        _reject_link(state)
        if PreprocessingState.load(state).fingerprint != plan.state.fingerprint:
            raise ValueError("Saved frozen preprocessing state differs from the run.")
        execution = metadata["execution"]
        previous = None
        receipts = []
        directories = sorted(plan.output.glob("epoch-*"))
        for epoch, directory in enumerate(directories, 1):
            _reject_link(directory)
            if epoch > plan.config.epochs or directory != epoch_directory(plan.output, epoch):
                raise ValueError("Epoch checkpoint sequence is invalid.")
            path = directory / "receipt.json"
            _reject_link(path)
            receipt = EpochReceipt.model_validate_json(path.read_bytes())
            if (receipt.epoch != epoch or receipt.previous != previous or receipt.compatibility != plan.compatibility
                    or receipt.execution != execution
                    or (receipt.validation_sha256 is not None) != should_evaluate(plan, epoch)):
                raise ValueError("Epoch receipt is incompatible with the run.")
            for filename, expected in (("model.pt", receipt.model_sha256), ("resume.pt", receipt.resume_sha256)):
                artifact = directory / filename
                _reject_link(artifact)
                if file_sha(artifact) != expected:
                    raise ValueError("An epoch checkpoint hash does not match its receipt.")
            if receipt.validation_sha256 is not None:
                path = directory / "validation.json"
                _reject_link(path)
                if file_sha(path) != receipt.validation_sha256:
                    raise ValueError("Validation artifact hash does not match its receipt.")
                report = read_artifact(path, ExperimentReport)
                _check_report(plan, report, receipt.model_sha256)
                if selection_metric(plan, report) != receipt.validation_metric:
                    raise ValueError("The recorded selection metric differs from the Validation report.")
            elif receipt.validation_metric is not None:
                raise ValueError("A selection metric requires an epoch Validation report.")
            receipts.append(receipt)
            previous = receipt.fingerprint
        if not 0 <= metadata["completed_epochs"] <= len(receipts):
            raise ValueError("Run status references missing epoch checkpoints.")
        if metadata["status"] == "completed":
            if len(receipts) != plan.config.epochs or selected_epoch(plan, tuple(receipts)) is None:
                raise ValueError("Completed run is missing the selected or final checkpoint.")
            raise ValueError("The product run already completed. Use its selected checkpoint.")
        return tuple(receipts)
    except (KeyError, TypeError, OSError) as exc:
        raise ValueError("Product run recovery is missing or corrupt. Check run metadata and committed epochs.") from exc


def _execution_identity(report: dict[str, object]) -> str:
    probe = report["device_probe"]
    return digest_json({
        **{key: report[key] for key in ("device", "cuda_build", "rocm_build", "nvidia_driver",
                                        "cpu_model", "cublas_workspace_config")},
        "device_name": probe["device_name"],
    })


def _save_run(
    store: RunStore, plan: ProductTrainingPlan, execution: str, receipts: tuple[EpochReceipt, ...], *,
    status: Literal["running", "failed", "interrupted", "completed"], stage: str,
    readiness: dict[str, object],
) -> None:
    selected = selected_epoch(plan, receipts)
    checkpoint = None
    validation_report = None
    if selected is not None:
        folder = epoch_directory(plan.output, selected.epoch).name
        checkpoint = {"epoch": selected.epoch, "path": f"{folder}/model.pt", "sha256": selected.model_sha256}
        if selected.validation_sha256 is not None:
            validation_report = {"path": f"{folder}/validation.json", "sha256": selected.validation_sha256}
    store.write_json("run.json", store.credentials.redact({
        **plan.manifest, "compatibility": plan.compatibility, "execution": execution,
        "execution_preflight": readiness, "status": status, "stage": stage, "updated_at": timestamp(),
        "completed_epochs": len(receipts), "selected_checkpoint": checkpoint,
        "validation_report": validation_report,
        "experiment_status": (("measured" if validation_report else "completed_unscored")
                              if status == "completed" else "pending"),
    }))


def execute_training(
    plan: ProductTrainingPlan, *, approved: bool, resume: bool, credentials: Credentials,
    progress: ProgressFactory,
) -> Path:
    if approved is not True:
        raise TrainingError("Product optimizer execution requires explicit --approve-train.")
    initial = inspect_recovery(plan, resume=resume)
    readiness = preflight(plan, remaining_epochs=plan.config.epochs - len(initial))
    if not readiness["ready"]:
        raise TrainingError("Product preflight failed. " + " ".join(readiness["issues"]))
    execution = _execution_identity(readiness)
    credentials.protect_path(str(plan.output), "Product run directory")
    credentials.protect_path(json.dumps(plan.manifest), "Product run configuration")
    store = RunStore(plan.output.parent, plan.output.name, credentials)
    if not resume:
        with fresh_output(plan.output, plan.config.source) as working:
            initial_store = RunStore(working.parent, working.name, credentials)
            initial_store.write_json("sensor-state.json", plan.state.payload())
            _save_run(initial_store, plan, execution, (), status="running", stage="initialize", readiness=readiness)

    with store.lock():
        receipts = inspect_recovery(plan, resume=True)
        if _run_metadata(plan.output)["execution"] != execution:
            raise TrainingError("The target device or runtime changed since this product run began.")
        stage = "initialize"
        _save_run(store, plan, execution, receipts, status="running", stage=stage, readiness=readiness)
        try:
            if len(receipts) < plan.config.epochs:
                from .training_torch import training_engine

                with training_engine(plan) as engine:
                    if receipts:
                        stage = "restore"
                        engine.restore(epoch_directory(plan.output, receipts[-1].epoch), receipts[-1])
                    for epoch in range(len(receipts) + 1, plan.config.epochs + 1):
                        started = monotonic()
                        stage = f"train epoch {epoch}"
                        _save_run(store, plan, execution, receipts, status="running", stage=stage, readiness=readiness)
                        loss = engine.train_epoch(epoch, progress(f"Train epoch {epoch}/{plan.config.epochs}"))
                        guard_training_write(plan.output, plan.checkpoint_bytes)
                        destination = epoch_directory(plan.output, epoch)
                        stage = f"checkpoint epoch {epoch}"
                        _save_run(store, plan, execution, receipts, status="running", stage=stage, readiness=readiness)
                        with fresh_output(destination, plan.config.source) as working:
                            engine.save(working, epoch)
                            model_hash = file_sha(working / "model.pt")
                            resume_hash = file_sha(working / "resume.pt")
                            report_hash, metric = None, None
                            if should_evaluate(plan, epoch):
                                stage = f"Validation epoch {epoch}"
                                _save_run(
                                    store, plan, execution, receipts, status="running", stage=stage, readiness=readiness,
                                )
                                report = engine.evaluate(model_hash, progress(f"Validation epoch {epoch}"))
                                _check_report(plan, report, model_hash)
                                write_artifact(working / "validation.json", report, credentials=credentials)
                                report_hash = file_sha(working / "validation.json")
                                metric = selection_metric(plan, report)
                            receipt = EpochReceipt(
                                compatibility=plan.compatibility, execution=execution, epoch=epoch,
                                previous=receipts[-1].fingerprint if receipts else None,
                                training_loss=loss, elapsed_seconds=monotonic() - started,
                                model_sha256=model_hash, resume_sha256=resume_hash,
                                validation_sha256=report_hash, validation_metric=metric,
                            )
                            RunStore(working.parent, working.name).write_json(
                                "receipt.json", receipt.model_dump(mode="json"),
                            )
                        receipts += (receipt,)
            stage = "checkpoint selection"
            if selected_epoch(plan, receipts) is None:
                raise TrainingError("The frozen Validation metric was undefined for every epoch.")
            _save_run(store, plan, execution, receipts, status="completed", stage="completed", readiness=readiness)
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
            _save_run(store, plan, execution, receipts, status=status, stage=stage, readiness=readiness)
            raise
    return store.directory / "run.json"
