"""Product training profiles and split-scoped, read-only planning."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import Field, model_validator

from ..data.model import Progress
from ..data.preprocessing.common import read_csv, require_separate_output
from ..data.preprocessing.console import protect_source_data
from ..data.preprocessing.leakage import Components, file_sha, is_candidate
from ..data.preprocessing.policy import label_policy
from ..data.preprocessing.revision import SEAL
from ..disk import MIB, RESERVE_BYTES, volume_at
from ..environment.doctor import collect_environment
from ..evaluation.product import read_artifact
from ..evaluation.product_protocol import FrozenValidation, ProductProtocol
from ..inspection.contracts import Contract, SchemaVersion
from ..pipeline.model import valid_id
from ..pipeline.store import Fingerprint, _reject_link, digest_json, fingerprint_path
from ..reporting.metadata import runtime_snapshot
from .alignment import EXTENSIONS
from .audit import ISOLATION_GROUPS, Cohort
from .dataset import MultimodalDataset
from .model_config import ArchitectureConfig, Branch, OptimizerConfig, image_preprocessing_fingerprint
from .preprocessing import CAUSAL_WINDOW_POLICY, PreprocessingState, training_fingerprint, transform_sensor


class SchedulerConfig(Contract):
    kind: Literal["constant", "step"] = "constant"
    step_size: int = Field(default=1, gt=0)
    gamma: float = Field(default=0.1, gt=0, le=1)


class ValidationConfig(Contract):
    cohort: Path
    fingerprint: Fingerprint
    frequency: Literal["final", "each_epoch"] = "final"


class DetectorInitialization(Contract):
    path: Path
    sha256: Fingerprint


class ProductTrainingConfig(Contract):
    schema_version: SchemaVersion = 1
    source: Path
    bundle: Path
    revision_id: Fingerprint
    cohort: Path
    cohort_fingerprint: Fingerprint
    preprocessing: Path
    architecture: ArchitectureConfig
    branch: Branch
    detector_initialization: DetectorInitialization | None = None
    missing_train: Literal["error", "drop"] = "error"
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    epochs: int = Field(default=1, gt=0)
    batch_size: int = Field(default=4, gt=0)
    gradient_accumulation: int = Field(default=1, gt=0)
    device: Literal["cpu", "cuda", "rocm", "mps"] = "cpu"
    cpu_threads: int = Field(default=1, gt=0)
    checkpoint_policy: Literal["final_epoch", "best_validation"] = "final_epoch"
    selection_metric: Literal["macro_f1", "macro_auroc"] = "macro_f1"
    validation: ValidationConfig | None = None

    @model_validator(mode="after")
    def check_policy(self) -> Self:
        if self.checkpoint_policy == "best_validation" and (
            self.validation is None or self.validation.frequency != "each_epoch"
        ):
            raise ValueError("Best Validation checkpoint selection requires frozen Validation every epoch.")
        if self.architecture.image.kind == "faster_rcnn_fpn":
            if self.architecture.image.detector is None or self.detector_initialization is None:
                raise ValueError("FPN training requires image.detector and a pinned detector initialization checkpoint.")
            if self.branch not in ("image", "paired"):
                raise ValueError("FPN training requires the image or paired branch.")
        elif self.detector_initialization is not None:
            raise ValueError("Detector initialization requires the FPN encoder.")
        if self.branch == "image" and self.architecture.sensor.kind != "mlp":
            raise ValueError("Use the default MLP placeholder for image-only training.")
        return self


@dataclass(frozen=True)
class ProductTrainingPlan:
    config: ProductTrainingConfig
    output: Path
    state: PreprocessingState
    train: MultimodalDataset
    validation: MultimodalDataset | None
    frozen: FrozenValidation | None
    manifest: dict[str, object]
    compatibility: str
    checkpoint_bytes: int
    estimated_host_bytes: int
    estimated_device_bytes: int

    image_input_contract: dict[str, object] = field(default_factory=dict)

    def validation_protocol(self) -> ProductProtocol:
        if self.frozen is None:
            raise ValueError("Configure a frozen Validation cohort before scoring.")
        architecture = self.config.architecture
        offline = self.state.config.window_seconds > 0 and self.state.config.window_policy != CAUSAL_WINDOW_POLICY
        return ProductProtocol(
            cohort=self.frozen.identity, target_policy=architecture.target_policy,
            threshold=architecture.fusion.threshold, preprocessing_fingerprint=self.state.fingerprint,
            image_preprocessing_fingerprint=image_preprocessing_fingerprint(architecture.image, self.image_input_contract),
            alignment_policy="audited_pair_decision_window-v1" if offline else "verified_receipt_at_capture-v1",
        )


def _record_hashes(record: dict) -> dict[str, str]:
    return {key: record[key.replace("_path", "_sha256")] for key in EXTENSIONS
            if record.get(key.replace("_path", "_sha256")) is not None}


def _check_isolation(records: list[dict], used: set[int]) -> None:
    components = Components(len(records))
    owners: dict[tuple[str, str], int] = {}
    candidates = []
    for index, record in enumerate(records):
        if record["detection_candidate"]:
            candidates.append(index)
        groups = {(kind, key) for kind, key in record["groups"] if kind in ISOLATION_GROUPS}
        groups.add(("sample_id", record["sample_id"].casefold()))
        groups.update((kind, record[field]) for kind, field in (
            ("image_sha256", "image_sha256"), ("sensor_workbook_sha256", "sensor_sha256"),
        ) if record.get(field))
        for group in groups:
            owner = owners.setdefault(group, index)
            if group[0] == "sample_id" and owner != index and (owner in used or index in used):
                raise ValueError("Ambiguous duplicate sample ID in the training cohort.")
            components.join(owner, index)
    for members in components.groups(candidates):
        if used.intersection(members) and len({records[i]["split"] for i in members}) > 1:
            raise ValueError("Cross-split sample, capture, or sensor evidence intersects the requested cohort.")


def _load_index(
    config: ProductTrainingConfig, frozen: FrozenValidation | None, progress: Progress,
) -> tuple[Cohort, str, int]:
    # Verify publication metadata. Hashing the detector-format trees here would
    # open reserved Test files, which this runner does not consume.
    revision = json.loads((config.bundle / "split.json").read_bytes())
    seal_path = config.bundle / "detection" / SEAL
    _reject_link(seal_path)
    seal = json.loads(seal_path.read_bytes())
    receipt = seal["record"]
    rows = read_csv(config.bundle / "manifest.csv")
    if (
        revision["schema_version"] != 1 or revision["revision_id"] != config.revision_id
        or digest_json({k: v for k, v in revision.items() if k != "revision_id"}) != config.revision_id
        or digest_json(rows) != revision["selected_manifest_fingerprint"]
        or digest_json(revision["label_policy"]) != digest_json(label_policy())
        or receipt["schema_version"] != 1 or digest_json(receipt) != seal["fingerprint"]
        or receipt["split"] != {k: v for k, v in revision.items() if k != "assignments"}
        or set(receipt["formats"]) != {"yolo", "coco"}
    ):
        raise ValueError("Use the explicitly pinned sealed split bundle and its original manifest.")

    raw = json.loads(config.cohort.read_bytes())
    payload = raw["cohort"]
    if raw["fingerprint"] != config.cohort_fingerprint or digest_json(payload) != config.cohort_fingerprint:
        raise ValueError("The audited cohort fingerprint differs from the pinned cohort.")
    records = sorted(payload["records"], key=lambda record: (
        record["provider_split"], record["sample_id"], record["annotation_path"],
    ))
    if any(record["split"] == "test" and (
        any(record.get(key) is not None for key in ("sequence", "targets", "capture_timestamp", "image_id",
                                                    "sensor_image_id", "alignment_method"))
        or record.get("alignment") != "unmatched"
    ) for record in records):
        raise ValueError("Use a product training index with Test evidence removed by Cohort.for_product_training().")
    rows.sort(key=lambda row: (row["original_split"], row["sample_id"], row["annotation_path"]))
    split = [[record[k] for k in ("sample_id", "annotation_path", "provider_split", "split",
                                "detection_candidate", "groups", "quarantined")]
             for record in payload["records"]]
    sources = []
    scoped = []
    used = set()
    selected = set(frozen.identity.sample_ids) if frozen else set()
    if len(rows) != len(records):
        raise ValueError("Published manifest and audited cohort membership differ.")
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        paths = {key: row[key] for key in EXTENSIONS}
        if (any(record[k] != row[k] for k in (*paths, "sample_id", "split", "decision"))
                or record["provider_split"] != row["original_split"]
                or record["detection_candidate"] != is_candidate(row)
                or dict(record["groups"]).get("revision_group", "") != row["group_id"]):
            raise ValueError("Published manifest and audited cohort assignments differ.")
        sources.append([paths, _record_hashes(record)])
        if record["split"] == "train" or (record["split"] == "validation" and record["sample_id"] in selected):
            scoped.append((index, record, paths))
            if record["detection_candidate"] and not record["quarantined"]:
                used.add(index)
    if (
        payload["revision_id"] != config.revision_id or payload["manifest_fingerprint"] != digest_json(rows)
        or payload["input_fingerprint"] != revision.get("selected_source_fingerprint", revision["dataset_fingerprint"])
        or digest_json(sources) != payload["input_fingerprint"] or digest_json(split) != payload["split_fingerprint"]
    ):
        raise ValueError("Source, manifest, or split fingerprint disagrees with the sealed revision.")
    _check_isolation(records, used)

    progress(0, len(scoped))
    max_image_bytes = 0
    for done, (index, record, paths) in enumerate(scoped, 1):
        if index in used:
            expected = _record_hashes(record)
            for key, relative in paths.items():
                actual = None
                if relative:
                    portable = PurePosixPath(relative)
                    path = (config.source / relative).resolve()
                    if portable.is_absolute() or ".." in portable.parts or not path.is_relative_to(config.source):
                        raise ValueError("An indexed source path escapes the extracted root.")
                    if path.is_file():
                        actual = file_sha(path)
                        if key == "image_path":
                            max_image_bytes = max(max_image_bytes, path.stat().st_size)
                if actual != expected.get(key):
                    raise ValueError(f"Source bytes changed for {record['sample_id']} ({key}).")
        progress(done, len(scoped))
    # Materialize only the permitted views. Reserved Test contributes
    # assignment/group/hash metadata to integrity checks.
    keys = {(record["provider_split"], record["sample_id"], record["annotation_path"]) for _, record, _ in scoped}
    visible = [record for record in payload["records"]
               if (record["provider_split"], record["sample_id"], record["annotation_path"]) in keys]
    return (
        Cohort.model_validate_json(json.dumps({**payload, "records": visible})),
        seal["fingerprint"], max_image_bytes,
    )


def _views(
    config: ProductTrainingConfig, cohort: Cohort, state: PreprocessingState, frozen: FrozenValidation | None,
) -> tuple[MultimodalDataset, MultimodalDataset | None, dict[str, object]]:
    sensor = config.branch != "image"
    if config.architecture.sensor.kind == "gru" and state.config.window_seconds == 0:
        raise ValueError("GRU requires a nonzero frozen sensor window and multiple observed steps.")
    if frozen and frozen != FrozenValidation.from_audit(cohort, frozen.identity.sample_ids):
        raise ValueError("Frozen Validation differs from the selected revision or audited labels and pairing.")
    if frozen and any(None in row.labels(config.architecture.target_policy) for row in frozen.samples):
        raise ValueError("Frozen Validation requires complete labels for product training.")

    views = {}
    counts = {}
    for split in ("train", "validation") if frozen else ("train",):
        records = tuple(row for row in cohort.records if row.split == split and row.detection_candidate)
        usable = []
        for row in records:
            if row.quarantined:
                continue
            if sensor and row.alignment == "ambiguous":
                raise ValueError(f"Ambiguous sensor pairing for {row.sample_id}. Review the audited cohort.")
            modality_ok = row.targets is not None and (config.branch == "sensor" or row.image_available)
            if modality_ok and sensor:
                window = (transform_sensor(row.sequence, row.capture_timestamp, state)
                          if row.sequence is not None and row.capture_timestamp is not None else None)
                minimum = config.architecture.sensor.min_steps if config.architecture.sensor.kind == "gru" else 1
                modality_ok = window is not None and sum(map(any, window.observed)) >= minimum
            if modality_ok:
                usable.append(row)
            elif split == "validation" or config.missing_train == "error":
                raise ValueError(f"Sample {row.sample_id} lacks valid targets, a requested modality, or enough observed steps.")
        if not usable:
            raise ValueError(f"No usable {config.branch} samples in {split}.")
        # Keep all eligible Train observations in the state identity even when
        # the profile explicitly drops rows for its own branch.
        selected_ids = frozenset(row.sample_id for row in usable)
        views[split] = MultimodalDataset(
            cohort, config.source, split, mode=config.branch, preprocessing=state, missing="error",
            sample_ids=selected_ids,
        )
        counts[split] = {"candidates": len(records), "used": len(usable),
                         "sample_fingerprint": digest_json([row.sample_id for row in views[split].records])}
    return views["train"], views.get("validation"), counts


def plan_training(
    config: ProductTrainingConfig, output: Path, *, workspace: Path, progress: Progress,
) -> ProductTrainingPlan:
    config = config.model_copy(update={
        name: (workspace / getattr(config, name).expanduser()).resolve()
        for name in ("source", "bundle", "cohort", "preprocessing")
    })
    if config.validation is not None:
        config = config.model_copy(update={"validation": config.validation.model_copy(update={
            "cohort": (workspace / config.validation.cohort.expanduser()).resolve(),
        })})
    initialization = config.detector_initialization
    if initialization is not None:
        initialization = initialization.model_copy(update={"path": (workspace / initialization.path.expanduser()).resolve()})
        config = config.model_copy(update={"detector_initialization": initialization})
    output = (workspace / output.expanduser()).absolute()
    _reject_link(output)
    output = output.resolve()
    if not valid_id(output.name):
        raise ValueError("Use a run directory name with up to 64 lowercase letters, digits, underscores, or hyphens.")
    protect_source_data(output, workspace)
    for path in (config.source, config.bundle, config.cohort, config.preprocessing,
                 *((config.validation.cohort,) if config.validation else ()),
                 *((initialization.path,) if initialization else ())):
        require_separate_output(path, output)
        require_separate_output(output, path)
    frozen = read_artifact(config.validation.cohort, FrozenValidation) if config.validation else None
    if frozen and frozen.identity.cohort_fingerprint != config.validation.fingerprint:
        raise ValueError("Frozen Validation fingerprint differs from the profile.")
    cohort, seal, image_bytes = _load_index(config, frozen, progress)
    state = PreprocessingState.load(config.preprocessing, expected_training_fingerprint=training_fingerprint(cohort))
    if state.fingerprint != config.architecture.preprocessing_fingerprint:
        raise ValueError("Architecture requires the pinned frozen Train preprocessing fingerprint.")
    if not any(channel.count for channel in state.statistics):
        raise ValueError("The frozen Train state contains no observed measurements.")
    train, validation, counts = _views(config, cohort, state, frozen)

    architecture = config.architecture
    image_contract: dict[str, object] = {}
    provenance: dict[str, object] = {}
    fpn_state_bytes = fpn_activations = detector_load_bytes = 0
    if initialization is not None and architecture.image.detector is not None:
        try:
            from .encoders import FasterRCNNImageEncoder
            from .fpn import ARCHITECTURE, FORMAT, load_detector

            detector = load_detector(initialization.path, initialization.sha256, architecture.image.detector)
            encoder = FasterRCNNImageEncoder(architecture.image, detector)
        except (ImportError, RuntimeError) as exc:
            raise ValueError("FPN preflight requires a working PyTorch and TorchVision installation.") from exc
        image_contract = encoder.input_contract
        fpn_state_bytes = sum(t.numel() * t.element_size() for t in encoder.backbone.state_dict().values())
        detector_load_bytes = 2 * initialization.path.stat().st_size
        padded_side = (architecture.image.detector.image_size + 31) // 32 * 32
        # Allow for feature maps and resize workspaces. The encoder processes
        # images serially, and a frozen backbone retains no activation graph.
        fpn_activations = padded_side**2 * (256 if architecture.image.freeze_backbone else 1024 * config.batch_size)
        provenance = {"format": FORMAT, "architecture": ARCHITECTURE, "sha256": initialization.sha256}

    package = Path(__file__).resolve().parents[1]
    code = digest_json({p.relative_to(package).as_posix(): fingerprint_path(p) for p in package.rglob("*.py")})
    runtime = runtime_snapshot(workspace, ("torch", "torchvision") if provenance else ("torch",))
    environment = {
        "python": runtime["python"], "packages": runtime["packages"], "os": runtime["os"],
    }
    settings = config.model_dump(mode="json")
    for key in ("source", "bundle", "cohort", "preprocessing"):
        settings.pop(key)
    if initialization is not None:
        settings["detector_initialization"].pop("path")
    if settings["validation"]:
        settings["validation"].pop("cohort")
    identity = {
        "profile": settings, "dataset_fingerprint": cohort.input_fingerprint,
        "split_fingerprint": cohort.split_fingerprint, "seal_fingerprint": seal,
        "preprocessing_fingerprint": state.fingerprint, "train_state_fingerprint": state.training_fingerprint,
        "architecture_fingerprint": digest_json([architecture.fingerprint, image_contract]),
        "validation_identity": frozen.identity.model_dump(mode="json") if frozen else None,
        "samples": counts, "code_fingerprint": code, "runtime_identity": environment,
        "label_policy": label_policy(),
    }
    if provenance:
        identity["detector_initialization"] = provenance
        identity["image_input_contract"] = image_contract
    # Conservative allowances for RGB decode, FP32 parameters, AdamW state,
    # activations and the two retained files in each committed epoch.
    sensor_width = architecture.sensor.hidden_dim
    sensor_dim = architecture.sensor.embedding_dim
    image_dim = architecture.image.embedding_dim
    fusion_width = architecture.fusion.hidden_dim
    params = (8 * sensor_width * (sensor_width + sensor_dim + 64)
              + image_dim * ((256 if provenance else 3 * architecture.image.grid_size**2) + 8)
              + 8 * (image_dim + sensor_dim + 8) * (fusion_width + image_dim + 8))
    max_pixels = max((row.image_size[0] * row.image_size[1] for view in (train, validation) if view
                      for row in view.records), default=0) if config.branch != "sensor" else 0
    max_steps = max((len(row.sequence.steps) for view in (train, validation) if view for row in view.records
                     if row.sequence is not None), default=0)
    model_bytes = params * 32 + config.batch_size * (max_steps * (sensor_width + 64) * 32
                                                    + architecture.image.grid_size**2 * 48)
    fpn_optimizer_bytes = 0 if architecture.image.freeze_backbone else 2 * fpn_state_bytes
    # Resizing transfers one decoded image to the device before scaling.
    model_bytes += 2 * fpn_state_bytes + 2 * fpn_optimizer_bytes + fpn_activations
    if provenance:
        model_bytes += max_pixels * 24
    host_bytes = 128 * MIB + model_bytes + detector_load_bytes + config.batch_size * (max_pixels * 8 + image_bytes)
    checkpoint_bytes = params * 24 + MIB + fpn_state_bytes + fpn_optimizer_bytes
    return ProductTrainingPlan(
        config, output, state, train, validation, frozen,
        {"schema_version": 1, "task": "product_classification", **identity,
         "configuration": config.model_dump(mode="json"), "runtime": runtime},
        digest_json(identity), checkpoint_bytes, host_bytes, 128 * MIB + model_bytes, image_contract,
    )


def preflight(plan: ProductTrainingPlan, *, remaining_epochs: int) -> dict[str, object]:
    volume = volume_at(plan.output)
    environment = collect_environment(volume.path)
    requested = plan.config.device
    backend = getattr(environment.torch, requested)
    disk = RESERVE_BYTES + plan.checkpoint_bytes * max(remaining_epochs, 1)
    ram = environment.resources.available_ram_bytes
    device_bytes = backend.free_device_bytes if requested in ("cuda", "rocm") else backend.shared_memory_limit_bytes
    issues = []
    if backend.readiness != "available" or "fp32" not in backend.precisions_tested:
        issues.append(f"{requested} failed the PyTorch FP32 readiness probe.")
    if requested == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        issues.append("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before deterministic CUDA product training.")
    if ram is not None and plan.estimated_host_bytes > ram * 85 // 100:
        issues.append("Estimated host memory exceeds 85% of available RAM. Reduce batch size or use a larger host.")
    if device_bytes is not None and plan.estimated_device_bytes > device_bytes * 85 // 100:
        issues.append("Estimated device memory exceeds 85% of available memory. Reduce batch size or use a larger device.")
    if volume.free_bytes < disk:
        issues.append("Insufficient free disk for retained epoch checkpoints. Choose a larger run volume.")
    return {
        "ready": not issues, "issues": issues, "device": requested, "device_probe": asdict(backend),
        "available_ram_bytes": ram, "estimated_host_bytes": plan.estimated_host_bytes,
        "available_device_bytes": device_bytes,
        "estimated_device_bytes": plan.estimated_device_bytes if requested != "cpu" else None,
        "free_disk_bytes": volume.free_bytes, "required_disk_bytes": disk,
        "cuda_build": environment.torch.cuda_build, "rocm_build": environment.torch.rocm_build,
        "nvidia_driver": environment.graphics.nvidia_driver,
        "cpu_model": environment.resources.cpu_model,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG") if requested == "cuda" else None,
    }
