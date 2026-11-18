"""Load once and coordinate causal, provenance-bound product inference."""

import hashlib
from collections.abc import Iterable
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from ..config.settings import Credentials
from ..inspection.contracts import (
    ImageSource, InspectionError, Modality, ProductInspection, ProductModelIdentity, ProductScoreRecord,
)
from ..models.contracts import ModelName, TrainConfig, TrainingError
from ..models.registry import REGISTRY, resolve_device
from ..pipeline.store import digest_json
from .inference_contracts import (
    MockProductModel, PairingReceipt, PairingRegistry, ProductInspectionInput, ProductModelSpec,
    capture_instant, causal_sequence, sequence_fingerprint,
)
from .model_config import image_preprocessing_fingerprint
from .preprocessing import WINDOW_POLICY, PreprocessingState, transform_sensor

if TYPE_CHECKING:
    from torch.nn import Module

    from .model import MultimodalModel


class ProductInputError(InspectionError):
    """The request lacks compatible or verified input evidence."""


def configured_file(root: Path, path: Path, credentials: Credentials) -> Path:
    if path.anchor or ".." in path.parts or any(":" in part for part in path.parts):
        raise InspectionError("Use an artifact relative to the configured model root.")
    resolved = (root / path).resolve()
    credentials.protect_path(str(resolved), "Multimodal model artifact")
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise InspectionError("A configured multimodal artifact is missing or outside the model root.")
    return resolved


class MultimodalInference:
    def __init__(
        self, model: "MultimodalModel | None", state: PreprocessingState | None,
        identity: ProductModelIdentity | None, pairings: Iterable[PairingReceipt],
    ) -> None:
        if (model is None) != (state is None) or (model is None) != (identity is None):
            raise ValueError("A real product model requires preprocessing and checkpoint provenance.")
        if model is not None and state is not None and identity is not None and (
            model.training or model.config.preprocessing_fingerprint != state.fingerprint
            or identity.preprocessing_fingerprint != state.fingerprint
            or identity.architecture_fingerprint != model.architecture_fingerprint
        ):
            raise ValueError("Product model, preprocessing and architecture provenance must match.")
        self._model = model
        self._state = state
        self.identity = identity
        self._pairings = frozenset(receipt.fingerprint for receipt in pairings)

    @classmethod
    def mock(cls, pairings: Iterable[PairingReceipt] = ()) -> "MultimodalInference":
        return cls(None, None, None, pairings)

    @classmethod
    def load(
        cls, spec: ProductModelSpec, root: Path, *, credentials: Credentials,
        detector: "Module | None" = None, scratch_root: Path | None = None,
    ) -> "MultimodalInference":
        root = root.resolve()
        try:
            receipts = () if spec.pairings is None else PairingRegistry.model_validate_json(
                configured_file(root, spec.pairings, credentials).read_bytes(),
            ).receipts
            if isinstance(spec, MockProductModel):
                return cls.mock(receipts)

            import torch

            from .checkpoint import FORMAT, load_checkpoint

            state = PreprocessingState.load(configured_file(root, spec.preprocessing, credentials))
            checkpoint = configured_file(root, spec.checkpoint, credentials)
            with TemporaryDirectory(prefix="ict-product-model-", dir=scratch_root) as directory:
                snapshot = Path(directory) / "checkpoint.pt"
                copyfile(checkpoint, snapshot)
                with snapshot.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != spec.checkpoint_sha256:
                        raise InspectionError("Multimodal checkpoint SHA-256 mismatch.")
                model = load_checkpoint(snapshot, state, detector=detector)
            if model.architecture_fingerprint != spec.architecture_fingerprint:
                raise InspectionError("Multimodal checkpoint architecture fingerprint mismatch.")
            config = model.config
            backend = ModelName.FASTER_RCNN if config.image.kind == "faster_rcnn_fpn" else ModelName.YOLO
            device = resolve_device(torch, TrainConfig(device=spec.device), REGISTRY[backend])
            model.to(device).eval()
            libraries = {name: version(name) for name in ("torch", "Pillow")}
            if config.image.kind == "faster_rcnn_fpn":
                libraries["torchvision"] = version("torchvision")
            identity = ProductModelIdentity(
                checkpoint_format=FORMAT, checkpoint_sha256=spec.checkpoint_sha256,
                architecture_version=config.schema_version, architecture_fingerprint=model.architecture_fingerprint,
                image_encoder=config.image.kind, image_encoder_version=config.image.schema_version,
                image_preprocessing_fingerprint=image_preprocessing_fingerprint(
                    config.image, model.image_encoder.input_contract,
                ),
                sensor_encoder=config.sensor.kind, sensor_encoder_version=config.sensor.schema_version,
                preprocessing_version=state.schema_version, preprocessing_fingerprint=state.fingerprint,
                window_policy=state.config.window_policy, fusion=config.fusion.kind,
                fusion_version=config.fusion.schema_version,
                fusion_fingerprint=digest_json(config.fusion.model_dump(mode="json")), device=spec.device,
                seed=config.seed, libraries=libraries,
            )
            return cls(model, state, identity, receipts)
        except (InspectionError, TrainingError) as exc:
            raise InspectionError(credentials.redact_text(str(exc))) from None
        except ImportError:
            raise InspectionError("A checkpoint requires the existing PyTorch and image runtime.") from None
        except Exception:
            raise InspectionError(
                "Cannot initialize product inference. Check the checkpoint, architecture, pairing registry and frozen preprocessing."
            ) from None

    def predict(
        self, request: ProductInspectionInput, image: ImageSource | None, data: bytes | None,
        credentials: Credentials,
    ) -> ProductInspection:
        evidence = request.sensor_evidence
        if (image is None) != (data is None) or (image is None and (evidence is None or request.expected_size)):
            raise ProductInputError("Supply at least one usable input type and image dimensions only with an image.")
        if image is not None and data is not None and hashlib.sha256(data).hexdigest() != image.sha256:
            raise ProductInputError("Image bytes do not match the image provenance.")

        input_mode: Modality = "paired" if evidence and image else "sensor" if evidence else "image"
        instant = capture_instant(request.capture_timestamp) if request.capture_timestamp is not None else None
        capture = instant.isoformat() if instant is not None else None
        sensor_hash = pairing_hash = None
        sensor_status = "absent"
        window = None
        if evidence is not None:
            try:
                assert instant is not None and capture is not None
                sequence = causal_sequence(evidence.sequence, instant)
                sensor_hash = sequence_fingerprint(sequence)
                if image is not None:
                    if sequence is None:
                        raise ProductInputError("Paired inspection requires sensor evidence available at capture.")
                    receipt = PairingReceipt(
                        sample_id=request.sample_id, image_source_id=request.source_id,
                        sensor_source_id=evidence.source_id, image_sha256=image.sha256,
                        sensor_fingerprint=sensor_hash, capture_timestamp=capture,
                    )
                    pairing_hash = receipt.fingerprint
                    if pairing_hash not in self._pairings:
                        raise ProductInputError("Image and sensor inputs require a matching trusted pairing receipt.")
                sensor_status = "available" if sequence is not None and sequence.available else "no_observations"
                if self._state is not None:
                    config = self._state.config
                    if config.window_policy == WINDOW_POLICY and config.window_seconds != 0:
                        raise ProductInputError("Use a frozen preprocessing state compatible with capture-time sensor inference.")
                    if sequence is not None:
                        window = transform_sensor(sequence, capture, self._state)
            except (ValueError, OverflowError):
                raise ProductInputError("Sensor evidence does not match the capture or preprocessing contract.") from None

        fields = {
            "sample_id": credentials.redact_text(request.sample_id), "input_mode": input_mode,
            "capture_timestamp": capture, "sensor_fingerprint": sensor_hash, "pairing_fingerprint": pairing_hash,
            "alignment": "verified_receipt" if pairing_hash is not None else "not_applicable",
        }
        if self._model is None:
            branch = ("paired" if image else "sensor") if sensor_status == "available" else "image" if image else None
            return ProductInspection(
                **fields, branch=branch, sensor_status=sensor_status, threshold=0.5,
                score_kind="simulated_fixture", weight_status="simulated" if branch else "unavailable",
                target_policy="smt-target-defect-presence-v1",
                scores=(ProductScoreRecord(name="any_target_defect", class_id=None, score=0.25,
                                           threshold_crossed=False),) if branch else (),
            )

        from PIL import Image

        from .model import ProductInput

        pixels = None
        if data is not None:
            with Image.open(BytesIO(data)) as original:
                pixels = original.convert("RGB")
        prediction, = self._model.predict((ProductInput(
            request.sample_id, pixels, window, (image.width, image.height) if image else None,
        ),))
        status = {"trained": "trusted_trained", "untrained": "untrained", None: "unavailable"}
        return ProductInspection(
            **fields, branch=prediction.branch, sensor_status=prediction.sensor if window else sensor_status,
            threshold=prediction.threshold, score_kind=prediction.score_kind,
            weight_status=status[prediction.weight_status], model=self.identity, target_policy=prediction.target_policy,
            scores=tuple(ProductScoreRecord(name=row.name, class_id=row.class_id, score=row.score,
                                            threshold_crossed=row.threshold_crossed) for row in prediction.scores),
        )
