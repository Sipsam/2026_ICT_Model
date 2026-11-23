"""Load one local model and inspect images through the shared predictors."""

import hashlib
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import BytesIO, StringIO
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from threading import Event, Lock
from typing import Literal, Protocol, cast

from ..config.settings import Credentials, SetupError, is_filesystem_path
from ..data.preprocessing.policy import TARGET_IDS
from ..evaluation.adapters import PredictorFactory, load_predictor
from ..evaluation.contracts import CLASSES, EvaluationConfig, EvaluationError, Prediction, Preprocessing
from ..evaluation.metrics import box_iou
from ..evaluation.runner import inspect_checkpoint
from ..models.contracts import ModelName
from ..models.dataset import Sample
from ..multimodal.inference import MultimodalInference, ProductInputError
from ..multimodal.inference_contracts import MockProductModel, ProductInspectionInput, ProductModelSpec
from ..pipeline.store import fingerprint_path
from .contracts import (
    MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, CheckpointKind, Detection, Device, Failure, ImageSource, InspectionError,
    InspectionInput, InspectionInterrupted,
    InspectionMetadataInput, InspectionRequest,
    InspectionResult, InspectionThresholds, MockScenario, ModelProvenance, TransientInspectionError,
)

SOURCE_IDS = {name: category for category, name in TARGET_IDS.items()}
# The shared loader and adapter diagnostics use process-wide streams.
_BACKEND_LOCK = Lock()


@contextmanager
def _use_backend(cancelled: Event | None) -> Iterator[None]:
    if cancelled is None:
        with _BACKEND_LOCK:
            yield
        return
    while not cancelled.is_set():
        if _BACKEND_LOCK.acquire(timeout=0.05):
            try:
                if cancelled.is_set():
                    break
                yield
                return
            finally:
                _BACKEND_LOCK.release()
    raise InspectionInterrupted("Inspection was cancelled before inference.")


class ImageError(Exception):
    def __init__(self, reason: Failure) -> None:
        self.reason = reason


class InspectionPredictor(Protocol):
    def predict(self, sample: Sample) -> Prediction: ...

    def synchronize(self) -> None: ...


class MockPredictor:
    """Deterministic simulated responses for exercising the inspection interface."""

    def __init__(self, scenario: MockScenario) -> None:
        self.scenario = scenario

    def predict(self, sample: Sample) -> Prediction:
        width, height = sample.width, sample.height
        size = (width, height)
        preprocessing = Preprocessing(size, size, "stretch")
        if self.scenario == "empty":
            return Prediction(sample.image.stem, (), (), (), preprocessing)
        box = (width * 0.25, height * 0.25, width * 0.75, height * 0.75)
        return Prediction(sample.image.stem, (box,), (1,), (0.95,), preprocessing)

    def synchronize(self) -> None:
        pass


def local_path(path: Path, base: Path, credentials: Credentials) -> Path:
    value = str(path)
    # Path construction collapses the slashes of URL schemes on some platforms.
    if not is_filesystem_path(value) or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]+:[/\\]", value):
        raise InspectionError("Inspection requires local filesystem paths.")
    credentials.protect_path(value, "inspection path")
    resolved = (base / path).resolve()
    credentials.protect_path(str(resolved), "resolved inspection path")
    return resolved


def _decode_image(
    data: bytes, *, expected_format: Literal["JPEG", "PNG"] | None = None,
    max_width: int = MAX_IMAGE_PIXELS, max_height: int = MAX_IMAGE_PIXELS,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> ImageSource:
    try:
        from PIL import Image
    except ImportError:
        raise InspectionError("Inspection requires Pillow from the existing preprocess or detector extra.") from None

    try:
        with Image.open(BytesIO(data), formats=("JPEG", "PNG")) as image:
            width, height = image.size
            if (not 0 < width <= max_width or not 0 < height <= max_height
                    or width * height > min(MAX_IMAGE_PIXELS, max_pixels)
                    or getattr(image, "is_animated", False)
                    or (expected_format is not None and image.format != expected_format)):
                raise ImageError("invalid_image")
            image_format = cast(Literal["JPEG", "PNG"], image.format)
            image.verify()
        with Image.open(BytesIO(data), formats=("JPEG", "PNG")) as image:
            image.load()
        return ImageSource(width=width, height=height, format=image_format, sha256=hashlib.sha256(data).hexdigest())
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageError("invalid_image") from None


def _read_image(path: Path) -> tuple[ImageSource, bytes]:
    try:
        if not path.exists():
            raise ImageError("missing_image")
        if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
            raise ImageError("invalid_image")
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE_BYTES + 1)
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageError("invalid_image")
        return _decode_image(data), data
    except FileNotFoundError:
        raise ImageError("missing_image") from None
    except (OSError, ValueError):
        raise ImageError("invalid_image") from None


def _detections(
    prediction: Prediction, sample: Sample, thresholds: InspectionThresholds,
) -> tuple[Detection, ...]:
    if (prediction.sample_id != sample.image.stem
            or prediction.preprocessing.original_size != (sample.width, sample.height)):
        raise ValueError("The prediction does not match the source image.")
    candidates = []
    for box, category, confidence in zip(
        prediction.boxes, prediction.class_ids, prediction.confidence_scores, strict=True,
    ):
        name = CLASSES[category]
        item = Detection(
            class_id=category, class_name=name, source_category_id=SOURCE_IDS[name],
            bbox_xyxy=box, confidence=confidence,
        )
        if box[2] > sample.width or box[3] > sample.height:
            raise ValueError("The prediction exceeds the source image bounds.")
        if confidence >= thresholds.confidence:
            candidates.append(item)

    retained: list[Detection] = []
    by_class: dict[int, list[Detection]] = {}
    for item in sorted(candidates, key=lambda row: (-row.confidence, row.class_id, row.bbox_xyxy)):
        same_class = by_class.setdefault(item.class_id, [])
        if all(box_iou(item.bbox_xyxy, other.bbox_xyxy) <= thresholds.nms_iou for other in same_class):
            same_class.append(item)
            retained.append(item)
    return tuple(retained)


class InspectionService:
    def __init__(
        self, predictor: InspectionPredictor | None, provenance: ModelProvenance,
        thresholds: InspectionThresholds, credentials: Credentials, *,
        product: MultimodalInference | None = None,
    ) -> None:
        self._predictor = predictor
        self.provenance = provenance
        self.thresholds = thresholds
        self.credentials = credentials
        self._product = product

    @classmethod
    def from_product(
        cls, spec: ProductModelSpec, root: Path, *, credentials: Credentials,
        detector: "InspectionService | None" = None, scratch_root: Path | None = None,
    ) -> "InspectionService":
        simulated = isinstance(spec, MockProductModel)
        if detector is not None and (simulated or detector.provenance.model is None):
            raise InspectionError("Product checkpoint inference may use a trusted detector for spatial evidence.")
        # The Faster R-CNN feature encoder copies its transform and backbone. Other detectors
        # remain independent sources of spatial evidence.
        fpn = (getattr(detector._predictor, "model", None)
               if detector is not None and detector.provenance.model == ModelName.FASTER_RCNN else None)
        with _BACKEND_LOCK, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            product = MultimodalInference.load(spec, root, credentials=credentials, detector=fpn,
                                               scratch_root=scratch_root)
        return cls(
            detector._predictor if detector is not None else None,
            detector.provenance if detector is not None else (
                ModelProvenance(mode="mock", mock_scenario="empty") if simulated
                else ModelProvenance(mode="real", product_only=True)
            ), detector.thresholds if detector is not None else InspectionThresholds(), credentials, product=product,
        )

    @classmethod
    def mock(
        cls, scenario: MockScenario = "empty", *, thresholds: InspectionThresholds = InspectionThresholds(),
        credentials: Credentials | None = None,
    ) -> "InspectionService":
        return cls(MockPredictor(scenario), ModelProvenance(mode="mock", mock_scenario=scenario),
                   thresholds, credentials or Credentials())

    @classmethod
    def from_run(
        cls, run: Path, *, trust_run: bool, thresholds: InspectionThresholds = InspectionThresholds(),
        checkpoint_kind: CheckpointKind = "best", expected_fingerprint: str | None = None,
        device: Device = "cpu", image_size: int | None = None, credentials: Credentials | None = None,
        factory: PredictorFactory | None = None,
        expected_model: ModelName | None = None, scratch_root: Path | None = None,
    ) -> "InspectionService":
        if trust_run is not True:
            raise InspectionError("Confirm the completed model directory comes from a trusted source.")
        credentials = credentials or Credentials()
        try:
            run = local_path(run, Path.cwd(), credentials)
            metadata = run / "run.json"
            metadata_fingerprint = fingerprint_path(metadata)
            checkpoint = inspect_checkpoint(run, checkpoint_kind, credentials)
            if expected_model is not None and checkpoint.name != expected_model:
                raise InspectionError("Checkpoint model differs from the configured model.")
            if expected_fingerprint is not None and checkpoint.fingerprint != expected_fingerprint:
                raise InspectionError("Checkpoint fingerprint differs from the supplied fingerprint.")
            config = EvaluationConfig(
                image_size=checkpoint.config.image_size if image_size is None else image_size,
                score_floor=min(thresholds.confidence, 0.999), confidence=thresholds.confidence,
                nms_iou=thresholds.nms_iou, warmup=0, device=device,
            )
            with TemporaryDirectory(prefix="ict-inspect-model-", dir=scratch_root) as directory:
                scratch = Path(directory)
                snapshot = scratch / checkpoint.path.name
                copyfile(checkpoint.path, snapshot)
                if (fingerprint_path(snapshot) != checkpoint.fingerprint
                        or fingerprint_path(metadata) != metadata_fingerprint):
                    raise InspectionError("The model run changed while preparing the checkpoint.")
                with _BACKEND_LOCK:
                    predictor = (factory or load_predictor)(checkpoint.name, snapshot, checkpoint.config, config, scratch)
            provenance = ModelProvenance(
                mode="real", model=checkpoint.name, checkpoint_kind=checkpoint_kind,
                checkpoint_fingerprint=checkpoint.fingerprint, run_metadata_fingerprint=metadata_fingerprint,
                device=predictor.device, image_size=config.image_size,
                libraries=cast(dict[str, str], credentials.redact(predictor.libraries)),
            )
        except (EvaluationError, SetupError) as exc:
            raise InspectionError(credentials.redact_text(str(exc))) from None
        except (OSError, ValueError, RuntimeError, AttributeError, TypeError):
            raise InspectionError("Cannot initialize inspection. Check run metadata, checkpoint, runtime and options.") from None
        return cls(predictor, provenance, thresholds, credentials)

    @classmethod
    def from_finalized(
        cls, directory: Path, *, trust_finalization: bool,
        thresholds: InspectionThresholds = InspectionThresholds(),
        device: Device = "cpu", credentials: Credentials | None = None,
        factory: PredictorFactory | None = None,
        expected_model: ModelName | None = None, scratch_root: Path | None = None,
    ) -> "InspectionService":
        if trust_finalization is not True:
            raise InspectionError("Confirm the finalized detector directory comes from a trusted source.")
        credentials = credentials or Credentials()
        try:
            directory = local_path(directory, Path.cwd(), credentials)
            from ..data.detector_finalization import inspect_finalized_detector
            inspected = inspect_finalized_detector(directory, credentials)
            if expected_model is not None and inspected["name"] != expected_model:
                raise InspectionError("Finalized checkpoint model differs from the configured model.")
            config = EvaluationConfig(
                image_size=inspected["image_size"], score_floor=min(thresholds.confidence, 0.999),
                confidence=thresholds.confidence, nms_iou=thresholds.nms_iou, warmup=0, device=device,
            )
            with TemporaryDirectory(prefix="ict-inspect-finalized-", dir=scratch_root) as temp:
                scratch = Path(temp)
                snapshot = scratch / "best.pt"
                copyfile(inspected["path"], snapshot)
                if (fingerprint_path(snapshot) != inspected["fingerprint"]
                        or fingerprint_path(directory / "finalization.json") != inspected["receipt_fingerprint"]):
                    raise InspectionError("The finalized model changed while preparing the checkpoint.")
                with _BACKEND_LOCK:
                    predictor = (factory or load_predictor)(
                        inspected["name"], snapshot, inspected["training"], config, scratch,
                    )
            provenance = ModelProvenance(
                mode="real", model=inspected["name"], checkpoint_kind="best",
                checkpoint_fingerprint=inspected["fingerprint"],
                run_metadata_fingerprint=inspected["metadata_fingerprint"],
                finalization_fingerprint=inspected["metadata_fingerprint"],
                device=predictor.device, image_size=config.image_size,
                libraries=cast(dict[str, str], credentials.redact(predictor.libraries)),
            )
        except InspectionError:
            raise
        except (EvaluationError, SetupError, ValueError) as exc:
            raise InspectionError(credentials.redact_text(str(exc))) from None
        except (OSError, RuntimeError, AttributeError, TypeError):
            raise InspectionError("Cannot initialize finalized inspection. Check finalization metadata, checkpoint, runtime and options.") from None
        return cls(predictor, provenance, thresholds, credentials)

    def _context(self, request: InspectionInput | None, input_line: int | None) -> dict[str, object]:
        context: dict[str, object] = {
            "provenance": self.provenance, "thresholds": self.thresholds, "input_line": input_line,
        }
        if request is not None:
            context.update(
                request_id=self.credentials.redact_text(request.request_id),
                source_id=self.credentials.redact_text(request.source_id),
                sensor_source_id=self.credentials.redact_text(request.sensor.source_id) if request.sensor else None,
            )
            if isinstance(request, InspectionMetadataInput):
                context["metadata"] = self.credentials.redact(request.metadata)
        return context

    def invalid_request(self, input_line: int) -> InspectionResult:
        return InspectionResult(outcome="inference_failed", reason="invalid_request",
                                **self._context(None, input_line))

    def inspect(
        self, request: InspectionRequest, *, base: Path | None = None, input_line: int | None = None,
        reuse: Callable[[ImageSource], InspectionResult | None] | None = None,
    ) -> InspectionResult:
        """Run a request on a private copy of the validated bytes."""
        context = self._context(request, input_line)
        try:
            path = local_path(request.image_path, base or Path.cwd(), self.credentials)
        except (InspectionError, SetupError, OSError, ValueError, RuntimeError):
            return InspectionResult(outcome="inference_failed", reason="invalid_request", **context)
        try:
            image, data = _read_image(path)
        except ImageError as exc:
            return InspectionResult(outcome="inference_failed", reason=exc.reason, **context)

        return self._predict_image(request, image, data, context, reuse=reuse)

    def inspect_bytes(
        self, request: InspectionInput, data: bytes, *, image_format: Literal["JPEG", "PNG"],
        max_width: int, max_height: int, max_pixels: int, scratch_root: Path | None = None,
        propagate_transient: bool = False, cancelled: Event | None = None,
    ) -> InspectionResult:
        """Inspect uploaded bytes without accepting a caller-supplied file path."""
        context = self._context(request, None)
        try:
            if len(data) > MAX_IMAGE_BYTES:
                raise ImageError("invalid_image")
            image = _decode_image(data, expected_format=image_format, max_width=max_width,
                                  max_height=max_height, max_pixels=max_pixels)
        except ImageError as exc:
            return InspectionResult(outcome="inference_failed", reason=exc.reason, **context)
        return self._predict_image(request, image, data, context, scratch_root=scratch_root,
                                   propagate_transient=propagate_transient, cancelled=cancelled)

    def inspect_product(
        self, request: ProductInspectionInput, image_path: Path | None = None,
    ) -> InspectionResult:
        context = self._product_context(request)
        image = data = None
        if image_path is not None:
            try:
                image, data = _read_image(local_path(image_path, Path.cwd(), self.credentials))
            except ImageError as exc:
                return InspectionResult(outcome="inference_failed", reason=exc.reason, **context)
            except (InspectionError, SetupError, OSError, ValueError, RuntimeError):
                return InspectionResult(outcome="inference_failed", reason="invalid_request", **context)
        return self._predict_product(request, image, data, context)

    def inspect_product_bytes(
        self, request: ProductInspectionInput, data: bytes | None, *, image_format: Literal["JPEG", "PNG"] | None,
        max_width: int, max_height: int, max_pixels: int, scratch_root: Path | None = None,
        cancelled: Event | None = None,
    ) -> InspectionResult:
        context = self._product_context(request)
        image = None
        if data is not None:
            try:
                if len(data) > MAX_IMAGE_BYTES:
                    raise ImageError("invalid_image")
                image = _decode_image(data, expected_format=image_format, max_width=max_width,
                                      max_height=max_height, max_pixels=max_pixels)
            except ImageError as exc:
                return InspectionResult(outcome="inference_failed", reason=exc.reason, **context)
        return self._predict_product(request, image, data, context, scratch_root=scratch_root, cancelled=cancelled)

    def _product_context(self, request: ProductInspectionInput) -> dict[str, object]:
        context = self._context(request, None)
        if request.sensor_evidence is not None:
            context["sensor_source_id"] = self.credentials.redact_text(request.sensor_evidence.source_id)
        return context

    def _predict_product(
        self, request: ProductInspectionInput, image: ImageSource | None, data: bytes | None,
        context: dict[str, object], *, scratch_root: Path | None = None, cancelled: Event | None = None,
    ) -> InspectionResult:
        context["image"] = image
        if self._product is None:
            return InspectionResult(outcome="inference_failed", reason="invalid_request", **context)
        if image is not None and request.expected_size and (
            image.width, image.height
        ) != (request.expected_size.width, request.expected_size.height):
            return InspectionResult(outcome="inference_failed", reason="dimension_mismatch", **context)
        try:
            with _use_backend(cancelled), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                product = self._product.predict(request, image, data, self.credentials)
                detections = ()
                if cancelled is not None and cancelled.is_set():
                    raise InspectionInterrupted("Inspection was cancelled before detection.")
                if image is not None and data is not None and self._predictor is not None:
                    detections = self._run_detector(image, data, scratch_root)
        except ProductInputError:
            return InspectionResult(outcome="inference_failed", reason="invalid_request", **context)
        except InspectionInterrupted:
            if cancelled is not None and cancelled.is_set():
                raise
            return InspectionResult(outcome="inference_failed", reason="prediction_failed", **context)
        except Exception:
            return InspectionResult(outcome="inference_failed", reason="prediction_failed", **context)
        reason = ("simulated_prediction" if self.provenance.mode == "mock" else
                  "product_review" if product.branch else "product_unavailable")
        return InspectionResult(outcome="review_required", reason=reason, detections=detections,
                                multimodal=product, **context)

    def _run_detector(self, image: ImageSource, data: bytes, scratch_root: Path | None) -> tuple[Detection, ...]:
        assert self._predictor is not None
        with TemporaryDirectory(prefix="ict-inspect-image-", dir=scratch_root) as directory:
            extension = "jpg" if image.format == "JPEG" else "png"
            snapshot = Path(directory) / f"image.{extension}"
            snapshot.write_bytes(data)
            sample = Sample(snapshot, image.width, image.height, ())
            prediction = self._predictor.predict(sample)
            self._predictor.synchronize()
        return _detections(prediction, sample, self.thresholds)

    def _predict_image(
        self, request: InspectionInput, image: ImageSource, data: bytes, context: dict[str, object],
        *, reuse: Callable[[ImageSource], InspectionResult | None] | None = None,
        scratch_root: Path | None = None, propagate_transient: bool = False, cancelled: Event | None = None,
    ) -> InspectionResult:
        context["image"] = image
        if self._predictor is None:
            return InspectionResult(outcome="inference_failed", reason="invalid_request", **context)
        if request.expected_size and (image.width, image.height) != (request.expected_size.width, request.expected_size.height):
            return InspectionResult(outcome="inference_failed", reason="dimension_mismatch", **context)
        if reuse is not None:
            cached = reuse(image)
            if (cached is not None and cached.outcome in ("defect_detected", "review_required")
                    and all(getattr(cached, key) == value for key, value in context.items())):
                return cached
        try:
            with _use_backend(cancelled), redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                detections = self._run_detector(image, data, scratch_root)
        except InspectionInterrupted:
            if cancelled is not None and cancelled.is_set():
                raise
            return InspectionResult(outcome="inference_failed", reason="prediction_failed", **context)
        except TransientInspectionError:
            if propagate_transient:
                raise TransientInspectionError("Inspection backend requested a retry.") from None
            return InspectionResult(outcome="inference_failed", reason="prediction_failed", **context)
        except Exception:
            return InspectionResult(outcome="inference_failed", reason="prediction_failed", **context)

        if self.provenance.mode == "mock":
            return InspectionResult(outcome="review_required", reason="simulated_prediction",
                                    detections=detections, **context)
        if detections:
            return InspectionResult(outcome="defect_detected", reason="detections_retained",
                                    detections=detections, **context)
        return InspectionResult(outcome="review_required", reason="no_detections", **context)
