"""Model lifecycle, bounded inference, and inspection persistence."""

import asyncio
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Literal
from uuid import UUID, uuid4

from starlette.concurrency import run_in_threadpool

from ..config.settings import Credentials
from ..evaluation.adapters import PredictorFactory
from ..inspection.contracts import InspectionInput, InspectionInterrupted
from ..inspection.service import InspectionService
from ..multimodal.inference_contracts import MockProductModel, ProductInspectionInput
from .errors import APIError
from .repository import InspectionRepository
from .schemas import AvailableModel, BatchItem, Health, StoredInspection
from .settings import APISettings, FinalizedModel, MockModel, ProductModel, TrustedModel

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(
        self, settings: APISettings, scratch: Path, credentials: Credentials, factory: PredictorFactory | None,
    ) -> None:
        self._services: dict[str, InspectionService] = {}
        catalog = []
        root = settings.model_root.resolve()
        for model_id, spec in settings.models.items():
            product = isinstance(spec, ProductModel)
            detector_model = None
            mode: Literal["real", "mock", "finalized"] = "real"
            mock_scenario = None
            if isinstance(spec, MockModel):
                mode, mock_scenario = "mock", spec.scenario
            elif isinstance(spec, TrustedModel):
                detector_model = spec.model
            elif isinstance(spec, FinalizedModel):
                mode, detector_model = "finalized", spec.model
            elif isinstance(spec.product, MockProductModel):
                mode, mock_scenario = "mock", "empty"
            elif spec.detector is not None:
                detector_model = spec.detector.model
            try:
                if isinstance(spec, MockModel):
                    service = InspectionService.mock(spec.scenario, thresholds=spec.thresholds, credentials=credentials)
                elif isinstance(spec, ProductModel):
                    detector = (self._load_detector(root, spec.detector, scratch, credentials, factory)
                                if spec.detector is not None else None)
                    service = InspectionService.from_product(
                        spec.product, root, credentials=credentials, detector=detector, scratch_root=scratch,
                    )
                else:
                    service = self._load_detector(root, spec, scratch, credentials, factory)
                self._services[model_id] = service
            except Exception:
                logger.warning('{"event":"inspection_model_initialization_failed"}')
            catalog.append(AvailableModel(
                model_id=model_id, mode=mode, available=model_id in self._services,
                model=detector_model, mock_scenario=mock_scenario, input_kind="multimodal" if product else None,
            ))
        self.catalog = tuple(catalog)
        self._configured = settings.models

    @staticmethod
    def _load_detector(
        root: Path, spec: TrustedModel | FinalizedModel, scratch: Path, credentials: Credentials,
        factory: PredictorFactory | None,
    ) -> InspectionService:
        run = (root / spec.run).resolve()
        if not run.is_relative_to(root):
            raise ValueError("Invalid model run location.")
        if isinstance(spec, FinalizedModel):
            if not (run / "finalization.json").resolve().is_relative_to(run):
                raise ValueError("Invalid finalized model location.")
            return InspectionService.from_finalized(
                run, trust_finalization=spec.trust_finalization, expected_model=spec.model,
                device=spec.device, thresholds=spec.thresholds, scratch_root=scratch,
                credentials=credentials, factory=factory,
            )
        if not (run / "run.json").resolve().is_relative_to(run):
            raise ValueError("Invalid model run location.")
        return InspectionService.from_run(
            run, trust_run=spec.trust_run, expected_model=spec.model, checkpoint_kind=spec.checkpoint,
            expected_fingerprint=spec.expected_fingerprint, device=spec.device, image_size=spec.image_size,
            thresholds=spec.thresholds, scratch_root=scratch, credentials=credentials, factory=factory,
        )

    def select(self, model_id: str, *, product: bool = False) -> InspectionService:
        if model_id not in self._configured:
            raise APIError("model_not_found")
        if model_id not in self._services:
            raise APIError("model_unavailable")
        if product != isinstance(self._configured[model_id], ProductModel):
            raise APIError("invalid_request")
        return self._services[model_id]

    def health(self) -> Health:
        if not self._services:
            return Health(status="unavailable")
        return Health(status="ok" if len(self._services) == len(self.catalog) else "degraded")

    def close(self) -> None:
        self._services.clear()


@dataclass(frozen=True)
class Upload:
    details: InspectionInput
    data: bytes | None
    format: Literal["JPEG", "PNG"] | None


class InspectionAPI:
    def __init__(
        self, settings: APISettings, models: ModelManager, repository: InspectionRepository, scratch: Path,
    ) -> None:
        self.settings = settings
        self.models = models
        self.repository = repository
        self.scratch = scratch
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ict-inspect")
        self._active: Future[list[StoredInspection | APIError]] | None = None

    def _inspect(
        self, service: InspectionService, model_id: str, items: list[Upload | APIError], cancelled: Event,
    ) -> list[StoredInspection | APIError]:
        results: list[StoredInspection | APIError] = []
        for upload in items:
            if cancelled.is_set():
                break
            if isinstance(upload, APIError):
                results.append(upload)
                continue
            try:
                if isinstance(upload.details, ProductInspectionInput):
                    result = service.inspect_product_bytes(
                        upload.details, upload.data, image_format=upload.format, max_width=self.settings.max_width,
                        max_height=self.settings.max_height, max_pixels=self.settings.max_pixels,
                        scratch_root=self.scratch, cancelled=cancelled,
                    )
                else:
                    assert upload.data is not None and upload.format is not None
                    result = service.inspect_bytes(
                        upload.details, upload.data, image_format=upload.format, max_width=self.settings.max_width,
                        max_height=self.settings.max_height, max_pixels=self.settings.max_pixels,
                        scratch_root=self.scratch, cancelled=cancelled,
                    )
            except InspectionInterrupted:
                break
            if result.outcome == "inference_failed":
                results.append(APIError(result.reason))
            else:
                results.append(StoredInspection(
                    inspection_id=uuid4(), created_at=datetime.now(UTC), model_id=model_id, result=result,
                ))
        return results

    async def inspect(
        self, model_id: str, items: list[Upload | APIError], *, product: bool = False,
    ) -> tuple[BatchItem, ...]:
        service = self.models.select(model_id, product=product) if product else self.models.select(model_id)
        if self._active is not None and not self._active.done():
            raise APIError("busy")
        cancelled = Event()
        self._active = self._executor.submit(self._inspect, service, model_id, items, cancelled)
        pending = asyncio.wrap_future(self._active)
        # Retrieve late exceptions even if the HTTP request has already returned.
        pending.add_done_callback(_observe_inference)
        try:
            results = await asyncio.wait_for(asyncio.shield(pending), self.settings.inference_timeout_seconds)
        except TimeoutError:
            raise APIError("inference_timeout") from None
        finally:
            cancelled.set()
        return await run_in_threadpool(self._persist, results)

    def _persist(self, results: list[StoredInspection | APIError]) -> tuple[BatchItem, ...]:
        records = []
        for index, result in enumerate(results):
            if isinstance(result, StoredInspection):
                try:
                    self.repository.add(result)
                except (OSError, ValueError):
                    result = APIError("storage_unavailable")
            if isinstance(result, APIError):
                records.append(BatchItem(index=index, status=result.status, error=result.info))
            else:
                records.append(BatchItem(index=index, status=201, inspection=result))
        return tuple(records)

    async def get(self, inspection_id: UUID) -> StoredInspection:
        try:
            record = await run_in_threadpool(self.repository.get, inspection_id)
        except (OSError, ValueError):
            raise APIError("storage_unavailable") from None
        if record is None:
            raise APIError("inspection_not_found")
        return record

    async def close(self) -> None:
        await run_in_threadpool(self._executor.shutdown)
        self._active = None
        self.models.close()


def _observe_inference(future: asyncio.Future[list[StoredInspection | APIError]]) -> None:
    if not future.cancelled():
        future.exception()
