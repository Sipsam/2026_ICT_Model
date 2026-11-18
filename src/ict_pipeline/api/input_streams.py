"""Connect server-owned input configuration to the shared inspection service."""

from pathlib import Path
from typing import cast

from ..config.settings import Credentials
from ..inspection.service import local_path
from ..streaming.camera import OpenCVCamera
from ..streaming.processing import FrameProcessor, StreamRepository
from ..streaming.replay import DirectoryReplay, SensorReplay
from ..streaming.runner import StreamRunner, StreamStatus
from ..streaming.settings import DirectorySettings
from .errors import APIError
from .repository import InspectionRepository
from .services import ModelManager
from .settings import APISettings


def build_stream(
    settings: APISettings, models: ModelManager, repository: InspectionRepository,
    credentials: Credentials,
) -> StreamRunner | StreamStatus:
    stream = settings.input_stream
    if stream is None:
        return StreamStatus()
    labels = {
        "stream_id": credentials.redact_text(stream.stream_id),
        "model_id": credentials.redact_text(stream.model_id),
    }
    try:
        service = models.select(stream.model_id)
    except APIError:
        return StreamStatus(state="failed", error="model_unavailable", **labels)
    if not callable(getattr(repository, "read_record", None)):
        return StreamStatus(state="failed", error="storage_unavailable", **labels)
    try:
        camera_settings = stream.camera
        sensor_settings = stream.sensor
        base = Path.cwd()
        limit = settings.max_image_bytes
        if isinstance(camera_settings, DirectorySettings):
            directory = local_path(camera_settings.directory, base, credentials)
            manifest = (local_path(camera_settings.manifest, base, credentials)
                        if camera_settings.manifest is not None else None)
            camera = DirectoryReplay(camera_settings.model_copy(update={"directory": directory, "manifest": manifest}),
                                     max_image_bytes=limit)
        else:
            camera = OpenCVCamera(
                camera_settings, max_image_bytes=limit, max_width=settings.max_width,
                max_height=settings.max_height, max_pixels=settings.max_pixels,
            )
        sensor = None
        if sensor_settings is not None:
            path = local_path(sensor_settings.path, base, credentials)
            sensor = SensorReplay(sensor_settings.model_copy(update={"path": path}))
        processor = FrameProcessor(
            stream_id=stream.stream_id, model_id=stream.model_id, service=service,
            repository=cast(StreamRepository, repository), alignment=stream.alignment, settings=stream.processing,
            limits=(limit, settings.max_width, settings.max_height, settings.max_pixels),
        )
        return StreamRunner(camera, sensor, processor,
                            max_observations=sensor_settings.max_observations if sensor_settings else 10_000)
    except Exception:
        return StreamStatus(state="failed", error="configuration_error", **labels)
