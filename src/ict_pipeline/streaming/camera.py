"""USB and RTSP capture in a killable OpenCV helper process."""

import multiprocessing
import os
import socket
import struct
from collections.abc import Iterator
from contextlib import redirect_stderr, redirect_stdout, suppress
from datetime import UTC, datetime
from threading import Event
from time import monotonic
from urllib.parse import urlsplit
from uuid import uuid4

from .contracts import CameraFrame, SourceIssue
from .settings import RTSPSettings, USBSettings


def _capture(
    channel: socket.socket, settings: USBSettings | RTSPSettings, limits: tuple[int, int, int, int],
) -> None:
    # OpenCV and its native video backends can print credential-bearing URLs.
    with channel, open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        url = ""
        if isinstance(settings, RTSPSettings):
            url = os.environ.get(settings.url_env, "")
            try:
                parsed = urlsplit(url)
                valid_url = parsed.scheme in {"rtsp", "rtsps"} and parsed.hostname and not any(c.isspace() for c in url)
            except ValueError:
                valid_url = False
            if not valid_url:
                channel.sendall(b"U")
                return
        try:
            import cv2
            cv2.setLogLevel(0)
            capture = cv2.VideoCapture()
        except (ImportError, AttributeError, OSError, RuntimeError):
            channel.sendall(b"D")
            return

        try:
            if isinstance(settings, RTSPSettings):
                timeout = int(settings.timeout_seconds * 1000)
                opened = capture.open(url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout,
                ])
            else:
                opened = capture.open(settings.device_index)
            if not opened:
                channel.sendall(b"X")
                return
            channel.sendall(b"O")
            while channel.recv(1) == b"R":
                ok, frame = capture.read()
                if not ok or frame is None:
                    channel.sendall(b"X")
                    return
                max_bytes, max_width, max_height, max_pixels = limits
                height, width = frame.shape[:2]
                if not (0 < width <= max_width and 0 < height <= max_height and width * height <= max_pixels):
                    channel.sendall(b"I")
                    continue
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok or not 0 < encoded.nbytes <= max_bytes:
                    channel.sendall(b"I")
                    continue
                channel.sendall(b"F" + struct.pack("!I", encoded.nbytes))
                channel.sendall(encoded.tobytes())
        except Exception:
            # Exit and let the parent classify EOF without reproducing diagnostics.
            return
        finally:
            with suppress(Exception):
                capture.release()


def _receive(channel: socket.socket, size: int, deadline: float, stop: Event) -> bytes:
    data = bytearray()
    while len(data) < size:
        remaining = deadline - monotonic()
        if stop.is_set():
            return b""
        if remaining <= 0:
            raise SourceIssue("timeout")
        channel.settimeout(min(0.1, remaining))
        try:
            chunk = channel.recv(size - len(data))
        except TimeoutError:
            continue
        if not chunk:
            raise SourceIssue("disconnected")
        data.extend(chunk)
    return bytes(data)


class OpenCVCamera:
    def __init__(
        self, settings: USBSettings | RTSPSettings, *, max_image_bytes: int, max_width: int,
        max_height: int, max_pixels: int,
    ) -> None:
        self.settings = settings
        self.source_id = settings.source_id
        self.kind = settings.kind
        self.limits = max_image_bytes, max_width, max_height, max_pixels

    def _session(self, stop: Event) -> Iterator[CameraFrame | SourceIssue]:
        context = multiprocessing.get_context("spawn")
        reader, writer = socket.socketpair()
        session = uuid4().hex
        with reader, writer:
            process = context.Process(target=_capture, args=(writer, self.settings, self.limits), daemon=True)
            try:
                process.start()
            except (OSError, RuntimeError):
                raise SourceIssue("source_unavailable") from None
            writer.close()
            try:
                tag = _receive(reader, 1, monotonic() + self.settings.timeout_seconds, stop)
                if stop.is_set():
                    return
                if tag == b"D":
                    raise SourceIssue("driver_unavailable")
                if tag == b"U":
                    raise SourceIssue("source_unavailable")
                if tag != b"O":
                    raise SourceIssue("disconnected")
                sequence = 0
                while not stop.is_set():
                    deadline = monotonic() + self.settings.timeout_seconds
                    reader.sendall(b"R")
                    tag = _receive(reader, 1, deadline, stop)
                    if stop.is_set():
                        return
                    if tag == b"I":
                        yield SourceIssue("invalid_frame")
                        continue
                    if tag != b"F":
                        raise SourceIssue("disconnected")
                    header = _receive(reader, 4, deadline, stop)
                    if stop.is_set():
                        return
                    size = struct.unpack("!I", header)[0]
                    if not 0 < size <= self.limits[0]:
                        raise SourceIssue("invalid_frame")
                    data = _receive(reader, size, deadline, stop)
                    if stop.is_set():
                        return
                    sequence += 1
                    yield CameraFrame(
                        source_id=self.source_id, sequence_id=f"{session}-{sequence}", kind=self.kind,
                        format="JPEG", data=data, timestamp=datetime.now(UTC).isoformat(),
                    )
            except OSError:
                raise SourceIssue("disconnected") from None
            finally:
                reader.close()
                process.join(0.15)
                if process.is_alive():
                    process.terminate()
                    process.join(0.25)
                if process.is_alive():
                    process.kill()
                    process.join(0.25)
                if not process.is_alive():
                    process.close()

    def frames(self, stop: Event) -> Iterator[CameraFrame | SourceIssue]:
        for attempt in range(self.settings.reconnect_attempts + 1):
            if stop.is_set():
                return
            try:
                yield from self._session(stop)
                return
            except SourceIssue as issue:
                if stop.is_set():
                    return
                if issue.code not in {"disconnected", "timeout"} or attempt == self.settings.reconnect_attempts:
                    raise
                yield issue
                if stop.wait(self.settings.reconnect_delay_seconds):
                    return
