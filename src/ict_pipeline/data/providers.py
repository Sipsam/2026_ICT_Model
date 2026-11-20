"""AI Hub v0.6 transport and a network-free provider for fixtures."""

import os
import re
import shutil
import subprocess
import tarfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.client import HTTPException, parse_headers
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import unquote, urljoin, urlsplit

from pydantic import SecretStr

from .model import DataError, Progress, Split

BASE_URL = "https://api.aihub.or.kr"
BLOCK = 1024 * 1024


class RetryableDownload(DataError):
    """A failed transport attempt that can be started in a fresh file."""


class DownloadProvider(Protocol):
    def preflight(self) -> None: ...

    def download(self, dataset_key: int, split: Split, target: Path, progress: Progress) -> None: ...


def _transport_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name.upper() != "AIHUB_API_KEY"}


def _probe(command: Sequence[str], timeout: int = 8) -> bytes | None:
    try:
        result = subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            timeout=timeout, check=False, env=_transport_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout if result.returncode == 0 else None


def _curl_command(windows: bool, distribution: str | None) -> tuple[str, ...]:
    if windows:
        wsl = shutil.which("wsl.exe")
        if wsl:
            listed = _probe((wsl, "--list", "--quiet"))
            if listed:
                # wsl.exe emits UTF-16LE when output is redirected.
                encoding = "utf-16-le" if b"\x00" in listed else "utf-8"
                names = listed.decode(encoding, errors="replace").lstrip("\ufeff").splitlines()
                for name in map(str.strip, names):
                    if not name or (distribution and name != distribution):
                        continue
                    if distribution is None and name in {"docker-desktop", "docker-desktop-data"}:
                        continue
                    command = (wsl, "--distribution", name, "--exec", "curl")
                    if _probe((*command, "-q", "--version"), timeout=25) is not None:
                        return command
        raise DataError("AI Hub download on Windows requires a WSL Linux distribution with curl. "
                        "Select one with ICT_WSL_DISTRIBUTION or use existing data.")
    curl = shutil.which("curl")
    if curl and _probe((curl, "-q", "--version")) is not None:
        return (curl,)
    raise DataError("AI Hub download requires curl. Use existing data if it is unavailable.")


@contextmanager
def _response(
    command: tuple[str, ...], url: str, header: bytes | None, timeout: int,
) -> Iterator[tuple[subprocess.Popen[bytes], BinaryIO, int, Mapping[str, str]]]:
    args = (
        *command, "-q", "--silent", "--no-buffer", "--include", "--suppress-connect-headers",
        "--proto", "=https", "--connect-timeout", "15", "--max-time", str(timeout),
        "--speed-time", "60", "--speed-limit", "1", "--output", "-",
    )
    if header is not None:
        args += ("--header", "@-")
    try:
        process = subprocess.Popen(
            (*args, "--", url), stdin=subprocess.PIPE if header is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            env=_transport_env(),
        )
    except OSError:
        raise RetryableDownload("Cannot start the AI Hub curl transport.") from None
    try:
        if header is not None:
            assert process.stdin is not None
            process.stdin.write(header)
            process.stdin.close()
        assert process.stdout is not None
        stream = process.stdout
        while True:
            line = stream.readline(8192)
            match = re.match(rb"HTTP/[0-9.]+ ([0-9]{3})(?: |\r?\n)", line)
            if match is None:
                raise RetryableDownload("AI Hub returned no usable HTTP response. Check network connectivity.")
            status = int(match[1])
            headers = parse_headers(stream)
            if status >= 200:
                break
        yield process, stream, status, headers
    except (BrokenPipeError, HTTPException):
        raise RetryableDownload("AI Hub returned an interrupted or invalid HTTP response.") from None
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        if process.stdin is not None:
            process.stdin.close()
        if process.stdout is not None:
            process.stdout.close()


@dataclass
class AIHubProvider:
    key: SecretStr | None = field(repr=False)
    windows: bool = False
    distribution: str | None = None
    _command: tuple[str, ...] = field(default=(), init=False, repr=False)

    def _credential_header(self) -> bytes:
        key = self.key.get_secret_value() if self.key is not None else ""
        if not key:
            raise DataError("Set AIHUB_API_KEY in the environment or .env.")
        if len(key) > 1024 or any(not 33 <= ord(char) <= 126 for char in key):
            raise DataError("AIHUB_API_KEY contains invalid header characters.")
        return f"apikey: {key}\n".encode("ascii")

    def preflight(self) -> None:
        self._credential_header()
        self._command = _curl_command(self.windows, self.distribution)
        with _response(self._command, f"{BASE_URL}/info/api.do", None, 15) as (_, _, status, _):
            if status != 200:
                raise DataError(f"AI Hub network preflight returned HTTP {status}. Check service access.")

    def download(self, dataset_key: int, split: Split, target: Path, progress: Progress) -> None:
        header = self._credential_header()
        query = ",".join(map(str, split.file_keys))
        url = f"{BASE_URL}/down/0.6/{dataset_key}.do?fileSn={query}"
        for _ in range(4):
            with _response(self._command, url, header, 6 * 3600) as (process, stream, status, headers):
                if status in {301, 302, 303, 307, 308}:
                    try:
                        url = urljoin(url, headers.get("Location", ""))
                        origin = urlsplit(url)
                    except ValueError:
                        raise DataError("AI Hub returned an invalid redirect.") from None
                    if header.removeprefix(b"apikey: ").strip().decode("ascii") in unquote(url):
                        raise DataError("AI Hub returned an unsafe redirect.")
                    if not (
                        origin.scheme == "https"
                        and (
                            origin.netloc == "api.aihub.or.kr"
                            or (
                                origin.netloc == "stream.aihub.or.kr"
                                and origin.path == "/shellStream.do"
                            )
                        )
                    ):
                        raise DataError("AI Hub redirected outside its HTTPS API origin.")
                    continue
                if status in {401, 403}:
                    raise DataError(f"AI Hub rejected the credential or dataset access (HTTP {status}). "
                                    "Check the key and approval for dataset 71900.")
                if status in {408, 429} or 500 <= status < 600:
                    raise RetryableDownload(f"AI Hub download returned HTTP {status}.")
                if status != 200:
                    raise DataError(f"AI Hub download returned unexpected HTTP {status}.")
                length = headers.get("Content-Length")
                total = int(length) if length and re.fullmatch(r"[0-9]{1,18}", length) else None
                if total is not None and total > split.max_download_bytes:
                    raise DataError("Provider archive exceeds the configured split download limit.")
                # Reject error documents before opening a partial archive.
                first = stream.read(tarfile.BLOCKSIZE)
                try:
                    tarfile.TarInfo.frombuf(first, "utf-8", "surrogateescape")
                except tarfile.HeaderError:
                    if len(first) < tarfile.BLOCKSIZE:
                        raise RetryableDownload("AI Hub archive ended before its first TAR header.") from None
                    raise DataError("AI Hub responded without a valid TAR archive.") from None
                done = 0
                with target.open("xb") as output:
                    chunk = first
                    while chunk:
                        done += len(chunk)
                        if done > split.max_download_bytes or (total is not None and done > total):
                            raise DataError("AI Hub transfer exceeded its declared size or split download limit.")
                        output.write(chunk)
                        progress(done, total)
                        chunk = stream.read(BLOCK)
                if process.wait() != 0 or (total is not None and done != total):
                    raise RetryableDownload("AI Hub transfer ended early. Partial archive retained.")
                return
        raise DataError("AI Hub returned too many redirects.")


@dataclass
class MockProvider:
    """Serve local TAR fixtures and optionally simulate interrupted transfers."""

    bundles: Mapping[str, Path]
    failures_remaining: int = 0
    requests: list[tuple[int, str]] = field(default_factory=list)

    def preflight(self) -> None:
        pass

    def download(self, dataset_key: int, split: Split, target: Path, progress: Progress) -> None:
        self.requests.append((dataset_key, split.name))
        source = self.bundles[split.name]
        total = source.stat().st_size
        if total > split.max_download_bytes:
            raise DataError("Mock archive exceeds the configured split download limit.")
        with source.open("rb") as stream, target.open("xb") as output:
            if self.failures_remaining:
                self.failures_remaining -= 1
                output.write(stream.read(512))
                raise RetryableDownload("Mock transfer interrupted. Partial archive retained.")
            for chunk in iter(lambda: stream.read(BLOCK), b""):
                output.write(chunk)
                progress(output.tell(), total)
