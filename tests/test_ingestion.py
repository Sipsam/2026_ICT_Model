from __future__ import annotations

import io
import json
import re
import stat
import tarfile
from collections import namedtuple
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest
from pydantic import SecretStr
from typer.testing import CliRunner

from ict_pipeline import cli
from ict_pipeline.cli import app
from ict_pipeline.data import commands, ingestion, providers
from ict_pipeline.data.archives import Archive, IncompleteArchive, bundle_archives
from ict_pipeline.data.ingestion import Acquisition, ArchiveSet, ExtractProvider, publish_dataset, verify_dataset
from ict_pipeline.data.model import DataError, Progress, Selection, Split, load_selection
from ict_pipeline.data.providers import AIHubProvider, MockProvider, RetryableDownload
from ict_pipeline.data.stages import DataStages
from ict_pipeline.pipeline import StageRunner
from ict_pipeline.pipeline.stages import pipeline_stages

STEM = "SD_DEF_CC_A_20250807-135503_00011"
CONTENT = {".jpg": b"\xff\xd8\xfffixture", ".json": b'{"annotations":[]}', ".xlsx": b"PK\x03\x04fixture"}
KOREAN = {
    "train_images": "TS_이미지데이터_납땜공정_불량_2.냉납.zip",
    "train_annotations": "TL_납땜공정_불량_2.냉납.zip",
    "train_sensors": "TS_센서데이터_납땜공정_불량_2.냉납.zip",
}


@pytest.fixture
def selection() -> Selection:
    folders = {
        f"{split}_{kind}_soldering_defect_cold_solder": (extension,)
        for split in ("train", "validation")
        for kind, extension in (("images", ".jpg"), ("annotations", ".json"), ("sensors", ".xlsx"))
    }
    return Selection(
        71900, (Split("train", (123,), 65536), Split("validation", (456,), 65536)), folders,
        "a" * 64, Path("data/raw"), Path("data/extracted"),
    )


def progress(done: int, total: int | None) -> None:
    assert done >= 0
    assert total is None or total >= done


def for_split(_: str) -> Progress:
    return progress


def add_tar_file(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, io.BytesIO(data))


def make_bundles(root: Path, selection: Selection, *, missing_part: bool = False) -> dict[str, Path]:
    root.mkdir()
    bundles = {}
    for split in selection.splits:
        path = root / f"{split.name}.tar"
        bundles[split.name] = path
        with tarfile.open(path, "w", format=tarfile.PAX_FORMAT) as tar:
            for folder, (extension,) in selection.folders.items():
                if not folder.startswith(f"{split.name}_"):
                    continue
                prefix = folder.split("_soldering")[0]
                name = KOREAN[prefix] if prefix in KOREAN else f"{folder}.zip"
                leaf = ("/" if prefix in KOREAN else "") + STEM + extension
                with io.BytesIO() as data:
                    with ZipFile(data, "w") as source:
                        source.writestr(leaf, CONTENT[extension])
                    payload = data.getvalue()
                parent = "Training" if split.name == "train" else "Validation"
                part = 2 if missing_part else 1
                # Write out of order and split through a ZIP structure.
                add_tar_file(tar, f"dataset/{parent}/{name}.part{part}", payload[37:])
                add_tar_file(tar, f"dataset/{parent}/{name}.part0", payload[:37])
    return bundles


def make_existing(root: Path, selection: Selection) -> None:
    for folder, (extension,) in selection.folders.items():
        destination = root / folder
        destination.mkdir(parents=True)
        (destination / (STEM + extension)).write_bytes(CONTENT[extension])


def test_selection_matches_legacy_script() -> None:
    root = Path(__file__).resolve().parents[1]
    script = (root / "scripts/download_data.sh").read_text(encoding="utf-8")
    selection = load_selection()
    assert selection.dataset_key == 71900
    assert len(selection.folders) == 60
    for split, legacy in zip(selection.splits, ("training", "validation"), strict=True):
        block = script.split(f"{legacy})", 1)[1].split(";;", 1)[0]
        assert tuple(map(int, re.findall(r"\b[0-9]{6}\b", block))) == split.file_keys


def test_missing_key_precedes_transport_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers, "_curl_command", lambda *_: pytest.fail("transport invoked"))
    with pytest.raises(DataError, match="Set AIHUB_API_KEY"):
        AIHubProvider(None).preflight()
    with pytest.raises(DataError, match="invalid header"):
        AIHubProvider(SecretStr("a\r\ninjected: b")).preflight()


def test_network_failure_is_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "fixture-credential"
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    class FakeProcess:
        stdin = io.BytesIO()
        stdout = io.BytesIO(b"")
        returncode = 6

        def poll(self) -> int:
            return 6

        def wait(self) -> int:
            return 6

    def fake_popen(args: tuple[str, ...], **kwargs: object) -> FakeProcess:
        calls.append((args, kwargs))
        return FakeProcess()

    monkeypatch.setattr(providers, "_curl_command", lambda *_: ("curl",))
    monkeypatch.setattr(providers.subprocess, "Popen", fake_popen)
    monkeypatch.setenv("AIHUB_API_KEY", secret)
    with pytest.raises(RetryableDownload, match="network connectivity") as error:
        AIHubProvider(SecretStr(secret)).preflight()
    assert secret not in str(error.value)
    assert secret not in repr(calls)
    assert not list(tmp_path.iterdir())


def test_wire_download_passes_key_only_via_stdin(
    tmp_path: Path, selection: Selection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = make_bundles(tmp_path / "fixtures", selection)["train"].read_bytes()
    secret = "test-only-provider-secret"
    observed: dict[str, object] = {}

    class FakeProcess:
        stdout = io.BytesIO(
            f"HTTP/2 200\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload
        )

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    class RecordingInput(io.BytesIO):
        def close(self) -> None:
            if not self.closed:
                observed["stdin"] = self.getvalue()
            super().close()

    def fake_popen(args: tuple[str, ...], **kwargs: object) -> FakeProcess:
        observed["args"] = args
        observed["options"] = kwargs
        fake = FakeProcess()
        fake.stdin = RecordingInput()
        return fake

    monkeypatch.setenv("AIHUB_API_KEY", secret)
    monkeypatch.setattr(providers.subprocess, "Popen", fake_popen)
    provider = AIHubProvider(SecretStr(secret))
    provider._command = ("curl",)
    target = tmp_path / "train.partial"

    provider.download(71900, selection.splits[0], target, progress)

    assert target.read_bytes() == payload
    assert observed.pop("stdin") == f"apikey: {secret}\n".encode()
    assert secret not in repr(observed)
    assert "--header" in observed["args"]
    assert observed["args"][-1] == "https://api.aihub.or.kr/down/0.6/71900.do?fileSn=123"


def test_wsl_discovers_available_curl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(providers.shutil, "which", lambda _name: "wsl.exe")
    calls = []

    def probe(command: tuple[str, ...], timeout: int = 8) -> bytes | None:
        calls.append(command)
        if "--list" in command:
            return "docker-desktop\r\nUbuntu\r\n".encode("utf-16-le")
        return b"curl fixture" if "Ubuntu" in command else None

    monkeypatch.setattr(providers, "_probe", probe)
    assert providers._curl_command(True, None) == ("wsl.exe", "--distribution", "Ubuntu", "--exec", "curl")
    assert len(calls) == 2
    with pytest.raises(DataError, match="requires a WSL"):
        providers._curl_command(True, "missing")


def test_existing_dataset_skips_download(
    tmp_path: Path, selection: Selection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(commands, "load_selection", lambda: selection)
    monkeypatch.setattr(AIHubProvider, "preflight", lambda _: pytest.fail("network invoked"))
    monkeypatch.delenv("AIHUB_API_KEY", raising=False)
    make_existing(tmp_path / selection.extracted_root, selection)
    for name in ("status", "download", "extract", "verify"):
        result = CliRunner().invoke(app, ["data", name])
        assert result.exit_code == 0, result.output
        assert "train 1 samples" in result.stdout
        assert "validation 1 samples" in result.stdout
    assert not (tmp_path / selection.raw_root).exists()


def test_cli_preview_and_environment_precedence(
    tmp_path: Path, selection: Selection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(commands, "load_selection", lambda: selection)
    (tmp_path / ".env").write_text("AIHUB_API_KEY=dotenv-fixture\n", encoding="utf-8")
    monkeypatch.setenv("AIHUB_API_KEY", "environment-fixture")

    def denied(provider: AIHubProvider) -> None:
        assert provider.key == SecretStr("environment-fixture")
        raise DataError("Test provider preflight failure.")

    monkeypatch.setattr(AIHubProvider, "preflight", denied)
    preview = CliRunner().invoke(app, ["data", "download"])
    assert preview.exit_code == 0
    assert "Preview only" in preview.stdout
    assert not (tmp_path / selection.raw_root).exists()
    approved = CliRunner().invoke(app, ["data", "download", "--approve-download"])
    assert approved.exit_code == 1
    assert "Test provider preflight failure" in approved.stderr
    assert "dotenv-fixture" not in approved.output
    assert "environment-fixture" not in approved.output


def test_bundle_retry_resume_and_atomic_publish(
    tmp_path: Path, selection: Selection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(ingestion, "sleep", lambda _: None)
    bundles = make_bundles(tmp_path / "fixtures", selection)
    provider = MockProvider(bundles, failures_remaining=1)
    acquisition = Acquisition(selection, tmp_path / "raw")
    archives = acquisition.acquire(provider, for_split, approved=True)
    assert provider.requests == [(71900, "train"), (71900, "train"), (71900, "validation")]
    partials = list(acquisition.cache.rglob("*.partial"))
    assert len(partials) == 1 and partials[0].stat().st_size == 512
    assert acquisition.acquire(provider, for_split).files == archives.files
    assert len(provider.requests) == 3

    destination = tmp_path / "extracted"
    report = publish_dataset(tmp_path, destination, archives, selection, progress)
    assert report.samples == {"train": 1, "validation": 1}
    manifest = json.loads((destination / ingestion.MANIFEST).read_text(encoding="utf-8"))
    assert manifest["tree_sha256"] == report.fingerprint
    assert publish_dataset(tmp_path, destination, archives, selection, progress) == report
    assert all(path.exists() for path in archives.files)
    label = destination / "train_annotations_soldering_defect_cold_solder" / f"{STEM}.json"
    label.write_bytes(b'{"edited": true}')
    with pytest.raises(DataError, match="differs from its completion manifest"):
        verify_dataset(destination, selection)


def test_missing_part_and_partial_tar_are_rejected(
    tmp_path: Path, selection: Selection,
) -> None:
    bundles = make_bundles(tmp_path / "fixtures", selection, missing_part=True)
    with pytest.raises(IncompleteArchive, match="gap"):
        bundle_archives(bundles["train"], selection, "train")
    partial = tmp_path / "partial.tar"
    partial.write_bytes(bundles["train"].read_bytes()[:1600])
    with pytest.raises(IncompleteArchive, match="partial"):
        bundle_archives(partial, selection, "train")


def test_resume_rechecks_checkpoints_and_reuses_other_split(
    tmp_path: Path, selection: Selection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = make_bundles(tmp_path / "fixtures", selection)
    provider = MockProvider(bundles)
    acquisition = Acquisition(selection, tmp_path / "raw")
    download = provider.download

    def interrupted(dataset_key: int, split: Split, path: Path, reporter: Progress) -> None:
        if split.name == "validation":
            raise KeyboardInterrupt
        download(dataset_key, split, path, reporter)

    monkeypatch.setattr(provider, "download", interrupted)
    with pytest.raises(KeyboardInterrupt):
        acquisition.acquire(provider, for_split, approved=True)
    train_checkpoint, = acquisition.find().files
    monkeypatch.setattr(provider, "download", download)
    acquisition.acquire(provider, for_split, approved=True)
    assert provider.requests == [(71900, "train"), (71900, "validation")]

    train_checkpoint.write_bytes(b"corrupt checkpoint")
    recovered = acquisition.acquire(provider, for_split, approved=True)
    assert recovered.by_split["train"][0].path != train_checkpoint
    assert train_checkpoint.read_bytes() == b"corrupt checkpoint"
    assert provider.requests == [(71900, "train"), (71900, "validation"), (71900, "train")]


def test_unsafe_tar_path(tmp_path: Path, selection: Selection) -> None:
    path = tmp_path / "unsafe.tar"
    with tarfile.open(path, "w") as archive:
        add_tar_file(archive, "../escape.zip", b"test")
    with pytest.raises(DataError, match="Unsafe archive member path"):
        bundle_archives(path, selection, "train")


@pytest.mark.parametrize("name", ["../escape.jpg", "/escape.jpg", "C:/escape.jpg", "..\\escape.jpg",
                                  "//server/evil.jpg", "aux.jpg", "p.jpg:stream"])
def test_unsafe_zip_paths(
    name: str, tmp_path: Path, selection: Selection,
) -> None:
    path = tmp_path / "train_images_soldering_defect_cold_solder.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr(name, CONTENT[".jpg"])
    source = Archive(path, path.name)
    with source.open() as archive, pytest.raises(DataError, match="Unsafe archive member path"):
        source.members(archive, (".jpg",))


def test_zip_link_and_case_collision(tmp_path: Path) -> None:
    path = tmp_path / "selected.zip"
    info = ZipInfo("link.jpg")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(path, "w") as archive:
        archive.writestr(info, "../../outside")
    source = Archive(path, path.name)
    with source.open() as archive, pytest.raises(DataError, match="link"):
        source.members(archive, (".jpg",))
    with ZipFile(path, "w") as archive:
        archive.writestr("A.jpg", b"one")
        archive.writestr("a.JPG", b"two")
    with source.open() as archive, pytest.raises(DataError, match="case-colliding"):
        source.members(archive, (".jpg",))


def test_disk_and_permission_preflight(
    tmp_path: Path, selection: Selection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundles = make_bundles(tmp_path / "fixtures", selection)
    provider = MockProvider(bundles)
    volume = namedtuple("Volume", "total used free")
    monkeypatch.setattr(ingestion.shutil, "disk_usage", lambda _: volume(1024, 1023, 1))
    acquisition = Acquisition(selection, tmp_path / "raw")
    with pytest.raises(DataError, match="Insufficient disk space"):
        acquisition.acquire(provider, for_split, approved=True)
    assert provider.requests == []
    assert not acquisition.raw_root.exists()
    archives = ArchiveSet({split: bundle_archives(path, selection, split) for split, path in bundles.items()})
    with pytest.raises(DataError, match="Insufficient disk space"):
        ExtractProvider().extract(archives, tmp_path / "extracted", selection, progress)
    assert not (tmp_path / "extracted").exists()

    def no_write(**_: object) -> None:
        raise PermissionError("unsafe diagnostics that should be hidden")

    monkeypatch.setattr(ingestion.shutil, "disk_usage", lambda _: volume(1 << 24, 0, 1 << 24))
    monkeypatch.setattr(ingestion, "NamedTemporaryFile", no_write)
    with pytest.raises(DataError, match="permissions"):
        acquisition.acquire(provider, for_split, approved=True)


def test_runner_ingests_and_reuses_artifacts(tmp_path: Path, selection: Selection) -> None:
    provider = MockProvider(make_bundles(tmp_path / "fixtures", selection))
    acquisition = Acquisition(selection, tmp_path / "raw")
    stages = DataStages(acquisition, provider, approved=True)
    runner = StageRunner(pipeline_stages(stages), tmp_path, {"dataset_source": "aihub"})
    state = runner.run("ingestion", ("download", "extract"))
    extract = state.stages["extract"]
    output = tmp_path / "runs" / "ingestion" / "artifacts" / "extract" / extract.attempt / "dataset"
    assert verify_dataset(output, selection).samples == {"train": 1, "validation": 1}
    assert len(provider.requests) == 2
    assert runner.resume("ingestion").stages["extract"].attempt == extract.attempt
    assert len(provider.requests) == 2


def test_runner_reuses_existing_and_rechecks_changed_input(tmp_path: Path, selection: Selection) -> None:
    existing = tmp_path / "existing"
    make_existing(existing, selection)
    provider = MockProvider({})
    stages = DataStages(Acquisition(selection, tmp_path / "raw"), provider, existing)
    runner = StageRunner(pipeline_stages(stages), tmp_path, {"dataset_source": "existing"})
    state = runner.run("local", ("download", "extract"))
    sample = next(existing.rglob("*.json"))
    sample.write_bytes(b'{"updated": true}')
    updated = runner.resume("local")
    assert updated.stages["extract"].attempt != state.stages["extract"].attempt
    assert provider.requests == []


def test_run_command_uses_configured_existing_dataset(
    tmp_path: Path, selection: Selection, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "load_selection", lambda: selection)
    monkeypatch.setattr(AIHubProvider, "preflight", lambda _: pytest.fail("network invoked"))
    make_existing(tmp_path / "existing", selection)
    (tmp_path / "profile.yaml").write_text(
        "dataset_source: existing\ndata_path: existing\n", encoding="utf-8",
    )
    client = CliRunner()
    result = client.invoke(app, [
        "run", "--config", "profile.yaml", "--until", "extract", "--run-id", "configured",
    ])
    assert result.exit_code == 0, result.output
    assert "Run configured completed." in result.stdout
    result = client.invoke(app, ["resume", "--config", "profile.yaml", "--run-id", "configured"])
    assert result.exit_code == 0, result.output
    assert "extract reused" in result.stdout
