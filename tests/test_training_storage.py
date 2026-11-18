import errno
from dataclasses import replace
from pathlib import Path

import pytest

from ict_pipeline import disk, training_storage
from ict_pipeline.config.settings import Credentials
from ict_pipeline.models import training
from ict_pipeline.models.contracts import Checkpoints, ModelName, TrainConfig, TrainingError, TrainingRequest
from ict_pipeline.models.dataset import DetectionData, Split
from ict_pipeline.models.yolo import YoloAdapter
from ict_pipeline.pipeline.model import PipelineError, Stage, StageContext, StageResult
from ict_pipeline.pipeline.runner import StageRunner
from ict_pipeline.pipeline.store import fingerprint_path
from ict_pipeline.training_storage import commit_checkpoint, storage_estimates


def test_estimates_count_periodic_copies_and_separate_cache_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "input"
    image = dataset / "yolo/images/train/fixture.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"data")
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("ICT_TRAIN_CACHE_ROOT", str(scratch))
    output = tmp_path / "output"
    plan = training.TrainingPlan(
        {ModelName.YOLO: TrainConfig(epochs=5, save_period=2)}, dataset, output,
        None, Credentials(), tmp_path, publish_checkpoints=True,
    )
    monkeypatch.setattr(training_storage, "volume_at", lambda path: disk.Volume(
        1 if path == output else 2, path, 5 * disk.MIB,
    ))
    runs, cache = storage_estimates(plan)
    assert runs.estimated_checkpoint_bytes == 8 * training_storage.CHECKPOINT_ALLOWANCE[ModelName.YOLO]
    assert runs.measured_dataset_copy_bytes == 0
    assert cache.measured_dataset_copy_bytes == 4
    assert cache.required_bytes == disk.RESERVE_BYTES + 4 + cache.estimated_cache_bytes
    before = fingerprint_path(dataset)
    monkeypatch.setenv("ICT_TRAIN_CACHE_ROOT", str(dataset / "unsafe"))
    with pytest.raises(TrainingError, match="outside protected or input"):
        storage_estimates(plan)
    assert fingerprint_path(dataset) == before


def test_low_disk_preflight_creates_no_attempt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "output"
    plan = training.TrainingPlan(
        {ModelName.RT_DETR: TrainConfig()}, tmp_path / "input", output, None, Credentials(), tmp_path,
    )
    monkeypatch.setattr(training_storage, "volume_at", lambda path: disk.Volume(1, path, 1))
    with pytest.raises(TrainingError, match="Measured free bytes: 1; estimated required bytes"):
        list(training.execute_training(plan, lambda *_: None))
    assert not output.exists()


def test_pipeline_low_disk_keeps_unpublished_work_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    full = False

    def finish(context: StageContext) -> StageResult:
        nonlocal full
        (context.output_dir / "receipt").write_text("finished")
        full = True
        return StageResult.SUCCESS

    monkeypatch.setattr(disk, "volume_at", lambda path: disk.Volume(
        1, path, disk.RESERVE_BYTES - 1 if full else 2 * disk.RESERVE_BYTES,
    ))
    stage = Stage("sample", (), (), (Path("receipt"),), (), "1", finish)
    runner = StageRunner((stage,), tmp_path, {})
    with pytest.raises(PipelineError, match="Low disk space for publishing pipeline stage sample"):
        runner.run("space", ("sample",))
    failed = runner.status("space")
    assert failed.status == "failed"
    assert not (tmp_path / "runs/space/artifacts").exists()
    partial = tmp_path / "runs/space/work/sample" / failed.stages["sample"].attempt / "receipt"
    assert partial.read_text() == "finished"
    monkeypatch.setattr(disk, "volume_at", lambda path: disk.Volume(1, path, 2 * disk.RESERVE_BYTES))
    complete = runner.resume("space")
    first = complete.stages["sample"].attempt
    assert first != failed.stages["sample"].attempt
    receipt = tmp_path / "runs/space/artifacts/sample" / first / "receipt"
    assert receipt.read_text() == "finished"


def test_external_yolo_cache_is_private_and_disposable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    image, label = tmp_path / "image.jpg", tmp_path / "image.txt"
    image.write_bytes(b"original")
    label.write_bytes(b"annotation")
    scratch, output = tmp_path / "cache", tmp_path / "model"
    output.mkdir()
    data = DetectionData({"train": Split((image,), (label,))}, {}, "")
    adapter = YoloAdapter()

    def run(request: TrainingRequest, directory: Path) -> Checkpoints:
        assert directory.parent == scratch
        copied = directory / "images/train/image.jpg"
        assert copied.read_bytes() == b"original"
        assert not copied.samefile(image)
        copied.write_bytes(b"backend rewrite")
        raise KeyboardInterrupt

    monkeypatch.setattr(adapter, "_train", run)
    with pytest.raises(KeyboardInterrupt):
        adapter.train(TrainingRequest(TrainConfig(), data, output, "cpu", lambda *_: None, cache_root=scratch))
    assert list(scratch.iterdir()) == []
    assert image.read_bytes() == b"original"


def test_snapshot_fallback_preserves_committed_checkpoints_on_low_space(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    last, best, periodic = (tmp_path / name for name in ("last.pt", "best.pt", "epoch-2.pt"))
    temporary = tmp_path / "last.tmp"
    temporary.write_bytes(b"first")
    commit_checkpoint(temporary, last, (best,))
    first = fingerprint_path(last)
    assert fingerprint_path(best) == first
    temporary.write_bytes(b"second")
    commit_checkpoint(temporary, last, ())
    assert fingerprint_path(best) == first
    second = fingerprint_path(last)
    assert second != first

    def unsupported(_self: Path, _source: Path) -> None:
        raise OSError(errno.EOPNOTSUPP, "fixture filesystem has no hard links")

    monkeypatch.setattr(Path, "hardlink_to", unsupported)
    monkeypatch.setattr(disk, "volume_at", lambda path: disk.Volume(1, path, disk.RESERVE_BYTES))
    temporary.write_bytes(b"third")
    with pytest.raises(TrainingError, match="Low disk space"):
        commit_checkpoint(temporary, last, (best, periodic))
    assert fingerprint_path(last) == second and fingerprint_path(best) == first
    assert not temporary.exists() and not periodic.exists()
    assert not list(tmp_path.glob("*.snapshot-partial"))

    volume = disk.Volume(1, tmp_path, 2 * disk.RESERVE_BYTES)
    monkeypatch.setattr(disk, "volume_at", lambda path: replace(volume, path=path))
    temporary.write_bytes(b"retry")
    commit_checkpoint(temporary, last, (best, periodic))
    assert last.read_bytes() == best.read_bytes() == periodic.read_bytes() == b"retry"
    assert not last.samefile(best)
