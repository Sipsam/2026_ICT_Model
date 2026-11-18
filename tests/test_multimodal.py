import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline.classification.sensors import SampleIssue
from ict_pipeline.cli import app
from ict_pipeline.data.preprocessing.audit import XLSX_FIELDS
from ict_pipeline.data.preprocessing.common import CsvRow, write_csv, write_json
from ict_pipeline.data.preprocessing.policy import SENSOR_FIELDS, label_policy
from ict_pipeline.inspection.contracts import SensorInput
from ict_pipeline.models.optimization.config import Augmentation, Geometric
from ict_pipeline.multimodal.audit import Cohort, audit_cohort, audit_report
from ict_pipeline.multimodal.contracts import SensorSequence, sensor_step
from ict_pipeline.multimodal.dataset import MultimodalDataset, collate
from ict_pipeline.multimodal.features import FEATURE_NAMES, statistical_features
from ict_pipeline.multimodal.preprocessing import (
    PreprocessingConfig, PreprocessingState, fit_preprocessor, training_fingerprint, transform_sensor,
)
from ict_pipeline.pipeline.store import digest_json
from ict_pipeline.streaming.contracts import SensorObservation

START = datetime(2025, 8, 7, 9, 3, 40)


def _workbook(path: Path, steps: list[dict]) -> None:
    workbook = Workbook()
    workbook.active.append(("timestamp", *XLSX_FIELDS))
    for step in steps:
        workbook.active.append((step["timestamp"], *(step[key] for key in SENSOR_FIELDS)))
    workbook.save(path)
    workbook.close()


def _sample(
    root: Path, index: int, *, split: str = "train", source: str = "A", capture: int | None = None,
    values: tuple[float | None, ...] = (0, 4, 8), image: bool = True, workbook: bool = True,
    session: str | None = None, normal: bool = False,
) -> CsvRow:
    instant = START + timedelta(minutes=index if capture is None else capture)
    status = "normal" if normal else "defect"
    stem = f"SD_{'NOR' if normal else 'DEF'}_CC_{source}_{instant:%Y%m%d-%H%M%S}_{index:05}"
    row = {
        "sample_id": stem, "split": split, "source_class": "CC", "manufacturer": source,
        "decision": "candidate", "use_for_detection": "candidate",
        **{f"{key}_path": f"{split}_{kind}_soldering_{status}_cold_solder/{stem}{ext}"
           for key, kind, ext in (("annotation", "annotations", ".json"), ("image", "images", ".JPG"),
                                  ("sensor", "sensors", ".xlsx"))},
    }
    for key in ("annotation_path", "image_path", "sensor_path"):
        (root / row[key]).parent.mkdir(parents=True, exist_ok=True)
    steps = [
        {"timestamp": (instant + timedelta(seconds=time)).isoformat(sep=" "),
         **dict(zip(SENSOR_FIELDS, (value, 70, 0, 0.01, 61), strict=True)),
         "defect_detected": int(not normal)}
        for time, value in enumerate(values)
    ]
    document = {
        "images": [{"id": index, "file_name": stem + ".JPG", "width": 24, "height": 18}],
        "annotations": [] if normal else [{"category_id": 23, "image_id": index, "bbox": [3, 4, 9, 8]}],
        "sensor_data": [{"image_id": index + 600, "sensor_sequence": steps}],
    }
    if session is not None:
        document["production_session_id"] = session
    (root / row["annotation_path"]).write_text(json.dumps(document), encoding="utf-8")
    if image:
        Image.new("RGB", (24, 18), (index * 16 % 256, 20, 60)).save(root / row["image_path"])
    if workbook:
        _workbook(root / row["sensor_path"], steps)
    return row


def _manifest(path: Path, *rows: CsvRow) -> Path:
    write_csv(path, list(rows), list(rows[0]))
    return path


def test_verified_pair_preserves_independent_ids_nullable_channels_and_legacy_interchange(tmp_path: Path) -> None:
    root = tmp_path / "source"
    row = _sample(root, 1, values=(0, None, 4))
    manifest = _manifest(tmp_path / "manifest.csv", row)
    cohort = audit_cohort(root, manifest)
    record = cohort.records[0]
    assert (record.image_id, record.sensor_image_id) == (1, 601)
    assert record.alignment_method == "provider_record_json_xlsx"
    assert record.sequence.steps[0].values[0] == 0.0
    assert record.sequence.steps[1].values[0] is None
    assert record.sequence.steps[0].timestamp.endswith("09:04:40")
    raw = record.sequence.model_dump_json()
    assert "defect_detected" not in raw
    step = record.sequence.steps[1]
    replay = SensorObservation(kind="json", source_id="fixture", sequence_id="row-2", timestamp=step.timestamp,
                               values=dict(zip(SENSOR_FIELDS, step.values, strict=True)))
    inspection = SensorInput(source_id=replay.source_id, values=replay.measurements)
    assert "temperature" not in inspection.values
    assert inspection.values["vibration"] == 0.0
    report = audit_report(cohort, preview=True)
    assert report["verified_pairs"] == 1
    assert report["timestamp_inferred_pairs"] == 0
    quality = report["by_split"]["train"]["sensor_quality"]
    assert quality["sequence_lengths"] == {"3": 1}
    assert quality["channels"]["temperature"]["missing"] == 1
    assert quality["channels"]["temperature"]["recorded_zeros"] == 1


@pytest.mark.parametrize("value", [True, "1.2", float("nan"), float("inf"), 1e31])
def test_invalid_sensor_values_are_rejected(value: object) -> None:
    with pytest.raises(SampleIssue, match="invalid_sensor_value"):
        sensor_step(START, (value, 0, 0, 0, 0))


@pytest.mark.parametrize("problem", ["ambiguous", "non_increasing", "capture", "missingness", "values", "timestamps"])
def test_invalid_relationship_retains_image_only(tmp_path: Path, problem: str) -> None:
    root = tmp_path / "source"
    row = _sample(root, 1)
    annotation = root / row["annotation_path"]
    document = json.loads(annotation.read_text())
    group = document["sensor_data"][0]
    sequence = group["sensor_sequence"]
    if problem == "ambiguous":
        document["sensor_data"].append(group)
    elif problem == "non_increasing":
        sequence[1]["timestamp"] = sequence[0]["timestamp"]
    elif problem == "capture":
        sequence.pop(0)
    annotation.write_text(json.dumps(document), encoding="utf-8")
    if problem == "capture":
        _workbook(root / row["sensor_path"], sequence)
    elif problem == "missingness":
        sequence[0]["temperature"] = None
        _workbook(root / row["sensor_path"], sequence)
    elif problem == "values":
        sequence[0]["temperature"] = 999
        _workbook(root / row["sensor_path"], sequence)
    elif problem == "timestamps":
        sequence[-1]["timestamp"] += ".5"
        _workbook(root / row["sensor_path"], sequence)
    cohort = audit_cohort(root, _manifest(tmp_path / "manifest.csv", row))
    record = cohort.records[0]
    expected = {
        "ambiguous": "ambiguous_json_sensor_group", "non_increasing": "non_increasing_or_incompatible_sensor_timestamps",
        "capture": "sensor_capture_timestamp_mismatch", "missingness": "sensor_missingness_mismatch",
        "values": "sensor_value_mismatch", "timestamps": "sensor_timestamp_mismatch",
    }
    assert record.alignment == ("ambiguous" if problem == "ambiguous" else "unmatched")
    assert expected[problem] in record.issues
    assert record.sequence is None
    assert MultimodalDataset(cohort, root, "train", mode="image")[0].modalities == (True, False)


def test_shared_time_cannot_replace_relationship_or_cross_group(tmp_path: Path) -> None:
    root = tmp_path / "source"
    first = _sample(root, 1, workbook=False)
    second = _sample(root, 2, capture=1, source="B", split="validation")
    first["sensor_path"] = second["sensor_path"]
    cohort = audit_cohort(root, _manifest(tmp_path / "manifest.csv", first, second))
    assert cohort.records[0].sequence is None
    assert "missing_or_conflicting_sensor_path" in cohort.records[0].issues

    # Even an explicitly named same-stem workbook cannot come from a different source folder.
    foreign_path = Path(second["sensor_path"]).with_name(Path(first["image_path"]).stem + ".xlsx")
    (root / foreign_path).write_bytes((root / second["sensor_path"]).read_bytes())
    first["sensor_path"] = foreign_path.as_posix()
    record = audit_cohort(root, _manifest(tmp_path / "second.csv", first, second)).records[0]
    assert "cross_group_sensor_path" in record.issues
    assert record.sequence is None


def test_cross_split_groups_and_duplicate_ids_are_quarantined(tmp_path: Path) -> None:
    root = tmp_path / "source"
    train = _sample(root, 1, session="lot-1")
    validation = _sample(root, 2, split="validation", capture=1, session="lot-1")
    duplicate = _sample(root, 1, split="validation", session="lot-2")
    other = _sample(root, 4, split="validation", source="B", session="lot-1")
    cohort = audit_cohort(root, _manifest(tmp_path / "manifest.csv", validation, other, duplicate, train))
    assert tuple(record.quarantined for record in cohort.records) == (True, True, True, False)
    assert cohort.records[0].alignment == "ambiguous"
    assert "cross_split_production_session" in cohort.records[0].issues
    assert "cross_split_sensor_exact" in cohort.records[0].issues
    assert len(MultimodalDataset(cohort, root, "train", mode="image")) == 0
    assert len(MultimodalDataset(cohort, root, "validation", mode="image")) == 1


def test_frozen_train_preprocessing_and_masks(tmp_path: Path) -> None:
    root = tmp_path / "source"
    train = _sample(root, 1, values=(0, None, 4))
    validation = _sample(root, 2, split="validation", values=(900, 1000, 1100))
    cohort = audit_cohort(root, _manifest(tmp_path / "manifest.csv", validation, train))
    config = PreprocessingConfig(scaling="standard", missing="train_median", window_seconds=2)
    state = fit_preprocessor(cohort, config)
    assert (state.statistics[0].center, state.statistics[0].scale, state.statistics[0].count) == (2.0, 2.0, 2)
    assert fit_preprocessor(cohort, PreprocessingConfig(scaling="robust", window_seconds=2)).statistics[0].scale == 2
    instant = fit_preprocessor(cohort, PreprocessingConfig(scaling="standard"))
    assert instant.statistics[0].center == 0.0
    assert instant.statistics[0].count == 1
    train_window = MultimodalDataset(cohort, root, "train", mode="sensor", preprocessing=state)[0].sensor
    val_window = MultimodalDataset(cohort, root, "validation", mode="sensor", preprocessing=state)[0].sensor
    assert tuple(row[0] for row in train_window.values) == (-1.0, 0.0, 1.0)
    assert tuple(row[0] for row in train_window.observed) == (True, False, True)
    assert val_window.values[0][0] == 449.0
    normalized = fit_preprocessor(cohort, PreprocessingConfig(
        scaling="robust", normalization="l2", window_seconds=2,
    ))
    record = cohort.records[0]
    assert transform_sensor(record.sequence, record.capture_timestamp, normalized).values[0][0] == -1.0
    with pytest.raises(ValueError, match="reject policy"):
        fit_preprocessor(cohort, PreprocessingConfig(missing="reject", window_seconds=2))

    train_only = cohort.model_copy(update={"records": cohort.records[:1]})
    assert fit_preprocessor(train_only, config) == state
    state_path = tmp_path / "state.json"
    write_json(state_path, state.payload())
    assert PreprocessingState.load(state_path, expected_config=config,
                                   expected_training_fingerprint=training_fingerprint(cohort)) == state
    with pytest.raises(ValueError, match="configuration fingerprint"):
        PreprocessingState.load(state_path, expected_config=PreprocessingConfig())
    with pytest.raises(ValueError, match="Train cohort fingerprint"):
        MultimodalDataset(cohort.model_copy(update={"records": cohort.records[1:]}), root,
                          "validation", mode="sensor", preprocessing=state)
    saved = state.payload()
    saved["state"]["statistics"][0]["center"] = 0.25
    corrupted_state = tmp_path / "corrupt.json"
    write_json(corrupted_state, saved)
    with pytest.raises(ValueError, match="state fingerprint"):
        PreprocessingState.load(corrupted_state)


def test_feature_order_masks_and_short_or_constant_sequences() -> None:
    series = SensorSequence(steps=(
        sensor_step(START, (0, None, None, 3, 7)),
        sensor_step(START + timedelta(seconds=1), (None, None, 1, 3, 7)),
        sensor_step(START + timedelta(seconds=4), (4, None, None, 3, 7)),
    ))
    result = statistical_features(series)
    assert result.names == FEATURE_NAMES
    assert len(result.values) == 30
    assert result.values[:6] == (2.0, 2.0, 0.0, 4.0, 4.0, 1.0)
    assert result.values[6:12] == (None,) * 6
    assert result.values[12:18] == (1.0, None, 1.0, 1.0, None, None)
    assert result.values[18:24] == (3.0, 0.0, 3.0, 3.0, 0.0, 0.0)
    assert result.observed[:18] == (True,) * 6 + (False,) * 6 + (True, False, True, True, False, False)


def test_dataset_views_and_deterministic_variable_length_batches(tmp_path: Path) -> None:
    root = tmp_path / "source"
    rows = (
        _sample(root, 1), _sample(root, 2, values=(5,), normal=True),
        _sample(root, 3, workbook=False), _sample(root, 4, image=False),
        _sample(root, 5, split="validation"),
    )
    cohort = audit_cohort(root, _manifest(tmp_path / "manifest.csv", *rows))
    state = fit_preprocessor(cohort, PreprocessingConfig(window_seconds=3))
    aug = Augmentation(geometric=Geometric(horizontal_flip_probability=1, flip_policy="verified_mirror_invariant"))
    pairs = MultimodalDataset(cohort, root, "train", mode="paired", preprocessing=state, augmentation=aug)
    assert len(pairs) == 2
    assert pairs[0].original_targets[0].box == (3.0, 4.0, 9.0, 8.0)
    assert pairs[0].image_targets[0].box == (12.0, 4.0, 9.0, 8.0)
    assert pairs[1].original_targets == ()
    val = MultimodalDataset(cohort, root, "validation", mode="paired", preprocessing=state, augmentation=aug)
    assert val[0].image_targets == val[0].original_targets
    assert pairs[0].image.tobytes() == pairs[0].image.tobytes()
    assert len(MultimodalDataset(cohort, root, "train", mode="sensor", preprocessing=state)) == 3
    assert len(MultimodalDataset(cohort, root, "train", mode="image")) == 3
    with pytest.raises(ValueError, match="lacks a requested modality"):
        MultimodalDataset(cohort, root, "train", mode="paired", preprocessing=state, missing="error")
    mixed = MultimodalDataset(cohort, root, "train", mode="paired", preprocessing=state, missing="preserve")
    items = tuple(mixed[index] for index in range(len(mixed)))
    batch = collate(items)
    assert batch == collate(items)
    assert batch.lengths == (3, 0, 3, 1)
    assert batch.modalities == ((True, True), (True, False), (False, True), (True, True))
    assert batch.timestep_valid[3] == (True, False, False)
    assert batch.sensor_values[3][1] == (0.0,) * 5
    assert batch.sensor_observed[1] == ((False,) * 5,) * 3
    with pytest.raises(ValueError, match="only one split"):
        collate((mixed[0], val[0]))
    with pytest.raises(ValueError, match="same frozen"):
        collate((mixed[0], replace(mixed[1], preprocessing_fingerprint=None)))
    (root / rows[0]["image_path"]).write_bytes(b"updated image")
    with pytest.raises(ValueError, match="Image bytes changed"):
        _ = mixed[0]


def test_cli_audit_and_resumable_index(tmp_path: Path) -> None:
    root = tmp_path / "source"
    rows = (_sample(root, 1), _sample(root, 2))
    manifest = _manifest(tmp_path / "manifest.csv", *rows)
    checkpoint = tmp_path / "cache.sqlite"

    def interrupt(done: int, total: int | None) -> None:
        if done == 1:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        audit_cohort(root, manifest, checkpoint=checkpoint, progress=interrupt)
    out = tmp_path / "output"
    args = ["multimodal", "audit", "--root", str(root), "--manifest", str(manifest), "--out", str(out),
            "--export-index", "--checkpoint", str(checkpoint)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["verified_pairs"] == 2
    assert "2/2" in result.stderr
    saved = Cohort.load(out / "cohort.json")
    assert saved == audit_cohort(root, manifest)
    assert CliRunner().invoke(app, args).exit_code == 2
    # A removed workbook invalidates its cached observation.
    (root / rows[0]["sensor_path"]).unlink()
    assert audit_cohort(root, manifest, checkpoint=checkpoint).records[0].sequence is None


def test_optional_cpu_tensor_shapes_and_dtypes(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    root = tmp_path / "source"
    first, second = _sample(root, 1), _sample(root, 2, values=(2,))
    cohort = audit_cohort(root, _manifest(tmp_path / "manifest.csv", first, second))
    state = fit_preprocessor(cohort, PreprocessingConfig(window_seconds=2))
    pairs = MultimodalDataset(cohort, root, "train", mode="paired", preprocessing=state)
    tensors = collate([pairs[0], pairs[1]]).sensor_tensors()
    assert tensors.values.shape == (2, 3, 5)
    assert tensors.values.dtype == torch.float64
    assert tensors.observed.dtype == tensors.timestep_valid.dtype == tensors.modalities.dtype == torch.bool
    assert tensors.lengths.dtype == torch.int64
    assert all(tensor.device.type == "cpu" for tensor in tensors)
    assert tensors.timestep_valid.tolist() == [[True, True, True], [True, False, False]]
    images = MultimodalDataset(cohort, root, "train", mode="image")
    assert collate([images[0], images[1]]).sensor_tensors().values.shape == (2, 0, 5)


def test_published_membership_and_source_fingerprints_cover_internal_test(tmp_path: Path) -> None:
    root = tmp_path / "source"
    train, validation = _sample(root, 1), _sample(root, 2, split="validation")
    source = audit_cohort(root, _manifest(tmp_path / "original.csv", train, validation))
    revised = [
        {**row, "original_split": row["split"], "split": split, "group_id": f"group-{split}"}
        for row, split in ((train, "train"), (validation, "test"))
    ]
    manifest = _manifest(tmp_path / "revised.csv", *revised)
    revision = {
        "schema_version": 1, "selected_manifest_fingerprint": digest_json(revised),
        "dataset_fingerprint": source.input_fingerprint, "label_policy": label_policy(),
    }
    path = tmp_path / "split.json"
    write_json(path, {**revision, "revision_id": digest_json(revision)})
    with pytest.raises(ValueError, match="published split.json"):
        audit_cohort(root, manifest)
    cohort = audit_cohort(root, manifest, revision_path=path)
    assert (cohort.records[1].split, cohort.records[1].provider_split) == ("test", "validation")
    state = fit_preprocessor(cohort, PreprocessingConfig())
    assert state.statistics[0].count == 1
    _ = MultimodalDataset(cohort, root, "test", mode="paired", preprocessing=state)[0]
    (root / validation["image_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source files changed"):
        audit_cohort(root, manifest, revision_path=path)
