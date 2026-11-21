"""Exercise Holdout integrity against a serialized Group Split publication."""

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from typer.testing import CliRunner

from ict_pipeline.cli import app
from ict_pipeline.data.model import Progress
from ict_pipeline.data.preprocessing.common import write_csv, write_json
from ict_pipeline.data.preprocessing import console
from ict_pipeline.data.preprocessing.leakage import AnalysisConfig, analyze_leakage
from ict_pipeline.data.preprocessing.revision import read_publication
from ict_pipeline.data.preprocessing.split import SplitConfig, plan_split, publish_split
from ict_pipeline.evaluation import limited_holdout as holdout
from ict_pipeline.evaluation.product import write_artifact
from ict_pipeline.evaluation.product_protocol import FrozenValidation
from ict_pipeline.multimodal.analysis import AnalysisRun, match_leakage_records
from ict_pipeline.multimodal.audit import audit_cohort
from ict_pipeline.multimodal.dataset import Batch, MultimodalDataset
from ict_pipeline.multimodal.model_config import ArchitectureConfig, image_preprocessing_fingerprint
from ict_pipeline.multimodal.preprocessing import PreprocessingConfig, fit_preprocessor
from ict_pipeline.pipeline.store import digest_json
from test_multimodal import _sample


def _publish(tmp_path: Path, scenario: str) -> tuple[Path, Path, str, str]:
    root = tmp_path / "source"
    train = _sample(root, 1, values=(20, 21, 22))
    validation = _sample(root, 2, split="validation", values=(40, 41, 42))
    test = _sample(
        root, 3, capture=1 if scenario == "metadata" else 3,
        values=(20, 21, 22) if scenario == "sensor-similarity" else (60, 61, 62),
    )
    for row, color in zip((train, validation, test), (0, 128, 4 if scenario == "image-similarity" else 255)):
        Image.new("RGB", (24, 18), (color,) * 3).save(root / row["image_path"])
    if scenario in ("image", "sensor"):
        field = scenario + "_path"
        (root / test[field]).write_bytes((root / train[field]).read_bytes())

    manifest = tmp_path / "provider.csv"
    write_csv(manifest, [train, validation, test], list(train))
    config = AnalysisConfig(verify_xlsx=True)
    provider = analyze_leakage(root, manifest, root, config)
    plan = plan_split(provider, SplitConfig(
        val_fraction=0.33, test_fraction=0.33, group_near_images=True, group_near_sensors=True,
    ))
    # Use the publication serializer with known assignments, including invalid
    # evidence-crossing assignments to test the independent post-publication check.
    roles = {train["sample_id"]: "train", validation["sample_id"]: "validation", test["sample_id"]: "test"}
    plan["counts"] = dict.fromkeys(roles.values(), 1)
    plan["assignments"] = [
        {"sample_index": index, "split": roles[row["sample_id"]], "group_id": digest_json([index])}
        for index, row in enumerate(provider["samples"])
    ]
    bundle = tmp_path / "bundle"
    publish_split(root, manifest, root, provider, plan, bundle, export_detection=True,
                  allow_missing_classes=True, progress=lambda _: lambda *_: None)
    revision_id = json.loads((bundle / "split.json").read_bytes())["revision_id"]
    return root, bundle, revision_id, test["sample_id"]


@pytest.mark.parametrize(("scenario", "error"), [
    ("metadata", None),
    ("image", 'hash duplication.*"kind": "image"'),
    ("sensor", 'hash duplication.*"kind": "sensor"'),
    ("image-similarity", 'cross-split similarity.*"method": "image_similarity"'),
    ("sensor-similarity", 'cross-split similarity.*"method": "sensor_similarity"'),
])
def test_published_groups_project_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, error: str | None,
) -> None:
    root, bundle, revision, test_id = _publish(tmp_path, scenario)
    manifest = bundle / "manifest.csv"
    original_bytes = manifest.read_bytes()
    cohort = audit_cohort(root, manifest, revision_path=bundle / "split.json", isolate_source_capture=False)
    if scenario != "sensor":  # The byte-identical workbook fixture deliberately breaks alignment.
        evidence = analyze_leakage(root, manifest, root, AnalysisConfig(verify_xlsx=True),
                                   provider_column="original_split")
        matched = match_leakage_records(evidence, cohort.records)
        assert {(r.provider_split, r.split) for r in matched.values()} == {
            ("train", "test"), ("train", "train"), ("validation", "validation"),
        }
        if scenario == "metadata":
            assert {"manufacturer", "source_day", "source_capture", "capture_timestamp"} <= {
                group["method"] for group in evidence["groups"]
            }
            assert any(r.quarantined for r in audit_cohort(
                root, manifest, revision_path=bundle / "split.json",
            ).records if r.split == "test")
    assert holdout.FIXED_TEST_COUNT == 1472
    with pytest.raises(ValueError, match="fixed Preflight or manifest"):
        holdout.snapshot_test_payload(root, bundle, revision, [test_id])
    monkeypatch.setattr(holdout, "FIXED_TEST_COUNT", 1)
    snapshot = holdout.snapshot_test_payload(root, bundle, revision, [test_id])
    inputs = dict(root=root, bundle=bundle, revision_id=revision, snapshot=snapshot, split_cohort=cohort)
    if error is None:
        assert holdout.verify_test_payload_integrity(**inputs)["coco_test_sample_count"] == 1
    else:
        with pytest.raises(ValueError, match="(?i)" + error):
            holdout.verify_test_payload_integrity(**inputs)
    assert manifest.read_bytes() == original_bytes


def test_cli_reuses_pre_fix_running_state_before_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, bundle, revision, test_id = _publish(tmp_path, "image-similarity")
    _, _, rows = read_publication(bundle, revision)
    cohort = audit_cohort(root, bundle / "manifest.csv", revision_path=bundle / "split.json")
    index = tmp_path / "cohort.json"
    write_json(index, {"cohort": cohort.model_dump(mode="json"), "fingerprint": cohort.fingerprint})
    monkeypatch.setattr(holdout, "FIXED_TEST_COUNT", 1)
    snapshot = holdout.snapshot_test_payload(root, bundle, revision, [test_id])
    assert snapshot["test_sample_ids"] == [row["sample_id"] for row in rows if row["split"] == "test"]
    plan = {"test_sample_ids": [test_id], "evaluation_plan": {"threshold": 0.5}}
    monkeypatch.setattr(holdout, "revalidate_limited_holdout", lambda **_: ({"fingerprint": "preflight"}, plan))
    monkeypatch.setattr(holdout, "score_run", lambda *args: pytest.fail("Integrity must pass before inference"))
    out = tmp_path / "interrupted"
    out.mkdir()
    state = out / "resume-state.json"
    # Serialize the plan fingerprint using the exact inputs from 9197e82.
    previous = {
        "schema_version": 1, "status": "running", "plan_fingerprint": digest_json({
            "preflight_fingerprint": "preflight", "live_evaluation_plan": plan["evaluation_plan"],
            "test_snapshot_fingerprint": snapshot["fingerprint"], "device": "cpu", "batch_size": 32,
            "seed": 42, "checkpoint_dir": "output-default",
        }),
        "preflight_fingerprint": "preflight", "test_snapshot_fingerprint": snapshot["fingerprint"],
        "device": "cpu", "batch_size": 32, "seed": 42,
    }
    write_json(state, previous)
    state_bytes = state.read_bytes()
    args = ["integrity", "limited-holdout-evaluate", "--root", str(root), "--bundle", str(bundle),
            "--revision-id", revision, "--cohort", str(index), "--out", str(out),
            "--approve-test-access", "--resume"]
    for option in ("preflight", "group-verification", "frozen-validation", "preprocessing", "image-run", "concat-run"):
        args += ["--" + option, str(tmp_path / option)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2, result.output
    assert "Cross-split similarity" in result.stderr
    assert "Snapshot Test payload 1/1 (100.0%)" in result.stderr
    assert result.stdout == ""
    changed = CliRunner().invoke(app, [*args, "--seed", "43"])
    assert changed.exit_code == 2
    assert "fresh --out without --resume" in changed.stderr
    assert state.read_bytes() == state_bytes
    assert set(out.iterdir()) == {state}
    partial = out / "limited-holdout.json"
    partial.write_bytes(b"existing report")
    blocked = CliRunner().invoke(app, args)
    assert blocked.exit_code == 2 and "reports already exist" in blocked.stderr
    assert partial.read_bytes() == b"existing report"


@pytest.mark.parametrize("interactive", [False, True])
def test_holdout_progress_throttles_and_uses_tty_updates(
    monkeypatch: pytest.MonkeyPatch, interactive: bool,
) -> None:
    class CapturedStream(StringIO):
        def isatty(self) -> bool:
            return interactive

    stream = CapturedStream()
    times = iter((0.0, 0.0, 0.5, 5.0, 5.1))
    monkeypatch.setattr(console, "monotonic", lambda: next(times))
    report = console.console_progress("Image-only inference", file=stream, percent=True)
    for completed in range(4):
        report(completed, 3)
    output = stream.getvalue()
    assert ("\r" in output) is interactive
    assert "1/3" not in output
    for text in ("0/3 (0.0%)", "2/3 (66.7%)", "3/3 (100.0%)"):
        assert text in output
    assert output.endswith("\n")


def test_preflight_preserves_all_1472_fixed_ids_and_strict_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ids = [f"sample-{i:04}" for i in range(1472)]
    live = {
        "test_sample_ids": ids, "test_sample_count": len(ids),
        "evaluation_plan": {
            "image_only_checkpoint": {"epoch": 27}, "sensor_concat_checkpoint": {"epoch": 15},
            "threshold": 0.5, "class_order": list(holdout.NAMES), "test_sample_fingerprint": digest_json(ids),
        },
    }
    receipt = {
        **live, "schema_version": 1, "stage": "preflight", "passed": True,
        "seal_verification_allowed": False, "final_evaluation_allowed": False,
    }
    receipt["fingerprint"] = digest_json(receipt)
    path = tmp_path / "preflight.json"
    write_json(path, receipt)
    before = path.read_bytes()
    monkeypatch.setattr(holdout, "preflight_test_release", lambda **_: live)
    inputs = {
        name: tmp_path / name for name in (
            "root", "bundle", "group_verification", "cohort", "frozen_validation", "preprocessing",
            "image_run", "concat_run",
        )
    }
    saved, checked = holdout.revalidate_limited_holdout(**inputs, preflight=path, revision_id="fixture")
    assert checked["test_sample_ids"] == ids
    assert saved["seal_verification_allowed"] is saved["final_evaluation_allowed"] is False
    live["test_sample_ids"] = ids[:-1] + ["sample-9999"]
    with pytest.raises(ValueError, match="disagrees"):
        holdout.revalidate_limited_holdout(**inputs, preflight=path, revision_id="fixture")
    assert path.read_bytes() == before


def test_cli_reports_actual_inference_and_restored_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    from ict_pipeline.data import integrity_commands
    from ict_pipeline.multimodal import checkpoint, training_torch
    from ict_pipeline.multimodal.model import ProductOutput

    root, bundle, revision, test_id = _publish(tmp_path, "metadata")
    monkeypatch.setattr(holdout, "FIXED_TEST_COUNT", 1)
    cohort = audit_cohort(root, bundle / "manifest.csv", revision_path=bundle / "split.json",
                          isolate_source_capture=False)
    index = tmp_path / "cohort.json"
    training = cohort.for_product_training()
    write_json(index, {"cohort": training.model_dump(mode="json"), "fingerprint": training.fingerprint})
    validation = MultimodalDataset(cohort, root, "validation", mode="image")
    frozen = tmp_path / "frozen.json"
    write_artifact(frozen, FrozenValidation.from_audit(training, tuple(row.sample_id for row in validation.records)))
    state = fit_preprocessor(training, PreprocessingConfig())
    preprocessing = tmp_path / "sensor-state.json"
    write_json(preprocessing, state.payload())
    plan = {
        "test_sample_ids": [test_id], "group_split_revision": revision,
        "evaluation_plan": {"detector_initialization": {}},
    }
    monkeypatch.setattr(holdout, "revalidate_limited_holdout", lambda **_: ({"fingerprint": "preflight"}, plan))
    architecture = ArchitectureConfig(
        preprocessing_fingerprint=state.fingerprint, target_policy="smt-view-class-presence-v1",
    )
    protocol = SimpleNamespace(image_preprocessing_fingerprint=image_preprocessing_fingerprint(architecture.image, {}))
    runs = {
        role: AnalysisRun(
            role, tmp_path / role, state, SimpleNamespace(branch=branch, architecture=architecture),
            (), validation, protocol, {"epoch": epoch}, "fixture-architecture",
        )
        for role, branch, epoch in (("image", "image", 27), ("concat", "paired", 15))
    }
    monkeypatch.setattr(holdout, "_inspect_run", lambda role, *args, **kwargs: runs[role])
    model_calls = []

    class FixtureModel:
        config = architecture
        architecture_fingerprint = "fixture-architecture"
        image_encoder = SimpleNamespace(input_contract={})

        def to(self, device: str) -> "FixtureModel":
            assert device == "cpu"
            return self

        def eval(self) -> "FixtureModel":
            return self

        def branch_status(self, branch: str) -> str:
            return "trained"

        def __call__(self, batch: Batch) -> ProductOutput:
            model_calls.extend(batch.modalities)
            logits = torch.zeros((len(batch.samples), len(self.config.output_names)))
            return ProductOutput(
                "test", tuple(row.sample_id for row in batch.samples), self.config.target_policy,
                self.config.output_names, logits, logits, logits,
                torch.tensor([image for image, _ in batch.modalities]),
                torch.tensor([sensor for _, sensor in batch.modalities]),
            )

    # The trainer keeps its own import-time binding to the real loader.
    assert training_torch.load_checkpoint is checkpoint.load_checkpoint
    monkeypatch.setattr(checkpoint, "load_checkpoint", lambda *args, **kwargs: FixtureModel())
    out = tmp_path / "out"
    args = ["integrity", "limited-holdout-evaluate", "--root", str(root), "--bundle", str(bundle),
            "--revision-id", revision, "--cohort", str(index), "--frozen-validation", str(frozen),
            "--preprocessing", str(preprocessing), "--out", str(out), "--approve-test-access"]
    for option in ("preflight", "group-verification", "image-run", "concat-run"):
        args += ["--" + option, str(tmp_path / option)]
    console = integrity_commands.console_progress

    with monkeypatch.context() as patch:
        def interrupt_after_batches(name: str, **kwargs: object) -> Progress:
            report = console(name, **kwargs)

            def emit(done: int, total: int | None) -> None:
                report(done, total)
                if name.startswith("Sensor Concat inference") and done == total:
                    raise KeyboardInterrupt

            return emit

        patch.setattr(integrity_commands, "console_progress", interrupt_after_batches)
        interrupted = CliRunner().invoke(app, args)
    assert interrupted.exit_code == 130, (interrupted.output, interrupted.exception)
    assert model_calls == [(True, False), (True, True)]
    cache_files = {path: path.read_bytes() for path in (out / "batch-checkpoints").glob("*/*.json")}
    assert len(cache_files) == 2
    resumed = CliRunner().invoke(app, [*args, "--resume"])
    assert resumed.exit_code == 0, (resumed.output, resumed.exception)
    assert json.loads(resumed.stdout)["fixed_test"]["sample_ids"] == [test_id]
    assert model_calls == [(True, False), (True, True)]
    assert all(path.read_bytes() == saved for path, saved in cache_files.items())
    for role in ("Image-only", "Sensor Concat"):
        for count in ("0/1 (0.0%)", "1/1 (100.0%)"):
            label = f"{role} inference (including restored batches) {count}"
            assert label in interrupted.stderr and label in resumed.stderr
    assert "Calculate metrics and save reports 1/1 (100.0%)" in resumed.stderr
