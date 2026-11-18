# ruff: noqa: E402
# The model imports below require the optional torch backend.
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

torch = pytest.importorskip("torch")

from ict_pipeline.evaluation.contracts import Prediction, Preprocessing
from ict_pipeline.models.dataset import Object
from ict_pipeline.multimodal.checkpoint import load_checkpoint, save_checkpoint
from ict_pipeline.multimodal.dataset import DatasetSample, collate
from ict_pipeline.multimodal.encoders import MLPSensorEncoder
from ict_pipeline.multimodal.fusion import LateFusion
from ict_pipeline.multimodal.model import (
    DetectorEvidence, MultimodalModel, ProductInput, product_loss, product_targets,
)
from ict_pipeline.multimodal.model_config import (
    ArchitectureConfig, FusionConfig, ImageEncoderConfig, SensorEncoderConfig, WeightStatus,
)
from ict_pipeline.multimodal.preprocessing import ChannelFit, PreprocessingConfig, PreprocessingState, SensorWindow


def frozen(window_seconds: int = 0, *, fingerprint: str = "a" * 64) -> PreprocessingState:
    config = PreprocessingConfig(window_seconds=window_seconds, scaling="standard")
    return PreprocessingState(
        config=config, config_fingerprint=config.fingerprint, training_fingerprint=fingerprint,
        statistics=(ChannelFit(count=3, center=0.0, scale=1.0, fill=0.0),) * 5,
    )


def window(state: PreprocessingState, steps: tuple[int, ...] = (0,)) -> SensorWindow:
    return SensorWindow(
        preprocessing_fingerprint=state.fingerprint,
        timestamps=tuple(f"2025-08-07 09:03:{40 + step}" for step in steps),
        values=tuple((float(step), 0.0, 200.0, 0.0, 0.0) for step in steps),
        observed=tuple((True, True, False, False, False) for _ in steps),
    )


def sample(
    key: str, state: PreprocessingState, *, image: bool = True, sensor: SensorWindow | None = None,
    targets: tuple[Object, ...] = (),
) -> DatasetSample:
    rgb = Image.new("RGB", (16, 12), (16, 32, 64)) if image else None
    return DatasetSample(key, "train", "train", "fixture", rgb, sensor, (16, 12), targets,
                         () if image else None, state.fingerprint)


@pytest.mark.parametrize("kind", ["late", "concat", "gated"])
def test_deterministic_forward_and_modality_fallback(kind: str) -> None:
    state = frozen()
    config = ArchitectureConfig(
        preprocessing_fingerprint=state.fingerprint, fusion=FusionConfig(kind=kind, threshold=0.0),
        image=ImageEncoderConfig(embedding_dim=7), sensor=SensorEncoderConfig(embedding_dim=3),
        target_policy="smt-view-class-presence-v1",
    )
    rng = torch.random.get_rng_state()
    model = MultimodalModel(config, state).eval()
    assert torch.equal(torch.random.get_rng_state(), rng)
    duplicate = MultimodalModel(config, state).eval()
    observed = window(state)
    missing = observed.model_copy(update={"observed": ((False,) * 5,)})
    batch = collate((
        sample("paired", state, sensor=observed), sample("image", state),
        sample("sensor", state, image=False, sensor=observed), sample("none", state, image=False, sensor=missing),
    ))
    output = model(batch)
    other = duplicate(batch)
    for branch, availability in (("image", [True, True, False, False]), ("sensor", [True, False, True, False]),
                                 ("paired", [True, False, False, False])):
        logits, mask = output.for_branch(branch)
        assert logits.shape == (4, 5)
        assert mask.tolist() == availability
        assert logits[~mask].count_nonzero() == 0
        assert torch.equal(logits, other.for_branch(branch)[0])
    changed = MultimodalModel(config.model_copy(update={"seed": 43}), state)
    assert not torch.equal(changed(batch).sensor_logits, output.sensor_logits)
    predictions = model.predict(batch)
    assert tuple(item.branch for item in predictions) == ("paired", "image", "sensor", None)
    assert tuple(item.sensor for item in predictions) == ("available", "absent", "available", "no_observations")
    for row in predictions[:3]:
        assert row.weight_status == "untrained"
        assert row.score_kind == "uncalibrated_sigmoid"
        assert row.outcome == "review_required"
        assert [score.class_id for score in row.scores] == [1, 2, 3, 4, 5]
        assert all(score.threshold_crossed for score in row.scores)
    assert not predictions[3].scores
    assert predictions[3].weight_status is None
    assert predictions[2].detection is None


@pytest.mark.parametrize("features", ["normalized_mean", "statistics"])
def test_sensor_masks_ignore_imputation_and_keep_observed_zero(features: str) -> None:
    state = frozen()
    encoder = MLPSensorEncoder(SensorEncoderConfig(features=features, embedding_dim=7), state)
    observed = window(state)
    imputed = observed.model_copy(update={"values": ((0.0, 0.0, 1e125, -1e125, 0.0),)})
    invalid = observed.model_copy(update={"values": ((1e125, 0.0, 0.0, 0.0, 0.0),)})
    empty = observed.model_copy(update={"observed": ((False,) * 5,)})
    output = encoder([observed, imputed, empty, None])
    assert output.embeddings.shape == (4, 7)
    assert output.usable.tolist() == [True, True, False, False]
    assert torch.equal(output.embeddings[0], output.embeddings[1])
    assert output.embeddings[2:].count_nonzero() == 0
    with pytest.raises(ValueError, match="numeric range"):
        encoder([invalid])


def test_temporal_steps_and_padding_use_only_observed_timestamps() -> None:
    state = frozen(4)
    config = ArchitectureConfig(
        preprocessing_fingerprint=state.fingerprint, sensor=SensorEncoderConfig(kind="gru", embedding_dim=9),
    )
    model = MultimodalModel(config, state).eval()
    gapped = window(state, (0, 1, 3))
    gapped = gapped.model_copy(update={"observed": (gapped.observed[0], (False,) * 5, gapped.observed[2])})
    same = window(state, (0, 3))
    changed = same.model_copy(update={"timestamps": window(state, (0, 2)).timestamps})
    short = window(state)
    enc = model.sensor_encoder([gapped, same, changed, short, None])
    assert enc.embeddings.shape == (5, 9)
    assert enc.usable.tolist() == [True, True, True, False, False]
    torch.testing.assert_close(enc.embeddings[0], enc.embeddings[1])
    assert not torch.allclose(enc.embeddings[1], enc.embeddings[2])
    batch = collate((sample("pair", state, sensor=gapped), sample("short", state, sensor=short),
                     sample("short-only", state, image=False, sensor=short)))
    assert batch.timestep_valid == ((True, True, True), (True, False, False), (True, False, False))
    predicted = model.predict(batch)
    assert [row.branch for row in predicted] == ["paired", "image", None]
    assert [row.sensor for row in predicted] == ["available", "insufficient_steps", "insufficient_steps"]
    with pytest.raises(ValueError, match="masks"):
        model(replace(batch, timestep_valid=((True, True, True),) * 3))
    with pytest.raises(ValueError, match="exceed the frozen"):
        model.sensor_encoder([window(state, (0, 5))])
    capture_only = frozen()
    with pytest.raises(ValueError, match="nonzero"):
        MultimodalModel(config.model_copy(update={"preprocessing_fingerprint": capture_only.fingerprint}), capture_only)
    with pytest.raises(ValueError, match="per-step"):
        SensorEncoderConfig(kind="gru", features="statistics")
    with pytest.raises(ValueError, match="min_steps"):
        SensorEncoderConfig(kind="gru", min_steps=1)


def test_weighted_late_fusion_and_semantic_loss() -> None:
    fusion = LateFusion(FusionConfig(image_weight=3.0, sensor_weight=1.0), 1)
    assert list(fusion.parameters()) == []
    assert fusion(torch.tensor([[2.0]]), torch.tensor([[-2.0]])).item() == 1.0
    with pytest.raises(ValueError, match="compatible"):
        fusion(torch.ones(2, 1), torch.ones(1, 1))
    state = frozen()
    config = ArchitectureConfig(preprocessing_fingerprint=state.fingerprint)
    multiclass = config.model_copy(update={"target_policy": "smt-view-class-presence-v1"})
    defect = (Object(1, (0, 0, 2, 2)), Object(1, (2, 2, 2, 2)), Object(5, (4, 4, 2, 2)))
    batch = collate((sample("defect", state, sensor=window(state), targets=defect),
                     sample("normal", state)))
    assert product_targets(batch, multiclass).values.tolist() == [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0]]
    targets = product_targets(batch, config)
    assert targets.values.tolist() == [[1], [0]]
    output = MultimodalModel(config, state)(batch)
    torch.testing.assert_close(product_loss(output, targets, branch="sensor"),
                               torch.nn.functional.softplus(-output.sensor_logits[0]).mean())
    assert product_loss(output, targets, branch="paired").requires_grad
    for wrong in (replace(targets, sample_ids=("normal", "defect")), product_targets(batch, multiclass),
                  replace(targets, values=torch.full((2, 1), 0.5)), replace(targets, split="validation")):
        with pytest.raises(ValueError, match="ordered binary"):
            product_loss(output, wrong, branch="paired")
    solo = collate((batch.samples[1],))
    with pytest.raises(ValueError, match="no usable rows"):
        product_loss(MultimodalModel(config, state)(solo), product_targets(solo, config), branch="paired")
    with pytest.raises(ValueError, match="one-based"):
        product_targets(collate((replace(batch.samples[0], original_targets=(Object(0, defect[0].box),)),)), config)
    with pytest.raises(ValueError, match="canonical SMT"):
        ArchitectureConfig(preprocessing_fingerprint=state.fingerprint, class_names=config.class_names[::-1])


def test_label_free_prediction_keeps_boxes_and_provenance_separate() -> None:
    state = frozen()
    model = MultimodalModel(ArchitectureConfig(preprocessing_fingerprint=state.fingerprint), state).eval()
    image = Image.new("RGB", (16, 12))
    paired = ProductInput("part", image, window(state))
    score_before = model.predict([paired])[0]
    boxes = Prediction("part", ((1, 2, 4, 5),), (2,), (0.91,), Preprocessing((16, 12), (64, 64), "letterbox"))
    evidence = DetectorEvidence("fixture-detector", boxes)
    prediction = model.predict([paired], detections=[evidence])[0]
    assert prediction.detection is evidence
    assert prediction.scores == score_before.scores
    assert prediction.scores[0].class_id is None
    assert prediction.preprocessing_fingerprint == state.fingerprint
    for invalid in (replace(paired, sample_id="other"), replace(paired, image=None)):
        with pytest.raises(ValueError, match="original image"):
            model.predict([invalid], detections=[evidence])
    with pytest.raises(ValueError, match="one optional"):
        model.predict([paired], detections=[])


def test_fingerprints_and_malformed_batch_rejected() -> None:
    state = frozen()
    other = frozen(fingerprint="b" * 64)
    config = ArchitectureConfig(preprocessing_fingerprint=state.fingerprint)
    model = MultimodalModel(config, state).eval()
    correct = sample("fixture", state, sensor=window(state))
    with pytest.raises(ValueError, match="Architecture preprocessing"):
        MultimodalModel(config, other)
    with pytest.raises(ValueError, match="Batch preprocessing"):
        model(collate((sample("bad", other, sensor=window(other)),)))
    with pytest.raises(ValueError, match="window preprocessing"):
        model.predict([ProductInput("bad", sensor=window(other))])
    with pytest.raises(ValueError, match="same frozen"):
        model(collate((replace(correct, preprocessing_fingerprint=None),)))
    with pytest.raises(ValueError, match="only one split"):
        model(collate((correct, replace(correct, split="validation"))))
    batch = collate((correct,))
    for malformed in (replace(batch, modalities=((False, True),)), replace(batch, lengths=(0,)),
                      replace(batch, sensor_observed=()), replace(batch, samples=())):
        with pytest.raises(ValueError):
            model(malformed)
    with pytest.raises(ValueError, match="canonical measured"):
        SensorEncoderConfig(channels=config.sensor.channels[::-1])


def test_checkpoint_round_trip_and_incompatible_state(tmp_path: Path) -> None:
    state = frozen()
    config = ArchitectureConfig(preprocessing_fingerprint=state.fingerprint, fusion=FusionConfig(kind="gated"))
    model = MultimodalModel(config, state).eval()
    model.weight_status = WeightStatus(sensor="trained")  # Exercise persisted caller declarations only.
    checkpoint = tmp_path / "fixture.pt"
    save_checkpoint(checkpoint, model)
    loaded = load_checkpoint(checkpoint, state, expected_config=config)
    inputs = [ProductInput("sensor", sensor=window(state)), ProductInput("image", Image.new("RGB", (8, 8)))]
    assert loaded.predict(inputs) == model.predict(inputs)
    assert [item.weight_status for item in loaded.predict(inputs)] == ["trained", "untrained"]
    with pytest.raises(FileExistsError):
        save_checkpoint(checkpoint, model)
    with pytest.raises(ValueError, match="preprocessing"):
        load_checkpoint(checkpoint, frozen(fingerprint="b" * 64))
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(checkpoint, state, expected_config=config.model_copy(update={"seed": 1}))
    payload = torch.load(checkpoint, weights_only=True)
    tensor = payload["state_dict"]["image_head.weight"]
    for label, malformed in (("shape", tensor[:, :1]), ("dtype", tensor.double()),
                             ("finite", torch.full_like(tensor, float("nan")))):
        payload["state_dict"]["image_head.weight"] = malformed
        invalid = tmp_path / f"{label}.pt"
        torch.save(payload, invalid)
        with pytest.raises(ValueError, match="image_head.weight"):
            load_checkpoint(invalid, state)
    payload["state_dict"]["image_head.weight"] = tensor
    payload["architecture_fingerprint"] = "f" * 64
    invalid = tmp_path / "fingerprint.pt"
    torch.save(payload, invalid)
    with pytest.raises(ValueError, match="architecture fingerprint"):
        load_checkpoint(invalid, state)


def test_opt_in_fpn_leaves_existing_faster_rcnn_predictions_intact() -> None:
    pytest.importorskip("torchvision")
    from ict_pipeline.models.coco import FasterRCNNAdapter
    from ict_pipeline.models.contracts import TrainConfig

    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        detector = FasterRCNNAdapter().build_model(TrainConfig(image_size=64)).eval()
        transforms = []
        detector.transform.register_forward_hook(lambda module, args, result: transforms.append(result[0].tensors.shape))
        pixels = torch.zeros(3, 16, 16)
        with torch.inference_mode():
            before = detector([pixels])[0]
        state = frozen()
        config = ArchitectureConfig(
            preprocessing_fingerprint=state.fingerprint,
            image=ImageEncoderConfig(kind="faster_rcnn_fpn", embedding_dim=3),
        )
        model = MultimodalModel(config, state, detector=detector).eval()
        model.train()
        assert not detector.backbone.training and model.image_encoder.backbone is not detector.backbone
        assert not model.image_encoder.transform.training
        model.eval()
        result = model.predict([ProductInput("image", Image.new("RGB", (16, 16)))])
        assert result[0].scores and result[0].branch == "image"
        assert len(transforms) == 1
        with torch.inference_mode():
            after = detector([pixels])[0]
        for key, value in before.items():
            assert torch.equal(value, after[key])
    finally:
        torch.set_num_threads(threads)
