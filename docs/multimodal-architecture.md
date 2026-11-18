# Multimodal model architecture

## Design and existing interfaces

The legacy classification stage fits independent binary logistic or centroid
models and can average their scores. Its sensor features assume complete
measurements and use its own scaling. The new architecture reuses the
capture-window, masks and observed-only statistics from
`ict_pipeline.multimodal`. Its default input contains at most one timestamped
step with temperature, humidity, vibration, acceleration and noise in that
order. A false observation bit excludes an imputed value. Constant observed
channels remain valid. `defect_detected` and source identifiers stay outside
the model inputs.

The detector evaluation adapters expose canonical one-based detections in the
original image coordinates. The YOLO adapter exposes decoded detections, and
the RT-DETR adapter uses query logits and boxes. Their current predictor
interfaces offer no supported pooled backbone embedding contract. Faster
R-CNN exposes its transform and FPN backbone. An opt-in adapter copies the
backbone from a caller-supplied detector, reconstructs its evaluation
transform independently, and pools one named FPN level.
The fixture image encoder uses downsampled RGB. Both implement an embedding
interface with a declared output width. Existing predictions can travel
alongside product scores with detector source provenance.

| Component | Design |
| --- | --- |
| Sensor MLP | Concatenate observed-only normalized channel means or the existing 30 statistical features with their masks. A small MLP produces an embedding. |
| Temporal sensor | An optional GRU consumes only steps with observations, with masks and elapsed seconds from the first observed step. Require a nonzero frozen window and at least two observed steps per usable row. A short row can use its image branch. |
| Late fusion | Weight and average product logits from compatible image and sensor heads. The average is parameter-free. |
| Feature fusion | Concatenate the two embeddings and apply a learned prediction head. |
| Gated fusion | Project embeddings to a common width, learn a per-feature gate from the paired embeddings, then apply a learned prediction head. |
| Fallback | Use the corresponding independent unimodal head when only one encoder is usable. An unavailable row has no product score. |

The MLP takes 10 inputs for normalized means or 60 for statistics, including
the masks. It uses values already transformed by the frozen preprocessing
state. A one-step window can provide means, minima and maxima. Standard
deviation, difference and rate features remain masked until their observation
requirements are met. The GRU takes 11 inputs per observed step. It skips
all-missing steps, retains elapsed time over gaps, and ignores batch padding.
Its minimum step count is configurable. The dataset's sensor presence flag
means at least one measurement was present. The model's `sensor_usable` flag
also incorporates the encoder's minimum step requirement.

Late fusion uses `(w_image * image_logit + w_sensor * sensor_logit) /
(w_image + w_sensor)` for each output independently. The weights apply to
paired late fusion. Feature fusion and gating each have a separate learned
head, while all three variants use the unimodal heads for fallback. A shared
score threshold applies independently to each output after the sigmoid.
An absent modality contributes neither a bias nor a placeholder score.

## Targets and prediction semantics

A dataset sample represents one captured product view. The binary target
records whether the original target boxes contain any SMT defect. The
five-output target is multi-label in canonical class order and records class
presence once per view. An empty target tuple yields zeros. Multiple views
of the same physical product require an independently defined aggregation
policy. Detection keeps original box coordinates, class IDs and detector
confidence. The forward interface exposes product logits and per-branch
masks. Prediction records expose sigmoid scores with weight provenance and
an explicit uncalibrated status. Threshold crossings are research
outputs. Production acceptance continues to require independent verification
under the inspection policy.

## Python interfaces

Use an existing PyTorch CPU environment with Pillow. The data modules and
configuration schemas can be imported independently of torch. TorchVision
is required only for the opt-in Faster R-CNN adapter. These interfaces
operate independently of the inspection service.

```python
from ict_pipeline.multimodal.dataset import collate
from ict_pipeline.multimodal.model import MultimodalModel, ProductInput, product_loss, product_targets
from ict_pipeline.multimodal.model_config import ArchitectureConfig, FusionConfig, SensorEncoderConfig

# state is the frozen Train-fitted PreprocessingState.
# samples come from a single MultimodalDataset split.
config = ArchitectureConfig(
    preprocessing_fingerprint=state.fingerprint,
    sensor=SensorEncoderConfig(features="statistics", embedding_dim=16),
    fusion=FusionConfig(kind="concat", threshold=0.5),
    target_policy="smt-view-class-presence-v1",
)
model = MultimodalModel(config, state).eval()
batch = collate(samples)
predictions = model.predict(batch)

# The forward and loss contracts can be checked without fitting.
outputs = model(batch)
targets = product_targets(batch, config)
loss = product_loss(outputs, targets, branch="paired")  # requires a usable pair

# Label-free inference accepts aligned, transformed windows or RGB images.
image_result = model.predict([ProductInput("capture-001", image=rgb_image)])
```

`SensorEncoder` receives a list of `SensorWindow | None` rows and returns
embeddings and a Boolean usability mask. Use `transform_sensor` on verified
capture-aligned raw sequences before calling the model. `ImageEncoder` accepts
a nonempty list of RGB images and returns a fixed-width embedding per image.
Live callers are responsible for verifying capture alignment and availability
at the chosen decision time.
The FPN adapter uses the caller's existing TorchVision Faster R-CNN object
via `MultimodalModel(config, state, detector=detector)`. Set
`ImageEncoderConfig(kind="faster_rcnn_fpn", fpn_level="0")` to pool the
corresponding spatial feature map. Embeddings are features for a learned
head. YOLO and RT-DETR are supported as sources of detached detector evidence.

Pass canonical `evaluation.contracts.Prediction` objects wrapped in
`DetectorEvidence(source, prediction)` to `model.predict(..., detections=...)`,
in batch order. The sample IDs and original image dimensions must agree.
The returned product record includes the branch used, sensor availability,
architecture fingerprint, window preprocessing fingerprint when supplied,
and caller-declared status of the selected weights. Sensor-only inference
produces no boxes. A row with neither usable input has an empty score list.

## State and future experiments

Version the architecture, target policy and preprocessing association. Use
deterministic CPU initialization and a separate tensor-only checkpoint format
with strict key, shape and dtype checks. Newly initialized components are
untrained. Persist explicit weight declarations per branch and preserve the
existing detector checkpoint formats.

`save_checkpoint(path, model)` publishes a new local version 1 artifact
in an existing output directory. Use a filesystem with hard-link support
for exclusive atomic publication. `load_checkpoint(path, state,
expected_config=config)` reconstructs the model, checks its architecture and
frozen preprocessing fingerprint, and loads tensors with
`torch.load(weights_only=True)`. The FPN variant also requires a compatible
detector object to reconstruct its backbone and image transform. Save the
FP32 architecture state after moving from mixed precision if needed.
`WeightStatus` defaults to untrained and can record a future runner's
declaration per branch. A late-fusion pair requires a trained declaration
for both unimodal heads. Accuracy and calibration still require validation.

Expose masked binary cross-entropy for each unimodal head and the paired
path. `OptimizerConfig` declares AdamW learning rate and weight decay for
a future runner. Future work can fit the sensor and
image heads independently on eligible Train views, then fit the paired head
or fine-tune the encoders with a recorded freeze policy. Compare variants on
the same paired Validation cohort and report coverage and metrics separately
for image-only and sensor-only fallback. Select thresholds and fit any
calibration under a recorded protocol with independent validation data.
Report product-level metrics separately from detector AP. Reserve internal
Test for a future evaluation after choices are frozen. Temporal experiments
require multiple observed timestamps and verified decision-time ingestion
latency. Model accuracy and calibration remain unverified.
