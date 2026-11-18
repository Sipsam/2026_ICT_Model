"""Train-only cohort selection and label frequencies."""

from collections import Counter
from random import Random

from ...data.preprocessing.policy import NAMES
from ...pipeline.store import digest_json
from ..contracts import TrainingError
from ..dataset import Sample
from .config import BACKGROUND, BUCKETS, Balancing


def image_buckets(sample: Sample) -> tuple[str, ...]:
    return tuple(NAMES[category - 1] for category in sorted({obj.category for obj in sample.objects})) or (BACKGROUND,)


def frequencies(samples: tuple[Sample, ...]) -> dict[str, object]:
    images = Counter(bucket for sample in samples for bucket in image_buckets(sample))
    objects = Counter(NAMES[obj.category - 1] for sample in samples for obj in sample.objects)
    return {
        "images": len(samples),
        "background_images": images[BACKGROUND],
        "multi_label_images": sum(len(image_buckets(sample)) > 1 for sample in samples),
        "image_counts": {name: images[name] for name in BUCKETS},
        "object_counts": {name: objects[name] for name in NAMES},
    }


def select_train(
    train: tuple[Sample, ...], config: Balancing | None, seed: int,
) -> tuple[Sample, ...]:
    """Draw one fixed, length-preserving training cohort with replacement when enabled."""
    if config is None:
        return train
    train = tuple(sorted(train, key=lambda sample: sample.image.name))
    buckets: dict[str, list[int]] = {name: [] for name in BUCKETS}
    memberships = [image_buckets(sample) for sample in train]
    for index, labels in enumerate(memberships):
        for label in labels:
            buckets[label].append(index)
    counts = {name: len(indices) for name, indices in buckets.items() if indices}
    if not counts:
        raise TrainingError("Balancing requires a nonempty Train split.")
    largest = max(counts.values())
    rng = Random(digest_json(["smt-cohort-v1", seed, config.model_dump()]))
    if config.method == "weighted":
        factors = {
            name: min(config.max_weight_ratio, (largest / count) ** config.strength)
            for name, count in counts.items()
        }
        weights = [max(factors[name] for name in labels) for labels in memberships]
        selected = tuple(rng.choices(train, weights=weights, k=len(train)))
    else:
        # Reweight bucket mass (starting at empirical membership mass), then sample uniformly within it.
        weights = [count * min(config.max_weight_ratio, (largest / count) ** config.strength)
                   for count in counts.values()]
        labels = rng.choices(list(counts), weights=weights, k=len(train))
        selected = tuple(train[rng.choice(buckets[name])] for name in labels)
    if not any(sample.objects for sample in selected):
        raise TrainingError("Balancing drew only background images. Change the training seed or balancing settings.")
    return selected


def cohort_record(train: tuple[Sample, ...], selected: tuple[Sample, ...]) -> dict[str, object]:
    ids = [sample.image.name for sample in selected]
    return {
        "algorithm": "smt-fixed-train-cohort-v1",
        "fingerprint": digest_json(ids),
        "unique_images": len(set(ids)),
        "original_train": frequencies(train),
        "selected_train": frequencies(selected),
    }
