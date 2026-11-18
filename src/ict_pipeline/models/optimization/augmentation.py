"""Deterministic SMT image and absolute COCO box augmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from random import Random
from typing import TYPE_CHECKING

from ...data.preprocessing.policy import NAMES
from ...pipeline.store import digest_json
from ..contracts import TrainingError
from ..dataset import Object, Sample
from .config import Augmentation, Geometric, Photometric

if TYPE_CHECKING:
    from PIL.Image import Image

Point = tuple[float, float]
Affine = tuple[float, float, float, float, float, float]
EPSILON = 1e-6


@dataclass(frozen=True)
class AnnotationChanges:
    clipped: tuple[int, ...] = ()
    removed: tuple[int, ...] = ()
    insufficient_visibility: tuple[int, ...] = ()
    geometry_rejected: bool = False
    photometric_rejected: tuple[int, ...] = ()


def read_rgb(sample: Sample) -> Image:
    from PIL import Image

    with Image.open(sample.image) as image:
        if image.size != (sample.width, sample.height):
            raise TrainingError("COCO image dimensions differ from the encoded image. Rerun detection validation.")
        return image.convert("RGB")


def _photometric(image: Image, config: Photometric, rng: Random) -> Image:
    from PIL import Image, ImageChops, ImageEnhance

    for enhance, delta in (
        (ImageEnhance.Brightness, config.brightness),
        (ImageEnhance.Contrast, config.contrast),
        (ImageEnhance.Color, config.saturation),
    ):
        if delta:
            image = enhance(image).enhance(1 + rng.uniform(-delta, delta))
    if config.noise:
        # One bounded, zero-mean noise value per pixel shared by RGB channels.
        # Stratify the 256 possible bytes symmetrically across signed levels.
        levels = [min(config.noise, i * (config.noise + 1) // 128) for i in range(128)]
        raw = rng.randbytes(image.width * image.height)
        positive = Image.frombytes("L", image.size, raw.translate(bytes(levels + [0] * 128)))
        negative = Image.frombytes("L", image.size, raw.translate(bytes([0] * 128 + levels)))
        image = ImageChops.subtract(ImageChops.add(image, positive.convert("RGB")), negative.convert("RGB"))
    return image


def _photo_evidence_loss(original: Image, changed: Image, objects: tuple[Object, ...]) -> tuple[int, ...]:
    """Flag substantial new saturation or contrast loss in labeled regions."""
    from PIL import ImageChops, ImageStat

    rejected = []
    saturated = [255] + [0] * 254 + [255]
    for index, obj in enumerate(objects):
        x, y, w, h = obj.box
        region = (math.floor(x), math.floor(y), min(original.width, math.ceil(x + w)),
                  min(original.height, math.ceil(y + h)))
        before, after = original.crop(region), changed.crop(region)
        added = ImageChops.subtract(after.point(saturated * 3), before.point(saturated * 3))
        added_fraction = sum(ImageStat.Stat(added).mean) / (3 * 255)
        before_contrast = ImageStat.Stat(before.convert("L")).stddev[0]
        after_contrast = ImageStat.Stat(after.convert("L")).stddev[0]
        if added_fraction > 0.02 or (before_contrast >= 2 and after_contrast < 0.85 * before_contrast):
            rejected.append(index)
    return tuple(rejected)


def _geometry(size: tuple[int, int], config: Geometric, rng: Random) -> tuple[Affine, tuple[int, int]]:
    width, height = size
    angle = math.radians(rng.uniform(-config.rotation_degrees, config.rotation_degrees))
    scale = 1 + rng.uniform(-config.scale_delta, config.scale_delta)
    a, d = scale * math.cos(angle), scale * math.sin(angle)
    b, e = -d, a
    c = width * (0.5 + rng.uniform(-config.translation_fraction, config.translation_fraction))
    f = height * (0.5 + rng.uniform(-config.translation_fraction, config.translation_fraction))
    c -= (a * width + b * height) / 2
    f -= (d * width + e * height) / 2
    if rng.random() < config.horizontal_flip_probability:
        a, b, c = -a, -b, width - c
    left, right = (rng.randint(0, int(width * config.crop_fraction)) for _ in range(2))
    top, bottom = (rng.randint(0, int(height * config.crop_fraction)) for _ in range(2))
    return (a, b, c - left, d, e, f - top), (width - left - right, height - top - bottom)


def _clip(points: list[Point], size: tuple[int, int]) -> list[Point]:
    """Clip a transformed quadrilateral to the output canvas."""
    width, height = size
    for axis, edge, side in ((0, 0, 1), (0, width, -1), (1, 0, 1), (1, height, -1)):
        clipped: list[Point] = []
        for before, after in zip(points[-1:] + points[:-1], points):
            before_inside = (before[axis] - edge) * side >= 0
            after_inside = (after[axis] - edge) * side >= 0
            if before_inside != after_inside:
                fraction = (edge - before[axis]) / (after[axis] - before[axis])
                other = 1 - axis
                value = before[other] + fraction * (after[other] - before[other])
                clipped.append((edge, value) if axis == 0 else (value, edge))
            if after_inside:
                clipped.append(after)
        points = clipped
    return points


def _area(points: list[Point]) -> float:
    if not points:
        return 0.0
    origin_x, origin_y = points[0]
    return abs(sum(
        (x - origin_x) * (next_y - origin_y) - (next_x - origin_x) * (y - origin_y)
        for (x, y), (next_x, next_y) in zip(points, points[1:] + points[:1], strict=True)
    )) / 2


def transform_objects(
    objects: tuple[Object, ...], matrix: Affine, size: tuple[int, int], min_visibility: float,
) -> tuple[tuple[Object, ...], AnnotationChanges]:
    """Describe a proposed affine transform. The caller rejects unsafe proposals."""
    a, b, c, d, e, f = matrix
    determinant = abs(a * e - d * b)
    retained, clipped, removed, insufficient = [], [], [], []
    for index, obj in enumerate(objects):
        x, y, w, h = obj.box
        polygon = [
            (a * px + b * py + c, d * px + e * py + f)
            for px, py in ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
        ]
        visible = _clip(polygon, size)
        area = _area(visible)
        if not area:
            removed.append(index)
            continue
        xs, ys = zip(*visible, strict=True)
        left, top = max(0.0, min(xs)), max(0.0, min(ys))
        right, bottom = min(size[0], max(xs)), min(size[1], max(ys))
        ratio = area / (determinant * w * h)
        if ratio < 1 - EPSILON:
            clipped.append(index)
        smallest = min(w, h)
        if (ratio < min_visibility - EPSILON
                or min(right - left, bottom - top, smallest * math.sqrt(determinant)) < min(smallest, 2) - EPSILON):
            insufficient.append(index)
        retained.append(Object(obj.category, (left, top, right - left, bottom - top)))
    rejected = bool(removed or insufficient)
    changes = AnnotationChanges(tuple(clipped), tuple(removed), tuple(insufficient), rejected)
    return (objects if rejected else tuple(retained)), changes


def _valid(obj: Object, sample: Sample) -> bool:
    if len(obj.box) != 4 or any(type(value) not in (int, float) for value in obj.box):
        return False
    x, y, w, h = obj.box
    return (
        type(obj.category) is int and 1 <= obj.category <= len(NAMES)
        and all(math.isfinite(value) for value in (x, y, w, h))
        and 0 <= x < min(sample.width, x + w) and 0 <= y < min(sample.height, y + h)
        and x + w <= sample.width + EPSILON and y + h <= sample.height + EPSILON
    )


def augment(
    image: Image, sample: Sample, config: Augmentation, seed: int,
) -> tuple[Image, Sample, AnnotationChanges]:
    from PIL import Image

    if image.size != (sample.width, sample.height) or image.mode != "RGB":
        raise TrainingError("Augmentation requires a decoded RGB image matching its COCO dimensions.")
    invalid = [index for index, obj in enumerate(sample.objects) if not _valid(obj, sample)]
    if invalid:
        raise TrainingError(f"Augmentation rejected invalid annotation indices {invalid}.")
    photo_rejected: tuple[int, ...] = ()
    if config.photometric is not None:
        photo = config.photometric
        rng = Random(digest_json(["smt-photo-v1", seed, sample.image.stem, photo.model_dump()]))
        proposed = _photometric(image, photo, rng)
        if proposed is not image:
            photo_rejected = _photo_evidence_loss(image, proposed, sample.objects)
            if not photo_rejected:
                image = proposed
    if config.geometric is None:
        return image, sample, AnnotationChanges(photometric_rejected=photo_rejected)

    geo = config.geometric
    rng = Random(digest_json(["smt-geometry-v1", seed, sample.image.stem, geo.model_dump()]))
    matrix, size = _geometry(image.size, geo, rng)
    if matrix == (1, 0, 0, 0, 1, 0) and size == image.size:
        return image, sample, AnnotationChanges(photometric_rejected=photo_rejected)
    objects, changes = transform_objects(sample.objects, matrix, size, geo.min_visibility)
    changes = replace(changes, photometric_rejected=photo_rejected)
    if changes.geometry_rejected:
        return image, sample, changes
    a, b, c, d, e, f = matrix
    determinant = a * e - d * b
    inverse = (e, -b, b * f - e * c, -d, a, d * c - a * f)
    transformed = image.transform(
        size, Image.Transform.AFFINE, tuple(value / determinant for value in inverse),
        resample=Image.Resampling.BILINEAR, fillcolor=(114, 114, 114),
    )
    return transformed, replace(sample, width=size[0], height=size[1], objects=objects), changes
