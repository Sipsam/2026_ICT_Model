"""Prepare a sealed, bounded SMT smoke experiment without training or networking."""

import argparse
import json
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import yaml

from .data.preprocessing.common import write_csv, write_json
from .data.preprocessing.console import console_progress, fresh_output, protect_source_data
from .data.preprocessing.convert import convert_detection
from .data.preprocessing.leakage import MODALITIES, file_sha, source_hashes, source_rows
from .data.preprocessing.policy import NAMES, label_policy
from .data.preprocessing.revision import SEAL, seal_dataset
from .data.preprocessing.split import SplitConfig, plan_split
from .data.preprocessing.validate import validate_detection
from .evaluation.contracts import EvaluationConfig
from .models.contracts import TrainConfig
from .models.weights import LAYOUTS
from .pipeline.store import digest_json

SPLIT_CONFIG = SplitConfig(group_near_images=True, group_near_sensors=True)
STRATA = (*NAMES, "background")
PER_STRATUM = 2
IMAGE_SIZE = 128
SOURCES = {
    "yolo": ("yolo11n.pt", "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
             "v8.3.0", "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
             "AGPL-3.0", "ultralytics_pickle"),
    "faster_rcnn": ("fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
                    "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
                    "COCO_V1", "907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5",
                    "TorchVision COCO V1 - upstream weight and COCO terms", "state_dict"),
    "rt_detr": ("model.safetensors",
                "https://huggingface.co/PekingU/rtdetr_r18vd/resolve/ac77a11ff0170a41b771c03264987f8ce2b0d753/model.safetensors",
                "ac77a11ff0170a41b771c03264987f8ce2b0d753",
                "fe87a5a30f5daf298d10794c7682a63b6107986f97d6a770ba948d89e4340093",
                "Apache-2.0", "safetensors"),
}


def select_samples(report: dict[str, Any], plan: dict[str, Any]) -> list[dict[str, Any]]:
    """Sample whole singleton groups within their assigned split."""
    sizes = Counter(entry["group_id"] for entry in plan["assignments"])
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in plan["assignments"]:
        sample = report["samples"][entry["sample_index"]]
        labels = sample["classes"]
        if sizes[entry["group_id"]] == 1 and len(labels) <= 1:
            label = labels[0] if labels else len(NAMES)
            buckets.setdefault((entry["split"], label), []).append(entry)
    selected = []
    for split in ("train", "validation", "test"):
        for label, name in enumerate(STRATA):
            pool = buckets.get((split, label), [])
            if len(pool) < PER_STRATUM:
                raise ValueError(f"Insufficient singleton groups for {split} / {name}.")
            pool.sort(key=lambda entry: digest_json([SPLIT_CONFIG.seed, entry["group_id"]]))
            selected.extend(pool[:PER_STRATUM])
    return sorted(selected, key=lambda entry: entry["sample_index"])


def smoke_profiles(cache: Path) -> dict[str, TrainConfig]:
    profiles = {}
    for name, (filename, origin, release, sha256, license_terms, fmt) in SOURCES.items():
        profiles[name] = TrainConfig(
            epochs=1, batch_size=2, image_size=IMAGE_SIZE, lr=0.0001, initialization="pretrained",
            freeze_backbone=True, checkpoint={
                "source": (cache.resolve() / filename).as_posix(), "origin": origin, "revision": release,
                "sha256": sha256, "license": license_terms, "architecture": LAYOUTS[name].architecture,
                "library_version": {"yolo": "8.3.228", "rt_detr": "4.57.6"}.get(name)
                or version("torchvision"), "format": fmt, "trusted_pickle": fmt == "ultralytics_pickle",
            },
        )
    return profiles


def prepare(root: Path, manifest: Path, base: Path, analysis: Path, out: Path, cache: Path) -> dict[str, Any]:
    root = root.resolve()
    protect_source_data(out, Path.cwd())
    if out.exists() or out.is_symlink():
        raise ValueError("Choose a fresh smoke output directory.")
    configs = smoke_profiles(cache)
    payload = json.loads(analysis.read_text(encoding="utf-8-sig"))
    report = payload["report"]
    if digest_json(report) != payload["fingerprint"]:
        raise ValueError("Leakage report fingerprint mismatch.")
    rows = source_rows(manifest, root, base)
    if digest_json(rows) != report["manifest_fingerprint"]:
        raise ValueError("Manifest changed since leakage analysis.")
    plan = plan_split(report, SPLIT_CONFIG)
    if any(plan["missing_classes"].values()) or set(plan["grouping_methods"]) & plan["remaining_cross_split_groups"].keys():
        raise ValueError("The parent split preview has missing classes or grouped cross-split findings.")
    assignments = select_samples(report, plan)
    progress = console_progress("Recheck leakage evidence bytes")
    progress(0, len(rows))
    for done, (row, sample) in enumerate(zip(rows, report["samples"], strict=True), 1):
        if source_hashes(root, row) != sample["hashes"]:
            raise ValueError("Source files changed since leakage analysis.")
        progress(done, len(rows))

    selected = []
    evidence = []
    hashes = set()
    for entry in assignments:
        index = entry["sample_index"]
        row = rows[index]
        sample = report["samples"][index]
        image_sha = sample["hashes"]["image_path"]
        if image_sha in hashes:
            raise ValueError("Smoke cohort contains duplicate image bytes.")
        hashes.add(image_sha)
        selected.append({**row, "original_split": row["split"], "split": entry["split"], "group_id": entry["group_id"]})
        evidence.append({**entry, "sample_id": row["sample_id"], "original_split": row["split"],
                         "classes": sample["classes"], "hashes": sample["hashes"]})

    revision = {
        "schema_version": 1, "label_policy": label_policy(), "test_origin": "internal_same_source",
        "analysis_fingerprint": payload["fingerprint"], "dataset_fingerprint": report["dataset_fingerprint"],
        "source_manifest_fingerprint": report["manifest_fingerprint"], "parent_preview_fingerprint": digest_json(plan),
        "configuration": plan["configuration"], "grouping_methods": plan["grouping_methods"],
        "selected_manifest_fingerprint": digest_json(selected),
        "selected_source_fingerprint": digest_json([
            [{key: row[key] for key in MODALITIES}, entry["hashes"]]
            for row, entry in zip(selected, evidence, strict=True)
        ]),
        "selected_images_fingerprint": digest_json([entry["hashes"]["image_path"] for entry in evidence]),
        "selected_annotations_fingerprint": digest_json([entry["hashes"]["annotation_path"] for entry in evidence]),
        "counts": dict(Counter(row["split"] for row in selected)),
        "class_samples": {split: dict.fromkeys(STRATA, PER_STRATUM) for split in plan["counts"]},
        "assignments": evidence,
    }
    revision["revision_id"] = digest_json(revision)
    with fresh_output(out, root) as working:
        selected_manifest = working / "manifest.csv"
        dataset = working / "detection"
        write_csv(selected_manifest, selected, list(selected[0]))
        convert_detection(root, selected_manifest, dataset, include_test=True, portable_yaml=True,
                          progress=console_progress("Convert smoke cohort"))
        validation = validate_detection(selected_manifest, dataset, working / "validation", include_test=True,
                                        progress=console_progress("Validate smoke labels"))
        if not validation["passed"]:
            raise ValueError("Smoke cohort failed YOLO/COCO validation.")
        for row, entry in zip(selected, evidence, strict=True):
            split = "val" if row["split"] == "validation" else row["split"]
            image = Path(row["image_path"]).name
            if any(file_sha(dataset / fmt / "images" / split / image) != entry["hashes"]["image_path"]
                   for fmt in ("yolo", "coco")) or file_sha(root / row["annotation_path"]) != entry["hashes"]["annotation_path"]:
                raise ValueError("Selected source changed during conversion.")
        write_json(dataset / SEAL, seal_dataset(dataset, revision))
        write_json(working / "split.json", revision)
        write_json(working / "preparation.json", {
            "provider_candidate_counts": report["candidate_split_counts"],
            "provider_preview": {k: v for k, v in plan_split(report, SplitConfig(
                strategy="aihub", group_near_images=True, group_near_sensors=True,
            )).items() if k != "assignments"},
            "group_aware_preview": {k: v for k, v in plan.items() if k != "assignments"},
            "validation": validation,
        })
        write_json(working / "protocol.json", {
            "split": "val", "checkpoint": "best", "dataset_revision": revision["revision_id"],
            "evaluation": EvaluationConfig(image_size=IMAGE_SIZE, warmup=1).model_dump(mode="json"),
        })
        profiles = working / "profiles"
        profiles.mkdir()
        for name, config in configs.items():
            (profiles / f"{name}.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    return {key: value for key, value in revision.items() if key != "assignments"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("root", "manifest", "path-base", "analysis", "out", "cache"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    try:
        revision = prepare(args.root, args.manifest, args.path_base, args.analysis, args.out, args.cache)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Smoke preparation failed ({type(exc).__name__}). Check the inputs and output directory.\n")
    print(json.dumps(revision, indent=2))


if __name__ == "__main__":
    main()
