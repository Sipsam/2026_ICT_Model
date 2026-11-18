"""Publish classification checkpoints and Validation comparisons through the DAG runner."""

import csv
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..data.preprocessing.common import write_csv, write_json
from ..data.stages import limited_progress
from ..pipeline.model import StageContext, StageFailure, StageResult
from ..pipeline.store import digest_json
from .contracts import LABEL_POLICY, SCHEMA_VERSION, ClassificationConfig, Modality, Sample
from .models import IMAGE_FEATURES, SENSOR_FEATURES, LinearModel, Unimodal, features, fit_model
from .prepare import prepare_samples


def read_samples(path: Path) -> Iterable[Sample]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield Sample.model_validate_json(line)


class Checkpoint(BaseModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    label_policy: str = LABEL_POLICY
    config: ClassificationConfig
    implementation_fingerprint: str
    source_fingerprint: str
    split_fingerprint: str
    prepared_fingerprints: dict[str, str]
    models: dict[Unimodal, LinearModel]
    train_counts: dict[Unimodal, dict[int, int]]
    train_cohorts: dict[Unimodal, str]


def _cohort_fingerprint(samples: Iterable[Sample]) -> str:
    return digest_json(sorted((sample.sample_id, sample.label) for sample in samples))


def _metrics(pairs: list[tuple[int, float]]) -> dict[str, int | float | None]:
    counts = Counter((label, int(score >= 0.5)) for label, score in pairs)
    tn, fp, fn, tp = (counts[0, 0], counts[0, 1], counts[1, 0], counts[1, 1])

    def ratio(top: int, bottom: int) -> float | None:
        return top / bottom if bottom else None

    recall = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "samples": len(pairs), "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": ratio(tp + tn, len(pairs)),
        "balanced_accuracy": (recall + specificity) / 2 if recall is not None and specificity is not None else None,
        "precision": ratio(tp, tp + fp), "recall": recall, "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


@dataclass(frozen=True)
class ClassificationStages:
    config: ClassificationConfig
    implementation_fingerprint: str
    approved: bool = False

    def prepare(self, context: StageContext) -> StageResult:
        try:
            prepare_samples(
                context.inputs["dataset"], context.output_dir, self.config,
                context.input_fingerprints["dataset"], limited_progress(context.progress),
            )
        except ImportError:
            raise StageFailure("Classification preparation needs the preprocessing extra for JPG and XLSX decoding.") from None
        except ValueError:
            raise StageFailure("Cannot prepare classification data. Check the SMT annotation directories and file format.") from None
        return StageResult.SUCCESS

    def train(self, context: StageContext) -> StageResult:
        if not self.approved:
            raise StageFailure("Review the classification dry run, then pass --approve-train to authorize training.")
        alignment = json.loads(context.inputs["alignment"].read_text(encoding="utf-8"))
        models: dict[Unimodal, LinearModel] = {}
        counts: dict[Unimodal, dict[int, int]] = {}
        cohorts: dict[Unimodal, str] = {}
        modalities: tuple[Unimodal, ...] = ("sensor",) if self.config.mode == "sensor" else ("sensor", "image")
        samples = [
            sample for sample in read_samples(context.inputs["samples"])
            if sample.split == "train" and (self.config.missing == "available" or (sample.sequence and sample.pixels))
        ]
        sensor_units = 1 if self.config.sensor_model == "centroid" else self.config.epochs
        total = sensor_units + (self.config.epochs if "image" in modalities else 0)
        progress = limited_progress(context.progress)
        progress(0, total)
        for modality in modalities:
            cohort = [(sample, vector) for sample in samples if (vector := features(sample, modality, self.config)) is not None]
            offset = sensor_units if modality == "image" else 0

            def report(done: int) -> None:
                progress(offset + done, total)

            models[modality] = fit_model(modality, [(vector, s.label) for s, vector in cohort], self.config, report)
            counts[modality] = dict(Counter(s.label for s, _ in cohort))
            cohorts[modality] = _cohort_fingerprint(s for s, _ in cohort)
        checkpoint = Checkpoint(
            config=self.config, implementation_fingerprint=self.implementation_fingerprint,
            source_fingerprint=alignment["source_fingerprint"],
            split_fingerprint=alignment["split_fingerprint"], prepared_fingerprints=dict(context.input_fingerprints),
            models=models, train_counts=counts, train_cohorts=cohorts,
        )
        with (context.output_dir / "checkpoint.json").open("x", encoding="utf-8") as stream:
            stream.write(checkpoint.model_dump_json(indent=2))
        return StageResult.SUCCESS

    def evaluate(self, context: StageContext) -> StageResult:
        checkpoint = Checkpoint.model_validate_json(context.inputs["checkpoint"].read_bytes())
        if (
            checkpoint.label_policy != LABEL_POLICY or checkpoint.config != self.config
            or checkpoint.implementation_fingerprint != self.implementation_fingerprint
            or any(checkpoint.prepared_fingerprints.get(key) != context.input_fingerprints[key] for key in ("samples", "alignment"))
            or set(checkpoint.models) != ({"sensor"} if self.config.mode == "sensor" else {"sensor", "image"})
            or any(
                model.feature_names != (SENSOR_FEATURES if modality == "sensor" else IMAGE_FEATURES)
                for modality, model in checkpoint.models.items()
            )
        ):
            raise StageFailure("The classification checkpoint does not match the prepared data or configuration.")

        validation = [sample for sample in read_samples(context.inputs["samples"]) if sample.split == "validation"]
        if not validation:
            raise StageFailure("No eligible Validation samples remain. Inspect alignment and group exclusions.")
        progress = limited_progress(context.progress)
        progress(0, len(validation))
        results: dict[tuple[str, Modality], list[tuple[int, float]]] = {
            (cohort, modality): [] for cohort in ("paired", "available", "sensor_only", "image_only")
            for modality in self.config.modalities
        }
        prediction_rows = []
        coverage: Counter[str] = Counter()
        paired = []
        for done, sample in enumerate(validation, 1):
            cohort = "paired" if sample.sequence and sample.pixels else "sensor_only" if sample.sequence else "image_only"
            coverage[cohort] += 1
            if cohort == "paired":
                paired.append(sample)
            if self.config.missing == "available" or cohort == "paired":
                scores: dict[Modality, float] = {}
                for modality, model in checkpoint.models.items():
                    vector = features(sample, modality, self.config)
                    if vector is not None:
                        scores[modality] = model.predict(vector)
                if "fusion" in self.config.modalities and scores:
                    scores["fusion"] = sum(scores.values()) / len(scores)
                for modality, score in scores.items():
                    results[cohort, modality].append((sample.label, score))
                    results["available", modality].append((sample.label, score))
                    prediction_rows.append({
                        "sample_id": sample.sample_id, "cohort": cohort, "label": sample.label,
                        "modality": modality, "score": str(score), "prediction": int(score >= 0.5),
                    })
            progress(done, len(validation))
        comparisons = [
            {"cohort": cohort, "modality": modality, **_metrics(pairs)}
            for (cohort, modality), pairs in results.items()
        ]
        report = {
            "schema_version": SCHEMA_VERSION, "split": "validation", "label_policy": LABEL_POLICY,
            "source_fingerprint": checkpoint.source_fingerprint, "split_fingerprint": checkpoint.split_fingerprint,
            "checkpoint_fingerprint": context.input_fingerprints["checkpoint"], "config": self.config.model_dump(),
            "decision_threshold": 0.5, "validation_samples": len(validation), "coverage": dict(coverage),
            "paired_sample_fingerprint": _cohort_fingerprint(paired), "train_counts": checkpoint.train_counts,
            "comparisons": comparisons,
        }
        write_json(context.output_dir / "comparison.json", report)
        with (context.output_dir / "comparison.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
            writer.writeheader()
            writer.writerows(comparisons)
        write_csv(context.output_dir / "predictions.csv", prediction_rows,
                  ["sample_id", "cohort", "label", "modality", "score", "prediction"])
        if not prediction_rows:
            raise StageFailure("No Validation samples are scorable for the configured modalities. Inspect coverage.")
        return StageResult.SUCCESS
