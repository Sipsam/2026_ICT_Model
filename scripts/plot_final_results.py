"""Plot aggregate metrics from the preserved final research receipts."""

from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes

CLASS_NAMES = {
    "cold_solder": "Cold\nsolder",
    "misalignment": "Misalignment",
    "short": "Short",
    "missing_component": "Missing\ncomponent",
    "tombstone": "Tombstone",
}
MODEL_NAMES = ("Image-only", "Sensor Concat")
COLORS = ("#27658a", "#b64d27")


def read_receipt(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return document


def render(revision: Path) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    verification = read_receipt(revision / "verification/verification.json")[
        "verification"
    ]
    (detector_receipt,) = sorted(
        revision.glob("finalized-detector/*/finalization.json")
    )
    detector = read_receipt(detector_receipt)["finalization"]
    detection = detector["checks"]["validation_inference"]
    holdout = read_receipt(revision / "limited-internal-holdout/limited-holdout.json")
    if not (
        verification["passed"]
        and detector["status"] == "finalized_early"
        and detection["status"] == "passed"
        and holdout["status"] == "completed"
        and holdout["evaluation_protocol"] == "limited_internal_holdout"
        and holdout["evaluation_origin"] == "internal_same_source"
        and holdout["group_split_revision"]
        == verification["revision_id"]
        == revision.name
    ):
        raise ValueError(
            "Expected a verified group revision and completed limited internal holdout."
        )

    roles = ("image_only", "sensor_concat")
    models = [holdout["models"][role]["overall"] for role in roles]
    validation_f1 = [
        read_receipt(revision / "product" / profile / "run.json")["validation_report"][
            "macro_f1"
        ]
        for profile in ("fpn-image-only", "fpn-sensor-concat")
    ]
    summary = read_receipt(revision / "fpn-analysis/summary.json")
    ablation = {
        row["group"]: row["macro_f1"]
        for row in summary["ablation_metrics"]
        if row["status"] == "measured"
    }
    classes = [{row["name"]: row for row in model["classes"]} for model in models]

    with matplotlib.rc_context(
        {"svg.hashsalt": "ict-group-final-v1", "svg.fonttype": "none", "font.size": 11}
    ):
        figure, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")
        figure.suptitle(
            "PCB SMT defect models | Final group revision",
            fontsize=19,
            fontweight="bold",
        )
        panels = axes.ravel()
        for axis in panels:
            axis.spines[["top", "right"]].set_visible(False)
            axis.set_axisbelow(True)

        def single(
            axis: Axes, labels: list[str], values: list[float], *, percent: bool
        ) -> None:
            bars = axis.bar(labels, values, color=COLORS[0], width=0.58)
            axis.bar_label(bars, fmt="%.2f%%" if percent else "%.0f", padding=3)
            axis.set_ylim(0, 115 if percent else max(values) * 1.2)
            axis.grid(axis="y", alpha=0.2)

        def paired(
            axis: Axes, labels: list[str], rows: list[list[float]], *, percent: bool
        ) -> None:
            for index, values in enumerate(rows):
                locations = [
                    position + (index - 0.5) * 0.34 for position in range(len(labels))
                ]
                bars = axis.bar(
                    locations,
                    values,
                    width=0.32,
                    color=COLORS[index],
                    label=MODEL_NAMES[index],
                )
                axis.bar_label(
                    bars, fmt="%.2f%%" if percent else "%.0f", padding=3, fontsize=9
                )
            axis.set_xticks(range(len(labels)), labels)
            axis.set_ylim(0, 120 if percent else max(map(max, rows)) * 1.22)
            axis.grid(axis="y", alpha=0.2)

        panels[0].set_title("Group split | Images")
        single(
            panels[0],
            ["Train", "Validation", "Test"],
            [verification["counts"][key] for key in ("train", "validation", "test")],
            percent=False,
        )
        panels[1].set_title("Faster R-CNN | Validation bbox AP")
        single(
            panels[1],
            ["mAP@50", "mAP@50:95"],
            [100 * detection[key] for key in ("map50", "map50_95")],
            percent=True,
        )
        panels[2].set_title("Classifier Macro F1 | Fixed threshold 0.5")
        paired(
            panels[2],
            ["Validation", "Internal Holdout"],
            [
                [100 * score, 100 * model["macro_f1"]]
                for score, model in zip(validation_f1, models, strict=True)
            ],
            percent=True,
        )
        panels[2].legend(loc="lower right", frameon=False)
        panels[3].set_title("Validation Macro F1 | Sensor ablation")
        single(
            panels[3],
            ["Original", "Within manufacturer", "Global shuffle"],
            [
                100 * ablation[key]
                for key in ("original", "within_manufacturer_shuffle", "global_shuffle")
            ],
            percent=True,
        )
        panels[4].set_title("Internal Holdout | Per-class F1")
        paired(
            panels[4],
            list(CLASS_NAMES.values()),
            [[100 * rows[key]["f1"] for key in CLASS_NAMES] for rows in classes],
            percent=True,
        )
        panels[4].tick_params(axis="x", labelsize=9)
        panels[5].set_title("Internal Holdout | Label decisions")
        paired(
            panels[5],
            ["False positives", "False negatives"],
            [
                [sum(row[key] for row in rows.values()) for key in ("fp", "fn")]
                for rows in classes
            ],
            percent=False,
        )

        output = BytesIO()
        figure.savefig(output, format="svg", metadata={"Date": None})
        plt.close(figure)
    return b"\n".join(line.rstrip() for line in output.getvalue().splitlines()) + b"\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision-dir",
        required=True,
        type=Path,
        help="Final group revision containing the receipts.",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="New SVG file outside the protected archive.",
    )
    args = parser.parse_args()
    revision = args.revision_dir.resolve()
    protected = Path(__file__).resolve().parents[1] / "outputs/final"
    output = args.out.resolve()
    if (
        output.suffix.lower() != ".svg"
        or output.is_relative_to(protected)
        or output.is_relative_to(revision)
    ):
        parser.error(
            "Choose an SVG destination outside the protected archive and source revision."
        )
    try:
        content = render(revision)
        with output.open("xb") as stream:
            stream.write(content)
    except (ImportError, KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
