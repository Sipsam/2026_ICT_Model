#!/usr/bin/env python3
"""Validate YOLO and COCO exports against the SMT manifest."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ict_pipeline.data.preprocessing.console import console_progress, report_output
from ict_pipeline.data.preprocessing.validate import validate_detection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--dataset", type=Path, default=Path("data/processed/detection"))
    parser.add_argument("--out", type=Path, default=Path("outputs/detection_validation"))
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--overwrite", action="store_true", help="replace existing validation reports")
    args = parser.parse_args()
    try:
        with report_output(
            args.out, ("issues.csv", "summary.json"), overwrite=args.overwrite, sources=(args.dataset,),
        ) as out:
            report = validate_detection(
                args.manifest, args.dataset, out, tolerance=args.tolerance, progress=console_progress("Validate"),
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Images {report['selected_images']}   issues {report['issue_count']}   {report['issue_types']}")
    print(f"Reports   {args.out.resolve()}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
