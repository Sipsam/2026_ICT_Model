#!/usr/bin/env python3
"""Audit the SMT dataset and write reports without editing source data."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ict_pipeline.data.preprocessing.audit import audit_dataset
from ict_pipeline.data.preprocessing.console import console_progress, report_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out", type=Path, default=Path("outputs/data_audit_english"))
    parser.add_argument("--deep", action="store_true", help="verify images and compare JSON/XLSX sensor rows")
    parser.add_argument("--hash-images", action="store_true", help="check for identical images across splits")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="sensor numerical comparison tolerance")
    parser.add_argument("--overwrite", action="store_true", help="replace existing audit reports")
    args = parser.parse_args()
    try:
        with report_output(
            args.out, ("samples.csv", "issues.csv", "summary.json"), overwrite=args.overwrite, sources=(args.root,),
        ) as out:
            report = audit_dataset(
                args.root, out, deep=args.deep, hash_images=args.hash_images, tolerance=args.tolerance,
                progress=console_progress("Audit"),
            )
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Samples {report['total_samples']}   statuses {report['status_counts']}")
    print(f"Reports   {args.out.resolve()}")


if __name__ == "__main__":
    main()
