#!/usr/bin/env python3
"""Build a traceable SMT manifest from a standalone audit."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ict_pipeline.data.preprocessing.console import console_progress, report_output
from ict_pipeline.data.preprocessing.manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/extracted"))
    parser.add_argument("--audit", type=Path, default=Path("outputs/data_audit_hash"))
    parser.add_argument("--out", type=Path, default=Path("data/processed"))
    parser.add_argument("--overwrite", action="store_true", help="replace existing manifest reports")
    args = parser.parse_args()
    try:
        with report_output(
            args.out, ("manifest.csv", "manifest_summary.json"), overwrite=args.overwrite, sources=(args.root, args.audit),
        ) as out:
            report = build_manifest(
                args.root, args.audit, out, path_base=args.root.resolve().parent.parent,
                progress=console_progress("Manifest"),
            )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if not report["audit_checks"].get("hash_images"):
        print("The audit did not check for identical image bytes across splits.")
    print(f"Samples {report['total_samples']}   decisions {report['by_decision']}")
    print(f"Reports   {args.out.resolve()}")


if __name__ == "__main__":
    main()
