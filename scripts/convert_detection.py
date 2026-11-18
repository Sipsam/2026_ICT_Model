#!/usr/bin/env python3
"""Export SMT candidate samples to YOLO and COCO."""

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ict_pipeline.data.preprocessing.console import console_progress, protect_source_data
from ict_pipeline.data.preprocessing.convert import convert_detection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="repository root")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/detection"))
    parser.add_argument("--image-mode", choices=("link", "copy"), default="link", help="hardlink or copy images")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="first N candidates, for testing")
    args = parser.parse_args()
    out = (args.root / args.out).resolve()
    try:
        if not args.dry_run:
            protect_source_data(out, args.root)
        report = convert_detection(
            args.root, args.root / args.manifest, out, image_mode=args.image_mode,
            dry_run=args.dry_run, limit=args.limit, restrict_sources=False,
            progress=console_progress("Convert"),
        )
    except FileExistsError:
        parser.error(f"Output already exists. Choose a new --out directory instead of {out}.")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.dry_run:
        print(f"Dry run   {report['selected_manifest_rows']} candidates   {report['by_split']}")
    else:
        print(f"Converted {report['selected_manifest_rows']} images   {report['counts']}")
        print(f"Dataset   {out}")


if __name__ == "__main__":
    main()
