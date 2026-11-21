"""Preview an explicitly selected image, JSON annotation, and sensor workbook."""

import argparse
import json
from pathlib import Path
from pprint import pprint


def inspect_sample(image: Path, annotation: Path, sensor: Path) -> None:
    from openpyxl import load_workbook

    print(
        "File existence:",
        {
            "image": image.is_file(),
            "json": annotation.is_file(),
            "sensor": sensor.is_file(),
        },
    )
    with annotation.open(encoding="utf-8-sig") as stream:
        document = json.load(stream)
    if not isinstance(document, dict):
        raise ValueError("The annotation must be a JSON object.")
    for name, value in document.items():
        print(f"\n{name}:")
        if isinstance(value, list):
            print(f"{len(value)} entries")
            if value:
                pprint(value[0])
        else:
            pprint(value)

    print("\nSensor XLSX (header and first five rows):")
    workbook = load_workbook(sensor, read_only=True, data_only=True)
    try:
        for row in workbook.active.iter_rows(max_row=6, values_only=True):
            pprint(row)
    finally:
        workbook.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--sensor", type=Path, required=True)
    args = parser.parse_args()
    try:
        inspect_sample(args.image, args.annotation, args.sensor)
    except (ImportError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
