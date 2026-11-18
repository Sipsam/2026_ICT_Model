import json
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT = Path("data/extracted")

offsets = Counter()
examples = {}

files = list(ROOT.rglob("*.json"))

for path in tqdm(files, desc="센서 ID 분석", unit="file"):
    try:
        with path.open(encoding="utf-8-sig") as f:
            data = json.load(f)

        images = data.get("images", [])
        sensors = data.get("sensor_data", [])

        if not images or not sensors:
            offsets["missing"] += 1
            continue

        image_id = images[0].get("id")
        sensor_id = sensors[0].get("image_id")

        if not isinstance(image_id, int) or not isinstance(sensor_id, int):
            offsets["invalid_id"] += 1
            continue

        offset = sensor_id - image_id
        offsets[offset] += 1

        if offset not in examples:
            examples[offset] = path.name

    except Exception as e:
        offsets["error"] += 1
        print(f"\n오류: {path.name}: {e}")

print("\n=== 센서 ID 오프셋 분포 ===")
for offset, count in offsets.most_common():
    print(f"offset={offset}: {count:,}개")
    if offset in examples:
        print(f"  예시: {examples[offset]}")
