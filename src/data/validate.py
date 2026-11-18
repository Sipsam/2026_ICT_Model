import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path("data/extracted")
OUTPUT = Path("outputs/data_audit")
OUTPUT.mkdir(parents=True, exist_ok=True)

CLASS_CODES = {
    "CC": 23,  # 냉납
    "ML": 24,  # 밀림
    "SH": 25,  # 쇼트
    "MS": 27,  # 미삽
    "ST": 30,  # 일어섬
}

CLASS_NAMES = {
    23: "냉납",
    24: "밀림",
    25: "쇼트",
    27: "미삽",
    30: "일어섬",
}


# 32개 클래스 중 1~16은 정상, 17~32는 불량
def category_status(category_id):
    if 1 <= category_id <= 16:
        return "normal"
    if 17 <= category_id <= 32:
        return "defect"
    return "unknown"


def main():
    labels = sorted(ROOT.rglob("*.json"))

    category_counts = Counter()
    mismatch_counts = Counter()
    issues = []
    matched = 0

    for path in labels:
        try:
            with path.open(encoding="utf-8-sig") as f:
                data = json.load(f)

            stem = path.stem
            parts = stem.split("_")

            if len(parts) < 6:
                issues.append([stem, "invalid_filename", ""])
                continue

            process, status, code = parts[:3]

            if process != "SD":
                issues.append([stem, "unexpected_process", process])

            # 라벨 파일과 동일한 파일명을 가진
            # 이미지·센서 파일이 존재하는지 확인
            image_folder = path.parent.name.replace(
                "TL_", "TS_이미지데이터_", 1
            ).replace(
                "VL_", "VS_이미지데이터_", 1
            )

            sensor_folder = path.parent.name.replace(
                "TL_", "TS_센서데이터_", 1
            ).replace(
                "VL_", "VS_센서데이터_", 1
            )

            image_dir = ROOT / image_folder
            sensor_dir = ROOT / sensor_folder

            image_exists = any(
                (image_dir / f"{stem}{ext}").exists()
                for ext in (".JPG", ".jpg", ".JPEG", ".jpeg")
            )

            sensor_exists = (
                    (sensor_dir / f"{stem}.xlsx").exists()
                    or (sensor_dir / f"{stem}.csv").exists()
            )

            if image_exists and sensor_exists:
                matched += 1
            else:
                issues.append([
                    stem,
                    "missing_pair",
                    f"image={image_exists},sensor={sensor_exists}"
                ])

            annotations = data.get("annotations", [])
            category_ids = {
                ann["category_id"]
                for ann in annotations
                if "category_id" in ann
            }

            for ann in annotations:
                cid = ann.get("category_id")
                if cid is not None:
                    category_counts[cid] += 1

            expected = CLASS_CODES.get(code)

            if expected is not None:
                # 정상 클래스 ID = 불량 클래스 ID - 16
                expected_id = (
                    expected if status == "DEF"
                    else expected - 16
                )

                if expected_id not in category_ids:
                    mismatch_counts[code] += 1
                    issues.append([
                        stem,
                        "expected_category_missing",
                        f"expected={expected_id}, actual={sorted(category_ids)}"
                    ])

            # 원본 이미지 정보와 파일명 비교
            images = data.get("images", [])
            if not images or images[0].get("file_name", "").rsplit(".", 1)[0] != stem:
                issues.append([
                    stem,
                    "image_metadata_mismatch",
                    ""
                ])

        except Exception as e:
            issues.append([path.name, "parse_error", str(e)])

    with (OUTPUT / "issues.csv").open(
            "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["file", "issue", "detail"])
        writer.writerows(issues)

    report = {
        "total_json": len(labels),
        "matched_image_sensor_pairs": matched,
        "annotation_category_counts": dict(
            sorted(category_counts.items())
        ),
        "expected_category_missing_counts": dict(
            mismatch_counts
        ),
        "issue_count": len(issues),
    }

    with (OUTPUT / "summary.json").open(
            "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=== 데이터 검증 결과 ===")
    print("JSON 수:", len(labels))
    print("JPG/XLSX 매칭 성공:", matched)
    print("문제 기록 수:", len(issues))

    print("\n=== 어노테이션 클래스 분포 ===")
    for cid, count in sorted(category_counts.items()):
        print(f"category_id={cid}: {count}")

    print("\n=== 파일명상 예상 클래스 누락 ===")
    for code, count in mismatch_counts.items():
        print(f"{code}: {count}개")

    print("\n리포트:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
