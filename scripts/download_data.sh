
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SHELL_PATH="$ROOT/tools/aihubshell"
DATA_DIR="$ROOT/data/raw"

if [[ ! -x "$SHELL_PATH" ]]; then
echo "aihubshell 실행 파일이 없습니다."
exit 1
fi

if [[ -z "${AIHUB_API_KEY:-}" ]]; then
read -rsp "AI Hub API Key: " AIHUB_API_KEY
echo
fi

MODE="${1:-validation}"

# 냉납, 밀림, 쇼트, 미삽, 일어섬
# 정상 + 불량 / 이미지 + 센서 + 라벨

case "$MODE" in
validation)
FILEKEYS=(
# 이미지
565803 565804 565805 565807 565810
565815 565816 565817 565819 565822

# 센서
565767 565768 565769 565771 565774
565779 565780 565781 565783 565786

# 라벨
565839 565840 565841 565843 565846
565851 565852 565853 565855 565858
)
;;

training)
FILEKEYS=(
    # 이미지
    565695 565696 565697 565699 565702
565707 565708 565709 565711 565714

# 센서
565659 565660 565661 565663 565666
565671 565672 565673 565675 565678

# 라벨
565731 565732 565733 565735 565738
565743 565744 565745 565747 565750
)
;;

*)
echo "사용법: bash scripts/download_data.sh [validation|training]"
exit 1 \
;;
esac

FILEKEY_CSV=$(IFS=,; echo "${FILEKEYS[*]}")

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "다운로드 모드: $MODE"
echo "선택한 파일 수: ${#FILEKEYS[@]}"

     "$SHELL_PATH" \
     -mode d \
           -datasetkey 71900 \
                       -filekey "$FILEKEY_CSV" \
                                -aihubapikey "$AIHUB_API_KEY"

echo "다운로드 완료"
