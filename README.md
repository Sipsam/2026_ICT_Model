# ICT Pipeline · PCB SMT 결함 탐지 및 분류

AI Hub SMT 납땜공정 데이터로 PCB 이미지의 결함 위치를 찾고, 이미지와 공정 센서를 이용해 검사 이미지에 해당하는 결함 종류를 예측한다. 결함 다섯 종류는 냉납, 밀림, 쇼트, 미삽, 일어섬이다. 객체 탐지는 상자와 클래스 점수를 반환하고, multi-label 분류는 이미지 한 장당 클래스별 존재 점수 다섯 개를 반환한다.

- AI Hub 데이터 취득·검증, Manifest 작성, YOLO/COCO 변환, 누출 점검, Group Split 발행 및 검증을 지원한다.
- Faster R-CNN/MobileNetV3-FPN, YOLO11n, RT-DETR-R18 탐지 어댑터와 Image-only, Sensor-only, 이미지–센서 분류를 제공한다.
- 실행 단계 재개, 고정된 전처리·체크포인트 검증, Validation 비교, 센서 셔플 분석, 제한 내부 Holdout 절차를 제공한다. 검사 CLI, REST API와 SQLite 기반 이력은 별도 운영 인터페이스다.

## 최종 결과

Group Split은 Train 11,038장, Validation 2,208장, Test 1,472장(총 14,718장)이다. 조기 확정한 Faster R-CNN의 Group Validation bbox mAP@50은 **94.25%**, mAP@50:95는 **77.16%**였다.

| 지표 | Image-only | Sensor Concat |
| --- | ---: | ---: |
| 선택 Epoch | 27 | 15 |
| Validation Macro AUROC | 98.42% | 99.53% |
| Validation Macro F1 | 90.00% | 93.96% |
| 내부 Holdout Macro AUROC | 98.67% | 99.54% |
| 내부 Holdout Macro F1 | 90.43% | 94.03% |
| 내부 Holdout Micro F1 | 89.77% | 93.63% |
| 내부 Holdout FP / FN | 45 / 144 | 45 / 77 |
| 내부 Holdout FPR | 0.70% | 0.70% |

Sensor Concat의 Validation Macro F1은 원본 짝 93.96%, 제조사 내 셔플 90.37%, 전역 셔플 83.66%였다. 최종 평가는 `limited_internal_holdout` / `internal_same_source`이다. 설정된 검사에서 중복이나 분할 간 유사성은 검출되지 않았다. 과거 Test 예약과 원본 봉인, 외부 출처 및 생산 세션 독립성은 확인되지 않았다. FP와 FN은 샘플별 클래스 판정 합계다. [최종 연구 보고서](docs/final-report.md)

<a id="install"></a>

## 설치 및 지원 환경

Python 3.11 이상과 가상환경을 사용한다. 로컬 CLI, 진단, 전처리 및 CPU 테스트는 Windows와 Linux에서 실행할 수 있다. WSL2 및 macOS 사용 절차와 플랫폼별 점검 범위는 별도 문서에 있다. GPU 재현은 해당 호스트에 맞는 PyTorch/TorchVision 빌드가 필요하며, 최종 연구 실험은 Ubuntu CUDA에서 수행했다. [플랫폼 점검](docs/platforms.md) · [실행 환경](docs/runtime.md)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[preprocess,test]"
```

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[preprocess,test]'
```

필요한 경우 백엔드별 extra(`train-faster-rcnn`, `train-yolo`, `train-rt-detr`)를 선택한다. REST API 테스트에는 `api,test-api`, 대시보드 JavaScript 테스트에는 Node.js가 필요하다. 설치와 데이터·가중치 다운로드, 학습은 각각 승인 후 실행한다. [모델 및 가중치](docs/models.md)

<a id="first-local-run"></a>

## CLI와 데이터 준비

AI Hub의 공개 데이터셋 번호는 **71900**이다. 승인된 JPG, JSON, XLSX를 사용하고 원본은 읽기 전용으로 취급한다. 다운로드가 필요한 경우 미리보기를 먼저 확인하고 키는 환경 또는 Git에서 제외되는 `.env`에 둔다. [데이터 접근](docs/data-access.md) · [라이선스](docs/licenses.md)

```text
ict --help
ict doctor
ict setup --non-interactive --dataset-source skip --data-path data/extracted --dry-run
ict data status
ict run --run-id smt --dry-run --until validate_detection
```

검토 후 `ict run --run-id smt --until validate_detection`으로 전처리를 실행한다. 이미 구성한 동일 작업은 `ict status --run-id smt`로 확인하고, `ict resume --run-id smt`로 검증 및 재개한다. 명령은 `runs/<run-id>/` 아래 단계별 상태와 산출물을 남긴다.

<a id="run-end-to-end-detection"></a>

## 탐지, 분류 및 최종 실험 재현

```text
ict run --run-id detector-preview --model faster_rcnn --dry-run
ict train --model faster_rcnn --dataset path/to/detection --dry-run
ict evaluate --help
ict multimodal train --config configs/local/fpn-image-only.yaml --out outputs/reproduction/image --dry-run
ict integrity verify-group --help
```

위의 `path/to/detection`에는 COCO/YOLO 변환 디렉터리를 지정하고 `configs/local/`에는 검증된 입력을 연결한 프로필을 둔다. 새 학습은 승인과 새 출력 경로가 필요하다. Group 모델의 재현에는 이미 발행된 revision, 검증 영수증, sanitized cohort, FrozenValidation 및 Train-fitted 센서 상태가 필요하다. 두 분류 프로필은 SHA-256으로 고정된 동일 탐지기에서 시작한다. 보존된 최종 실험을 확인하려면 원래 결과와 체크포인트 해시를 대조한다. Test 관련 명령은 별도의 승인과 기록을 요구한다. [Group 검증, 학습 및 조기 체크포인트 확정](docs/group-retraining.md) · [그룹 분석과 셔플](docs/multimodal-analysis.md) · [Test 절차](docs/dataset-integrity.md)

```text
python -m pytest -ra -m "not cpu_backend and not real_data_smoke"
node --test tests/dashboard.test.mjs
```

## 저장소와 산출물

| 경로 | 역할 |
| --- | --- |
| `src/ict_pipeline/cli.py`, `config/` | Typer CLI와 패키지 기본 설정 |
| `data/`, `data/preprocessing/` (패키지 내) | 취득, 감사, 변환, 누출 점검, 분할, Test 영수증 |
| `models/`, `classification/`, `multimodal/` (패키지 내) | 탐지, 기존 분류 작업, 센서 전처리와 멀티모달 학습·추론 |
| `evaluation/`, `reporting/`, `pipeline/`, `runtime/` (패키지 내) | 지표, 보고서, 단계 상태 및 실행 환경 |
| `inspection/`, `api/`, `streaming/` (패키지 내) | 검사 요청, HTTP API·대시보드, 재생·스트리밍 |
| `scripts/`, `tests/`, `docs/profiles/` | 독립 도구, 계약 테스트, 버전 관리 예제 설정 |
| `data/{raw,extracted,processed}/`, `runs/`, `outputs/` | Git 외부의 데이터와 생성 산출물 |

`outputs/final/` 전체는 보호된 로컬 연구 아카이브다. 보존된 보고서를 읽어 검증하고 별도 경로에 파생 보고서를 생성한다. 원본 데이터, 회원 키, 환경 정보, 개별 예측 및 체크포인트는 Git 밖에서 관리한다. [산출물 관리](docs/artifact-management.md)

## 문서

- 연구 방법, 실험 이력, 출처 및 그림은 [최종 연구 보고서](docs/final-report.md).
- 코드 경계, 스크립트 상태 및 변경 근거는 [유지보수 점검](docs/maintenance-audit.md).
- 학습·재현·서비스 안내 전체는 [문서 색인](docs/README.md).
