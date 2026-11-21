# 최종화 유지보수 점검

## 작업 전 상태와 디렉터리

작업은 `main`의 `fad06e3`에서 시작했다. 당시 로컬 HEAD와 확인한 원격 `main`은 같은 커밋이었다. 스테이징된 변경은 없었고 `.gitignore`의 최종 아카이브 무시 규칙만 수정되어 있었다. 최종 규칙은 보호 범위를 `/outputs/final/` 전체로 적용한다.

| 경로 | 점검 결과 |
| --- | --- |
| `src/ict_pipeline/` | `pyproject.toml`이 설치하는 패키지. Typer 계층은 `cli.py`와 각 도메인의 `commands.py`에 있다. |
| `data/`, `models/`, `multimodal/`, `evaluation/` (패키지 내) | 학습·평가 경계와 산출물 호환성을 유지한다. 파일 바이트 SHA-256 구현은 감사, 누출 검사, Test release 및 holdout 사이에서 공통화했다. 트리 지문과 JSON 지문은 각각의 직렬화 계약을 유지한다. |
| `runtime/`, `pipeline/`, `reporting/` (패키지 내) | 작업·시도 기록, 재개, 원격·컨테이너 어댑터 및 리포트. 상위 CLI에는 구현된 명령 그룹이 연결된다. |
| `inspection/`, `api/`, `streaming/` (패키지 내) | 추론과 소규모 제품 인터페이스. 연구용 평가와 수명주기가 달라 이동하지 않았다. API의 별도 의존성은 `api,test-api`다. |
| `tests/` | 데이터, CLI, backend, 모델, 무결성 및 복구의 공개 계약을 유지한다. synthetic JPG/JSON/XLSX와 product profile fixture가 테스트 모듈 상호 import 대신 전용 지원 모듈에 있다. |
| `src/ict_pipeline/config/default.yaml`, `data/aihub_smt.yaml`, `models/*.yaml` (패키지 내) | 각각 런타임, 데이터 선택·경로, detector 기본 프로필의 버전 관리 원본이다. |
| `docs/profiles/`, `configs/` | Git에 포함되는 문서 예제와 로컬 설정 영역. `configs/local.yaml`은 setup 산출물이고 `configs/local/`은 비공개 재현 프로필용이다. |
| `data/{raw,extracted,interim,processed}/`, `runs/`, `outputs/` | Git에 없는 원본·파생·실행 디렉터리. `outputs/final/`에는 보호 아카이브 및 추출된 결과가 있다. 다른 로컬 출력도 비교·재현 근거로 보존한다. |
| `.venv/`, `*.egg-info/`, IDE, 테스트 및 Python 캐시 | 가상환경 및 editable-install 메타데이터는 현재 로컬 실행에 필요하다. 재생성되는 개발 캐시는 출력·데이터와 구분한다. |
| `PLANS.md`, `tools/aihubshell` | 로컬 초기 설계와 외부 AI Hub 공급자 도구. 현행 인터페이스는 CLI 도움말과 테스트, 도구 사용 조건은 공급자 문서를 확인한다. |

추적 파일에 원본 데이터, 모델, 연구 아카이브, 캐시 및 패키지 빌드는 없었다. 작은 `docs/inspection-e2e-verification.json`은 합성 검사 데모의 명시된 재현 증거로 유지했다. `classification/`과 `multimodal/`은 서로 다른 입력·체크포인트와 호출자를 가진다. 패키지 이동이나 모듈 이름 변경은 이 계약을 검증할 근거가 없는 영역에 적용하지 않았다.

직접적인 최상위 Python import 관계에서 순환은 발견되지 않았다. 소스 바이트로 계산하는 구현 지문은 코드 수정에 따라 달라지므로 이전 작업의 strict resume에는 해당 작업의 저장된 코드 버전과 동일한 실행 설정을 사용한다. 최종 체크포인트, 분할, 센서 상태와 평가 영수증은 기존 지문을 사용한다.

## 스크립트와 호환성

| 진입점 | 분류와 조치 |
| --- | --- |
| `scripts/validate_dataset.py`, `build_manifest.py`, `convert_detection.py`, `validate_detection.py` | 지원되는 독립 전처리. CLI와 공통 도메인 모듈을 이미 사용하지만 인자 및 출력 계약이 달라 보존한다. |
| `scripts/download_data.sh`, `tools/aihubshell` | 공급자 접근의 기존 경로와 외부 스크립트. 현재 문서는 안전한 자격 증명 처리를 포함하는 `ict data`를 사용한다. |
| `scripts/migrate_dataset_paths.py` | 이전 원본 레이아웃용 1회성 도구. 경로 정책은 `data/naming.py`를 재사용하고 기본 미리보기 및 충돌 검사를 유지한다. |
| `scripts/check_sensor_ids.py`, `scripts/visualize_issues.py` | 센서 ID 오프셋 진단과 감사 이슈 시각 검토. 연구 사례 재검토에 사용되므로 유지한다. |
| `scripts/inspect_sample.py` | 로컬 진단. 고정 샘플 대신 `--image`, `--annotation`, `--sensor`를 명시한다. 기존 openpyxl extra로 XLSX를 읽는다. |
| `scripts/inspection_e2e.py` | 서비스와 이력의 합성 검증 데모. 문서와 테스트가 사용한다. |
| `src/data/validate.py` | 이전 한국어 폴더 배치에서 동작한 사전 패키지 감사. 현재 지원 경로는 packaged audit와 standalone wrapper다. 폴더 이름 및 이전 보고서 계약을 해석할 근거로 보존한다. |
| `scripts/plot_final_results.py` | 보존된 최종 집계 영수증에서 SVG를 재현한다. CLI 및 모델 실행과 분리되어 있다. |

계약 테스트는 삭제하지 않았다. 모델 실행, API, 과거 데이터 사용 흐름과 스크립트는 유지한다. 실제 모델 분할·봉인·최종화의 수치와 제한 사항은 [최종 연구 보고서](final-report.md)에 있다.

## 최종 검증

- **Passed** — Windows Python 3.11의 `python -m pytest -ra -p no:cacheprovider`에서 391개 통과, 11개 건너뜀. `test_rest_api`, `test_inspection_dashboard`, `test_inspection_e2e`, `test_product_api`는 이미 준비된 ASGI 의존성으로 별도 실행하여 44개 모두 통과했다.
- **Passed** — `node --test tests/dashboard.test.mjs` 7개. Python 소스 198개 컴파일, `ruff check --no-cache src scripts tests`, CLI 도움말과 dry run, 샘플 YAML 파싱, 상대 문서 링크 163개를 확인했다.
- **Passed** — `outputs/final/` 파일 103,923개의 SHA-256, 목록과 수정 시각을 작업 전후 대조했다. 선택된 세 가중치의 원본 digest, 동결 센서 상태, 두 Test 프로토콜 영수증 및 결과 표를 확인했다. 집계 SVG는 같은 환경에서 바이트 단위로 재생성됐다.
- **Failed** — 전체 `ruff format --check --no-cache src scripts tests`에서 기존 스타일 차이 172개 파일이 보고됐다. 신규 파일과 다시 작성한 Inspector의 포맷 검사는 통과했다.
- **Not run** — 실제 데이터·학습 opt-in, Windows 디렉터리 symlink 권한이 필요한 테스트, Linux detached worker, Ubuntu GPU 재실험 및 Test 재추론.
