# 웨이퍼 결함 분석 데모 (기획서 F-01 ~ F-07)

MixedWM38 웨이퍼 결함 **탐지 + 다중라벨 분류 + Graph-RAG 원인 역추적** 통합 데모.

## 구성
| 파일 | 역할 |
|---|---|
| `app.py` | 조립 전용 진입점 — 화면은 `ui/`, 추론은 `models/` |
| `models/labels.py` | 클래스 목록(label_map.json) + 학습 입력 규격 |
| `models/weights.py` | 가중치 경로 탐색 / 없으면 내려받기 |
| `models/preprocess.py` | 이미지 → 마스크 → 학습 팔레트 재렌더 |
| `models/classifier.py` | Swin 다중라벨 분류 (+ 휴리스틱 폴백) |
| `models/detector.py` | Superb 배포모델 → YOLOv8 → 휴리스틱 폴백 |
| `ui/theme.py` | 색·크기 상수 + 전역 CSS (UI 톤은 여기만 고치면 됨) |
| `ui/panels.py` | 헤더 / 좌·중·우 패널 렌더링 |
| `ui/charts.py` | 웨이퍼 맵·3D FFT (plotly=화면, matplotlib=PDF) |
| `ui/io_utils.py` | 업로드 파싱·검증 (PNG·JPG·BMP·NPZ) |
| `ui/report_pdf.py` | F-07 PDF 생성 |
| `graph_rag.py` | 공정 지식그래프 + 원인 역추적 리포트 (F-04) |
| `superb_client.py` | Superb 업로드/배포모델 추론/피드백 (F-01, F-02, F-05) |
| `config.py` | `.env` + Streamlit Secrets 로더 (의존성 없음) |
| `make_samples.py` | 테스트용 웨이퍼 맵 생성/내려받기 |
| `selftest.py` | 터미널 전 구간 점검 (모델·리포트·PDF·Superb 연동) |
| `robustness.py` | 분포 밖 입력(색·밝기·회전·해상도) 강건성 측정 |
| `share.py` | ngrok 터널로 외부 공유 |
| `upload.py` | MixedWM38 → Superb 바운딩박스 일괄 업로드 스크립트 |
| `WaferDefect_AllInOne.ipynb` | 학습 + Graph-RAG + 앱 생성 올인원 노트북 (Colab GPU) |
| `runs/detect/*/weights/best.pt` | 학습된 YOLOv8 가중치 (자동 탐색) |
| `swin_multilabel.pt` | 학습된 Swin 다중라벨 분류기 |

## 실행
```bash
pip install -r requirements.txt
cp .env.example .env      # 키/ID 입력 (아래 'Superb 연동' 참고)
streamlit run app.py      # → http://localhost:8501
```
로컬 실행에는 ngrok 같은 터널이 **필요 없습니다**. Colab에서 띄울 때만 터널이 필요합니다(맨 아래 참고).

## 단위테스트
```bash
pip install pytest
python3 -m pytest tests -q        # 86개, 약 30초. 실제 Superb API는 호출하지 않음(목 서버)
```
| 파일 | 개수 | 범위 |
|---|---|---|
| `tests/test_config.py` | 6 | `.env` 파싱(따옴표·export·주석), 셸 환경변수 우선순위, 값 비노출 |
| `tests/test_models.py` | 27 | Otsu, 팔레트 6종 마스크 추출, 정상 웨이퍼, 좌표 환산, 분류/탐지 폴백 |
| `tests/test_superb_client.py` | 28 | 배포 조회, `image_b64` predict, 워밍업 재시도, 업로드 3단계, 어노테이션 |
| `tests/test_pipeline.py` | 25 | Graph-RAG, PDF, 샘플 생성기, 교란 함수, 파일 검증 |

Superb 테스트는 `tests/conftest.py`의 목 HTTP 서버를 띄우고 환경변수를 갈아끼운 뒤
모듈을 reload 합니다 — `.env`의 실제 키를 쓰지 않으므로 안전하게 반복 실행할 수 있습니다.

## 데이터 준비 & 통합 점검
```bash
python3 make_samples.py --from-superb --n 16   # Superb에 올려둔 실데이터 + 정답 라벨 내려받기 (권장)
python3 make_samples.py --npz MixedWM38.npz    # 로컬에 npz가 있을 때
python3 make_samples.py                        # 둘 다 없으면 합성 샘플 (파이프라인 점검용)

python3 selftest.py                            # 모델·리포트·PDF 전 구간 점검 + 정확도
python3 selftest.py --superb                   # 위 + Superb 실제 API 연동 점검
python3 selftest.py --superb-only              # 연동만 (키 확인용)
```
`--from-superb`는 프로젝트의 **클래스별 어노테이션을 역조회**해서 8종이 골고루 섞이도록 뽑고,
그 어노테이션을 정답으로 `samples/labels.csv`에 저장합니다. `selftest.py`가 이를 읽어 완전일치율을 계산합니다.

실측(2026-08-08, 실데이터 16장): 분류 완전일치 **15/16 (94%)**, 탐지는 Superb 배포모델 22~25ms.

> `selftest.py --superb`는 점검용 자산 1개(`selftest_*.png`)와 어노테이션을 실제 프로젝트에 만듭니다.
> 콘솔에서 파일명으로 찾아 지우면 됩니다.

## 일반화(강건성) — 실사용 입력에 대한 검증
사용자가 넣는 이미지는 학습 데이터와 똑같이 생기지 않습니다. 그래서 교란을 걸어 점수 하락을 측정합니다:
```bash
python3 robustness.py
```
실측(실데이터 16장 기준):

| 교란 | 분류 완전일치 |
|---|---|
| 원본 (in-distribution) | 94% |
| 회전 90° · 좌우반전 | 94% |
| 노이즈 +3% · JPEG q40 · 저해상도 64px | 94% |
| **다른 팔레트(파랑 결함) · 흑백 웨이퍼맵 · 밝기 0.6배** | **94%** |
| 축소 70%(여백 있음) | 88% |
| 회전 30° | 81% |

색·밝기 교란 3종은 원래 **0%** 였습니다. 원인은 모델이 아니라 `image_to_mask`의 하드코딩 임계값
(`r>150 & g<130 & b<130`)이었고, 색을 고정하지 않는 방식으로 바꿔 해결했습니다:
1. 배경색을 이미지 테두리에서 추정 (검정 배경이든 흰 배경이든 무관)
2. 채도축·명도축에 각각 Otsu를 걸어 **더 잘 갈라지는 축**을 선택
   — 회색 웨이퍼 + 컬러 결함이면 채도로, 흑백 웨이퍼맵이면 명도로 분리
3. 두 군집이 충분히 안 벌어지면 '결함 없음'(정상 웨이퍼)으로 판단

덕분에 학습 팔레트와 다른 이미지(실제 업로드본은 배경이 `#0f172a`, 정상이 `#64748b`로 노트북과 다릅니다)도
그대로 동작합니다. 결함이 정상보다 **어둡게** 그려진 이미지를 쓴다면 `.env`에 `WAFER_DEFECT_BRIGHT=0`.

남은 한계: 30° 같은 임의 각도 회전(81%)과 여백이 큰 화면 캡처(88%)는 다소 떨어집니다. 실제 웨이퍼 맵은
축 정렬·꽉 찬 프레임으로 들어오므로 데모에는 영향이 적지만, 더 올리려면 회전/스케일 증강으로 재학습이 필요합니다.
(TTA(회전·반전 8종 평균)도 시도했으나 81%→75%로 **악화**되어 넣지 않았습니다.)

## 학습 모델 연결
앱이 시작할 때 아래 순서로 **자동 탐색**하므로 별도 설정이 필요 없습니다.
- 분류: `swin_multilabel.pt` (환경변수 `WAFER_CLS_PT`로 변경 가능)
- 탐지: `best.pt` → `runs/detect/wafer_det*/weights/best.pt` → `runs/detect/*/weights/best.pt` 중 최신
  (환경변수 `WAFER_DET_PT`로 변경 가능)
- 클래스 순서는 `label_map.json`을 따릅니다.

**입력 표현을 학습과 동일하게 맞춥니다.** 업로드된 임의 이미지는 `image_to_mask()`로 52×52
(0 빈칸 / 1 정상 / 2 불량) 정규화 후 학습 팔레트(정상 `#94a3b8`, 불량 `#ef4444`)로 다시 렌더링해서
분류는 224px, 탐지는 256px(학습 `imgsz`)로 넣고, 박스 좌표는 원본 크기로 환산합니다.
가중치가 없으면 휴리스틱 폴백으로 계속 동작합니다.

현재 가중치 성능(학습 로그 기준): 탐지 `mAP50 0.967` / `mAP50-95 0.956` (`runs/detect/wafer_det-2`).

## 기능 ↔ 기획서 매핑
- **F-01** 업로드 + Superb 저장 — 좌측 업로더, `superb_client.upload_image`
- **F-02** 결함 자동 탐지(빨간 박스) — 중앙, Superb 배포모델 → YOLOv8 → 휴리스틱
- **F-03** 3D 주파수 신호 시각화 — 중앙, 웨이퍼 맵 2D FFT → 3D surface
- **F-04** 공정 판단 + 원인 역추적 리포트 — 우측, Graph-RAG
- **F-05** AI 판단 강제 수정 + 피드백 루프 — 우측 multiselect → `push_feedback`
- **F-06** 파일 업로드 예외 처리 — 확장자/손상 검증 + 경고
- **F-07** 결함 분석 리포트 PDF — 우측 다운로드 버튼 (reportlab)

## Superb 연동
설정은 프로젝트 루트의 **`.env` 파일**에서 읽습니다 (`config.py`가 import 시 자동 로드, 별도 패키지 불필요).
`.env`는 `.gitignore`에 등록되어 있으니 **절대 커밋하지 마세요**.
```ini
SUPERB_AI_API_KEY=sbd_pk_...
SUPERB_AI_TENANT=spot
SUPERB_DEPLOYMENT_ID=6008abc6-71b6-4c45-9d9c-a89cd14266fe   # 배포 ID (권장)
SUPERB_PROJECT_ID=7b267a4e-eea0-4764-9a61-6d74515a4985
SUPERB_DATASET_ID=ded47876-a675-492e-a548-c296f80fd151
SUPERB_CLASS_ID=9fd96ef0-0b00-4157-8633-fabd20a4e6c3        # defect_pattern_v2
SUPERB_PREDICT_MAX_WAIT=90
SUPERB_ANN_DATA_KEY=answer
```
- 우선순위는 **셸 환경변수 > .env** 입니다. 임시로 다른 키를 쓰려면 `SUPERB_AI_API_KEY=... streamlit run app.py`.
- `SUPERB_DEPLOYMENT_ID`가 없으면 `SUPERB_MODEL_ID`와 일치하는 배포를,
  그것도 없으면 테넌트의 실행 중인 배포 하나를 자동 선택합니다.
- 키가 없으면 앱은 오프라인 모드로 정상 동작(⚪ 배지).
- 키가 유출되면 콘솔에서 폐기(revoke) 후 새로 발급해 `.env`만 교체하면 됩니다.

동작 방식 (SDK 0.4.x 계약):
```python
result = client.deployments.predict(deployment_id, image_b64=..., confidence=0.2659)
for p in result.predictions:
    p.class_name, p.geometry, p.confidence      # geometry: {"type":"bbox","x","y","w","h"}
```
- 이미지는 **base64 바이트**(`image_b64`)로 전송하고, 좌표는 보낸 이미지의 픽셀 좌표계입니다.
  앱은 256px 캔버스로 보낸 뒤 응답의 `image.width/height`로 원본 크기에 맞춰 환산합니다.
- 모델 워밍업 중(`MODEL_LOADING`/`MODEL_STARTING`)이면 안내된 주기로 자동 재시도합니다
  (최대 대기: `SUPERB_PREDICT_MAX_WAIT`, 기본 90초).
- 탐지 신뢰도 슬라이더 기본값은 배포모델의 **추천 임계값**(`capability.params.confidence.default`)을 사용합니다.
- 오토라벨 적재는 `source="model"`, F-05 엔지니어 수정은 `source="manual"`로 구분해 저장합니다.
- 업로드된 자산 행은 서버가 비동기로 생성하므로, 앱은 파일명으로 잠시 폴링해서 `asset_id`를 확보합니다.

> **Python 버전 주의**: `superb-ai` SDK는 **Python ≥ 3.12**를 요구합니다.
> 그 미만 환경에서는 `superb_client.py`가 동일한 REST 엔드포인트를 `requests`로 직접 호출하는
> 폴백으로 자동 전환됩니다(헤더 배지에 `Superb 연동(rest)`로 표시). 기능 차이는 없습니다.

## 학습 (Colab)
`WaferDefect_AllInOne.ipynb` 실행 → `best.pt`, `swin_multilabel.pt`, `label_map.json` 생성 → 이 폴더로 복사.
- 탐지: 단일 패턴 이미지의 결함 픽셀 바운딩박스 자동 생성 → YOLOv8n 파인튜닝 (`imgsz=256`, 8클래스)
- 분류: `arr_1` 멀티핫을 타깃으로 Swin Transformer 학습 (혼합 패턴 포함 8종)

## 배포 & CI/CD

**GitHub Pages 에는 올라가지 않습니다.** Pages 는 정적 파일만 서빙하는데 이 앱은
파이썬 서버 + PyTorch 추론 + API 키가 필요합니다. 대신 아래 조합으로 "push 하면 자동 반영"을 만듭니다.

| 역할 | 도구 |
|---|---|
| 테스트 자동 실행 | GitHub Actions (`.github/workflows/ci.yml`) |
| 앱 자동 재배포 | Streamlit Community Cloud (레포 연결 시 push 마다 자동) |
| 모델 가중치 | GitHub Releases + `models/weights.py` (또는 Git LFS) |
| 비밀 키 | Streamlit Cloud Secrets (`.env` 는 절대 커밋 금지) |

### 1) 가중치 (105MB 문제)
`swin_multilabel.pt` 가 105MB 라 GitHub 일반 파일 상한(100MB)을 넘습니다. 두 방법 중 하나:

**(A) GitHub Releases — 권장.** LFS 쿼터와 무관하고 파일당 2GB까지 됩니다.
```bash
gh release create weights-v1 swin_multilabel.pt runs/detect/wafer_det-2/weights/best.pt
```
배포 환경 변수에 URL 만 넣으면 앱이 첫 실행 때 자동으로 받아옵니다:
```
WAFER_CLS_URL=https://github.com/<user>/<repo>/releases/download/weights-v1/swin_multilabel.pt
WAFER_DET_URL=https://github.com/<user>/<repo>/releases/download/weights-v1/best.pt
```

**(B) Git LFS.** 간단하지만 무료 한도가 **저장 1GB / 대역폭 1GB per month** 이고,
105MB 모델은 클론·재배포 1회마다 대역폭을 먹어 **월 9회쯤이면 소진**됩니다.
```bash
brew install git-lfs && git lfs install
# .gitignore 에서 *.pt 두 줄을 주석 처리한 뒤
git add .gitattributes swin_multilabel.pt
```

가중치가 없어도 앱은 휴리스틱 폴백으로 정상 기동합니다(헤더 배지가 ⚪ 로 표시).

### 2) 첫 배포
```bash
git init && git add . && git commit -m "wafer defect demo"
git remote add origin https://github.com/<user>/<repo>.git
git push -u origin main
```
`.env` 는 `.gitignore` 에 있어 커밋되지 않습니다. 반드시 확인하세요:
```bash
git status --porcelain | grep -c '\.env$'      # 0 이어야 함
```
### 3) API 키를 배포 환경에 넣기
`.env` 는 커밋되지 않으므로 배포처에는 **Secrets** 로 넣습니다.
[share.streamlit.io](https://share.streamlit.io) → 레포 연결 → main / `app.py` 선택 →
**Settings → Secrets** 에 `.streamlit/secrets.toml.example` 내용을 TOML 로 붙여넣기:
```toml
SUPERB_AI_API_KEY = "sbd_pk_..."
SUPERB_AI_TENANT = "spot"
SUPERB_DEPLOYMENT_ID = "..."
SUPERB_PROJECT_ID = "..."
SUPERB_DATASET_ID = "..."
SUPERB_CLASS_ID = "..."
SUPERB_ANN_DATA_KEY = "answer"
WAFER_CLS_URL = "https://github.com/.../swin_multilabel.pt"
```
`config.py` 가 이 값들을 `os.environ` 으로 넣어 주므로 **코드는 로컬(.env)과 완전히 동일하게 동작**합니다.
설정 출처 우선순위는 **셸 환경변수 > .env > Secrets** 입니다.

로컬에서 Secrets 방식을 테스트하려면 `.streamlit/secrets.toml.example` 을
`.streamlit/secrets.toml` 로 복사하세요 (이 파일도 `.gitignore` 에 있습니다).

### 3) 이후 자동 반영
`main` 에 push → Actions 가 테스트를 돌리고, Streamlit Cloud 가 자동으로 재배포합니다.
UI 만 고칠 때는 `ui/` 안의 파일만 건드리면 됩니다.

> 메모리 주의: Streamlit Community Cloud 는 앱당 RAM 1GB 입니다. torch + Swin 추론이
> 빠듯할 수 있어 `requirements.txt` 상단에서 CPU 전용 휠을 받도록 지정해 뒀습니다
> (기본 휠은 CUDA 포함 ~2.5GB). OOM 이 나면 RAM 16GB 를 주는 Hugging Face Spaces
> (Streamlit SDK) 로 옮기면 같은 코드가 그대로 돕니다.

## 다른 사람에게 접속시키기 (터널)
```bash
pip install pyngrok
# .env 에 토큰 추가: NGROK_AUTHTOKEN=2abc...
#   https://dashboard.ngrok.com/get-started/your-authtoken (무료 가입)
python3 share.py            # 앱 실행 + 터널 → 공개 URL 출력, Ctrl+C 로 종료
python3 share.py --no-run   # 이미 streamlit 이 떠 있으면 터널만
```
링크를 아는 사람은 누구나 들어올 수 있으니 **데모가 끝나면 반드시 종료**하세요.

| 상황 | 방법 |
|---|---|
| 혼자 로컬 확인 | `streamlit run app.py` → localhost:8501, 터널 불필요 |
| 같은 사무실 네트워크 | `streamlit run app.py --server.address 0.0.0.0` → Network URL 공유 |
| **다른 지역 친구/심사위원** | `python3 share.py` (ngrok) |
| 계정 만들기 싫을 때 | `brew install cloudflared && cloudflared tunnel --url http://localhost:8501` |
| Colab | 터널 필수 — 노트북은 localtunnel 사용 (접속 시 비밀번호=출력된 IP 입력) |

Colab에서 ngrok을 쓰려면:
```python
!pip -q install pyngrok
!streamlit run app.py --server.port 8501 &>/content/st_log.txt &
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_TOKEN")
print(ngrok.connect(8501))
```

## PDF 한글
PDF 본문 한글이 필요하면 한글 폰트를 설치하면 자동 인식합니다:
```bash
sudo apt-get install -y fonts-nanum      # Colab: !apt-get install -y fonts-nanum
```
없으면 그림/영문은 정상, 한글 본문만 공백 처리됩니다.
