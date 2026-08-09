"""
반도체 웨이퍼 결함 분석 & 3D 주파수 모니터링 — 기획서 F-01 ~ F-07 (Streamlit)

실행:
    python3 -m streamlit run app.py

레이아웃
  헤더            제목 + 부제 + 연동 상태 + 우측 상단 PDF 다운로드(F-07)
  좌 (입력&설정)   업로드(F-01/F-06) · 공정 정답 교정(F-05) · 고급 설정
  중 (인식 메인)   웨이퍼 맵 + Bounding Box(F-02) · 분석 시작 버튼
  우 (원인 리포트)  감지 결함 / 추정 공정 / 자동 생성 원인 문장(F-04) · 피드백(F-05)
  하단            3D 공간 주파수 입체 히트맵(F-03)
"""
import io
import os
import tempfile
import datetime
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import models
import graph_rag as gr
import superb_client as sb

st.set_page_config(page_title="반도체 웨이퍼 결함 분석 & 3D 주파수 모니터링",
                   page_icon="🔬", layout="wide")

ALLOWED = {"png", "jpg", "jpeg", "bmp"}         # F-06 이미지 확장자
ALLOWED_DATA = {"npz"}                          # MixedWM38 원본 배열
ALLOWED_ALL = ALLOWED | ALLOWED_DATA

# 화면 표시용 팔레트 (모델 입력용 팔레트와는 별개)
C_BG, C_NORMAL, C_DEFECT = "#0f172a", "#3b82f6", "#ef4444"

COL_H = 720          # 1·2·3열 카드 높이를 고정해 아래 단차를 없앤다
PLOT_H = 520         # 중앙 웨이퍼 맵 높이 (카드를 채우도록)

st.markdown("""
<style>
  .wd-title   { font-size: 2.75rem; font-weight: 800; margin: 0 0 .3rem 0; letter-spacing: -1.2px;
                line-height: 1.15; }
  .wd-sub     { opacity: .62; font-size: 1.02rem; margin: 0; }
  .wd-sec     { font-size: 1.3rem; font-weight: 700; margin: .1rem 0 .6rem 0; }
  .wd-sub2    { font-size: 1.02rem; font-weight: 650; margin: .2rem 0 .5rem 0; }
  .wd-card    { border-radius: .55rem; padding: .7rem .9rem; margin-bottom: .55rem;
                border: 1px solid rgba(128,128,128,.18); font-size: .95rem; }
  .wd-defect  { background: rgba(239, 68, 68, .10); }
  .wd-defect  b { color: #dc2626; }
  .wd-proc    { background: rgba(234, 179,  8, .13); }
  .wd-proc    b { color: #b45309; }
  .wd-label   { opacity: .7; font-size: .88rem; margin: .6rem 0 .25rem 0; }
  div[data-testid="stButton"] > button[kind="primary"] {
      background: #ef4444; border-color: #ef4444; font-weight: 700; }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
      background: #dc2626; border-color: #dc2626; }
</style>
""", unsafe_allow_html=True)


# ============================ 입력 처리 (F-01, F-06) ============================
def load_upload(up, npz_index: int = 0):
    """업로드 파일 → (PIL.Image, 설명). 실패 시 (None, 사유)."""
    ext = up.name.rsplit(".", 1)[-1].lower()
    if ext in ALLOWED_DATA:
        try:
            data = np.load(io.BytesIO(up.getvalue()), allow_pickle=False)
            key = "arr_0" if "arr_0" in data.files else data.files[0]
            X = data[key]
        except Exception as e:
            return None, f"NPZ를 읽을 수 없습니다: {e}"
        arr = np.asarray(X[npz_index] if X.ndim == 3 else X)
        if arr.ndim != 2:
            return None, "NPZ 안에서 2D 웨이퍼 맵을 찾지 못했습니다."
        m = arr.astype(int)
        rgb = np.zeros((*m.shape, 3), np.uint8)
        rgb[m == 1] = (148, 163, 184)                 # 학습 팔레트로 렌더
        rgb[m == 2] = (239, 68, 68)
        img = Image.fromarray(rgb).resize((256, 256), Image.NEAREST)
        return img, f"{key}[{npz_index}] · {m.shape[0]}×{m.shape[1]}"
    if ext not in ALLOWED:
        return None, "지원하지 않는 파일 형식입니다. 웨이퍼 맵 이미지(.png/.jpg) 또는 .npz 를 올려주세요."
    try:
        return Image.open(up).convert("RGB"), up.name
    except Exception:
        return None, "이미지를 열 수 없습니다. 손상되지 않은 웨이퍼 맵 파일인지 확인해 주세요."


def npz_length(up) -> int:
    try:
        data = np.load(io.BytesIO(up.getvalue()), allow_pickle=False)
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        return int(data[key].shape[0]) if data[key].ndim == 3 else 1
    except Exception:
        return 1


# ============================ 시각화 ============================
def wafer_figure(img: Image.Image, boxes=()):
    """F-02: 웨이퍼 맵(52 격자) + Bounding Box."""
    m = models.image_to_mask(img, 52)
    W, H = img.size
    fig = go.Figure(go.Heatmap(
        z=m, zmin=0, zmax=2, showscale=False,
        colorscale=[[0.0, C_BG], [0.33, C_BG], [0.33, C_NORMAL],
                    [0.66, C_NORMAL], [0.66, C_DEFECT], [1.0, C_DEFECT]],
        hovertemplate="x=%{x} y=%{y}<extra></extra>"))
    sx, sy = 52.0 / max(W, 1), 52.0 / max(H, 1)          # 원본 px → 52 격자
    for name, x0, y0, x1, y1, conf in boxes:
        fig.add_shape(type="rect", x0=x0 * sx, y0=y0 * sy, x1=x1 * sx, y1=y1 * sy,
                      line=dict(color="#f87171", width=2))
        fig.add_annotation(x=x0 * sx, y=y1 * sy, text=f"{name} {conf:.2f}",
                           showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=11, color="#ffffff"),
                           bgcolor="rgba(239,68,68,.85)", borderpad=2)
    fig.update_layout(height=PLOT_H, margin=dict(l=0, r=0, t=6, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def _fft_map(img: Image.Image) -> np.ndarray:
    m = models.image_to_mask(img, 52).astype(float)
    return np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(m))))


def fft_surface_3d(img: Image.Image):
    """F-03: 2D FFT → 3D Surface (화면용, plotly)."""
    F = _fft_map(img)
    Z = (F - F.mean()) / (F.std() + 1e-9)
    fig = go.Figure(go.Surface(z=Z, colorscale="Viridis",
                               colorbar=dict(thickness=14, len=.7)))
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=6, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                                 aspectratio=dict(x=1, y=1, z=.55)))
    return fig


# ---- PDF용 정적 그림 (kaleido 없이 동작하도록 matplotlib 유지) ----
def draw_boxes(img: Image.Image, boxes):
    """F-02: 원본 위에 빨간 박스."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img); ax.axis("off")
    import matplotlib.patches as patches
    for name, x0, y0, x1, y1, conf in boxes:
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                     fill=False, edgecolor="#ef4444", lw=2.2))
        ax.text(x0, max(y0 - 4, 0), f"{name} {conf:.2f}", color="white",
                fontsize=8, bbox=dict(facecolor="#ef4444", pad=1, edgecolor="none"))
    fig.tight_layout()
    return fig


def freq_surface(img: Image.Image):
    """F-03: 웨이퍼 맵의 2D FFT 크기 스펙트럼을 3D surface로 (PDF용)."""
    F = _fft_map(img)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid(np.arange(F.shape[1]), np.arange(F.shape[0]))
    ax.plot_surface(xx, yy, F, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title("3D Frequency Spectrum (log|FFT|)", fontsize=9)
    ax.set_xlabel("u"); ax.set_ylabel("v"); ax.set_zlabel("mag")
    fig.tight_layout()
    return fig


def build_pdf(img, boxes, report_text, proc, preds, scores, src):
    """F-07: 리포트 + 시각화 종합 PDF (bytes)."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4
    # 한글 폰트 (있으면 등록, 없으면 기본)
    font = "Helvetica"
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
                  "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
            if os.path.exists(p):
                pdfmetrics.registerFont(TTFont("KR", p)); font = "KR"; break
    except Exception:
        pass

    c.setFont(font, 16); c.drawString(20 * mm, H - 25 * mm, "웨이퍼 결함 분석 리포트")
    c.setFont(font, 9)
    c.drawString(20 * mm, H - 32 * mm,
                 f"생성: {datetime.datetime.now():%Y-%m-%d %H:%M}   |   탐지 소스: {src}")

    # 이미지 (박스 포함)
    fig = draw_boxes(img, boxes)
    ib = io.BytesIO(); fig.savefig(ib, format="png", dpi=120, bbox_inches="tight"); plt.close(fig); ib.seek(0)
    c.drawImage(ImageReader(ib), 20 * mm, H - 120 * mm, width=80 * mm, height=80 * mm, preserveAspectRatio=True)

    # 3D 주파수
    fig2 = freq_surface(img)
    ib2 = io.BytesIO(); fig2.savefig(ib2, format="png", dpi=120, bbox_inches="tight"); plt.close(fig2); ib2.seek(0)
    c.drawImage(ImageReader(ib2), 110 * mm, H - 120 * mm, width=80 * mm, height=70 * mm, preserveAspectRatio=True)

    # 텍스트 리포트
    c.setFont(font, 11); c.drawString(20 * mm, H - 130 * mm, f"판단 공정: {proc}")
    c.setFont(font, 9)
    y = H - 138 * mm
    c.drawString(20 * mm, y, "예측 패턴: " + (", ".join(preds) if preds else "정상(결함 없음)")); y -= 6 * mm
    for line in report_text.split("\n"):
        for chunk in [line[i:i + 90] for i in range(0, len(line), 90)] or [""]:
            c.drawString(20 * mm, y, chunk); y -= 5 * mm
            if y < 20 * mm:
                c.showPage(); c.setFont(font, 9); y = H - 20 * mm
    c.showPage(); c.save(); buf.seek(0)
    return buf.getvalue()


# ============================ 상태 ============================
if "asset_id" not in st.session_state:
    st.session_state.asset_id = None
if "result" not in st.session_state:
    st.session_state.result = None


@st.cache_data(ttl=60, show_spinner=False)
def superb_status():
    return sb.status()


S = superb_status()
dep = S.get("deployment") or {}

# ============================ 헤더 ============================
h1, h2 = st.columns([3.2, 1])
with h1:
    st.markdown('<p class="wd-title">🔬 반도체 웨이퍼 결함 분석 &amp; 3D 주파수 모니터링</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="wd-sub">MixedWM38 데이터셋 기반 복합 결함 패턴 및 3D 신호 처리 시스템</p>',
                unsafe_allow_html=True)
with h2:
    st.markdown('<div style="height:1.1rem"></div>', unsafe_allow_html=True)   # 제목과 수직 정렬
    pdf_slot = st.empty()          # 분석 후 F-07 다운로드 버튼으로 교체됨
    pdf_slot.button("📥 PDF 리포트 다운로드 (F-07)", disabled=True, use_container_width=True)

b_sb = f"🟢 Superb 연동({S['transport']})" if S.get("connected") else "⚪ Superb 오프라인"
if dep.get("ok"):
    b_dp = f"{'🟢' if dep['status'] == 'ready' else '🟡'} 배포모델 {dep['status']} · {dep['name'] or dep['id'][:8]}"
else:
    b_dp = "⚪ 배포모델 미설정"
_run_dir = os.path.basename(os.path.dirname(os.path.dirname(models.DET_PATH)))
b_yo = f"🟢 YOLO 학습모델({_run_dir})" if models.yolo_available() else f"⚪ YOLO 없음({models.yolo_error()})"
b_cl = "🟢 Swin 분류" if models.cls_available() else f"⚪ Swin 없음({models.cls_error()})"
st.caption(f"{b_sb} · {b_dp} · {b_yo} · {b_cl}")
st.divider()

left, center, right = st.columns([1.05, 1.65, 1.25], gap="large")
# 세 열을 같은 높이의 카드로 고정 — 내용 길이가 달라도 아래 단차가 생기지 않는다.
# 나중에(분석 실행 후) 추가되는 메시지도 이 컨테이너 안으로 들어간다.
left_box = left.container(height=COL_H, border=True)
center_box = center.container(height=COL_H, border=True)
right_box = right.container(height=COL_H, border=True)

# ==================== 좌: 입력 & 설정 (F-01, F-05, F-06) ====================
with left_box:
    st.markdown('<p class="wd-sec">👉 입력 &amp; 설정</p>', unsafe_allow_html=True)
    up = st.file_uploader("웨이퍼 이미지/데이터 업로드", type=sorted(ALLOWED_ALL),
                          help="드래그&드롭 지원 · MixedWM38 원본 .npz 도 가능 (F-01)")

    npz_index = 0
    if up is not None and up.name.lower().endswith(".npz"):
        n = npz_length(up)
        if n > 1:
            npz_index = st.number_input(f"샘플 인덱스 (0 ~ {n - 1})", 0, n - 1, 0, 1)

    st.divider()
    st.markdown('<p class="wd-sub2">🔧 엔지니어 공정 정답 교정 (F-05)</p>', unsafe_allow_html=True)
    proc_override = st.selectbox("추정 공정 수정 선택", gr.PROCESS_CHOICES, index=0,
                                 format_func=lambda s: "자동 추정 유지" if s == "자동 인식" else s)

    with st.expander("⚙️ 고급 설정"):
        det_src = st.radio("탐지 엔진 (F-02)", ["auto", "superb", "yolo", "heuristic"],
                           format_func=lambda s: {"auto": "자동 (Superb→YOLO→폴백)",
                                                  "superb": "Superb 배포모델",
                                                  "yolo": "로컬 YOLOv8",
                                                  "heuristic": "휴리스틱"}[s], index=0)
        thr = st.slider("분류 임계값", 0.2, 0.8, 0.5, 0.05)
        rec = dep.get("recommended_conf") if dep.get("ok") else None
        dconf = st.slider("탐지 신뢰도 임계값", 0.05, 0.9, float(rec) if rec else 0.25, 0.05,
                          help=f"배포모델 추천값 {rec:.4f}" if rec else None)

# ==================== 중앙: 인식 메인 (F-02) ====================
with center_box:
    st.markdown('<p class="wd-sec">🎯 웨이퍼 화면 인식 메인 (F-02)</p>', unsafe_allow_html=True)
    st.markdown('<p class="wd-sub2">🖼️ 인식된 웨이퍼 맵 (Bounding Box 표시)</p>', unsafe_allow_html=True)
    plot_slot = st.empty()
    caption_slot = st.empty()
    run = st.button("🔍 웨이퍼 결함 및 주파수 분석 시작", use_container_width=True, type="primary")

# ==================== 분석 실행 ====================
img, note = None, None
if up is not None:
    img, note = load_upload(up, npz_index)
    if img is None:
        with left_box:
            st.error(note)                      # F-06 예외 처리

if run and img is None:
    with center_box:
        st.warning("먼저 웨이퍼 이미지 또는 .npz 파일을 업로드하세요.")
elif run:
    # F-01: Superb 저장소 업로드 (연동 시)
    upres = {"ok": False, "reason": "offline"}
    if sb.available():
        with st.spinner("Superb 업로드 중…"):
            stem = os.path.basename(up.name).replace(os.sep, "_").rsplit(".", 1)[0]
            tmp = os.path.join(tempfile.gettempdir(), f"{stem}.png")
            img.save(tmp)
            upres = sb.upload_image(tmp, key=f"{stem}.png")
    st.session_state.asset_id = upres.get("asset_id")

    with st.spinner("모델 추론 중…"):
        preds, scores, csrc = models.classify(img, thr=thr)
        boxes, dsrc = models.detect(img, source=det_src, conf=dconf)
    conf = max([scores[p] for p in preds], default=None)
    st.session_state.result = dict(img=img, preds=preds, scores=scores, csrc=csrc,
                                   boxes=boxes, dsrc=dsrc, conf=conf, note=note)

    # 오토라벨링 결과 Superb 적재 (source=model)
    push = sb.push_prediction(st.session_state.asset_id, preds) \
        if (st.session_state.asset_id and preds) else None
    with left_box:
        st.success(f"분석 완료 · 분류:{csrc} · 탐지:{dsrc}")
        if upres.get("ok"):
            aid = st.session_state.asset_id
            st.caption(f"Superb 업로드 완료 · asset={aid[:8] + '…' if aid else '생성 대기중'}"
                       + (" · 오토라벨 적재됨" if push and push.get("ok") else ""))
            if push and not push.get("ok"):
                st.caption(f"↳ 어노테이션 적재 실패: {push.get('reason')}")
            elif upres.get("pending"):
                st.caption(f"↳ {upres.get('reason')}")
        elif sb.available():
            st.caption(f"Superb 업로드 실패: {upres.get('reason')}")

R = st.session_state.result

# ==================== 중앙 플롯 채우기 ====================
if R:
    plot_slot.plotly_chart(wafer_figure(R["img"], R["boxes"]), use_container_width=True)
    caption_slot.caption(f"탐지 소스: {R['dsrc']} · Bounding Box {len(R['boxes'])}개"
                         + (f" · {R['note']}" if R.get("note") else ""))
elif img is not None:
    plot_slot.plotly_chart(wafer_figure(img), use_container_width=True)
    caption_slot.caption(f"미리보기 · {note} — [분석 시작]을 누르면 결함을 탐지합니다.")
else:
    plot_slot.info("좌측에서 웨이퍼 맵 이미지(.png/.jpg) 또는 MixedWM38 .npz 를 업로드하세요.")

# ==================== 우: 원인 분석 리포트 (F-04, F-05) ====================
with right_box:
    st.markdown('<p class="wd-sec">👉 원인 분석 리포트 (F-04)</p>', unsafe_allow_html=True)
    st.markdown('<p class="wd-sub2">📋 AI 분석 결과</p>', unsafe_allow_html=True)
    if R:
        preds = R["preds"]
        defect_txt = (" + ".join(preds) + (" (복합)" if len(preds) > 1 else "")) \
            if preds else "없음 (정상 웨이퍼)"
        auto_proc = gr.primary_process(preds) if preds else "정상"
        proc = auto_proc if proc_override == "자동 인식" else proc_override
        st.markdown(f'<div class="wd-card wd-defect">감지 결함: <b>{defect_txt}</b></div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="wd-card wd-proc">추정 공정: <b>{proc}</b>'
                    + ("" if proc_override == "자동 인식" else " <small>(엔지니어 교정)</small>")
                    + "</div>", unsafe_allow_html=True)

        use_llm = st.toggle("로컬 LLM 문장화(GPU 필요)", value=False)
        report_text = gr.report(preds, confidence=R["conf"], use_llm=use_llm)
        st.markdown('<p class="wd-label">자동 생성 원인 문장</p>', unsafe_allow_html=True)
        st.text_area("자동 생성 원인 문장", report_text, height=150, label_visibility="collapsed")

        if R["conf"] is not None:
            st.progress(min(R["conf"], 1.0), text=f"분류 신뢰도 {R['conf']:.0%}")
        with st.expander("패턴별 점수"):
            st.bar_chart({k: v for k, v in R["scores"].items()})

        # F-05: 공정 강제 수정 + 피드백
        st.divider()
        st.markdown('<p class="wd-sub2">🔁 AI 판단 강제 수정 (액티브 러닝)</p>', unsafe_allow_html=True)
        corrected = st.multiselect("실제 결함 패턴으로 수정", models.LABELS, default=preds)
        if st.button("피드백 전송", use_container_width=True):
            if st.session_state.asset_id:
                fr = sb.push_feedback(st.session_state.asset_id, corrected)
                st.success("Superb로 피드백 전송됨(source=manual)" if fr.get("ok")
                           else f"전송 실패: {fr.get('reason')}")
            elif sb.available():
                st.warning("자산 생성이 아직 진행 중입니다 — 잠시 후 다시 분석하면 적재됩니다.")
            else:
                st.info("Superb 오프라인 — 수정값은 로컬 리포트에만 반영됩니다.")
            R["preds"] = corrected
            R["conf"] = max([R["scores"].get(p, 0.6) for p in corrected], default=None)
    else:
        st.info("분석 결과가 여기에 표시됩니다.")

# ==================== F-07 PDF (헤더 우측 슬롯) ====================
if R:
    final_preds = R["preds"]
    final_proc = (gr.primary_process(final_preds) if final_preds else "정상") \
        if proc_override == "자동 인식" else proc_override
    pdf_slot.download_button(
        "📥 PDF 리포트 다운로드 (F-07)",
        build_pdf(R["img"], R["boxes"],
                  gr.report(final_preds, confidence=R["conf"], use_llm=False),
                  final_proc, final_preds, R["scores"], R["dsrc"]),
        file_name="wafer_report.pdf", mime="application/pdf", use_container_width=True)

# ==================== 하단: 3D 주파수 (F-03) ====================
st.divider()
st.markdown('<p class="wd-sec">⬇️ 3D 공간 주파수 입체 히트맵 (FFT Surface) (F-03)</p>',
            unsafe_allow_html=True)
st.markdown('<p class="wd-sub2">2D FFT ➜ 3D 공간 주파수 Surface 시각화</p>', unsafe_allow_html=True)
if R or img is not None:
    st.plotly_chart(fft_surface_3d(R["img"] if R else img), use_container_width=True)
    st.caption("규칙적 패턴일수록 특정 주파수에 에너지가 집중됩니다 · 드래그로 회전, 스크롤로 확대")
else:
    st.info("웨이퍼 맵을 업로드하면 2D FFT 기반 3D 주파수 표면이 여기에 표시됩니다.")
