"""화면 상수 + 전역 스타일. UI 톤을 바꾸려면 이 파일만 건드리면 된다."""
from __future__ import annotations

import streamlit as st

PAGE_TITLE = "반도체 웨이퍼 결함 분석 & 3D 주파수 모니터링"
PAGE_ICON = "🔬"
SUBTITLE = "MixedWM38 데이터셋 기반 복합 결함 패턴 및 3D 신호 처리 시스템"

# 업로드 허용 확장자
ALLOWED = {"png", "jpg", "jpeg", "bmp"}         # F-06 이미지
ALLOWED_DATA = {"npz"}                          # MixedWM38 원본 배열
ALLOWED_ALL = ALLOWED | ALLOWED_DATA

# 화면 표시용 팔레트 (모델 입력용 팔레트와는 별개)
C_BG, C_NORMAL, C_DEFECT = "#0f172a", "#3b82f6", "#ef4444"

# ---------------- 화면 밀도 (와이드 모드) ----------------
# 넓은 모니터에서 3열이 과하게 벌어지는 걸 막는다.
#   max_w : 본문 최대 폭(px). None 이면 화면 전체 사용
#   scale : 글자·카드 높이 배율 (브라우저 확대/축소와 비슷한 효과)
LAYOUT_MODES = {
    "컴팩트": dict(max_w=1180, scale=0.88, desc="좁고 촘촘하게 (브라우저 80% 축소와 비슷)"),
    "표준":   dict(max_w=1560, scale=1.00, desc="가운데 정렬 · 기본"),
    "와이드": dict(max_w=None, scale=1.06, desc="모니터 전체 폭 사용"),
}
DEFAULT_MODE = "표준"

# 기준 크기 (scale 이 곱해진다)
COL_H = 720          # 1·2·3열 카드 높이 — 고정해서 아래 단차를 없앤다
PLOT_H = 520         # 중앙 웨이퍼 맵 높이 (카드를 채우도록)
FFT_H = 560          # 하단 3D 주파수 표면 높이
REPORT_H = 150       # 원인 문장 textarea 높이


def mode_of(name: str) -> dict:
    return LAYOUT_MODES.get(name, LAYOUT_MODES[DEFAULT_MODE])


def sizes(name: str = DEFAULT_MODE) -> dict:
    """선택된 밀도에 맞춘 픽셀 크기들."""
    s = mode_of(name)["scale"]
    return {"col_h": int(COL_H * s), "plot_h": int(PLOT_H * s),
            "fft_h": int(FFT_H * s), "report_h": int(REPORT_H * s), "scale": s}


def _css(name: str) -> str:
    m = mode_of(name)
    s = m["scale"]
    # 본문 최대 폭 — 클래스와 testid 를 함께 노려 버전 차이를 흡수한다.
    width_rule = (f'div.block-container, div[data-testid="stMainBlockContainer"] '
                  f'{{ max-width: {m["max_w"]}px !important; }}') if m["max_w"] else ""
    return f"""
<style>
  /* Streamlit 이 [data-testid="stMarkdownContainer"] p 로 문단 글자 크기를 강제한다.
     그 선택자가 클래스 선택자보다 우선순위가 높아서, 크기는 !important 로 눌러야 먹는다. */

  {width_rule}

  /* 창 너비에 맞춰 커지되 상한에서 멈춘다 (좁은 창에서 두 줄로 깨지지 않게). */
  .wd-title   {{ font-size: clamp({1.8 * s:.2f}rem, {3 * s:.2f}vw, {3.4 * s:.2f}rem) !important;
                font-weight: 800; margin: 0 0 .5rem 0; letter-spacing: -.02em;
                line-height: 1.1; text-wrap: balance; }}
  .wd-sub     {{ opacity: .62; font-size: {1.1 * s:.2f}rem !important; margin: 0; }}
  .wd-sec     {{ font-size: {1.35 * s:.2f}rem !important; font-weight: 700;
                margin: .1rem 0 .6rem 0; }}
  .wd-sub2    {{ font-size: {1.05 * s:.2f}rem !important; font-weight: 650;
                margin: .2rem 0 .5rem 0; }}
  .wd-card    {{ border-radius: .55rem; padding: {.7 * s:.2f}rem {.9 * s:.2f}rem;
                margin-bottom: .55rem; border: 1px solid rgba(128,128,128,.18);
                font-size: {.95 * s:.2f}rem; }}
  .wd-defect  {{ background: rgba(239, 68, 68, .10); }}
  .wd-defect  b {{ color: #dc2626; }}
  .wd-proc    {{ background: rgba(234, 179,  8, .13); }}
  .wd-proc    b {{ color: #b45309; }}
  .wd-label   {{ opacity: .7; font-size: {.88 * s:.2f}rem !important;
                margin: .6rem 0 .25rem 0; }}
  div[data-testid="stButton"] > button[kind="primary"] {{
      background: #ef4444; border-color: #ef4444; font-weight: 700; }}
  div[data-testid="stButton"] > button[kind="primary"]:hover {{
      background: #dc2626; border-color: #dc2626; }}
</style>
"""


def setup_page(mode: str = DEFAULT_MODE):
    """st.set_page_config + 전역 CSS. 앱 진입점에서 가장 먼저 한 번 호출."""
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
    st.markdown(_css(mode), unsafe_allow_html=True)


def sec(text: str):
    """열 제목 (👉 입력 & 설정 …)"""
    st.markdown(f'<p class="wd-sec">{text}</p>', unsafe_allow_html=True)


def sub(text: str):
    """소제목 (🖼️ 인식된 웨이퍼 맵 …)"""
    st.markdown(f'<p class="wd-sub2">{text}</p>', unsafe_allow_html=True)


def card(label: str, value: str, kind: str = "defect", note: str = ""):
    """분홍(결함) / 노랑(공정) 요약 카드."""
    st.markdown(f'<div class="wd-card wd-{kind}">{label}: <b>{value}</b>'
                + (f" <small>{note}</small>" if note else "") + "</div>",
                unsafe_allow_html=True)
