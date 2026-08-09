"""
반도체 웨이퍼 결함 분석 & 3D 주파수 모니터링 — 기획서 F-01 ~ F-07 (Streamlit)

실행:
    python3 -m streamlit run app.py

이 파일은 '조립'만 한다 — 화면은 ui/ 패키지, 추론은 models.py,
연동은 superb_client.py, 원인 역추적은 graph_rag.py 가 담당.
(리팩터링 전 단일 파일 버전은 app_backup.py 에 보관)

  F-01 업로드 + Superb 저장     F-02 결함 자동 탐지(Bounding Box)
  F-03 3D 주파수 시각화         F-04 공정 판단 + 원인 역추적 리포트
  F-05 판단 강제 수정 + 피드백    F-06 파일 예외 처리
  F-07 결함 분석 리포트 PDF
"""
import os
import tempfile

import streamlit as st

import graph_rag as gr
import models
import superb_client as sb
from ui import panels
from ui.charts import draw_boxes, fft_surface_3d, freq_surface, wafer_figure   # noqa: F401
from ui.io_utils import load_upload, npz_length                                # noqa: F401
from ui.report_pdf import build_pdf
from ui.theme import (ALLOWED, ALLOWED_ALL, ALLOWED_DATA, sec, setup_page,     # noqa: F401
                      sizes, sub)

# 화면 밀도(컴팩트/표준/와이드)는 세션에 저장돼 있다 — CSS 를 주입하기 전에 읽어야 한다.
MODE = panels.current_mode()
SZ = sizes(MODE)
setup_page(MODE)


# 배포 환경용: 가중치가 없고 WAFER_*_URL 이 설정돼 있으면 최초 1회 내려받는다.
# (로컬은 파일이 이미 있으므로 no-op) — models.* 를 처음 호출하기 전에 실행해야 한다.
# 결과를 stdout 에 남긴다: 배포 로그에서 'skipped'(URL 없음) / 'failed'(URL 오류) /
# 'downloaded' 를 구분할 수 있어야 원인을 찾을 수 있다.
@st.cache_resource(show_spinner="모델 가중치 준비 중…")
def _ensure_weights():
    try:
        res = models.ensure_weights()
    except Exception as e:
        res = {"error": str(e)}
    print("[weights]", ", ".join(f"{k}={v}" for k, v in res.items()), flush=True)
    for name, url_key in (("swin_multilabel.pt", "WAFER_CLS_URL"), ("best.pt", "WAFER_DET_URL")):
        if res.get(name) == "skipped":
            print(f"[weights] {name} 없음 — {url_key} 환경변수/Secrets 가 비어 있습니다.", flush=True)
    return res


WEIGHTS = _ensure_weights()

# ---------------------------- 상태 ----------------------------
st.session_state.setdefault("asset_id", None)
st.session_state.setdefault("result", None)


@st.cache_data(ttl=60, show_spinner=False)
def superb_status():
    return sb.status()


S = superb_status()
dep = S.get("deployment") or {}

# ---------------------------- 화면 뼈대 ----------------------------
pdf_slot = panels.render_header(panels.status_badges(S, dep) + panels.weights_hint(WEIGHTS))
left_box, center_box, right_box = panels.make_columns(SZ["col_h"])

ctl = panels.render_input(left_box, dep)
plot_slot, caption_slot, run = panels.render_center(center_box)

# ---------------------------- 입력 로드 (F-01, F-06) ----------------------------
img, note = None, None
if ctl["up"] is not None:
    img, note = load_upload(ctl["up"], ctl["npz_index"])
    if img is None:
        with left_box:
            st.error(note)

# ---------------------------- 분석 실행 ----------------------------
if run and img is None:
    with center_box:
        st.warning("먼저 웨이퍼 이미지 또는 .npz 파일을 업로드하세요.")
elif run:
    # F-01: Superb 저장소 업로드 (연동 시)
    upres = {"ok": False, "reason": "offline"}
    if sb.available():
        with st.spinner("Superb 업로드 중…"):
            stem = os.path.basename(ctl["up"].name).replace(os.sep, "_").rsplit(".", 1)[0]
            tmp = os.path.join(tempfile.gettempdir(), f"{stem}.png")
            img.save(tmp)
            upres = sb.upload_image(tmp, key=f"{stem}.png")
    st.session_state.asset_id = upres.get("asset_id")

    # 분류 + F-02 탐지
    with st.spinner("모델 추론 중…"):
        preds, scores, csrc = models.classify(img, thr=ctl["thr"])
        boxes, dsrc = models.detect(img, source=ctl["det_src"], conf=ctl["dconf"])
    st.session_state.result = dict(
        img=img, preds=preds, scores=scores, csrc=csrc, boxes=boxes, dsrc=dsrc,
        conf=max([scores[p] for p in preds], default=None), note=note)

    # 오토라벨링 결과 Superb 적재 (source=model)
    push = sb.push_prediction(st.session_state.asset_id, preds) \
        if (st.session_state.asset_id and preds) else None
    panels.render_upload_status(left_box, upres, st.session_state.asset_id,
                                push, csrc, dsrc, sb.available())

R = st.session_state.result

# ---------------------------- 중앙 플롯 (F-02) ----------------------------
if R:
    plot_slot.plotly_chart(wafer_figure(R["img"], R["boxes"], SZ["plot_h"]))
    caption_slot.caption(f"탐지 소스: {R['dsrc']} · Bounding Box {len(R['boxes'])}개"
                         + (f" · {R['note']}" if R.get("note") else ""))
elif img is not None:
    plot_slot.plotly_chart(wafer_figure(img, height=SZ["plot_h"]))
    caption_slot.caption(f"미리보기 · {note} — [분석 시작]을 누르면 결함을 탐지합니다.")
else:
    plot_slot.info("좌측에서 웨이퍼 맵 이미지(.png/.jpg) 또는 MixedWM38 .npz 를 업로드하세요.")

# ---------------------------- 우: 리포트 (F-04) + 피드백 (F-05) ----------------------------
corrected, feedback_clicked = panels.render_report(right_box, R, ctl["proc_override"],
                                                   SZ["report_h"])
if feedback_clicked and R:
    sent = sb.push_feedback(st.session_state.asset_id, corrected) \
        if st.session_state.asset_id else None
    panels.render_feedback_result(right_box, sent, sb.available(),
                                  bool(st.session_state.asset_id))
    R["preds"] = corrected
    R["conf"] = max([R["scores"].get(p, 0.6) for p in corrected], default=None)

# ---------------------------- F-07 PDF (헤더 우측 슬롯) ----------------------------
if R:
    final_preds = R["preds"]
    pdf_slot.download_button(
        "📥 PDF 리포트 다운로드",
        build_pdf(R["img"], R["boxes"],
                  gr.report(final_preds, confidence=R["conf"], use_llm=False),
                  panels.resolve_process(final_preds, ctl["proc_override"]),
                  final_preds, R["scores"], R["dsrc"]),
        file_name="wafer_report.pdf", mime="application/pdf", width="stretch")

# ---------------------------- 하단: 3D 주파수 (F-03) ----------------------------
st.divider()
sec("⬇️ 3D 공간 주파수 입체 히트맵 (FFT Surface)")
sub("2D FFT ➜ 3D 공간 주파수 Surface 시각화")
if R or img is not None:
    st.plotly_chart(fft_surface_3d(R["img"] if R else img, SZ["fft_h"]))
    st.caption("규칙적 패턴일수록 특정 주파수에 에너지가 집중됩니다 · 드래그로 회전, 스크롤로 확대")
else:
    st.info("웨이퍼 맵을 업로드하면 2D FFT 기반 3D 주파수 표면이 여기에 표시됩니다.")
