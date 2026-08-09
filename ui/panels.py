"""
화면 구획(패널) — 순수 렌더링 + 위젯 값 반환만 담당한다.
모델 추론 / Superb 호출 같은 부수효과는 app.py 가 맡는다.
"""
from __future__ import annotations

import os

import streamlit as st

import graph_rag as gr
import models

from .io_utils import npz_length
from .theme import (ALLOWED_ALL, COL_H, DEFAULT_MODE, LAYOUT_MODES, REPORT_H,
                    SUBTITLE, card, sec, sub)

AUTO = "자동 인식"          # gr.PROCESS_CHOICES[0] — '자동 추정 유지'로 표시
MODE_KEY = "layout_mode"    # 화면 밀도 (세션에 저장 → 다음 rerun 의 CSS 에 반영)


# ---------------- 헤더 ----------------
def status_badges(status: dict, dep: dict) -> str:
    """연동/모델 상태 한 줄."""
    b_sb = f"🟢 Superb 연동({status['transport']})" if status.get("connected") else "⚪ Superb 오프라인"
    if dep.get("ok"):
        icon = "🟢" if dep["status"] == "ready" else "🟡"
        b_dp = f"{icon} 배포모델 {dep['status']} · {dep['name'] or dep['id'][:8]}"
    else:
        b_dp = "⚪ 배포모델 미설정"
    run_dir = os.path.basename(os.path.dirname(os.path.dirname(models.DET_PATH)))
    b_yo = f"🟢 YOLO 학습모델({run_dir})" if models.yolo_available() else f"⚪ YOLO 없음({models.yolo_error()})"
    b_cl = "🟢 Swin 분류" if models.cls_available() else f"⚪ Swin 없음({models.cls_error()})"
    return f"{b_sb} · {b_dp} · {b_yo} · {b_cl}"


def render_header(badges: str):
    """제목 + 부제 + 우측 상단 화면 밀도/PDF 슬롯. 반환된 슬롯을 분석 후 채운다."""
    h1, h2 = st.columns([4, 1])          # 제목이 한 줄에 들어가도록 넓게
    with h1:
        st.markdown('<p class="wd-title">🔬 반도체 웨이퍼 결함 분석 &amp; 3D 주파수 모니터링</p>',
                    unsafe_allow_html=True)
        st.markdown(f'<p class="wd-sub">{SUBTITLE}</p>', unsafe_allow_html=True)
    with h2:
        st.markdown('<div style="height:1.6rem"></div>', unsafe_allow_html=True)   # 커진 제목과 수직 정렬
        _render_layout_picker()
        pdf_slot = st.empty()
        pdf_slot.button("📥 PDF 리포트 다운로드", disabled=True, use_container_width=True)
    st.caption(badges)
    st.divider()
    return pdf_slot


def _render_layout_picker():
    """화면 밀도 선택 — 넓은 모니터에서 3열이 과하게 벌어지는 걸 조절한다."""
    st.session_state.setdefault(MODE_KEY, DEFAULT_MODE)
    options = list(LAYOUT_MODES)
    with st.popover(f"🖥️ 화면: {st.session_state[MODE_KEY]}", use_container_width=True):
        picker = getattr(st, "segmented_control", None)     # 구버전 Streamlit 폴백
        if picker:
            picker("화면 밀도", options, key=MODE_KEY, selection_mode="single")
        else:
            st.radio("화면 밀도", options, key=MODE_KEY, horizontal=True)
        st.caption(LAYOUT_MODES[st.session_state[MODE_KEY] or DEFAULT_MODE]["desc"])


def current_mode() -> str:
    """이번 rerun 에 적용할 화면 밀도 (CSS 주입 전에 읽는다)."""
    return st.session_state.get(MODE_KEY) or DEFAULT_MODE


def make_columns(height: int = COL_H):
    """1·2·3열을 같은 높이의 카드로. 나중에 추가되는 메시지도 같은 카드에 들어간다."""
    left, center, right = st.columns([1.05, 1.65, 1.25], gap="large")
    return (left.container(height=height, border=True),
            center.container(height=height, border=True),
            right.container(height=height, border=True))


# ---------------- 좌: 입력 & 설정 ----------------
def render_input(box, dep: dict) -> dict:
    """업로드(F-01/F-06) + 공정 교정(F-05) + 고급 설정. 위젯 값들을 dict 로 반환."""
    with box:
        sec("👉 입력 &amp; 설정")
        up = st.file_uploader("웨이퍼 이미지/데이터 업로드", type=sorted(ALLOWED_ALL),
                              help="드래그&드롭 지원 · MixedWM38 원본 .npz 도 가능")

        npz_index = 0
        if up is not None and up.name.lower().endswith(".npz"):
            n = npz_length(up)
            if n > 1:
                npz_index = st.number_input(f"샘플 인덱스 (0 ~ {n - 1})", 0, n - 1, 0, 1)

        st.divider()
        sub("🔧 엔지니어 공정 정답 교정")
        proc_override = st.selectbox(
            "추정 공정 수정 선택", gr.PROCESS_CHOICES, index=0,
            format_func=lambda s: "자동 추정 유지" if s == AUTO else s)

        with st.expander("⚙️ 고급 설정"):
            det_src = st.radio("탐지 엔진", ["auto", "superb", "yolo", "heuristic"],
                               format_func=lambda s: {"auto": "자동 (Superb→YOLO→폴백)",
                                                      "superb": "Superb 배포모델",
                                                      "yolo": "로컬 YOLOv8",
                                                      "heuristic": "휴리스틱"}[s], index=0)
            thr = st.slider("분류 임계값", 0.2, 0.8, 0.5, 0.05)
            rec = dep.get("recommended_conf") if dep.get("ok") else None
            dconf = st.slider("탐지 신뢰도 임계값", 0.05, 0.9, float(rec) if rec else 0.25, 0.05,
                              help=f"배포모델 추천값 {rec:.4f}" if rec else None)
    return dict(up=up, npz_index=npz_index, proc_override=proc_override,
                det_src=det_src, thr=thr, dconf=dconf)


def render_upload_status(box, upres: dict, asset_id, push, csrc: str, dsrc: str, online: bool):
    """분석 직후 좌측 카드에 붙는 결과/적재 메시지."""
    with box:
        st.success(f"분석 완료 · 분류:{csrc} · 탐지:{dsrc}")
        if upres.get("ok"):
            st.caption(f"Superb 업로드 완료 · asset={asset_id[:8] + '…' if asset_id else '생성 대기중'}"
                       + (" · 오토라벨 적재됨" if push and push.get("ok") else ""))
            if push and not push.get("ok"):
                st.caption(f"↳ 어노테이션 적재 실패: {push.get('reason')}")
            elif upres.get("pending"):
                st.caption(f"↳ {upres.get('reason')}")
        elif online:
            st.caption(f"Superb 업로드 실패: {upres.get('reason')}")


# ---------------- 중앙: 인식 메인 ----------------
def render_center(box):
    """제목 + 플롯 슬롯 + 분석 버튼. (plot_slot, caption_slot, run) 반환."""
    with box:
        sec("🎯 웨이퍼 화면 인식 메인")
        sub("🖼️ 인식된 웨이퍼 맵 (Bounding Box 표시)")
        plot_slot = st.empty()
        caption_slot = st.empty()
        run = st.button("🔍 웨이퍼 결함 및 주파수 분석 시작",
                        use_container_width=True, type="primary")
    return plot_slot, caption_slot, run


# ---------------- 우: 원인 분석 리포트 ----------------
def resolve_process(preds, proc_override: str) -> str:
    """자동 추정 공정 또는 엔지니어가 교정한 공정."""
    if proc_override != AUTO:
        return proc_override
    return gr.primary_process(preds) if preds else "정상"


def render_report(box, R: dict, proc_override: str, report_h: int = REPORT_H):
    """
    감지 결함 / 추정 공정 / 원인 문장 (F-04) + 강제 수정 (F-05).
    반환: (corrected_labels, feedback_clicked)
    """
    with box:
        sec("👉 원인 분석 리포트")
        sub("📋 AI 분석 결과")
        if not R:
            st.info("분석 결과가 여기에 표시됩니다.")
            return [], False

        preds = R["preds"]
        defect_txt = (" + ".join(preds) + (" (복합)" if len(preds) > 1 else "")) \
            if preds else "없음 (정상 웨이퍼)"
        card("감지 결함", defect_txt, "defect")
        card("추정 공정", resolve_process(preds, proc_override), "proc",
             note="" if proc_override == AUTO else "(엔지니어 교정)")

        use_llm = st.toggle("로컬 LLM 문장화(GPU 필요)", value=False)
        report_text = gr.report(preds, confidence=R["conf"], use_llm=use_llm)
        st.markdown('<p class="wd-label">자동 생성 원인 문장</p>', unsafe_allow_html=True)
        st.text_area("자동 생성 원인 문장", report_text,
                     height=report_h, label_visibility="collapsed")

        if R["conf"] is not None:
            st.progress(min(R["conf"], 1.0), text=f"분류 신뢰도 {R['conf']:.0%}")
        with st.expander("패턴별 점수"):
            st.bar_chart({k: v for k, v in R["scores"].items()})

        st.divider()
        sub("🔁 AI 판단 강제 수정 (액티브 러닝)")
        corrected = st.multiselect("실제 결함 패턴으로 수정", models.LABELS, default=preds)
        clicked = st.button("피드백 전송", use_container_width=True)
    return corrected, clicked


def render_feedback_result(box, sent: dict | None, online: bool, has_asset: bool):
    """피드백 전송 결과 메시지."""
    with box:
        if sent is not None:
            st.success("Superb로 피드백 전송됨(source=manual)" if sent.get("ok")
                       else f"전송 실패: {sent.get('reason')}")
        elif online and not has_asset:
            st.warning("자산 생성이 아직 진행 중입니다 — 잠시 후 다시 분석하면 적재됩니다.")
        elif not online:
            st.info("Superb 오프라인 — 수정값은 로컬 리포트에만 반영됩니다.")
