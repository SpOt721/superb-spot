"""앱 전 구간(리포트·PDF·시각화) + 샘플 생성기 단위테스트."""
from __future__ import annotations
import csv
import logging
import os

import numpy as np
import pytest
from PIL import Image

import graph_rag as gr
import make_samples
import models
import robustness
from conftest import render, wafer_mask

logging.getLogger("streamlit").setLevel(logging.CRITICAL)   # bare-mode 경고 제거
import app                                                  # noqa: E402


# ---------------- Graph-RAG ----------------
def test_리포트에_예측패턴과_공정이_들어간다():
    proc = gr.primary_process(["Center"])
    rep = gr.report(["Center"], confidence=0.9, use_llm=False)
    assert isinstance(proc, str) and proc
    assert "Center" in rep


def test_예측이_없어도_리포트가_깨지지_않는다():
    rep = gr.report([], confidence=None, use_llm=False)
    assert isinstance(rep, str) and rep.strip()


def test_혼합패턴도_처리한다():
    rep = gr.report(["Center", "Scratch"], confidence=0.8, use_llm=False)
    assert "Center" in rep and "Scratch" in rep


# ---------------- 시각화 / PDF ----------------
def test_박스_그리기(wafer_img):
    fig = app.draw_boxes(wafer_img, [("Center", 10, 10, 50, 50, 0.9)])
    assert fig is not None


def test_3D_주파수_시각화(wafer_img):
    assert app.freq_surface(wafer_img) is not None


def test_PDF가_유효한_바이트로_생성된다(wafer_img):
    preds, scores, _ = models.classify(wafer_img, thr=0.5)
    boxes, dsrc = models.detect(wafer_img, source="heuristic")
    pdf = app.build_pdf(wafer_img, boxes,
                        gr.report(preds, confidence=0.9, use_llm=False),
                        gr.primary_process(preds) if preds else "정상",
                        preds, scores, dsrc)
    assert pdf[:5] == b"%PDF-"
    assert len(pdf) > 10_000


def test_박스가_없어도_PDF가_생성된다(normal_img):
    _, scores, _ = models.classify(normal_img, thr=0.5)
    pdf = app.build_pdf(normal_img, [], "정상입니다", "정상", [], scores, "휴리스틱")
    assert pdf[:5] == b"%PDF-"


# ---------------- 샘플 생성기 ----------------
def test_합성샘플_12장과_정답라벨을_만든다(tmp_path):
    rows = make_samples.build_synthetic(str(tmp_path))
    assert len(rows) == len(make_samples.PLAN)
    for row in rows:
        assert (tmp_path / row[0]).exists()
        assert len(row) == 1 + len(make_samples.LABELS)


def test_합성샘플의_라벨과_그림이_일치한다(tmp_path):
    """Normal 은 결함 픽셀이 거의 없고, 패턴 샘플은 확실히 있어야 한다."""
    make_samples.build_synthetic(str(tmp_path))
    normal = models.image_to_mask(Image.open(tmp_path / "Normal.png"), 52)
    center = models.image_to_mask(Image.open(tmp_path / "Center.png"), 52)
    assert (normal == 2).sum() <= 30          # 배경 노이즈만
    assert (center == 2).sum() > 30


def test_to_png는_학습_팔레트를_쓴다():
    img = make_samples.to_png(wafer_mask("Center"))
    cols = {tuple(c) for c in np.unique(np.array(img).reshape(-1, 3), axis=0)}
    assert (239, 68, 68) in cols and (148, 163, 184) in cols


def test_라벨_순서가_모든_모듈에서_같다():
    assert make_samples.LABELS == models.LABELS
    import superb_client as sb
    assert sb.LABELS == models.LABELS


# ---------------- 강건성 스크립트의 교란 함수 ----------------
@pytest.mark.parametrize("name,fn", robustness.CASES)
def test_교란함수는_같은_크기의_RGB를_돌려준다(name, fn):
    img = render(wafer_mask("Center"), size=128)
    out = fn(img)
    assert isinstance(out, Image.Image)
    assert out.size == img.size
    assert out.convert("RGB").mode == "RGB"


def test_색교란_후에도_마스크가_비지_않는다():
    """색·밝기 교란에서 마스크가 통째로 비었던 회귀 케이스."""
    img = render(wafer_mask("Center"), size=128)
    for name, fn in robustness.CASES:
        m = models.image_to_mask(fn(img), 52)
        assert (m == 2).sum() > 0, f"{name}: 결함 픽셀 0개"
        assert (m == 1).sum() > 0, f"{name}: 정상 픽셀 0개"


# ---------------- 화면 밀도 (와이드 모드) ----------------
def test_밀도별_크기가_배율대로_계산된다():
    from ui import theme
    c, s, w = theme.sizes("컴팩트"), theme.sizes("표준"), theme.sizes("와이드")
    assert c["col_h"] < s["col_h"] < w["col_h"]
    assert c["plot_h"] < s["plot_h"] < w["plot_h"]
    assert s["scale"] == 1.0
    assert s["col_h"] == theme.COL_H


def test_알수없는_밀도는_기본값으로_떨어진다():
    from ui import theme
    assert theme.sizes("없는모드") == theme.sizes(theme.DEFAULT_MODE)


def test_컴팩트_표준은_본문폭을_제한하고_와이드는_전체폭():
    from ui import theme
    assert "max-width: 1180px" in theme._css("컴팩트")
    assert "max-width: 1560px" in theme._css("표준")
    assert "max-width" not in theme._css("와이드")


def test_밀도가_바뀌면_글자크기도_바뀐다():
    from ui import theme
    assert theme._css("컴팩트") != theme._css("와이드")
    for mode in theme.LAYOUT_MODES:
        assert "!important" in theme._css(mode)      # Streamlit 기본 규칙을 이겨야 함


def test_차트_높이를_인자로_바꿀_수_있다(wafer_img):
    from ui.charts import fft_surface_3d, wafer_figure
    assert wafer_figure(wafer_img, height=333).layout.height == 333
    assert fft_surface_3d(wafer_img, height=444).layout.height == 444
    assert wafer_figure(wafer_img).layout.height == models_theme_plot_h()


def models_theme_plot_h():
    from ui.theme import PLOT_H
    return PLOT_H


# ---------------- 파일 검증 (F-06) ----------------
def test_허용_확장자_목록():
    assert app.ALLOWED == {"png", "jpg", "jpeg", "bmp"}


def test_깨진_이미지는_열리지_않는다(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes("이건 PNG 가 아님".encode("utf-8"))
    with pytest.raises(Exception):
        Image.open(bad).convert("RGB")
