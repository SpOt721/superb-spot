"""전처리 / 라벨 / 좌표 환산 / 추론 폴백 단위테스트."""
from __future__ import annotations
import json
import os

import numpy as np
import pytest
from PIL import Image

import models
from conftest import PAL_DEFECT, PAL_NORMAL, render, wafer_mask


# ---------------- _otsu ----------------
def test_otsu는_두_군집의_중점을_임계값으로_준다():
    v = np.concatenate([np.full(100, 0.2), np.full(100, 0.8)])
    t, sep = models.otsu(v)
    assert 0.45 < t < 0.55            # 두 값 사이 한가운데
    assert sep == pytest.approx(0.6, abs=0.05)


def test_otsu_단일군집이면_분리도가_0에_가깝다():
    _, sep = models.otsu(np.full(200, 0.42))
    assert sep < 0.05


def test_otsu_군집값이_구간경계에_걸려도_한쪽으로_쏠리지_않는다():
    """실팔레트에서 전 픽셀이 결함으로 뒤집혔던 회귀 케이스."""
    lo, hi = 0.28, 0.715                      # 실제 업로드본의 채도값
    v = np.concatenate([np.full(500, lo), np.full(100, hi)])
    t, _ = models.otsu(v)
    assert lo < t < hi


# ---------------- image_to_mask ----------------
def test_학습_팔레트를_정확히_분해():
    m0 = wafer_mask("Center")
    m = models.image_to_mask(render(m0), 52)
    assert (m == 2).sum() > 0
    assert (m == 1).sum() > (m == 2).sum()
    assert (m == 0).sum() > 0                 # 원판 밖 배경


@pytest.mark.parametrize("normal,defect,bg,name", [
    ((148, 163, 184), (239, 68, 68), (0, 0, 0), "학습 팔레트"),
    ((100, 116, 139), (239, 68, 68), (15, 23, 42), "실제 업로드본 팔레트"),
    ((200, 200, 200), (60, 90, 240), (0, 0, 0), "파랑 결함"),
    ((110, 110, 110), (255, 255, 255), (0, 0, 0), "흑백 웨이퍼맵"),
    ((89, 98, 110), (143, 41, 41), (0, 0, 0), "밝기 0.6배"),
    ((90, 100, 115), (240, 70, 70), (250, 250, 250), "흰 배경"),
])
def test_팔레트가_달라도_결함을_찾는다(normal, defect, bg, name):
    truth = wafer_mask("Center")
    m = models.image_to_mask(render(truth, normal, defect, bg), 52)
    got, want = (m == 2), (truth == 2)
    iou = (got & want).sum() / max((got | want).sum(), 1)
    assert iou > 0.8, f"{name}: IoU={iou:.2f}"


def test_정상_웨이퍼는_결함을_만들지_않는다():
    m = models.image_to_mask(render(wafer_mask("Normal")), 52)
    assert (m == 2).sum() == 0
    assert (m == 1).sum() > 100


def test_빈_이미지는_전부_배경():
    m = models.image_to_mask(Image.new("RGB", (256, 256), (0, 0, 0)), 52)
    assert (m == 0).all()


def test_결함이_어두운_표현은_환경변수로_뒤집는다(monkeypatch):
    """WAFER_DEFECT_BRIGHT=0 → 명도축에서 어두운 쪽을 결함으로."""
    truth = wafer_mask("Center")
    img = render(truth, normal=(230, 230, 230), defect=(60, 60, 60), bg=(0, 0, 0))
    monkeypatch.setattr(models.preprocess, "DEFECT_IS_BRIGHT", False)
    m = models.image_to_mask(img, 52)
    got, want = (m == 2), (truth == 2)
    assert (got & want).sum() / max((got | want).sum(), 1) > 0.8


# ---------------- render_canonical ----------------
def test_render_canonical은_학습_팔레트로_되돌린다():
    img = render(wafer_mask("Center"), normal=(100, 116, 139), bg=(15, 23, 42))
    out = models.render_canonical(img, 224)
    assert out.size == (224, 224)
    cols = {tuple(c) for c in np.unique(np.array(out).reshape(-1, 3), axis=0)}
    assert PAL_DEFECT in cols and PAL_NORMAL in cols and (0, 0, 0) in cols


def test_render_canonical은_입력크기와_무관하게_같은_결과():
    m = wafer_mask("Edge_Ring")
    a = np.array(models.render_canonical(render(m, size=128), 224))
    b = np.array(models.render_canonical(render(m, size=512), 224))
    assert np.array_equal(a, b)


# ---------------- 좌표 환산 ----------------
def test_scale_boxes_확대():
    boxes = [("Center", 10, 20, 30, 40, 0.9)]
    out = models.scale_boxes(boxes, 256, 256, 512, 128)
    assert out == [("Center", 20.0, 10.0, 60.0, 20.0, 0.9)]


def test_scale_boxes_동일크기면_그대로():
    boxes = [("Center", 10, 20, 30, 40, 0.9)]
    assert models.scale_boxes(boxes, 256, 256, 256, 256) == boxes


def test_scale_boxes_빈입력_안전():
    assert models.scale_boxes([], 256, 256, 512, 512) == []
    assert models.scale_boxes([("a", 1, 2, 3, 4, 0.5)], 0, 0, 10, 10) == [("a", 1, 2, 3, 4, 0.5)]


# ---------------- 라벨 / 가중치 ----------------
def test_라벨은_label_map_json을_따른다():
    with open(os.path.join(models.HERE, "label_map.json"), encoding="utf-8") as f:
        assert models.LABELS == json.load(f)["labels"]
    assert models.NUM_CLASSES == len(models.LABELS)


def test_탐지_가중치_자동탐색이_wafer_det을_고른다():
    if not os.path.exists(models.DET_PATH):
        pytest.skip("학습 가중치 없음")
    assert models.DET_PATH.endswith("best.pt")
    assert "wafer_det" in models.DET_PATH or models.DET_PATH.endswith(
        os.path.join(models.HERE, "best.pt"))


# ---------------- 분류 / 탐지 ----------------
def test_classify는_라벨전체_점수를_돌려준다(wafer_img):
    preds, scores, src = models.classify(wafer_img, thr=0.5)
    assert set(scores) == set(models.LABELS)
    assert all(0.0 <= v <= 1.0 for v in scores.values())
    assert set(preds) <= set(models.LABELS)
    assert isinstance(src, str) and src


def test_classify_임계값이_높으면_예측이_줄어든다(wafer_img):
    lo, _, _ = models.classify(wafer_img, thr=0.2)
    hi, _, _ = models.classify(wafer_img, thr=0.95)
    assert set(hi) <= set(lo)


def test_학습모델이_있으면_Center를_맞힌다(wafer_img):
    if not models.cls_available():
        pytest.skip("swin_multilabel.pt 없음")
    preds, scores, src = models.classify(wafer_img, thr=0.5)
    assert preds == ["Center"], f"{preds} / {scores}"
    assert "학습모델" in src


def test_휴리스틱_탐지는_원본좌표계_박스를_준다():
    img = render(wafer_mask("Center"), size=300)
    boxes, src = models.detect(img, source="heuristic")
    assert boxes and "휴리스틱" in src
    for _, x0, y0, x1, y1, c in boxes:
        assert 0 <= x0 < x1 <= 300 and 0 <= y0 < y1 <= 300
        assert 0.0 <= c <= 1.0


def test_yolo_탐지_박스가_이미지_안에_있다():
    if not models.yolo_available():
        pytest.skip("best.pt 없음")
    img = render(wafer_mask("Center"), size=300)
    boxes, src = models.detect(img, source="yolo", conf=0.25)
    assert "YOLO" in src
    for name, x0, y0, x1, y1, c in boxes:
        assert name in models.LABELS
        assert -1 <= x0 < x1 <= 301 and -1 <= y0 < y1 <= 301


def test_정상_웨이퍼는_휴리스틱_탐지가_박스를_안만든다(normal_img):
    boxes, src = models.detect(normal_img, source="heuristic")
    assert boxes == []
    assert "탐지없음" in src


def test_superb_미설정이면_명확한_사유를_돌려준다(normal_img, superb_offline):
    boxes, src = models.detect(normal_img, source="superb")
    assert boxes == []
    assert "미설정" in src or "실패" in src
