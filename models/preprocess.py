"""
입력 전처리 — 업로드된 임의 이미지를 학습과 같은 표현으로 정규화한다.

  image_to_mask()     이미지 → 52x52 마스크(0 빈칸 / 1 정상 / 2 불량)
  render_canonical()  마스크 → 학습 팔레트로 재렌더 (모델 입력용)

색을 하드코딩하지 않는 게 핵심이다. 툴·캡처마다 팔레트와 밝기가 달라서,
고정 임계값을 쓰면 색이 조금만 달라도 마스크가 통째로 비어 버린다.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .labels import PALETTE

GRID = 52               # MixedWM38 격자

# 결함이 정상 영역보다 '밝은' 표현을 기본으로 본다. 반대로 그린 툴이면 0으로.
DEFECT_IS_BRIGHT = os.environ.get("WAFER_DEFECT_BRIGHT", "1") != "0"
SEP_MIN = 0.14          # 두 군집이 이만큼도 안 벌어지면 '결함 없음'으로 판단


def otsu(v: np.ndarray):
    """
    [0,1] 값 배열을 두 군집으로 나눈다 → (임계값, 분리도).
    임계값은 히스토그램 경계가 아니라 '두 군집 평균의 중점' — 군집 값이 경계에
    딱 걸려 전체가 한쪽으로 쏠리는 사고를 막는다.
    """
    if v.size < 8:
        return 1.0, 0.0
    hist, edges = np.histogram(v, bins=64, range=(0.0, 1.0))
    p = hist.astype(float) / max(hist.sum(), 1)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    csum = np.cumsum(p * centers)
    mu0 = csum / np.maximum(w0, 1e-9)
    mu1 = (csum[-1] - csum) / np.maximum(w1, 1e-9)
    # 한쪽 군집이 비어 있는 분할은 후보에서 제외한다. 그러지 않으면 값이 하나뿐인
    # (= 결함이 전혀 없는 정상 웨이퍼) 입력에서 분리도가 크게 나와 전체를 결함으로 칠한다.
    valid = (w0 > 1e-6) & (w1 > 1e-6)
    if not valid.any():
        return 1.0, 0.0
    var_b = np.where(valid, w0 * w1 * (mu0 - mu1) ** 2, -1.0)
    k = int(np.argmax(var_b))
    return float((mu0[k] + mu1[k]) / 2), float(abs(mu1[k] - mu0[k]))


def image_to_mask(img: Image.Image, size: int = GRID) -> np.ndarray:
    """
    웨이퍼 이미지 → 마스크(0 빈칸, 1 정상, 2 불량).

      1) 배경색은 이미지 테두리에서 추정 (웨이퍼 원판은 항상 안쪽에 있음)
      2) 채도(S)와 명도(V) 각각에 Otsu를 걸어 '더 잘 갈라지는 축'을 선택
         - 회색 웨이퍼에 컬러 결함  → 채도로 분리
         - 흑백 웨이퍼맵            → 명도로 분리
      3) 두 군집이 충분히 안 벌어지면 결함 없음(정상 웨이퍼)으로 처리
    """
    a = np.array(img.convert("RGB").resize((size, size), Image.NEAREST)).astype(np.float32) / 255.0
    mx, mn = a.max(-1), a.min(-1)
    val = mx
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    # 1) 배경: 테두리 2px의 중앙값 색과 가까운 픽셀
    frame = np.concatenate([a[:2].reshape(-1, 3), a[-2:].reshape(-1, 3),
                            a[:, :2].reshape(-1, 3), a[:, -2:].reshape(-1, 3)])
    bg_color = np.median(frame, axis=0)
    bg = np.linalg.norm(a - bg_color, axis=-1) < 0.20
    wafer = ~bg
    m = np.zeros((size, size), int)
    if wafer.sum() < 8:
        return m
    m[wafer] = 1

    # 2) 채도축 / 명도축 중 분리가 잘 되는 쪽으로 결함 판정
    t_s, sep_s = otsu(sat[wafer])
    t_v, sep_v = otsu(val[wafer])
    if max(sep_s, sep_v) < SEP_MIN:              # 3) 단일 군집 = 결함 없음
        return m
    if sep_s >= sep_v:
        defect = wafer & (sat > t_s)             # 더 진한 색 = 결함
    else:
        defect = wafer & ((val > t_v) if DEFECT_IS_BRIGHT else (val < t_v))
    m[defect] = 2
    return m


def render_canonical(img: Image.Image, size: int) -> Image.Image:
    """업로드 이미지 → 마스크 → 학습과 동일한 팔레트/크기로 재렌더."""
    m = image_to_mask(img, GRID)
    rgb = np.zeros((*m.shape, 3), np.uint8)
    for v, color in PALETTE.items():
        rgb[m == v] = color
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)
