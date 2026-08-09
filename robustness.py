"""
일반화(강건성) 측정 — samples/ 의 실데이터에 교란을 걸어 정확도 변화를 본다.

  python3 robustness.py
  python3 robustness.py --dir samples --thr 0.5

"실제 사용자가 우리가 학습한 것과 똑같이 생긴 이미지를 넣지는 않는다"를 검증하는 용도.
색/밝기/해상도 교란에서 점수가 무너지면 그건 대개 모델이 아니라 image_to_mask 전처리 문제다.
"""
from __future__ import annotations
import argparse
import csv
import io
import os

import numpy as np
from PIL import Image, ImageEnhance

import config          # noqa: F401  (.env 로드)
import models


def _rotate(img, deg):
    return img.rotate(deg, resample=Image.NEAREST, fillcolor=(0, 0, 0))


def _rescale(img, f):
    """웨이퍼를 작게 그리고 여백을 남김 (화면 캡처로 올린 상황)."""
    w, h = img.size
    small = img.resize((max(int(w * f), 8), max(int(h * f), 8)), Image.NEAREST)
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(small, ((w - small.width) // 2, (h - small.height) // 2))
    return canvas


def _recolor(img, defect, normal):
    """다른 팔레트로 그린 웨이퍼 맵 (툴마다 색이 다름)."""
    m = models.image_to_mask(img, 52)
    out = np.zeros((52, 52, 3), np.uint8)
    out[m == 1] = normal
    out[m == 2] = defect
    return Image.fromarray(out).resize(img.size, Image.NEAREST)


def _noisy(img, p, seed=0):
    a = np.array(img).copy()
    rng = np.random.default_rng(seed)
    hit = (rng.random(a.shape[:2]) < p) & (a.sum(-1) > 60)
    a[hit] = (239, 68, 68)
    return Image.fromarray(a)


def _jpeg(img, q):
    b = io.BytesIO()
    img.save(b, format="JPEG", quality=q)
    b.seek(0)
    return Image.open(b).convert("RGB")


def _lowres(img, s):
    return img.resize((s, s), Image.BILINEAR).resize(img.size, Image.BILINEAR)


CASES = [
    ("원본 (in-distribution)", lambda im: im),
    ("회전 90°", lambda im: _rotate(im, 90)),
    ("회전 30°", lambda im: _rotate(im, 30)),
    ("좌우 반전", lambda im: im.transpose(Image.FLIP_LEFT_RIGHT)),
    ("축소 70% (여백)", lambda im: _rescale(im, 0.7)),
    ("다른 팔레트(파랑 결함)", lambda im: _recolor(im, (60, 90, 240), (200, 200, 200))),
    ("흑백 웨이퍼맵", lambda im: _recolor(im, (255, 255, 255), (110, 110, 110))),
    ("노이즈 +3%", lambda im: _noisy(im, 0.03)),
    ("JPEG q40", lambda im: _jpeg(im, 40)),
    ("저해상도 64px", lambda im: _lowres(im, 64)),
    ("밝기 0.6배", lambda im: ImageEnhance.Brightness(im).enhance(0.6)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="samples")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    lp = os.path.join(a.dir, "labels.csv")
    if not os.path.exists(lp):
        raise SystemExit(f"{lp} 가 없습니다. python3 make_samples.py --from-superb 를 먼저 실행하세요.")
    truth = {}
    with open(lp, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth[row["file"]] = {l for l in models.LABELS if row.get(l) == "1"}
    files = [f for f in sorted(truth) if os.path.exists(os.path.join(a.dir, f))]
    if not files:
        raise SystemExit("이미지가 없습니다.")

    print(f"{len(files)}장 기준 · 분류 임계값 {a.thr} · 탐지 로컬 YOLO {a.conf}\n")
    print(f"{'교란':28s} {'분류 완전일치':>13s} {'탐지 박스':>10s}")
    print("-" * 56)
    base = None
    for name, fn in CASES:
        hit = det = 0
        for f in files:
            img = fn(Image.open(os.path.join(a.dir, f)).convert("RGB"))
            preds, _, _ = models.classify(img, thr=a.thr)
            hit += set(preds) == truth[f]
            boxes, _ = models.detect(img, source="yolo", conf=a.conf)
            det += bool(boxes)
        acc = hit / len(files)
        base = acc if base is None else base
        flag = "  ⚠️" if base - acc >= 0.15 else ""
        print(f"{name:28s} {hit:>3}/{len(files)} = {acc:4.0%} {det:>6}/{len(files)}{flag}")
    print("\n⚠️ 는 원본 대비 15%p 이상 하락 — 대개 전처리(image_to_mask)가 색/밝기를 못 읽은 경우입니다.")


if __name__ == "__main__":
    main()
