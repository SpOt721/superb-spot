"""
다중라벨 분류 — Swin(swin_multilabel.pt) → 휴리스틱 폴백.

학습과 동일한 입력 표현을 쓴다: 마스크 → 학습 팔레트 224px 렌더 → (x/255-0.5)/0.5
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .labels import CLS_SIZE, LABELS, NUM_CLASSES
from .preprocess import image_to_mask, render_canonical
from .weights import CLS_PATH

_swin = None
_swin_err = None


def _load_swin():
    global _swin, _swin_err
    if _swin is not None or _swin_err is not None:
        return _swin
    if not os.path.exists(CLS_PATH):
        _swin_err = "swin_multilabel.pt 없음"
        return None
    try:
        import torch, timm
        net = timm.create_model("swin_tiny_patch4_window7_224",
                                pretrained=False, num_classes=NUM_CLASSES)
        sd = torch.load(CLS_PATH, map_location="cpu")
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        net.load_state_dict(sd)
        net.eval()
        _swin = net
        return net
    except Exception as e:
        _swin_err = f"{type(e).__name__}: {e}"
        return None


def cls_available() -> bool:
    return _load_swin() is not None


def cls_error():
    _load_swin()
    return _swin_err


def classify(img: Image.Image, thr: float = 0.5):
    """반환: (labels, scores, source)"""
    net = _load_swin()
    if net is not None:
        import torch
        canon = render_canonical(img, CLS_SIZE)                 # 학습과 동일한 입력 표현
        arr = np.array(canon).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        x = torch.tensor(arr).permute(2, 0, 1).unsqueeze(0)
        with torch.no_grad():
            p = torch.sigmoid(net(x)).numpy()[0]
        scores = {LABELS[k]: float(p[k]) for k in range(min(NUM_CLASSES, len(p)))}
        preds = [l for l, s in scores.items() if s > thr]
        return preds, scores, "Swin 다중라벨(학습모델)"
    return heuristic_classify(image_to_mask(img))


def heuristic_classify(m: np.ndarray):
    """가중치가 없을 때의 규칙 기반 분류 (데모가 멈추지 않도록)."""
    H = W = m.shape[0]
    defect = (m == 2)
    total = defect.sum()
    scores = {l: 0.0 for l in LABELS}
    if total < 5:
        return [], scores, "휴리스틱(정상)"
    ys, xs = np.where(defect)
    yy, xx = np.mgrid[0:H, 0:W]
    rr = np.sqrt((yy - H / 2) ** 2 + (xx - W / 2) ** 2) / (H / 2)
    edge_ratio = defect[rr > 0.75].sum() / max(total, 1)
    center_ratio = defect[rr < 0.35].sum() / max(total, 1)
    density = total / (np.pi * (H / 2) ** 2)
    if total >= 8:
        c = np.cov(np.stack([xs, ys]))
        ev = np.sort(np.linalg.eigvalsh(c))
        elong = ev[1] / max(ev[0], 1e-6)
    else:
        elong = 1.0
    scores["Near_Full"] = min(density * 2.5, 1.0)
    scores["Edge_Ring"] = edge_ratio
    scores["Edge_Loc"] = edge_ratio * 0.7
    scores["Center"] = center_ratio
    scores["Loc"] = center_ratio * 0.6
    scores["Scratch"] = min(elong / 12.0, 1.0)
    scores["Random"] = min(density * 1.2, 1.0) \
        if elong < 3 and edge_ratio < 0.4 and center_ratio < 0.4 else 0.2
    preds = [l for l, s in scores.items() if s > 0.5] or [max(scores, key=scores.get)]
    return preds, scores, "휴리스틱(데모)"
