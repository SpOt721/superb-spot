"""
결함 탐지 — Superb 배포모델 → 로컬 YOLOv8(best.pt) → 휴리스틱 폴백.

박스 좌표는 항상 '원본 이미지 픽셀' 기준으로 돌려준다.
모델에는 학습과 같은 256px 캔버스를 보내고, 결과를 원본 크기로 환산한다.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .classifier import heuristic_classify
from .labels import DET_SIZE, LABELS, ROOT
from .preprocess import GRID, image_to_mask, render_canonical
from .weights import DET_PATH

_yolo = None
_yolo_err = None


def _load_yolo():
    global _yolo, _yolo_err
    if _yolo is not None or _yolo_err is not None:
        return _yolo
    if not os.path.exists(DET_PATH):
        _yolo_err = "best.pt 없음"
        return None
    try:
        from ultralytics import YOLO
        _yolo = YOLO(DET_PATH)
        return _yolo
    except Exception as e:
        _yolo_err = f"{type(e).__name__}: {e}"
        return None


def yolo_available() -> bool:
    return _load_yolo() is not None


def yolo_error():
    _load_yolo()
    return _yolo_err


def scale_boxes(boxes, src_w: float, src_h: float, dst_w: float, dst_h: float):
    """모델 입력 캔버스 좌표 → 원본 이미지 좌표."""
    if not boxes or not src_w or not src_h:
        return boxes
    sx, sy = dst_w / float(src_w), dst_h / float(src_h)
    if abs(sx - 1) < 1e-9 and abs(sy - 1) < 1e-9:
        return boxes
    return [(n, x0 * sx, y0 * sy, x1 * sx, y1 * sy, c) for (n, x0, y0, x1, y1, c) in boxes]


def detect(img: Image.Image, source: str = "auto", conf: float = 0.25):
    """
    source: 'auto' | 'superb' | 'yolo' | 'heuristic'
    반환: (boxes[(label,x0,y0,x1,y1,conf)], source_str)
    """
    W, H = img.size

    # 1) Superb 배포 모델
    if source in ("auto", "superb"):
        try:
            import superb_client as sb
            if sb.deploy_available():
                canon = render_canonical(img, DET_SIZE)      # 학습 팔레트로 렌더해서 전송
                r = sb.detect_remote(canon, conf=conf)
                if r.get("ok"):
                    sw, sh = r.get("size") or (DET_SIZE, DET_SIZE)
                    boxes = scale_boxes(r["boxes"], sw or DET_SIZE, sh or DET_SIZE, W, H)
                    if boxes or source == "superb":
                        return boxes, f"Superb 배포모델 ({r.get('inference_ms', 0)}ms)"
                elif source == "superb":
                    return [], f"Superb 호출 실패: {r.get('reason', '?')}"
            elif source == "superb":
                d = sb.resolve_deployment()
                return [], f"Superb 배포모델 미설정: {d.get('reason', 'SUPERB_DEPLOYMENT_ID 확인')}"
        except Exception as e:
            if source == "superb":
                return [], f"Superb 오류({type(e).__name__}: {e})"

    # 2) 로컬 YOLO (학습 imgsz=256 캔버스에서 추론 후 원본 좌표로 환산)
    if source in ("auto", "yolo"):
        model = _load_yolo()
        if model is not None:
            canon = render_canonical(img, DET_SIZE)
            # PIL을 그대로 넘긴다 — numpy로 넘기면 ultralytics가 BGR로 간주해 R/B가 뒤집힌다.
            r = model.predict(canon, imgsz=DET_SIZE, conf=conf, verbose=False)[0]
            names = getattr(model, "names", None) or {i: l for i, l in enumerate(LABELS)}
            boxes = []
            for b in r.boxes:
                x0, y0, x1, y1 = b.xyxy[0].tolist()
                k = int(b.cls[0])
                boxes.append((str(names.get(k, LABELS[k] if k < len(LABELS) else k)),
                              x0, y0, x1, y1, float(b.conf[0])))
            boxes = scale_boxes(boxes, DET_SIZE, DET_SIZE, W, H)
            if boxes or source == "yolo":
                return boxes, f"YOLOv8 학습모델 ({os.path.relpath(DET_PATH, ROOT)})"
        elif source == "yolo":
            return [], f"YOLO 가중치 사용 불가: {yolo_error()}"

    # 3) 휴리스틱
    return heuristic_detect(img)


def heuristic_detect(img: Image.Image):
    """가중치가 없을 때: 불량 픽셀 연결요소를 박스로."""
    from scipy import ndimage
    W0, H0 = img.size
    m = image_to_mask(img, GRID)
    mask = (m == 2)
    if mask.sum() == 0:
        return [], "휴리스틱(탐지없음)"
    lbl, n = ndimage.label(mask)
    sx, sy = W0 / float(GRID), H0 / float(GRID)
    preds, _, _ = heuristic_classify(m)
    name = preds[0] if preds else "Defect"
    boxes = []
    for c in range(1, n + 1):
        ys, xs = np.where(lbl == c)
        if len(xs) < 4:
            continue
        boxes.append((name, xs.min() * sx, ys.min() * sy,
                      (xs.max() + 1) * sx, (ys.max() + 1) * sy, 0.5))
    return boxes, "휴리스틱(데모)"
