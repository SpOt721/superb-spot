"""
모델 패키지 — 로딩 & 추론.

  labels      클래스 목록(label_map.json) + 학습 입력 규격
  weights     가중치 경로 탐색 / 없으면 내려받기
  preprocess  이미지 → 마스크 → 학습 팔레트 재렌더
  classifier  Swin 다중라벨 분류 (+ 휴리스틱 폴백)
  detector    Superb 배포모델 → YOLOv8 → 휴리스틱 폴백

학습(WaferDefect_AllInOne.ipynb)과 '입력 표현'을 정확히 맞춘다:
  MixedWM38 arr_0 (0 빈칸 / 1 정상 / 2 불량) → RGB 팔레트 렌더
    1 → (148,163,184),  2 → (239,68,68),  0 → (0,0,0)
  분류 Swin : 224x224 NEAREST,  (x/255 - 0.5)/0.5
  탐지 YOLO : 256x256 NEAREST  (imgsz=256, nc=8)

호출부는 `import models` 만 하면 된다 — 아래 이름들이 그대로 노출된다.
"""
from __future__ import annotations

import config  # noqa: F401  — .env / Streamlit Secrets 를 os.environ 으로 (import 부수효과)

from .classifier import classify, cls_available, cls_error, heuristic_classify
from .detector import detect, heuristic_detect, scale_boxes, yolo_available, yolo_error
from .labels import CLS_SIZE, DET_SIZE, LABELS, NUM_CLASSES, PALETTE, ROOT
from .preprocess import GRID, image_to_mask, otsu, render_canonical
from .weights import CLS_PATH, DET_PATH, ensure_weights, find_det_weights

HERE = ROOT             # 하위 호환: 예전 models.py 는 루트에 있었다

__all__ = [
    "LABELS", "NUM_CLASSES", "PALETTE", "CLS_SIZE", "DET_SIZE", "GRID", "ROOT", "HERE",
    "CLS_PATH", "DET_PATH", "find_det_weights", "ensure_weights",
    "image_to_mask", "render_canonical", "otsu",
    "classify", "cls_available", "cls_error", "heuristic_classify",
    "detect", "yolo_available", "yolo_error", "heuristic_detect", "scale_boxes",
]
