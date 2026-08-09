"""클래스 라벨 + 학습 시 입력 규격. label_map.json 이 있으면 그걸 따른다."""
from __future__ import annotations

import json
import os

# 이 패키지는 프로젝트 루트 아래에 있다 — 가중치/라벨 파일은 루트 기준으로 찾는다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_DEFAULT_LABELS = ["Center", "Donut", "Edge_Loc", "Edge_Ring",
                   "Loc", "Near_Full", "Scratch", "Random"]


def load_labels(path: str = None) -> list:
    path = path or os.path.join(ROOT, "label_map.json")
    try:
        with open(path, encoding="utf-8") as f:
            labels = json.load(f).get("labels")
        if isinstance(labels, list) and labels:
            return [str(x) for x in labels]
    except Exception:
        pass
    return list(_DEFAULT_LABELS)


LABELS = load_labels()
NUM_CLASSES = len(LABELS)

# 학습(WaferDefect_AllInOne.ipynb)과 동일한 팔레트 / 입력 크기
#   MixedWM38 arr_0: 0 빈칸 / 1 정상 / 2 불량
PALETTE = {1: (148, 163, 184), 2: (239, 68, 68)}
CLS_SIZE = 224          # Swin 입력
DET_SIZE = 256          # YOLO 학습 imgsz
