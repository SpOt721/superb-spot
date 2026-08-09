"""업로드 파일 → PIL.Image 변환 (F-01) + 형식 검증 (F-06)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from .theme import ALLOWED, ALLOWED_DATA

# MixedWM38 학습 팔레트 (npz 배열을 그림으로 되돌릴 때 사용)
_PAL_NORMAL = (148, 163, 184)
_PAL_DEFECT = (239, 68, 68)
_RENDER_SIZE = 256


def load_upload(up, npz_index: int = 0):
    """업로드 파일 → (PIL.Image, 설명). 실패 시 (None, 사유)."""
    ext = up.name.rsplit(".", 1)[-1].lower()
    if ext in ALLOWED_DATA:
        return _load_npz(up, npz_index)
    if ext not in ALLOWED:
        return None, "지원하지 않는 파일 형식입니다. 웨이퍼 맵 이미지(.png/.jpg) 또는 .npz 를 올려주세요."
    try:
        return Image.open(up).convert("RGB"), up.name
    except Exception:
        return None, "이미지를 열 수 없습니다. 손상되지 않은 웨이퍼 맵 파일인지 확인해 주세요."


def _load_npz(up, npz_index: int):
    try:
        data = np.load(io.BytesIO(up.getvalue()), allow_pickle=False)
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        X = data[key]
    except Exception as e:
        return None, f"NPZ를 읽을 수 없습니다: {e}"
    arr = np.asarray(X[npz_index] if X.ndim == 3 else X)
    if arr.ndim != 2:
        return None, "NPZ 안에서 2D 웨이퍼 맵을 찾지 못했습니다."
    m = arr.astype(int)
    rgb = np.zeros((*m.shape, 3), np.uint8)
    rgb[m == 1] = _PAL_NORMAL
    rgb[m == 2] = _PAL_DEFECT
    img = Image.fromarray(rgb).resize((_RENDER_SIZE, _RENDER_SIZE), Image.NEAREST)
    return img, f"{key}[{npz_index}] · {m.shape[0]}×{m.shape[1]}"


def npz_length(up) -> int:
    """npz 안에 들어 있는 웨이퍼 맵 장수 (인덱스 선택 UI용)."""
    try:
        data = np.load(io.BytesIO(up.getvalue()), allow_pickle=False)
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        return int(data[key].shape[0]) if data[key].ndim == 3 else 1
    except Exception:
        return 1
