"""
학습 가중치의 위치를 정하고, 없으면 내려받는다.

  swin_multilabel.pt 는 105MB 로 GitHub 일반 파일 상한(100MB)을 넘는다.
  레포에 넣는 대신 GitHub Releases 에 올려두고 배포 환경에서 받아오는 방식.

    export WAFER_CLS_URL=https://github.com/<user>/<repo>/releases/download/weights-v1/swin_multilabel.pt
    export WAFER_DET_URL=https://github.com/<user>/<repo>/releases/download/weights-v1/best.pt
    python3 -m models.weights

  - 이미 파일이 있으면 건너뛴다 (로컬 개발 환경은 아무 영향 없음).
  - URL 이 없으면 아무것도 하지 않는다.
  - 실패해도 앱은 휴리스틱 폴백으로 뜨므로 데모가 죽지는 않는다.
"""
from __future__ import annotations

import glob
import os
import sys

from .labels import ROOT


def find_det_weights() -> str:
    """best.pt 탐색: 환경변수 → 프로젝트 루트 → runs/detect/*/weights/best.pt (최신)."""
    env = os.environ.get("WAFER_DET_PT")
    if env:
        return env
    root = os.path.join(ROOT, "best.pt")
    if os.path.exists(root):
        return root
    # 노트북 학습 run(wafer_det*)을 우선, 없으면 아무 detect run 중 최신
    for pattern in ("wafer_det*", "*"):
        cands = glob.glob(os.path.join(ROOT, "runs", "detect", pattern, "weights", "best.pt"))
        if cands:
            return max(cands, key=os.path.getmtime)
    return root                      # 없으면 존재하지 않는 경로 그대로(폴백 유도)


DET_PATH = find_det_weights()
CLS_PATH = os.environ.get("WAFER_CLS_PT", os.path.join(ROOT, "swin_multilabel.pt"))

TARGETS = [("WAFER_CLS_URL", CLS_PATH), ("WAFER_DET_URL", DET_PATH)]


def _download(url: str, dest: str):
    import requests
    tmp = dest + ".part"
    try:
        with requests.get(url, stream=True, timeout=300, allow_redirects=True) as r:
            if r.status_code >= 400:
                return False, f"HTTP {r.status_code}"
            os.makedirs(os.path.dirname(os.path.abspath(dest)) or ".", exist_ok=True)
            size = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
                    size += len(chunk)
        if size < 1024:                     # 포인터 파일/에러 페이지를 받은 경우
            os.remove(tmp)
            return False, f"파일이 너무 작음({size}B) — URL 확인 필요"
        os.replace(tmp, dest)
        return True, f"{size / 1e6:.1f}MB"
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return False, f"{type(e).__name__}: {e}"


def ensure_weights(verbose: bool = False) -> dict:
    """가중치가 없고 URL 이 있으면 내려받는다. {파일명: 상태} 반환."""
    out = {}
    for env_key, dest in TARGETS:
        name = os.path.basename(dest)
        if os.path.exists(dest):
            out[name] = "exists"
        elif not os.environ.get(env_key, "").strip():
            out[name] = "skipped"           # URL 없음 → 휴리스틱 폴백
        else:
            ok, info = _download(os.environ[env_key].strip(), dest)
            out[name] = f"downloaded({info})" if ok else f"failed({info})"
        if verbose:
            print(f"  {name:24s} {out[name]}")
    return out


if __name__ == "__main__":
    import config  # noqa: F401  — .env / Secrets 로드
    print("가중치 확인 중…")
    res = ensure_weights(verbose=True)
    sys.exit(1 if any(v.startswith("failed") for v in res.values()) else 0)
