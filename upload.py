"""
MixedWM38 → Superb '결함 위치' 바운딩 박스 자동 생성 & 업로드
====================================================================
전략: 분류(이미지 8-hot)가 정답의 중심, 박스는 "결함 위치" 표시용.
  - 박스: PNG의 빨간(불량) 픽셀 → 연결요소별 바운딩 박스 → 'Defect' 1클래스로 업로드.
          (어느 픽셀이 어느 패턴인지 정답이 없으므로 8클래스로 나누지 않음 = 정직)
  - 8-hot: labels_ref.csv 그대로 분류기(Swin) 학습에 사용.

설정은 전부 .env / 환경변수에서 읽는다 (superb_client 와 동일한 값을 공유).
    SUPERB_AI_API_KEY / SUPERB_AI_TENANT / SUPERB_PROJECT_ID

준비물:
  pip install pillow numpy scipy pandas
  객체 검출 스키마에 'Defect' 클래스 1개 (어노테이션 타입 = 바운딩 박스)

실행:
  python3 upload.py                 # PREVIEW — overlay 몇 장만 저장하고 끝 (기본)
  python3 upload.py --upload        # 실제 업로드
  python3 upload.py --dir ./superb_png --class-name Defect --replace
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import uuid

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

import config                      # noqa: F401  — .env / Secrets 를 os.environ 으로
import superb_client as sc         # 인증·전송은 앱과 동일한 경로를 재사용

# ============================ 기본값 (CLI 로 덮어쓸 수 있음) ============================
PNG_DIR = "./superb_png"                      # wafer_*.png 폴더
LABELS_CSV = "./superb_png/labels_ref.csv"    # file + 8-hot
DEFECT_CLASS = "Defect"                       # 객체 검출 스키마의 위치표시 클래스명

# 빨간(불량) 픽셀 판정 (R 높고 G/B 낮음) — PREVIEW 로 확인 후 조정
RED_MIN, GB_MAX = 120, 120
MIN_BOX_PX = 3        # 이보다 작은 박스는 노이즈로 버림
DILATE_ITERS = 1      # 인접 결함을 살짝 합쳐 박스 파편화 방지
MAX_BOXES = 8         # 박스가 이보다 많으면(=Random/Near_Full) 전체 1박스로 대체
PREVIEW_N = 12
PREVIEW_DIR = "./_preview"
BATCH = 1000          # /annotations/batch-create 의 서버 상한

CLASS_ORDER = ["Center", "Donut", "Edge_Loc", "Edge_Ring",
               "Loc", "Near_Full", "Scratch", "Random"]


def red_mask(png_path):
    im = np.asarray(Image.open(png_path).convert("RGB"))
    R, G, B = im[..., 0].astype(int), im[..., 1].astype(int), im[..., 2].astype(int)
    return (R >= RED_MIN) & (G <= GB_MAX) & (B <= GB_MAX)


def boxes_from_mask(mask):
    """연결요소별 박스 리스트 [(x,y,w,h)]; 너무 많으면 전체 1박스."""
    m = ndimage.binary_dilation(mask, iterations=DILATE_ITERS) if DILATE_ITERS else mask
    lbl, n = ndimage.label(m)
    out = []
    for c in range(1, n + 1):
        ys, xs = np.where(lbl == c)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w, h = x1 - x0 + 1, y1 - y0 + 1
        if w >= MIN_BOX_PX and h >= MIN_BOX_PX:
            out.append((int(x0), int(y0), int(w), int(h)))
    if len(out) > MAX_BOXES:                     # scattered → 전체 하나로
        ys, xs = np.where(mask)
        out = [(int(xs.min()), int(ys.min()),
                int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))]
    return out


def collect_rows(png_dir: str, labels_csv: str, class_name: str, preview: bool):
    """PNG 들을 훑어 어노테이션 행을 만든다. (rows, empties, preview_saved)"""
    df = pd.read_csv(labels_csv)
    present = {os.path.basename(r["file"]): [c for c in CLASS_ORDER if int(r[c]) == 1]
               for _, r in df.iterrows()}

    rows, empties, preview_saved = [], [], 0
    if preview:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    for png in sorted(glob.glob(os.path.join(png_dir, "*.png"))):
        fn = os.path.basename(png)
        classes = present.get(fn)
        if classes is None or len(classes) == 0:
            continue                      # CSV 에 없거나 정상(결함 0) → 박스 없음

        mask = red_mask(png)
        if not mask.any():
            empties.append(fn); continue
        bxs = boxes_from_mask(mask)
        if not bxs:
            empties.append(fn); continue

        for (x, y, w, h) in bxs:
            rows.append({"filename": fn, "type": "bbox",
                         "geometry": {"type": "bbox", "x": float(x), "y": float(y),
                                      "w": float(w), "h": float(h)},
                         "_class_name": class_name})

        if preview and preview_saved < PREVIEW_N:
            im = Image.open(png).convert("RGB")
            d = ImageDraw.Draw(im)
            for (x, y, w, h) in bxs:
                d.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=1)
            im.save(os.path.join(PREVIEW_DIR, fn))
            preview_saved += 1

    return rows, empties, preview_saved


def upload(rows, class_name: str, replace: bool) -> int:
    """superb_client 의 전송 계층으로 업로드 (SDK 없으면 REST 폴백)."""
    cls = sc.list_classes()
    if not cls.get("ok"):
        raise SystemExit(f"클래스 조회 실패: {cls.get('reason')}")
    class_id = cls["object"].get(class_name)
    if not class_id:
        raise SystemExit(f"'{class_name}' 객체 클래스가 프로젝트에 없습니다. "
                         f"있는 것: {sorted(cls['object'])}")

    for r in rows:
        r.pop("_class_name", None)
        r["class_id"] = class_id
        # 재실행해도 같은 id → 중복 생성 방지
        r["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, r["filename"] + str(r["geometry"])))

    created = 0
    for i in range(0, len(rows), BATCH):        # 서버 상한(1000)씩 끊어 보낸다
        chunk = rows[i:i + BATCH]
        ok, payload = sc._api().request(
            "POST", f"/projects/{sc.PROJECT_ID}/annotations/batch-create",
            json={"annotations": chunk, "source": "imported", "replace": replace},
            timeout=120.0)
        if not ok:
            raise SystemExit(f"업로드 실패({i}~{i + len(chunk)}): {payload}")
        created += payload.get("created", len(chunk))
        print(f"  {i + len(chunk)}/{len(rows)} 전송")
    return created


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=PNG_DIR, help="wafer_*.png 폴더")
    ap.add_argument("--csv", default=None, help="8-hot 라벨 CSV (기본: <dir>/labels_ref.csv)")
    ap.add_argument("--class-name", default=DEFECT_CLASS, help="객체 검출 클래스명")
    ap.add_argument("--upload", action="store_true", help="실제 업로드 (없으면 PREVIEW)")
    ap.add_argument("--replace", action="store_true", help="대상 에셋의 기존 어노테이션 삭제 후 재삽입")
    a = ap.parse_args()

    labels_csv = a.csv or os.path.join(a.dir, "labels_ref.csv")
    if not os.path.exists(labels_csv):
        raise SystemExit(f"라벨 CSV 가 없습니다: {labels_csv}")

    rows, empties, saved = collect_rows(a.dir, labels_csv, a.class_name, not a.upload)
    print(f"이미지 처리 완료 | 생성 박스: {len(rows)} | 빨간픽셀 못찾음: {len(empties)}")
    if empties[:8]:
        print("  확인 필요(임계값?):", empties[:8])

    if not a.upload:
        print(f"\n[PREVIEW] overlay {saved}장을 {PREVIEW_DIR}/ 에 저장했습니다.")
        print("박스가 결함을 잘 감싸면 --upload 를 붙여 다시 실행하세요.")
        return 0

    if not sc.available():
        raise SystemExit("Superb 연결 불가 — .env 의 SUPERB_AI_API_KEY 를 확인하세요.")
    if not sc.PROJECT_ID:
        raise SystemExit("SUPERB_PROJECT_ID 가 비어 있습니다.")
    print(f"\n업로드 대상: tenant={sc.TENANT} project={sc.PROJECT_ID} "
          f"(transport={sc.transport()}, replace={a.replace})")

    created = upload(rows, a.class_name, a.replace)
    print(f"업로드 완료: {created}건")

    ok, payload = sc._api().request(
        "GET", f"/projects/{sc.PROJECT_ID}/annotations",
        params={"limit": 1, "include_total": True}, timeout=30.0)
    if ok:
        print("프로젝트 총 어노테이션:", payload.get("total"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
