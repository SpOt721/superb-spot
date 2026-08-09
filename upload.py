"""
MixedWM38 -> Superb SPOT  '결함 위치' 바운딩 박스 자동 생성 & 업로드
====================================================================
전략(사용자 확정): 분류(이미지 8-hot)가 정답의 중심, 박스는 "결함 위치" 표시용.
  - 박스: PNG의 빨간(불량) 픽셀 -> 연결요소별 바운딩 박스 -> 'Defect' 1클래스로 업로드.
          (어느 픽셀이 어느 패턴인지 정답이 없으므로 8클래스로 나누지 않음 = 정직)
  - 8-hot: labels_ref.csv 그대로 분류기(Swin) 학습에 사용. (Superb 분류 업로드는 선택)

준비물:
  pip install superb-ai pillow numpy scipy pandas
스키마:
  객체 검출 탭에 'Defect' 클래스 1개 (어노테이션 타입 = 바운딩 박스) 만들어 두기.

먼저 PREVIEW=True로 돌려 overlay 몇 장 확인 -> 임계값 OK면 PREVIEW=False로 실제 업로드.
"""

import os, glob, uuid
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage
from superb_ai import Client

# ============================ 설정 ============================
TENANT     = os.environ.get("SUPERB_AI_TENANT", "spot")  # superb.ai 계정명(tenant)
API_KEY    = os.environ.get("SUPERB_AI_API_KEY", "sbd_pk_...")   # 설정 > API 키에서 발급
PROJECT_ID = "3ae6b512-4b15-4d0a-aae0-60ab67a1XXXX"             # poc 프로젝트 UUID (URL 뒷부분 전체)

PNG_DIR        = "./superb_png"              # wafer_*.png 폴더 (Colab의 superb_upload 압축 푼 곳)
LABELS_CSV     = "./superb_png/labels_ref.csv"   # file + 8-hot
DEFECT_CLASS   = "Defect"             # 객체 검출 스키마의 위치표시 클래스명

# 빨간(불량) 픽셀 판정 (R 높고 G/B 낮음) — PREVIEW로 확인 후 조정
RED_MIN, GB_MAX = 120, 120
MIN_BOX_PX      = 3       # 이보다 작은 박스는 노이즈로 버림
DILATE_ITERS    = 1       # 인접 결함을 살짝 합쳐 박스 파편화 방지
MAX_BOXES       = 8       # 한 이미지 박스가 이보다 많으면(=Random/Near_Full) 전체 1박스로 대체

PREVIEW      = True       # True: 업로드 안 하고 overlay 미리보기 저장 / False: 실제 업로드
PREVIEW_N    = 12
PREVIEW_DIR  = "./_preview"
REPLACE      = False      # True면 대상 에셋의 기존 어노테이션 삭제 후 재삽입(재실행 시 중복 방지)
# =============================================================

CLASS_ORDER = ["Center","Donut","Edge_Loc","Edge_Ring","Loc","Near_Full","Scratch","Random"]


def red_mask(png_path):
    im = np.asarray(Image.open(png_path).convert("RGB"))
    R, G, B = im[...,0].astype(int), im[...,1].astype(int), im[...,2].astype(int)
    return (R >= RED_MIN) & (G <= GB_MAX) & (B <= GB_MAX)


def boxes_from_mask(mask):
    """연결요소별 박스 리스트 [(x,y,w,h)]; 너무 많으면 전체 1박스."""
    m = ndimage.binary_dilation(mask, iterations=DILATE_ITERS) if DILATE_ITERS else mask
    lbl, n = ndimage.label(m)
    out = []
    for c in range(1, n+1):
        ys, xs = np.where(lbl == c)
        x0,x1,y0,y1 = xs.min(), xs.max(), ys.min(), ys.max()
        w,h = x1-x0+1, y1-y0+1
        if w >= MIN_BOX_PX and h >= MIN_BOX_PX:
            out.append((int(x0),int(y0),int(w),int(h)))
    if len(out) > MAX_BOXES:                     # scattered -> 전체 하나로
        ys, xs = np.where(mask)
        out = [(int(xs.min()),int(ys.min()),int(xs.max()-xs.min()+1),int(ys.max()-ys.min()+1))]
    return out


def main():
    df = pd.read_csv(LABELS_CSV)
    present = {os.path.basename(r["file"]): [c for c in CLASS_ORDER if int(r[c])==1]
              for _, r in df.iterrows()}

    rows, empties, preview_saved = [], [], 0
    if PREVIEW:
        os.makedirs(PREVIEW_DIR, exist_ok=True)

    for png in sorted(glob.glob(os.path.join(PNG_DIR, "*.png"))):
        fn = os.path.basename(png)
        classes = present.get(fn)
        if classes is None:
            continue                      # CSV에 없는 파일 skip
        if len(classes) == 0:
            continue                      # 정상(결함0) -> 박스 없음

        mask = red_mask(png)
        if not mask.any():
            empties.append(fn); continue
        bxs = boxes_from_mask(mask)
        if not bxs:
            empties.append(fn); continue

        for (x,y,w,h) in bxs:
            rows.append({"filename": fn,
                         "type": "bbox",
                         "geometry": {"type":"bbox","x":float(x),"y":float(y),"w":float(w),"h":float(h)},
                         "_class_name": DEFECT_CLASS})

        if PREVIEW and preview_saved < PREVIEW_N:
            im = Image.open(png).convert("RGB"); d = ImageDraw.Draw(im)
            for (x,y,w,h) in bxs:
                d.rectangle([x,y,x+w,y+h], outline=(0,255,0), width=1)
            im.save(os.path.join(PREVIEW_DIR, fn))
            preview_saved += 1

    print(f"이미지 처리 완료 | 생성 박스: {len(rows)} | 빨간픽셀 못찾음: {len(empties)}")
    if empties[:8]:
        print("  확인 필요(임계값?):", empties[:8])

    if PREVIEW:
        print(f"\n[PREVIEW] overlay {preview_saved}장을 {PREVIEW_DIR}/ 에 저장했어요.")
        print("박스가 결함을 잘 감싸면 PREVIEW=False 로 바꿔 다시 실행하세요.")
        return

    # ---- 실제 업로드 ----
    client = Client(tenant=TENANT, api_key=API_KEY)
    class_id = {c.name: c.id for c in client.classes.list(PROJECT_ID).classes}
    if DEFECT_CLASS not in class_id:
        raise SystemExit(f"'{DEFECT_CLASS}' 클래스가 프로젝트에 없어요. 객체검출 스키마에 만들어 주세요.")
    for r in rows:
        r["class_id"] = class_id[r.pop("_class_name")]
        r["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, r["filename"]+str(r["geometry"])))  # 재실행 대비 결정적 id

    if len(rows) <= 1000:
        res = client.annotations.project_batch_create(PROJECT_ID, annotations=rows,
                                                      source="imported", replace=REPLACE)
        print("동기 업로드 완료:", res)
    else:
        job = client.annotations.import_ndjson(PROJECT_ID, rows, source="imported", replace=REPLACE)
        job.wait(timeout=900)
        print("비동기 업로드 완료")

    total = client.annotations.list(PROJECT_ID, limit=1, include_total=True).total
    print("프로젝트 총 어노테이션:", total)


if __name__ == "__main__":
    main()