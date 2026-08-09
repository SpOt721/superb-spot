"""
파이프라인 자체 점검 — Streamlit 없이 터미널에서 전부 확인.

  python3 selftest.py                    # 모델(분류/탐지) + 리포트 + PDF, samples/ 기준 정확도
  python3 selftest.py --superb           # 위 + Superb 연동(.env 키로 실제 API 호출)
  python3 selftest.py --superb-only      # Superb 연동만
  python3 selftest.py --dir samples --det auto --thr 0.5 --conf 0.25
"""
from __future__ import annotations
import argparse
import csv
import logging
import os
import sys
import warnings

warnings.filterwarnings("ignore")   # urllib3/LibreSSL 등 실행에 무관한 경고

from PIL import Image               # noqa: E402

import config                       # noqa: E402  (.env 로드)
import models                       # noqa: E402
import graph_rag as gr              # noqa: E402

OK, NO = "✅", "❌"


def _silence_streamlit():
    """
    app.py 를 import 하면 streamlit 이 bare mode 경고를 쏟아낸다.
    streamlit 은 import 시점에 자기 로거들의 레벨을 직접 세팅하므로,
    import '이후에' 등록된 streamlit* 로거를 전부 눌러야 조용해진다.
    """
    for name in list(logging.root.manager.loggerDict):
        if name == "streamlit" or name.startswith("streamlit."):
            logging.getLogger(name).setLevel(logging.CRITICAL)
    logging.getLogger("streamlit").setLevel(logging.CRITICAL)


def _labels_csv(d: str) -> dict:
    p = os.path.join(d, "labels.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["file"]] = [l for l in models.LABELS if row.get(l) == "1"]
    return out


def test_models(d: str, det_src: str, thr: float, conf: float) -> int:
    print("=" * 74)
    print("모델 로딩")
    print(f"  분류 Swin : {OK if models.cls_available() else NO} {models.CLS_PATH}"
          + (f"  ({models.cls_error()})" if not models.cls_available() else ""))
    print(f"  탐지 YOLO : {OK if models.yolo_available() else NO} {models.DET_PATH}"
          + (f"  ({models.yolo_error()})" if not models.yolo_available() else ""))
    print(f"  클래스    : {models.LABELS}")

    if not os.path.isdir(d):
        print(f"\n{NO} '{d}' 폴더가 없습니다.  python3 make_samples.py  로 먼저 생성하세요.")
        return 1
    files = sorted(f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
    if not files:
        print(f"\n{NO} '{d}' 에 이미지가 없습니다.")
        return 1
    truth = _labels_csv(d)

    print("=" * 74)
    print(f"추론 — {len(files)}장 (분류 임계값 {thr}, 탐지 {det_src}/{conf})\n")
    exact = graded = 0
    for f in files:
        img = Image.open(os.path.join(d, f)).convert("RGB")
        preds, scores, csrc = models.classify(img, thr=thr)
        boxes, dsrc = models.detect(img, source=det_src, conf=conf)
        gt = truth.get(f)
        mark = ""
        if gt is not None:
            graded += 1
            hit = set(preds) == set(gt)
            exact += hit
            mark = f"  {OK if hit else NO} 정답={gt or ['Normal']}"
        top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
        print(f"  {f[:38]:38s} 예측={preds or ['Normal']}{mark}")
        print(f"  {'':38s} 상위={[(k, round(v, 2)) for k, v in top]} · 박스 {len(boxes)}개 ({dsrc})")
    if graded:
        print(f"\n  완전일치(exact match): {exact}/{graded} = {exact / graded:.0%}")
        src_file = os.path.join(d, "SOURCE.txt")
        src = open(src_file, encoding="utf-8").read().strip() if os.path.exists(src_file) else "?"
        if src == "synthetic":
            print("  ※ 합성 샘플 기준입니다. 모델 성능 판단은 MixedWM38 실데이터로 하세요"
                  "  (make_samples.py --npz MixedWM38.npz)")

    # 리포트 + PDF 왕복
    print("=" * 74)
    img = Image.open(os.path.join(d, files[0])).convert("RGB")
    preds, scores, _ = models.classify(img, thr=thr)
    boxes, dsrc = models.detect(img, source=det_src, conf=conf)
    c = max([scores[p] for p in preds], default=None)
    proc = gr.primary_process(preds) if preds else "정상"
    rep = gr.report(preds, confidence=c, use_llm=False)
    print(f"Graph-RAG  판단 공정: {proc}")
    print("  " + rep.replace("\n", "\n  ")[:400])
    try:
        import streamlit                     # noqa: F401  로거를 먼저 등록시킨 뒤
        _silence_streamlit()                 # 눌러야 app.py 의 st.* 경고까지 조용해진다
        import app
        _silence_streamlit()
        pdf = app.build_pdf(img, boxes, rep, proc, preds, scores, dsrc)
        print(f"\nPDF 생성   : {OK} {len(pdf):,} bytes")
    except Exception as e:
        print(f"\nPDF 생성   : {NO} {type(e).__name__}: {e}")
        return 1
    return 0


def test_superb(d: str) -> int:
    import superb_client as sb
    print("=" * 74)
    env = config.env_status()
    print(f"환경설정   : .env {'있음' if env['exists'] else '없음'} ({env['path']})")
    print(f"             .env 에서 로드: {env['keys'] or '(없음 — 셸 환경변수 사용)'}")
    if env.get("secrets"):
        print(f"             Streamlit Secrets: {env['secrets']}")
    print(f"             tenant={sb.TENANT}  project={sb.PROJECT_ID[:8]}…  dataset={sb.DATASET_ID[:8]}…")
    print(f"             API키={'설정됨(' + sb.API_KEY[:10] + '…)' if sb.API_KEY else '없음'}")

    s = sb.status()
    print(f"연결       : {OK if s.get('connected') else NO} transport={s.get('transport')}")
    if not s.get("connected"):
        print(f"             {s.get('reason')}")
        return 1

    info = sb.deployment_info()
    if not info.get("ok"):
        print(f"배포모델   : {NO} {info.get('reason')}")
        print("             → 배포 ID를 .env 의 SUPERB_DEPLOYMENT_ID 에 넣거나, 콘솔에서 모델을 배포하세요.")
        return 1
    print(f"배포모델   : {OK} {info['name'] or info['id']} · status={info['status']} · task={info['task']}")
    print(f"             선택경로={info['via']}  추천임계값={info['recommended_conf']}")
    print(f"             클래스맵={info['class_map']}")

    files = sorted(f for f in os.listdir(d) if f.lower().endswith(".png")) if os.path.isdir(d) else []
    if not files:
        print(f"{NO} '{d}' 에 테스트 이미지가 없습니다. make_samples.py 를 먼저 실행하세요.")
        return 1
    path = os.path.join(d, files[0])
    img = Image.open(path).convert("RGB")

    canon = models.render_canonical(img, models.DET_SIZE)
    r = sb.detect_remote(canon, conf=info["recommended_conf"])
    if r.get("ok"):
        print(f"추론       : {OK} {len(r['boxes'])}개 · {r['inference_ms']}ms · 입력 {r['size']}")
        for b in r["boxes"][:5]:
            print(f"             {b[0]} conf={b[5]:.2f} box=({b[1]:.0f},{b[2]:.0f})-({b[3]:.0f},{b[4]:.0f})")
    else:
        print(f"추론       : {NO} {r.get('reason')}")

    # 점검용 업로드는 원본과 구분되게 selftest_ 접두어로 (콘솔에서 찾아 지우기 쉽게)
    import shutil
    import tempfile
    tmp = os.path.join(tempfile.gettempdir(), f"selftest_{files[0]}")
    shutil.copyfile(path, tmp)
    up = sb.upload_image(tmp, key=os.path.basename(tmp))
    print(f"업로드     : {OK if up.get('ok') else NO} {up}")
    print("             ※ 점검용 자산이 프로젝트에 1개 생성됩니다 (파일명 selftest_*)")
    aid = up.get("asset_id")
    if aid:
        preds, _, _ = models.classify(img, thr=0.5)
        pr = sb.push_prediction(aid, preds or ["Center"])
        print(f"오토라벨   : {OK if pr.get('ok') else NO} {pr}")
        fb = sb.push_feedback(aid, ["Scratch"])
        print(f"피드백F-05 : {OK if fb.get('ok') else NO} {fb}")
    else:
        print("오토라벨   : ⏳ 자산 생성 대기 — 잠시 후 다시 실행하면 적재됩니다.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="samples")
    ap.add_argument("--det", default="auto", choices=["auto", "superb", "yolo", "heuristic"])
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--superb", action="store_true", help="Superb 연동까지 점검")
    ap.add_argument("--superb-only", action="store_true")
    a = ap.parse_args()

    rc = 0
    if not a.superb_only:
        rc |= test_models(a.dir, a.det, a.thr, a.conf)
    if a.superb or a.superb_only:
        rc |= test_superb(a.dir)
    print("=" * 74)
    print("완료" if rc == 0 else "일부 항목 실패 — 위 ❌ 메시지를 확인하세요.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
