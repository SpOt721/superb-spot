"""
테스트용 웨이퍼 맵 생성기.

  python3 make_samples.py                 # 합성 샘플 생성 (기본 samples/)
  python3 make_samples.py --npz MixedWM38.npz --n 24
        # 진짜 MixedWM38 이 있으면 거기서 뽑아 정답 라벨까지 그대로 저장 (권장)

출력: samples/*.png  +  samples/labels.csv (file, 정답 라벨 8-hot)
이미지는 학습과 동일한 팔레트로 렌더 — 정상 #94a3b8, 불량 #ef4444, 빈칸 검정.
"""
from __future__ import annotations
import argparse
import csv
import os

import numpy as np
from PIL import Image

LABELS = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_Full", "Scratch", "Random"]
G = 52                      # MixedWM38 격자
OUT_SIZE = 256              # 저장 해상도


def to_png(m: np.ndarray, size: int = OUT_SIZE) -> Image.Image:
    """0/1/2 마스크 → 학습 팔레트 RGB."""
    rgb = np.zeros((*m.shape, 3), np.uint8)
    rgb[m == 1] = (148, 163, 184)
    rgb[m == 2] = (239, 68, 68)
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)


# ---------------- 합성 (npz 없을 때) ----------------
def _base(rng):
    """웨이퍼 원판 + 옅은 배경 노이즈(실데이터에 늘 있는 산발 결함)."""
    yy, xx = np.mgrid[0:G, 0:G]
    rr = np.sqrt((yy - G / 2 + 0.5) ** 2 + (xx - G / 2 + 0.5) ** 2)
    m = np.zeros((G, G), int)
    wafer = rr <= G / 2 - 0.5
    m[wafer] = 1
    noise = wafer & (rng.random((G, G)) < 0.012)
    m[noise] = 2
    return m, rr, wafer


def synth(kind: str, rng):
    m, rr, wafer = _base(rng)
    yy, xx = np.mgrid[0:G, 0:G]

    def put(sel):
        m[wafer & sel] = 2

    if kind == "Center":
        put(rr < rng.uniform(5, 9))
    elif kind == "Donut":
        r0 = rng.uniform(8, 11)
        put((rr > r0) & (rr < r0 + rng.uniform(3, 5)))
    elif kind == "Edge_Ring":
        put(rr > G / 2 - rng.uniform(3.5, 5.5))
    elif kind == "Edge_Loc":
        th = np.arctan2(yy - G / 2, xx - G / 2)
        a = rng.uniform(-np.pi, np.pi)
        put((rr > G / 2 - 6) & (np.abs(np.angle(np.exp(1j * (th - a)))) < 0.5))
    elif kind == "Loc":
        # 중심부(Center)나 가장자리(Edge_*)와 헷갈리지 않게 중간 반경에 배치
        a = rng.uniform(0, 2 * np.pi)
        d = rng.uniform(10, 16)
        cy, cx = G / 2 + d * np.sin(a), G / 2 + d * np.cos(a)
        put(np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) < rng.uniform(4, 6.5))
    elif kind == "Near_Full":
        # 실데이터의 Near_Full 은 '거의 전면이 통짜로' 불량 (산발 노이즈가 아님)
        holes = rng.random((G, G)) < 0.06
        put((rr < G / 2 - rng.uniform(0.5, 2.0)) & ~holes)
    elif kind == "Scratch":
        a = rng.uniform(0, np.pi)
        c = rng.uniform(-6, 6)
        d = np.abs((xx - G / 2) * np.sin(a) - (yy - G / 2) * np.cos(a) - c)
        put(d < rng.uniform(0.8, 1.6))
    elif kind == "Random":
        put(rng.random((G, G)) < rng.uniform(0.18, 0.28))
    elif kind == "Normal":
        pass
    return m


# 단일 8종 + 혼합 3종 + 정상 1종 = 12장 (혼합이 이 데모의 핵심)
PLAN = [(l, [l]) for l in LABELS] + [
    ("mix_Center+Edge_Ring", ["Center", "Edge_Ring"]),
    ("mix_Loc+Scratch", ["Loc", "Scratch"]),
    ("mix_Donut+Edge_Loc", ["Donut", "Edge_Loc"]),
    ("Normal", []),
]


def build_synthetic(outdir: str, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = []
    for name, parts in PLAN:
        m = synth(parts[0] if parts else "Normal", rng)
        for extra in parts[1:]:                       # 혼합: 패턴을 겹쳐 그림
            m2 = synth(extra, rng)
            m[m2 == 2] = 2
        f = f"{name}.png"
        to_png(m).save(os.path.join(outdir, f))
        rows.append([f] + [1 if l in parts else 0 for l in LABELS])
    return rows


# ---------------- 진짜 MixedWM38 에서 추출 ----------------
def build_from_npz(npz: str, outdir: str, n: int, seed: int = 7):
    data = np.load(npz)
    X, Y = data["arr_0"], data["arr_1"].astype(int)
    rng = np.random.default_rng(seed)
    single = np.where(Y.sum(1) == 1)[0]
    mixed = np.where(Y.sum(1) >= 2)[0]
    normal = np.where(Y.sum(1) == 0)[0]
    pick = np.concatenate([
        rng.choice(single, min(len(single), n // 2), replace=False),
        rng.choice(mixed, min(len(mixed), n - n // 2 - 2), replace=False),
        rng.choice(normal, min(len(normal), 2), replace=False)])
    rows = []
    for i in pick:
        tags = [LABELS[k] for k in range(8) if Y[i, k] == 1] or ["Normal"]
        f = f"wafer_{i}_{'+'.join(tags)}.png"
        to_png(np.asarray(X[i]).astype(int)).save(os.path.join(outdir, f))
        rows.append([f] + [int(Y[i, k]) for k in range(8)])
    return rows


# ---------------- Superb 데이터셋에서 실데이터 내려받기 ----------------
def build_from_superb(outdir: str, n: int, pages: int = 12):
    """
    이미 Superb에 올려둔 MixedWM38 이미지 + 기존 어노테이션(정답)을 그대로 사용.
    데이터셋 순서가 패턴별로 뭉쳐 있어서, '클래스별 어노테이션'을 역으로 조회해 균등 샘플링한다.
    """
    import superb_client as sb
    cls = sb.list_classes()
    if not cls.get("ok"):
        raise SystemExit(f"클래스 조회 실패: {cls.get('reason')}")
    obj = {k: v for k, v in cls["object"].items() if k in LABELS}
    if not obj:
        raise SystemExit("프로젝트에 8종 객체 클래스가 없습니다.")

    per = max(1, n // max(len(obj), 1)) + 2
    gt, fname = {}, {}
    for name, cid in sorted(obj.items()):
        r = sb.annotations_of_class(cid, limit=per * 3)
        if not r.get("ok"):
            print(f"  {name}: 조회 실패 {r.get('reason')}")
            continue
        picked = 0
        for it in r["items"]:
            aid = it["asset_id"]
            gt.setdefault(aid, set()).add(it["class_name"])
            fname[aid] = it["filename"] or f"{aid}.png"
            picked += 1
            if picked >= per:
                break
    if not gt:
        raise SystemExit("어노테이션이 없습니다. upload.py 로 먼저 박스를 올리세요.")

    # 라벨 조합별로 돌아가며 골라 다양성 확보
    buckets = {}
    for aid, tags in gt.items():
        buckets.setdefault("+".join(sorted(tags)), []).append(aid)
    chosen, keys = [], sorted(buckets)
    while len(chosen) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k] and len(chosen) < n:
                chosen.append(buckets[k].pop(0))
    print(f"  패턴 분포: { {k: len(v) for k, v in sorted(buckets.items())} } → {len(chosen)}장 선택")

    rows = []
    for aid in chosen:
        f = fname[aid]
        dl = sb.download_asset(aid, os.path.join(outdir, f))
        if not dl.get("ok"):
            print(f"  건너뜀 {f}: {dl.get('reason')}")
            continue
        tags = [t for t in gt[aid] if t in LABELS]
        rows.append([f] + [1 if l in tags else 0 for l in LABELS])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="MixedWM38.npz", help="MixedWM38.npz 경로 (있으면 사용)")
    ap.add_argument("--from-superb", action="store_true",
                    help="Superb 데이터셋에서 실이미지+정답을 내려받아 사용 (권장)")
    ap.add_argument("--pages", type=int, default=12, help="--from-superb 시 훑을 자산 페이지 수(100장/페이지)")
    ap.add_argument("--out", default="samples")
    ap.add_argument("--n", type=int, default=24, help="뽑을 장수")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    if a.from_superb:
        rows = build_from_superb(a.out, a.n, a.pages)
        src = "Superb 데이터셋 실데이터 + 기존 어노테이션"
    elif os.path.exists(a.npz):
        rows = build_from_npz(a.npz, a.out, a.n, a.seed)
        src = f"MixedWM38 실데이터({a.npz})"
    else:
        rows = build_synthetic(a.out, a.seed)
        src = "합성(npz 없음 — 실데이터가 있으면 --npz 로 지정하세요)"

    with open(os.path.join(a.out, "labels.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file"] + LABELS)
        w.writerows(rows)
    kind = "superb" if a.from_superb else ("MixedWM38" if os.path.exists(a.npz) else "synthetic")
    with open(os.path.join(a.out, "SOURCE.txt"), "w", encoding="utf-8") as f:
        f.write(kind + "\n")
    print(f"{len(rows)}장 생성 → {a.out}/  ({src})")
    print(f"정답 라벨: {a.out}/labels.csv")
    if kind == "synthetic":
        print("주의: 합성 샘플은 실데이터 분포와 달라 정확도가 낮게 나올 수 있습니다.\n"
              "      학습에 쓴 MixedWM38.npz 를 이 폴더에 두고 다시 실행하면 실데이터로 평가합니다.")


if __name__ == "__main__":
    main()
