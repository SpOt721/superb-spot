"""
Superb AI 프로젝트 어노테이션 전수 QA (Superb Platform REST API)

실행:
    python3 spot_qa.py

인증/대상은 .env 에서 읽습니다 (config.py 가 import 시점에 os.environ 으로 주입).
    SUPERB_AI_API_KEY=sbd_pk_...
    SUPERB_AI_TENANT=spot
    SUPERB_PROJECT_ID=...

검사 항목
    1. 클래스명    — bbox 는 class.name, classification 은 data.answer 값을 정규 클래스와 대조
    2. BBox 크기   — w/h 누락 또는 0 이하
    3. BBox 범위   — 이미지 경계를 벗어난 좌표
"""
import sys

import config  # noqa: F401  — import 만으로 .env 를 os.environ 에 주입
import superb_client as sc

# =========================================================
# 1. 설정
# =========================================================
PAGE_SIZE = 200          # /annotations 엔드포인트 상한
BOUNDS_TOLERANCE = 1.0   # 부동소수 오차 허용치(px)

# WM-811K 8개 정규 클래스
VALID_CLASSES = {
    'Center', 'Donut', 'Edge_Loc', 'Edge_Ring',
    'Loc', 'Near_Full', 'Random', 'Scratch'
}


def iter_annotations():
    """커서 페이징으로 프로젝트의 모든 어노테이션을 순회한다.

    응답은 {'items': [...], 'next_cursor': str|None, 'total': None} 형태이고
    total 이 항상 None 이라 커서가 빌 때까지 도는 것 말고는 종료 조건이 없다.
    """
    cursor = None
    while True:
        params = {"include": "asset,class", "limit": PAGE_SIZE}
        if cursor:
            params["cursor"] = cursor
        ok, payload = sc._api().request(
            "GET", f"/projects/{sc.PROJECT_ID}/annotations", params=params, timeout=60.0
        )
        if not ok:
            raise RuntimeError(payload)
        items = payload.get("items") or []
        if not items:
            return
        yield from items
        cursor = payload.get("next_cursor")
        if not cursor:
            return


def check_class(ann):
    """클래스명 오류 사유를 반환(정상이면 None).

    bbox 는 class.name 자체가 결함 클래스지만, classification 은 class.name 이
    스키마 이름(defect_pattern_v2)이고 실제 값이 data.answer 에 들어 있다.
    """
    ann_type = ann.get("type")
    class_name = (ann.get("class") or {}).get("name")

    if ann_type == "classification":
        data = ann.get("data") or {}
        answers = data.get("answer") or data.get("value") or []
        if not answers:
            return "분류 값 비어 있음"
        bad = [a for a in answers if a not in VALID_CLASSES]
        return f"{class_name} → {', '.join(bad)}" if bad else None

    if class_name not in VALID_CLASSES:
        return class_name if class_name else "(클래스명 없음)"
    return None


def check_geometry(ann):
    """(크기 오류, 범위 오류) 사유를 반환. bbox 가 아니면 (None, None)."""
    if ann.get("type") != "bbox":
        return None, None

    g = ann.get("geometry")
    if not isinstance(g, dict):
        return "geometry 누락", None

    x, y, w, h = g.get("x"), g.get("y"), g.get("w"), g.get("h")
    if w is None or h is None:
        return "w/h 누락", None
    if w <= 0 or h <= 0:
        return f"w={w:.1f}, h={h:.1f}", None

    # 이미지 경계 검사 — asset.properties 에 원본 크기가 실려 온다
    props = (ann.get("asset") or {}).get("properties") or {}
    iw, ih = props.get("width"), props.get("height")
    if iw and ih and x is not None and y is not None:
        t = BOUNDS_TOLERANCE
        if x < -t or y < -t or x + w > iw + t or y + h > ih + t:
            return None, f"({x:.1f}, {y:.1f}, {w:.1f}×{h:.1f}) vs {iw}×{ih}"
    return None, None


def report(title, errors):
    if not errors:
        return
    print("-" * 60)
    print(f"[{title}] 상위 {min(10, len(errors))}건")
    for filename, detail in errors[:10]:
        print(f"  - {filename}: {detail}")
    if len(errors) > 10:
        print(f"  … 외 {len(errors) - 10}건")


def main():
    print("=" * 60)
    print("🚀 Superb AI 어노테이션 전수 QA 시작")
    print("=" * 60)

    if not sc.available():
        print("[중단] API 키 또는 전송수단 없음. .env 의 SUPERB_AI_API_KEY 를 확인하세요.")
        return 1
    if not sc.PROJECT_ID:
        print("[중단] SUPERB_PROJECT_ID 가 비어 있습니다.")
        return 1

    print(f"• 테넌트: {sc.TENANT} / 프로젝트: {sc.PROJECT_ID}")
    print(f"• 어노테이션 수집 중... ({PAGE_SIZE}건씩 페이징)")

    assets = set()
    type_counts = {}
    class_errors, size_errors, bounds_errors = [], [], []
    total = 0

    try:
        for ann in iter_annotations():
            total += 1
            type_counts[ann.get("type")] = type_counts.get(ann.get("type"), 0) + 1

            asset = ann.get("asset") or {}
            filename = asset.get("filename") or str(ann.get("asset_id"))
            assets.add(ann.get("asset_id"))

            if (err := check_class(ann)):
                class_errors.append((filename, err))
            size_err, bounds_err = check_geometry(ann)
            if size_err:
                size_errors.append((filename, size_err))
            if bounds_err:
                bounds_errors.append((filename, bounds_err))

            if total % 1000 == 0:
                print(f"  … {total}건 처리")
    except Exception as e:
        print(f"[중단] 수집 중 오류: {type(e).__name__}: {e}")
        print(f"  (중단 시점까지 {total}건 처리 — 아래 집계는 부분 결과입니다)")
        return 1

    # 결과 리포트
    breakdown = ", ".join(f"{k} {v}" for k, v in sorted(type_counts.items(), key=lambda kv: -kv[1]))
    print("\n" + "=" * 60)
    print("📊 어노테이션 Data QA 최종 결과")
    print("=" * 60)
    print(f"• 검수 대상 에셋 수         : {len(assets)}개")
    print(f"• 총 어노테이션 수           : {total}개 ({breakdown})")
    print("-" * 60)
    print(f"1. 잘못된 클래스명(오타/미지정) : {len(class_errors)}건")
    print(f"2. 크기가 0 이하인 찌그러진 BBox : {len(size_errors)}건")
    print(f"3. 이미지 경계를 벗어난 BBox     : {len(bounds_errors)}건")

    report("클래스명 오류", class_errors)
    report("BBox 크기 오류", size_errors)
    report("BBox 범위 오류", bounds_errors)
    print("-" * 60)

    if total == 0:
        print("⚠️ [최종 판정] 어노테이션이 0건입니다. 프로젝트 ID를 확인해 주세요.")
        return 1

    total_errors = len(class_errors) + len(size_errors) + len(bounds_errors)
    if total_errors == 0:
        print("✅ [최종 판정] PASS: 라벨 어노테이션 무결성 100% 정상 확인!")
        return 0
    print(f"⚠️ [최종 판정] FAIL: 총 {total_errors}건의 라벨 오류 발견")
    return 1


if __name__ == "__main__":
    sys.exit(main())
