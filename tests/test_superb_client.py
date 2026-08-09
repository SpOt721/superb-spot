"""Superb 연동 단위테스트 — 전부 로컬 목 서버 대상 (실 API 호출 없음)."""
from __future__ import annotations
import time

import pytest
from PIL import Image

from conftest import CID, DEP, DS, PID


# ---------------- 오프라인 가드 ----------------
def test_키가_없으면_모든_호출이_조용히_실패(superb_offline):
    sb = superb_offline
    assert sb.available() is False
    assert sb.deploy_available() is False
    for r in (sb.detect_remote("x.png"), sb.upload_image("x.png"),
              sb.push_prediction("a", ["Center"]), sb.list_assets(), sb.list_classes()):
        assert r["ok"] is False
        assert r["reason"] == "no_api_key_or_transport"


def test_오프라인_status는_이유를_알려준다(superb_offline):
    st = superb_offline.status()
    assert st["connected"] is False
    assert "SUPERB_AI_API_KEY" in st["reason"]


# ---------------- 배포 조회 ----------------
def test_배포ID는_환경변수를_최우선으로_쓴다(superb):
    sb, state = superb
    r = sb.resolve_deployment()
    assert r["ok"] and r["deployment_id"] == DEP
    assert "env" in r["via"]
    assert not any(c[1] == "/tenants/testco/deployments" for c in state.calls)   # 목록조회 불필요


def test_배포ID가_없으면_실행중인_배포를_자동선택(superb, monkeypatch):
    sb, _ = superb
    monkeypatch.setattr(sb, "DEPLOY_ID", "")
    sb._dep_cache.clear()
    r = sb.resolve_deployment(force=True)
    assert r["ok"] and r["deployment_id"] == DEP
    assert "자동선택" in r["via"]


def test_모델ID로_배포를_찾는다(superb, monkeypatch):
    sb, _ = superb
    monkeypatch.setattr(sb, "DEPLOY_ID", "")
    monkeypatch.setattr(sb, "MODEL_ID", "model-abc")
    sb._dep_cache.clear()
    r = sb.resolve_deployment(force=True)
    assert r["ok"] and "model_id" in r["via"]


def test_모델ID가_안맞으면_사유를_돌려준다(superb, monkeypatch):
    sb, _ = superb
    monkeypatch.setattr(sb, "DEPLOY_ID", "")
    monkeypatch.setattr(sb, "MODEL_ID", "존재하지-않는-모델")
    sb._dep_cache.clear()
    r = sb.resolve_deployment(force=True)
    assert r["ok"] is False and "배포된 deployment 없음" in r["reason"]


def test_deployment_info가_클래스맵과_추천임계값을_읽는다(superb):
    sb, _ = superb
    info = sb.deployment_info()
    assert info["ok"] and info["status"] == "ready"
    assert info["recommended_conf"] == pytest.approx(0.2659)
    assert info["class_map"] == {0: "Center", 1: "Donut"}


# ---------------- 추론 ----------------
def test_predict는_image_b64로_보낸다(superb, wafer_img):
    sb, state = superb
    r = sb.detect_remote(wafer_img, conf=0.3)
    assert r["ok"]
    body = [c[2] for c in state.calls if c[0] == "POST" and c[1].endswith("/predict")][0]
    assert "image_b64" in body and body["confidence"] == 0.3
    assert "image_url" not in body


def test_predict_응답의_bbox와_polygon을_모두_파싱(superb, wafer_img):
    sb, _ = superb
    r = sb.detect_remote(wafer_img)
    assert r["size"] == (128, 64)
    assert r["inference_ms"] == 12
    assert ("Center", 10.0, 20.0, 40.0, 60.0, 0.9) in r["boxes"]      # x,y,w,h → x0,y0,x1,y1
    donut = [b for b in r["boxes"] if b[0] == "Donut"][0]
    assert donut[1:5] == (5.0, 5.0, 50.0, 60.0)                       # 폴리곤 → 외접 박스


def test_conf를_안주면_바디에_넣지_않는다_모델기본값_사용(superb, wafer_img):
    sb, state = superb
    sb.detect_remote(wafer_img, conf=None)
    body = [c[2] for c in state.calls if c[0] == "POST" and c[1].endswith("/predict")][0]
    assert "confidence" not in body


def test_워밍업중이면_재시도한다(superb, wafer_img, monkeypatch):
    sb, state = superb
    state.warmup_errors = 1
    monkeypatch.setattr(time, "sleep", lambda s: None)      # 대기 없이
    r = sb.detect_remote(wafer_img, max_wait=60)
    assert r["ok"] and state.predict_calls == 2


def test_워밍업이_안끝나면_사유와_함께_포기(superb, wafer_img, monkeypatch):
    sb, state = superb
    state.warmup_errors = 99
    monkeypatch.setattr(time, "sleep", lambda s: None)
    r = sb.detect_remote(wafer_img, max_wait=0)
    assert r["ok"] is False and "MODEL_LOADING" in r["reason"]


def test_경로_bytes_PIL_셋다_받는다(superb, wafer_img, tmp_path):
    sb, _ = superb
    p = tmp_path / "w.png"
    wafer_img.save(p)
    assert sb.detect_remote(str(p))["ok"]
    assert sb.detect_remote(p.read_bytes())["ok"]
    assert sb.detect_remote(wafer_img)["ok"]


# ---------------- 업로드 ----------------
def test_업로드는_init_PUT_조회_프로젝트편입_순서로_진행(superb, wafer_img, tmp_path):
    sb, state = superb
    p = tmp_path / "w.png"
    wafer_img.save(p)
    r = sb.upload_image(str(p), key="w.png")
    assert r["ok"] and r["asset_id"] == "asset-1" and r["pending"] is False
    assert r["in_project"] is True
    seq = [c[1] for c in state.calls]
    assert any("upload-init/batch" in s for s in seq)
    assert "/s3put" in seq
    assert any(s.endswith("/assets/batch-add") for s in seq)


def test_자산생성이_늦으면_pending으로_돌려준다(superb, wafer_img, tmp_path):
    sb, state = superb
    state.asset_ready = False                       # 서버가 아직 자산 행을 안 만든 상태
    p = tmp_path / "w.png"
    wafer_img.save(p)
    r = sb.upload_image(str(p), key="w.png", wait_ready=0)
    assert r["ok"] and r["asset_id"] is None and r["pending"] is True


def test_없는_파일은_업로드_실패(superb):
    sb, _ = superb
    r = sb.upload_image("/존재하지/않는/파일.png")
    assert r["ok"] is False


# ---------------- 어노테이션 ----------------
def test_data키를_자동탐색하고_성공한_키를_기억한다(superb):
    sb, state = superb
    r1 = sb.push_prediction("asset-1", ["Center"])
    assert r1["ok"] and r1["data_key"] == "answer"       # value 는 422 → answer 로 성공
    n1 = len([c for c in state.calls if c[1].endswith("/annotations/batch-create")])
    assert n1 == 2                                       # 첫 호출은 두 번 시도

    r2 = sb.push_prediction("asset-1", ["Donut"])
    n2 = len([c for c in state.calls if c[1].endswith("/annotations/batch-create")])
    assert r2["ok"] and n2 - n1 == 1                     # 두 번째부터는 한 번만


def test_오토라벨과_피드백의_source가_구분된다(superb):
    sb, state = superb
    sb.push_prediction("asset-1", ["Center"])
    sb.push_feedback("asset-1", ["Scratch"])
    sources = [s for (_, s, _) in state.annotations]
    assert sources == ["model", "manual"]


def test_어노테이션_payload_모양(superb):
    sb, state = superb
    sb.push_prediction("asset-1", ["Center", "Donut"])
    ann, source, replace = state.annotations[-1]
    assert ann["asset_id"] == "asset-1"
    assert ann["class_id"] == CID
    assert ann["type"] == "classification"
    assert ann["data"] == {"answer": ["Center", "Donut"]}
    assert replace is True


def test_라벨목록에_없는_값은_걸러낸다(superb):
    sb, state = superb
    sb.push_prediction("asset-1", ["Center", "존재하지않는패턴"])
    ann, _, _ = state.annotations[-1]
    assert ann["data"]["answer"] == ["Center"]


def test_유효한_라벨이_하나도_없으면_전송하지_않는다(superb):
    sb, state = superb
    before = len(state.calls)
    r = sb.push_prediction("asset-1", ["없는패턴"])
    assert r["ok"] is False and r["reason"] == "empty_prediction"
    assert len(state.calls) == before


def test_asset_id가_없으면_전송하지_않는다(superb):
    sb, _ = superb
    r = sb.push_prediction(None, ["Center"])
    assert r["ok"] is False and "asset_id" in r["reason"]


# ---------------- 테스트데이터 조회 ----------------
def test_list_assets_와_download(superb, tmp_path):
    sb, state = superb
    state.uploaded = True
    lst = sb.list_assets(limit=10)
    assert lst["ok"] and lst["items"][0]["filename"] == "w.png"
    dest = tmp_path / "out" / "w.png"
    dl = sb.download_asset("asset-1", str(dest))
    assert dl["ok"] and dest.read_bytes() == b"PNG!"
    # 프리사인드 URL 요청엔 Authorization 헤더가 붙으면 안 된다
    s3 = [c for c in state.calls if c[1] == "/s3get"][0]
    assert s3[2] is None


def test_list_classes는_object와_classification을_분리(superb):
    sb, _ = superb
    r = sb.list_classes()
    assert r["ok"]
    assert r["object"] == {"Center": "cls-center", "Donut": "cls-donut"}
    assert r["classification"]["defect_pattern_v2"] == CID


def test_annotations_of_class는_파일명과_클래스명을_붙여준다(superb):
    sb, _ = superb
    r = sb.annotations_of_class("cls-center", limit=5)
    assert r["ok"]
    it = r["items"][0]
    assert it["asset_id"] == "asset-1" and it["filename"] == "w.png"
    assert it["class_name"] == "Center"


def test_get_annotations는_자산별_라벨집합을_준다(superb):
    sb, _ = superb
    r = sb.get_annotations(["asset-1"])
    assert r["ok"] and r["labels"]["asset-1"] == ["Center"]


# ---------------- 에러 처리 ----------------
def test_서버_에러envelope를_사람이_읽을_수_있게_변환(superb):
    sb, _ = superb
    ok, text = sb._api().request("GET", "/없는/경로")
    assert ok is False and text.startswith("NOT_FOUND:")


def test_status_요약(superb):
    sb, _ = superb
    st = sb.status()
    assert st["connected"] is True
    assert st["transport"] in ("sdk", "rest")
    assert st["deployment"]["id"] == DEP
