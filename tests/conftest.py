"""
공용 픽스처.

원칙: 테스트는 **실제 Superb API를 절대 호출하지 않는다**.
프로젝트 루트의 .env 가 자동 로드되므로, Superb 관련 테스트는 환경변수를 목 서버로
갈아끼운 뒤 모듈을 reload 한다(설정을 모듈 상수로 읽기 때문).
"""
from __future__ import annotations
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DS = "11111111-1111-1111-1111-111111111111"
PID = "22222222-2222-2222-2222-222222222222"
CID = "33333333-3333-3333-3333-333333333333"
DEP = "44444444-4444-4444-4444-444444444444"

G = 52
PAL_NORMAL = (148, 163, 184)
PAL_DEFECT = (239, 68, 68)


# ---------------- 웨이퍼 이미지 헬퍼 ----------------
def wafer_mask(pattern: str = "Center") -> np.ndarray:
    """0 빈칸 / 1 정상 / 2 불량 마스크."""
    yy, xx = np.mgrid[0:G, 0:G]
    rr = np.sqrt((yy - G / 2 + 0.5) ** 2 + (xx - G / 2 + 0.5) ** 2)
    m = np.zeros((G, G), int)
    wafer = rr <= G / 2 - 0.5
    m[wafer] = 1
    if pattern == "Center":
        m[wafer & (rr < 7)] = 2
    elif pattern == "Edge_Ring":
        m[wafer & (rr > G / 2 - 4)] = 2
    elif pattern == "Normal":
        pass
    return m


def render(m: np.ndarray, normal=PAL_NORMAL, defect=PAL_DEFECT, bg=(0, 0, 0), size=256) -> Image.Image:
    rgb = np.zeros((*m.shape, 3), np.uint8)
    rgb[:] = bg
    rgb[m == 1] = normal
    rgb[m == 2] = defect
    return Image.fromarray(rgb).resize((size, size), Image.NEAREST)


@pytest.fixture
def wafer_img():
    return render(wafer_mask("Center"))


@pytest.fixture
def normal_img():
    return render(wafer_mask("Normal"))


# ---------------- Superb 목 서버 ----------------
class _State:
    def __init__(self):
        self.calls = []              # (method, path, body)
        self.predict_calls = 0
        self.warmup_errors = 0       # 이 횟수만큼 MODEL_LOADING 을 먼저 돌려준다
        self.uploaded = False
        self.asset_ready = True
        self.reject_value_key = True  # data={"value":...} 는 422 (이 프로젝트는 answer 스키마)
        self.annotations = []


def _make_handler(state: _State):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        # -- GET --
        def do_GET(self):
            path = self.path.split("?")[0]
            state.calls.append(("GET", path, self.headers.get("Authorization")))
            if path == "/tenants/testco/deployments":
                return self._send(200, {"items": [
                    {"id": DEP, "model_id": "model-abc", "status": "ready", "name": "wafer-det"}]})
            if path == f"/tenants/testco/deployments/{DEP}":
                return self._send(200, {
                    "id": DEP, "model_id": "model-abc", "status": "ready", "name": "wafer-det",
                    "task": "detection",
                    "capability": {"task": "detection", "output": "bbox",
                                   "class_map": [{"class_id": 0, "name": "Center"},
                                                 {"class_id": 1, "name": "Donut"}],
                                   "params": {"confidence": {"min": 0, "max": 1, "default": 0.2659}}}})
            if path == f"/tenants/testco/datasets/images/{DS}/assets":
                if not (state.uploaded and state.asset_ready):
                    return self._send(200, {"items": [], "next_cursor": None})
                return self._send(200, {"items": [{"id": "asset-1", "filename": "w.png",
                                                   "status": "ready"}], "next_cursor": None})
            if path.endswith("/download-url"):
                return self._send(200, {"url": f"http://127.0.0.1:{self.server.server_port}/s3get"})
            if path == f"/tenants/testco/projects/{PID}/classes":
                return self._send(200, {"object_classes": [
                    {"kind": "object", "name": "Center", "id": "cls-center"},
                    {"kind": "object", "name": "Donut", "id": "cls-donut"}],
                    "classifications": [{"kind": "classification", "name": "defect_pattern_v2", "id": CID}]})
            if path == f"/tenants/testco/projects/{PID}/annotations":
                return self._send(200, {"items": [
                    {"id": "ann-1", "asset_id": "asset-1", "class_id": "cls-center",
                     "type": "bbox", "geometry": {"type": "bbox", "x": 1, "y": 2, "w": 3, "h": 4},
                     "asset": {"filename": "w.png"}, "class": {"name": "Center"}}]})
            if path == "/s3get":
                self.send_response(200)
                self.send_header("Content-Length", "4")
                self.end_headers()
                self.wfile.write(b"PNG!")
                return
            return self._send(404, {"error": {"code": "NOT_FOUND", "message": path}})

        # -- PUT (S3 프리사인드) --
        def do_PUT(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            state.calls.append(("PUT", self.path, self.headers.get("Authorization")))
            state.uploaded = True
            self.send_response(200)
            self.end_headers()

        # -- POST --
        def do_POST(self):
            path = self.path.split("?")[0]
            body = self._body()
            state.calls.append(("POST", path, body))
            if path == f"/tenants/testco/deployments/{DEP}/predict":
                state.predict_calls += 1
                if state.predict_calls <= state.warmup_errors:
                    return self._send(503, {"error": {"code": "MODEL_LOADING", "message": "warming"}})
                if "image_b64" not in body:
                    return self._send(422, {"error": {"code": "VALIDATION_ERROR",
                                                      "message": "no image source"}})
                base64.b64decode(body["image_b64"])          # 디코딩 가능해야 함
                return self._send(200, {"predictions": [
                    {"type": "bbox", "class_id": 0, "class_name": "Center", "confidence": 0.9,
                     "geometry": {"type": "bbox", "x": 10, "y": 20, "w": 30, "h": 40}},
                    {"type": "polygon", "class_id": 1, "class_name": "Donut", "confidence": 0.5,
                     "geometry": {"type": "polygon",
                                  "polygons": [{"exterior": [[5, 5], [50, 5], [50, 60]]}]}}],
                    "image": {"width": 128, "height": 64}, "inference_ms": 12})
            if path == f"/tenants/testco/datasets/images/{DS}/assets/upload-init/batch":
                return self._send(200, {"items": [{
                    "client_ref": "w.png", "upload_type": "single", "s3_key": "k",
                    "upload_url": f"http://127.0.0.1:{self.server.server_port}/s3put"}]})
            if path == f"/tenants/testco/projects/{PID}/assets/batch-add":
                return self._send(200, {"added": 1, "skipped": 0})
            if path == f"/tenants/testco/projects/{PID}/annotations/batch-create":
                ann = body["annotations"][0]
                if state.reject_value_key and "value" in (ann.get("data") or {}):
                    return self._send(422, {"error": {"code": "VALIDATION_ERROR", "message": "bad key"}})
                state.annotations.append((ann, body.get("source"), body.get("replace")))
                return self._send(201, {"created": 1})
            return self._send(404, {"error": {"code": "NOT_FOUND", "message": path}})

    return H


@pytest.fixture
def superb(monkeypatch):
    """
    목 서버를 띄우고 superb_client 를 그 서버로 reload 해서 돌려준다.
    사용: def test_x(superb): sb, state = superb
    """
    import importlib
    state = _State()
    srv = HTTPServer(("127.0.0.1", 0), _make_handler(state))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_port

    for k, v in {
        "SUPERB_AI_BASE_URL": f"http://127.0.0.1:{port}",
        "SUPERB_AI_TENANT": "testco",
        "SUPERB_AI_API_KEY": "sbd_pk_test",
        "SUPERB_PROJECT_ID": PID,
        "SUPERB_DATASET_ID": DS,
        "SUPERB_CLASS_ID": CID,
        "SUPERB_DEPLOYMENT_ID": DEP,
        "SUPERB_PREDICT_MAX_WAIT": "5",
        "SUPERB_ANN_DATA_KEY": "",
        "SUPERB_MODEL_ID": "",
    }.items():
        monkeypatch.setenv(k, v)

    import superb_client
    sb = importlib.reload(superb_client)
    try:
        yield sb, state
    finally:
        srv.shutdown()
        srv.server_close()
        importlib.reload(superb_client)      # 다른 테스트에 영향 주지 않게 원복


@pytest.fixture
def superb_offline(monkeypatch):
    """키가 전혀 없는 오프라인 상태의 superb_client."""
    import importlib
    for k in ("SUPERB_AI_API_KEY", "SUPERB_DEPLOYMENT_ID", "SUPERB_MODEL_ID"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SUPERB_AI_BASE_URL", "http://127.0.0.1:1")   # 실수로도 못 나가게
    import superb_client
    sb = importlib.reload(superb_client)
    try:
        yield sb
    finally:
        importlib.reload(superb_client)
