"""
Superb AI 연동 모듈 (SDK 0.4.x 계약 기준)
  F-01 업로드 / F-02 배포 모델 추론(오토라벨링) / F-05 피드백 루프

- API 키는 절대 코드에 넣지 말 것. 환경변수로만 주입.
    export SUPERB_AI_API_KEY=sbd_pk_...
    export SUPERB_AI_TENANT=spot
    export SUPERB_DEPLOYMENT_ID=6008abc6-71b6-4c45-9d9c-a89cd14266fe
- 키가 없으면 모든 함수가 조용히 실패(ok=False)해서 데모가 오프라인으로도 동작합니다.

전송 계층은 두 가지:
  1) superb_ai SDK (Python >= 3.12 필요) — 있으면 우선 사용(재시도/에러매핑 포함)
  2) requests 직접 호출 — SDK를 못 쓰는 환경(예: Python 3.9)용 폴백.
     동일한 REST 엔드포인트(/tenants/{tenant}/...)를 그대로 호출합니다.

핵심 API (SDK 0.4.0):
    client.deployments.predict(deployment_id, image_b64=..., confidence=...)
      -> PredictResponse(predictions=[Prediction(class_name, geometry{x,y,w,h}, confidence)],
                         image={width,height}, inference_ms)
    * 이미지 소스는 base64 바이트(image_b64). 좌표는 '보낸 이미지'의 픽셀 좌표계.
"""
from __future__ import annotations

import base64
import io
import os
import time
from typing import Optional, Union

import config          # .env 를 os.environ 으로 로드 (import 부수효과)

# ---------------- 설정 (환경변수 / .env) ----------------
BASE_URL = os.environ.get("SUPERB_AI_BASE_URL", "https://api.bdai.superb-ai.com").rstrip("/")
TENANT = os.environ.get("SUPERB_AI_TENANT") or os.environ.get("SUPERB_TENANT", "spot")
API_KEY = os.environ.get("SUPERB_AI_API_KEY", "")
PROJECT_ID = os.environ.get("SUPERB_PROJECT_ID", "7b267a4e-eea0-4764-9a61-6d74515a4985")
DATASET_ID = os.environ.get("SUPERB_DATASET_ID", "ded47876-a675-492e-a548-c296f80fd151")
CLS_ID = os.environ.get("SUPERB_CLASS_ID", "9fd96ef0-0b00-4157-8633-fabd20a4e6c3")
# 배포 ID(권장). 없으면 모델 ID로 배포 목록에서 찾고, 그것도 없으면 ready 배포 하나를 자동 선택.
DEPLOY_ID = os.environ.get("SUPERB_DEPLOYMENT_ID", "")
MODEL_ID = os.environ.get("SUPERB_MODEL_ID", "")
# 모델이 워밍업(MODEL_LOADING/MODEL_STARTING) 중일 때 최대 대기 시간(초)
PREDICT_MAX_WAIT = float(os.environ.get("SUPERB_PREDICT_MAX_WAIT", "90"))

LABELS = ["Center", "Donut", "Edge_Loc", "Edge_Ring", "Loc", "Near_Full", "Scratch", "Random"]


# ---------------- 전송 계층 ----------------
class _Api:
    """/tenants/{tenant} 스코프 REST 호출. SDK가 있으면 SDK 트랜스포트를 사용."""

    def __init__(self):
        self.mode = "none"
        self._sdk = None
        if not API_KEY:
            return
        try:                                    # 1) SDK (Python >= 3.12)
            from superb_ai import Client
            self._sdk = Client(tenant=TENANT, api_key=API_KEY, base_url=BASE_URL)
            self.mode = "sdk"
            return
        except Exception:
            self._sdk = None
        try:                                    # 2) requests 폴백
            import requests  # noqa: F401
            self.mode = "rest"
        except Exception:
            self.mode = "none"

    @property
    def sdk(self):
        return self._sdk

    def request(self, method: str, path: str, json=None, params=None, timeout: float = 60.0):
        """반환: (ok, payload) — 실패 시 payload는 사람이 읽을 수 있는 에러 문자열."""
        if self.mode == "sdk":
            try:
                r = self._sdk.request(method, path, json=json, params=params, timeout=timeout)
                return True, (r.json() if r.content else {})
            except Exception as e:
                return False, _err_text(e)
        if self.mode == "rest":
            import requests
            url = f"{BASE_URL}/tenants/{TENANT}{path}"
            try:
                r = requests.request(
                    method, url, json=json, params=params, timeout=timeout,
                    headers={"Authorization": f"Bearer {API_KEY}",
                             "User-Agent": "wafer-demo/1.0"},
                )
            except Exception as e:
                return False, f"{type(e).__name__}: {e}"
            if r.status_code >= 400:
                return False, _http_err_text(r.status_code, r.text)
            try:
                return True, (r.json() if r.content else {})
            except ValueError:
                return True, {}
        return False, "no_api_key_or_transport"


def _err_text(e: Exception) -> str:
    code = getattr(e, "code", None)
    msg = str(e)
    return f"{code}: {msg}" if code else f"{type(e).__name__}: {msg}"


def _http_err_text(status: int, body: str) -> str:
    try:
        import json as _json
        env = _json.loads(body)
        err = env.get("error") or {}
        if err:
            return f"{err.get('code', status)}: {err.get('message', '')[:200]}"
    except Exception:
        pass
    return f"HTTP {status}: {body[:200]}"


def _error_code(text: str) -> str:
    return text.split(":", 1)[0].strip() if ":" in text else ""


_api_cache: Optional[_Api] = None


def _api() -> _Api:
    global _api_cache
    if _api_cache is None:
        _api_cache = _Api()
    return _api_cache


def available() -> bool:
    """API 키 + 전송 계층이 준비됐는지."""
    return _api().mode != "none"


def transport() -> str:
    return _api().mode          # 'sdk' | 'rest' | 'none'


# ---------------- 배포(deployment) 조회 ----------------
_dep_cache: dict = {}


def resolve_deployment(force: bool = False) -> dict:
    """
    사용할 배포 ID를 확정.
      1) SUPERB_DEPLOYMENT_ID
      2) SUPERB_MODEL_ID 와 model_id 가 일치하는 배포
      3) 테넌트의 ready 배포 중 최신 1개
    반환: {'ok':bool, 'deployment_id':str, 'via':str} | {'ok':False,'reason':str}
    """
    if not force and _dep_cache.get("id"):
        return {"ok": True, "deployment_id": _dep_cache["id"], "via": _dep_cache.get("via", "cache")}
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    if DEPLOY_ID:
        _dep_cache.update(id=DEPLOY_ID, via="env:SUPERB_DEPLOYMENT_ID")
        return {"ok": True, "deployment_id": DEPLOY_ID, "via": _dep_cache["via"]}

    ok, payload = _api().request("GET", "/deployments", params={"limit": 50}, timeout=10.0)
    if not ok:
        return {"ok": False, "reason": payload}
    items = payload.get("items") or payload.get("data") or []
    if MODEL_ID:
        for d in items:
            if str(d.get("model_id")) == MODEL_ID:
                _dep_cache.update(id=str(d["id"]), via="model_id 매칭")
                return {"ok": True, "deployment_id": _dep_cache["id"], "via": _dep_cache["via"]}
        return {"ok": False, "reason": f"SUPERB_MODEL_ID({MODEL_ID[:8]}…)로 배포된 deployment 없음"}
    for d in items:
        if str(d.get("status")) in ("ready", "deploying"):
            _dep_cache.update(id=str(d["id"]), via=f"자동선택({d.get('name','')})")
            return {"ok": True, "deployment_id": _dep_cache["id"], "via": _dep_cache["via"]}
    return {"ok": False, "reason": "실행 중인 배포 없음 (SUPERB_DEPLOYMENT_ID 설정 필요)"}


def deployment_info(force: bool = False) -> dict:
    """
    배포 상세: 상태 / 클래스맵 / 추천 임계값.
    반환: {'ok':True,'id','status','task','class_map':{id:name},'recommended_conf':float|None}
    """
    if not force and _dep_cache.get("info"):
        return _dep_cache["info"]
    r = resolve_deployment(force=force)
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("reason")}
    dep_id = r["deployment_id"]
    ok, payload = _api().request("GET", f"/deployments/{dep_id}", timeout=10.0)
    if not ok:
        return {"ok": False, "reason": payload, "id": dep_id}
    cap = payload.get("capability") or {}
    class_map = {int(c["class_id"]): c["name"] for c in (cap.get("class_map") or [])
                 if c.get("class_id") is not None}
    params = cap.get("params") or {}
    rec = params.get("confidence", {}).get("default") if isinstance(params.get("confidence"), dict) else None
    info = {"ok": True, "id": dep_id, "via": r.get("via"),
            "status": str(payload.get("status", "?")),
            "task": str(payload.get("task", "?")),
            "name": str(payload.get("name", "")),
            "class_map": class_map,
            "recommended_conf": float(rec) if rec is not None else None}
    _dep_cache["info"] = info
    return info


def deploy_available() -> bool:
    """배포 모델 추론이 가능한 상태인지(키 + 배포 ID 확정)."""
    return available() and resolve_deployment().get("ok", False)


# ---------------- F-02 배포 모델 추론 ----------------
def _to_png_b64(image: Union[str, bytes, "object"]) -> bytes:
    """경로 / bytes / PIL.Image → PNG 바이트."""
    if isinstance(image, bytes):
        return image
    if isinstance(image, str):
        with open(image, "rb") as f:
            return f.read()
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")     # PIL.Image
    return buf.getvalue()


def detect_remote(image, conf: Optional[float] = None, iou: Optional[float] = None,
                  max_detections: Optional[int] = None, max_wait: Optional[float] = None) -> dict:
    """
    배포된 Superb 탐지 모델로 추론 (deployments.predict, image_b64).

    image: 파일 경로 | PNG bytes | PIL.Image
    conf : 신뢰도 임계값. None이면 모델의 추천 기본값 사용.
    반환 : {'ok':True, 'boxes':[(label,x0,y0,x1,y1,conf),...],
            'size':(W,H),           # 모델이 디코딩한 이미지 크기(= 보낸 이미지 크기)
            'inference_ms':int, 'deployment_id':str}
           또는 {'ok':False, 'reason':str}
    """
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    r = resolve_deployment()
    if not r.get("ok"):
        return {"ok": False, "reason": r.get("reason")}
    dep_id = r["deployment_id"]

    try:
        b64 = base64.b64encode(_to_png_b64(image)).decode("ascii")
    except Exception as e:
        return {"ok": False, "reason": f"이미지 인코딩 실패: {e}"}

    body = {"image_b64": b64}
    if conf is not None:
        body["confidence"] = float(conf)
    if iou is not None:
        body["iou"] = float(iou)
    if max_detections is not None:
        body["max_detections"] = int(max_detections)

    deadline = time.time() + (PREDICT_MAX_WAIT if max_wait is None else max_wait)
    last = ""
    while True:
        ok, payload = _api().request("POST", f"/deployments/{dep_id}/predict", json=body, timeout=120.0)
        if ok:
            return _parse_predict(payload, dep_id)
        last = payload
        code = _error_code(payload)
        # 모델 워밍업 중이면 안내된 주기로 재시도
        if code in ("MODEL_LOADING", "MODEL_STARTING") and time.time() < deadline:
            time.sleep(15.0 if code == "MODEL_LOADING" else 30.0)
            continue
        return {"ok": False, "reason": last, "deployment_id": dep_id}


def _parse_predict(payload: dict, dep_id: str) -> dict:
    """PredictResponse → (label, x0,y0,x1,y1, conf) 리스트. bbox/polygon 모두 처리."""
    boxes = []
    for p in payload.get("predictions") or []:
        score = float(p.get("confidence", 0.0) or 0.0)
        name = p.get("class_name") or str(p.get("class_id", "Defect"))
        g = p.get("geometry") or {}
        gtype = g.get("type")
        if gtype == "bbox":
            x, y, w, h = float(g["x"]), float(g["y"]), float(g["w"]), float(g["h"])
            boxes.append((str(name), x, y, x + w, y + h, score))
        elif gtype == "polygon":                      # 폴리곤은 외접 박스로 환산
            pts = [pt for part in g.get("polygons", []) for pt in part.get("exterior", [])]
            if not pts:
                continue
            xs = [float(p_[0]) for p_ in pts]; ys = [float(p_[1]) for p_ in pts]
            boxes.append((str(name), min(xs), min(ys), max(xs), max(ys), score))
    img = payload.get("image") or {}
    size = (int(img.get("width", 0) or 0), int(img.get("height", 0) or 0))
    return {"ok": True, "boxes": boxes, "size": size,
            "inference_ms": int(payload.get("inference_ms", 0) or 0),
            "deployment_id": dep_id}


# ---------------- F-01 업로드 ----------------
def upload_image(path: str, key: Optional[str] = None, wait_ready: float = 12.0) -> dict:
    """
    데이터셋에 업로드 → 프로젝트 스코프에 편입.
    자산 행(asset row)은 서버가 '비동기'로 만들기 때문에, 업로드 직후엔 asset_id가 없을 수 있음.
    wait_ready 초 동안 폴링해서 확보하고, 못 찾으면 pending=True 로 반환.
    """
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    filename = key or os.path.basename(path)

    api = _api()
    if api.mode == "sdk":
        try:
            res = api.sdk.assets.upload_paths(DATASET_ID, [path], concurrency=1)
            r0 = res[0] if isinstance(res, list) and res else res
            if not getattr(r0, "ok", False):
                return {"ok": False, "reason": getattr(r0, "error", "upload failed")}
        except Exception as e:
            return {"ok": False, "reason": _err_text(e)}
    else:
        up = _rest_upload(path, filename)
        if not up.get("ok"):
            return up

    asset_id = _find_asset_id(filename, wait_ready)
    if not asset_id:
        return {"ok": True, "asset_id": None, "pending": True, "key": filename,
                "reason": "자산 생성이 아직 진행 중(서버 비동기) — 어노테이션은 나중에 가능"}
    ok, payload = _api().request("POST", f"/projects/{PROJECT_ID}/assets/batch-add",
                                 json={"asset_ids": [asset_id]})
    return {"ok": True, "asset_id": asset_id, "pending": False, "key": filename,
            "in_project": bool(ok), "project_reason": None if ok else payload}


def _rest_upload(path: str, filename: str) -> dict:
    """SDK 없이 3단계 프리사인드 업로드(init → PUT)."""
    import mimetypes
    import requests
    ctype = mimetypes.guess_type(filename)[0] or "image/png"
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return {"ok": False, "reason": str(e)}
    ok, payload = _api().request(
        "POST", f"/datasets/images/{DATASET_ID}/assets/upload-init/batch",
        json={"items": [{"client_ref": filename, "filename": filename,
                         "size_bytes": size, "content_type": ctype}]})
    if not ok:
        return {"ok": False, "reason": payload}
    items = payload.get("items") or []
    if not items:
        return {"ok": False, "reason": "upload-init 응답 비어 있음"}
    it = items[0]
    if it.get("upload_type") == "error" or not it.get("upload_url"):
        return {"ok": False, "reason": it.get("error_message") or it.get("error_code") or "init failed"}
    try:
        with open(path, "rb") as f:                     # 프리사인드 URL엔 인증 헤더 금지
            r = requests.put(it["upload_url"], data=f, headers={"Content-Type": ctype}, timeout=120)
        if r.status_code >= 400:
            return {"ok": False, "reason": f"S3 PUT {r.status_code}"}
    except Exception as e:
        return {"ok": False, "reason": f"S3 PUT 실패: {e}"}
    return {"ok": True}


def _find_asset_id(filename: str, wait_s: float) -> Optional[str]:
    """파일명으로 자산을 폴링 조회 (업로드 직후엔 아직 없을 수 있음)."""
    deadline = time.time() + max(wait_s, 0.0)
    while True:
        ok, payload = _api().request(
            "GET", f"/datasets/images/{DATASET_ID}/assets",
            params={"q": filename, "status": "ready", "limit": 10}, timeout=15.0)
        if ok:
            for a in (payload.get("items") or payload.get("data") or []):
                if a.get("filename") == filename:
                    return str(a["id"])
        if time.time() >= deadline:
            return None
        time.sleep(1.5)


# ---------------- 데이터셋에서 테스트 이미지 내려받기 ----------------
def list_assets(limit: int = 100, q: Optional[str] = None, max_pages: int = 1) -> dict:
    """
    데이터셋의 ready 자산 목록 (커서 페이지네이션).
    limit: 페이지당 개수(서버 상한 100), max_pages: 몇 페이지까지 이어서 가져올지.
    반환 {'ok':True,'items':[{'id','filename'},...]}
    """
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    items, cursor = [], None
    for _ in range(max(max_pages, 1)):
        params = {"status": "ready", "limit": min(limit, 100)}
        if q:
            params["q"] = q
        if cursor:
            params["cursor"] = cursor
        ok, payload = _api().request("GET", f"/datasets/images/{DATASET_ID}/assets",
                                     params=params, timeout=20.0)
        if not ok:
            return {"ok": False, "reason": payload, "items": items}
        items += [{"id": str(a["id"]), "filename": a.get("filename", "")}
                  for a in (payload.get("items") or [])]
        cursor = payload.get("next_cursor")
        if not cursor:
            break
    return {"ok": True, "items": items}


def download_asset(asset_id: str, dest_path: str) -> dict:
    """자산 원본을 로컬로 저장 (프리사인드 URL은 인증 헤더 없이 받아야 함)."""
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    ok, payload = _api().request("GET", f"/datasets/images/{DATASET_ID}/assets/{asset_id}/download-url",
                                 timeout=20.0)
    if not ok:
        return {"ok": False, "reason": payload}
    url = payload.get("url")
    if not url:
        return {"ok": False, "reason": "download-url 응답에 url 없음"}
    try:
        import requests
        r = requests.get(url, timeout=60)          # S3: Authorization 헤더 금지
        if r.status_code >= 400:
            return {"ok": False, "reason": f"S3 GET {r.status_code}"}
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(r.content)
    except Exception as e:
        return {"ok": False, "reason": f"{type(e).__name__}: {e}"}
    return {"ok": True, "path": dest_path, "bytes": len(r.content)}


def list_classes() -> dict:
    """프로젝트 클래스 목록 → {'ok':True,'object':{name:id}, 'classification':{name:id}}"""
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    ok, payload = _api().request("GET", f"/projects/{PROJECT_ID}/classes", timeout=20.0)
    if not ok:
        return {"ok": False, "reason": payload}
    obj, clf = {}, {}
    for v in payload.values():
        if not isinstance(v, list):
            continue
        for c in v:
            (obj if c.get("kind") == "object" else clf)[c.get("name")] = str(c.get("id"))
    return {"ok": True, "object": obj, "classification": clf}


def annotations_of_class(class_id: str, limit: int = 20) -> dict:
    """
    특정 클래스의 어노테이션 조회(자산 파일명 포함).
    반환 {'ok':True,'items':[{'asset_id','filename','class_name','geometry'},...]}
    """
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    ok, payload = _api().request(
        "GET", f"/projects/{PROJECT_ID}/annotations",
        params={"class_id": class_id, "include": "asset,class", "limit": min(limit, 200)},
        timeout=20.0)
    if not ok:
        return {"ok": False, "reason": payload}
    items = []
    for a in (payload.get("items") or []):
        items.append({"asset_id": str(a.get("asset_id")),
                      "filename": (a.get("asset") or {}).get("filename", ""),
                      "class_name": (a.get("class") or {}).get("name", ""),
                      "geometry": a.get("geometry") or {}})
    return {"ok": True, "items": items}


def get_annotations(asset_ids: list, limit: int = 200) -> dict:
    """자산들의 기존 어노테이션(정답 라벨) 조회 → {asset_id: [class_name, ...]}"""
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    ok, payload = _api().request(
        "GET", f"/projects/{PROJECT_ID}/annotations",
        params={"asset_ids": [str(a) for a in asset_ids], "include": "class", "limit": limit},
        timeout=20.0)
    if not ok:
        return {"ok": False, "reason": payload}
    by_asset: dict = {}
    for a in (payload.get("items") or []):
        cls = (a.get("class") or {}).get("name")
        if cls:
            by_asset.setdefault(str(a["asset_id"]), set()).add(cls)
    return {"ok": True, "labels": {k: sorted(v) for k, v in by_asset.items()}}


# ---------------- 어노테이션 주입 / F-05 피드백 ----------------
_ann_key = {"k": os.environ.get("SUPERB_ANN_DATA_KEY", "")}   # 'value' | 'answer'


def push_prediction(asset_id: str, patterns: list, source: str = "model") -> dict:
    """
    이미지 레벨 다중라벨(체크박스) 분류 결과를 프로젝트에 적재.
    source: 'model'(오토라벨링) | 'manual'(엔지니어 수정)
    """
    if not available():
        return {"ok": False, "reason": "no_api_key_or_transport"}
    if not asset_id:
        return {"ok": False, "reason": "asset_id 없음(업로드 미완료)"}
    answer = [p for p in patterns if p in LABELS]
    if not answer:
        return {"ok": False, "reason": "empty_prediction"}

    last = ""
    # data 키는 프로젝트 스키마(체크박스)에 따라 value / answer 중 하나 — 처음 한 번만 탐색하고 기억.
    keys = [_ann_key["k"]] if _ann_key["k"] else ["value", "answer"]
    for data in ({k: answer} for k in keys):
        row = {"asset_id": str(asset_id), "class_id": CLS_ID,
               "type": "classification", "data": data}
        ok, payload = _api().request(
            "POST", f"/projects/{PROJECT_ID}/annotations/batch-create",
            json={"annotations": [row], "source": source, "replace": True})
        if ok:
            _ann_key["k"] = list(data)[0]
            return {"ok": True, "created": payload.get("created", 1),
                    "source": source, "data_key": _ann_key["k"]}
        last = payload
        if _error_code(payload) != "VALIDATION_ERROR":
            break
    return {"ok": False, "reason": last}


def push_feedback(asset_id: str, corrected_patterns: list) -> dict:
    """F-05: 엔지니어 수정 라벨을 Superb로 재주입 (액티브 러닝)."""
    return push_prediction(asset_id, corrected_patterns, source="manual")


# ---------------- 상태 요약 (UI 배지용) ----------------
def status() -> dict:
    """앱 헤더에 뿌릴 연동 상태 한 방에."""
    if not API_KEY:
        return {"connected": False, "reason": "SUPERB_AI_API_KEY 미설정", "transport": "none"}
    mode = transport()
    if mode == "none":
        return {"connected": False, "transport": "none",
                "reason": "superb-ai SDK(Python≥3.12) 또는 requests 중 하나가 필요합니다"}
    info = deployment_info()
    return {"connected": True, "transport": mode, "tenant": TENANT,
            "deployment": info if info.get("ok") else None,
            "reason": None if info.get("ok") else info.get("reason")}
