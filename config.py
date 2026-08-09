"""
설정 로더 (의존성 없음) — models / superb_client / app 이 import 시점에 한 번 실행.

우선순위: 실제 환경변수 > .env 파일 > Streamlit Secrets
  - 로컬     : .env 파일
  - 배포(클라우드) : .env 를 커밋하지 않으므로 Streamlit Secrets 에서 채운다

.env 지원 형식:
    KEY=value
    export KEY=value
    KEY="value with spaces"      # 따옴표 제거
    # 주석 / 빈 줄 무시
"""
from __future__ import annotations
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.environ.get("WAFER_ENV_FILE", os.path.join(HERE, ".env"))

_loaded = []        # .env 에서 채운 키
_secrets = []       # Streamlit Secrets 에서 채운 키


def load_env(path: str = None) -> list:
    """.env를 os.environ에 주입하고 '새로 채워진 키' 목록을 반환."""
    global _loaded
    path = path or ENV_PATH
    if not os.path.exists(path):
        return []
    filled = []
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip()
                if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
                    val = val[1:-1]
                if key and key not in os.environ:      # 셸 환경변수가 우선
                    os.environ[key] = val
                    filled.append(key)
    except OSError:
        return []
    _loaded = filled
    return filled


def load_secrets() -> list:
    """
    Streamlit Secrets(.streamlit/secrets.toml 또는 클라우드 Secrets)를 os.environ 으로.

    배포 환경에는 .env 가 없으므로 이쪽이 설정 통로다.
    참고: Streamlit 은 최상위 스칼라 시크릿을 '이미' 환경변수로 내보낸다. 그래서 보통
    여기서 새로 채울 게 없다 — 그래도 두는 이유는 그 동작에 의존하지 않기 위해서다.
    Streamlit 이 없거나 secrets 가 없는 환경(테스트/CLI)에서는 조용히 넘어간다.
    """
    global _secrets
    seen = []
    try:
        import streamlit as st
        for key in st.secrets:                      # 최상위 스칼라만 사용
            val = st.secrets[key]
            if isinstance(val, (str, int, float, bool)):
                seen.append(key)
                os.environ.setdefault(key, str(val))
    except Exception:
        return []
    _secrets = seen
    return seen


def loaded_keys() -> list:
    return list(_loaded)


def secret_keys() -> list:
    return list(_secrets)


def env_status() -> dict:
    """UI 표시용: 설정 출처와 채워진 키(값은 절대 노출하지 않음)."""
    return {"path": ENV_PATH, "exists": os.path.exists(ENV_PATH),
            "keys": loaded_keys(), "secrets": secret_keys()}


load_env()          # import 시 자동 실행
load_secrets()      # .env 에 없는 값만 Secrets 로 보충
