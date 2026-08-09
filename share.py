"""
데모를 외부에 공유 — Streamlit 실행 + ngrok 터널.

  python3 share.py                 # 앱 띄우고 공개 URL 출력
  python3 share.py --port 8502
  python3 share.py --no-run        # 이미 streamlit 이 떠 있을 때 터널만

준비:
  pip install pyngrok
  https://dashboard.ngrok.com/get-started/your-authtoken 에서 토큰 복사 → .env 에 추가
      NGROK_AUTHTOKEN=2abc...

토큰 없이 바로 쓰려면 (설치만 하면 계정 불필요):
  brew install cloudflared && cloudflared tunnel --url http://localhost:8501
"""
from __future__ import annotations
import argparse
import os
import socket
import subprocess
import sys
import time

import config          # .env 로드


def _port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8501)
    ap.add_argument("--no-run", action="store_true", help="터널만 열기(앱은 이미 실행 중)")
    ap.add_argument("--wait", type=int, default=40, help="앱이 뜰 때까지 최대 대기(초)")
    a = ap.parse_args()

    token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not token:
        print("❌ NGROK_AUTHTOKEN 이 없습니다.\n"
              "   1) https://dashboard.ngrok.com/get-started/your-authtoken 에서 토큰 복사\n"
              "   2) .env 에 한 줄 추가:  NGROK_AUTHTOKEN=2abc...\n"
              "   (계정 없이 쓰려면: cloudflared tunnel --url http://localhost:%d)" % a.port)
        return 1
    try:
        from pyngrok import conf as ngrok_conf, ngrok
    except ImportError:
        print("❌ pyngrok 이 없습니다.  pip install pyngrok")
        return 1

    proc = None
    if not a.no_run and not _port_open(a.port):
        print(f"▶ streamlit 실행 (port {a.port}) …")
        proc = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py",
             "--server.headless", "true", "--server.port", str(a.port)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        for _ in range(a.wait * 2):
            if _port_open(a.port):
                break
            time.sleep(0.5)
    if not _port_open(a.port):
        print(f"❌ {a.port} 포트에 앱이 뜨지 않았습니다. 'streamlit run app.py' 를 직접 확인하세요.")
        if proc:
            proc.terminate()
        return 1

    ngrok_conf.get_default().auth_token = token
    tunnel = ngrok.connect(a.port, "http")
    print("\n" + "=" * 60)
    print(f"  공개 URL :  {tunnel.public_url}")
    print(f"  로컬     :  http://localhost:{a.port}")
    print("=" * 60)
    print("  이 링크를 그대로 보내면 됩니다. 창을 닫으면(Ctrl+C) 링크도 끊깁니다.")
    print("  ※ 링크를 아는 사람은 누구나 접속 가능 — 데모 끝나면 꼭 종료하세요.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n종료 중…")
    finally:
        try:
            ngrok.disconnect(tunnel.public_url)
            ngrok.kill()
        except Exception:
            pass
        if proc:
            proc.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
