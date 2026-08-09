""".env 로더 단위테스트."""
from __future__ import annotations
import importlib
import os

import config


def _load(tmp_path, text, monkeypatch):
    p = tmp_path / ".env"
    p.write_text(text, encoding="utf-8")
    monkeypatch.setenv("WAFER_ENV_FILE", str(p))
    return importlib.reload(config)


def test_기본_KEY_VALUE_파싱(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO_A", raising=False)
    c = _load(tmp_path, "FOO_A=hello\n", monkeypatch)
    assert os.environ["FOO_A"] == "hello"
    assert "FOO_A" in c.loaded_keys()


def test_export_접두어와_따옴표와_주석(tmp_path, monkeypatch):
    for k in ("FOO_B", "FOO_C", "FOO_D", "FOO_E"):
        monkeypatch.delenv(k, raising=False)
    _load(tmp_path, """
# 주석은 무시
export FOO_B=exported
FOO_C="quoted value"
FOO_D='single'

FOO_E=has=equals=inside
""", monkeypatch)
    assert os.environ["FOO_B"] == "exported"
    assert os.environ["FOO_C"] == "quoted value"
    assert os.environ["FOO_D"] == "single"
    assert os.environ["FOO_E"] == "has=equals=inside"


def test_셸_환경변수가_env파일보다_우선(tmp_path, monkeypatch):
    monkeypatch.setenv("FOO_F", "from-shell")
    c = _load(tmp_path, "FOO_F=from-file\n", monkeypatch)
    assert os.environ["FOO_F"] == "from-shell"
    assert "FOO_F" not in c.loaded_keys()


def test_파일이_없으면_조용히_무시(tmp_path, monkeypatch):
    monkeypatch.setenv("WAFER_ENV_FILE", str(tmp_path / "없는파일.env"))
    c = importlib.reload(config)
    assert c.load_env() == []
    assert c.env_status()["exists"] is False


def test_값이_없는_줄은_건너뜀(tmp_path, monkeypatch):
    monkeypatch.delenv("FOO_G", raising=False)
    _load(tmp_path, "쓰레기줄\nFOO_G=ok\n", monkeypatch)
    assert os.environ["FOO_G"] == "ok"


def test_env_status는_값을_노출하지_않음(tmp_path, monkeypatch):
    monkeypatch.delenv("SECRET_X", raising=False)
    c = _load(tmp_path, "SECRET_X=super-secret\n", monkeypatch)
    st = c.env_status()
    assert "SECRET_X" in st["keys"]
    assert "super-secret" not in repr(st)
