"""
웨이퍼 데모 UI 패키지.

  theme       화면 상수 + 전역 CSS + 작은 마크업 헬퍼
  io_utils    업로드 파싱/검증 (PNG·JPG·BMP·NPZ)
  charts      웨이퍼 맵 / 3D FFT (plotly=화면, matplotlib=PDF)
  report_pdf  F-07 PDF 생성
  panels      헤더·좌·중·우 패널 렌더링

app.py 는 이 패키지를 조립하고 모델/Superb 호출만 담당한다.
"""
from .charts import draw_boxes, fft_surface_3d, freq_surface, wafer_figure
from .io_utils import load_upload, npz_length
from .report_pdf import build_pdf
from .theme import (ALLOWED, ALLOWED_ALL, ALLOWED_DATA, card, sec, setup_page, sub)

__all__ = [
    "ALLOWED", "ALLOWED_ALL", "ALLOWED_DATA",
    "setup_page", "sec", "sub", "card",
    "load_upload", "npz_length",
    "wafer_figure", "fft_surface_3d", "draw_boxes", "freq_surface",
    "build_pdf",
]
