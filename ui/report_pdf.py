"""
F-07: 결함 분석 리포트 PDF.

화면(Streamlit)과 같은 구성·색으로 문서를 만든다.
  표지 헤더 → 요약 카드(감지 결함 / 추정 공정) → 신뢰도
  1. 결함 탐지 (웨이퍼 맵 + 박스 목록)
  2. 주파수 분석 (3D FFT)
  3. 원인 역추적 (Graph-RAG 문장)
  4. 패턴별 점수 (막대 표)
"""
from __future__ import annotations

import datetime
import io
import os

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image as RLImage, KeepTogether,
                                PageTemplate, Paragraph, Spacer, Table, TableStyle)

from .charts import fft_report_figure, wafer_report_figure

# 화면과 동일한 팔레트
RED = colors.HexColor("#ef4444")
RED_DK = colors.HexColor("#dc2626")
RED_BG = colors.HexColor("#fdecec")
AMBER = colors.HexColor("#b45309")
AMBER_BG = colors.HexColor("#fdf6e3")
INK = colors.HexColor("#0f172a")
MUTED = colors.HexColor("#64748b")
LINE = colors.HexColor("#e2e8f0")
BLUE = colors.HexColor("#3b82f6")

# 한글 폰트 후보 (path, ttc 안에서의 폰트 번호).
# Helvetica 로 폴백되면 한글이 전부 깨지므로 OS별 경로를 모두 훑는다.
# 환경변수 WAFER_PDF_FONT 로 직접 지정 가능.
_KR_FONTS = [
    # Linux (Streamlit Cloud 등 — packages.txt 의 fonts-nanum)
    ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 0),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", 0),
    # macOS
    ("/System/Library/Fonts/Supplemental/AppleGothic.ttf", 0),
    ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0),
    ("/Library/Fonts/Arial Unicode.ttf", 0),
    # Windows
    ("C:/Windows/Fonts/malgun.ttf", 0),
]

_registered = None          # 프로세스당 한 번만 등록


def _register_font() -> str:
    """한글 TTF/TTC 를 'KR' 로 등록하고 폰트명을 반환. 실패 시 'Helvetica'."""
    global _registered
    if _registered is not None:
        return _registered

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    env = os.environ.get("WAFER_PDF_FONT", "").strip()
    for path, idx in ([(env, 0)] if env else []) + _KR_FONTS:
        if not os.path.exists(path):
            continue
        try:
            font = TTFont("KR", path, subfontIndex=idx) if path.lower().endswith(".ttc") \
                else TTFont("KR", path)
            pdfmetrics.registerFont(font)
            pdfmetrics.stringWidth("웨이퍼", "KR", 10)      # 한글 글리프 실제 확인
            # 굵은 글꼴이 따로 없으므로 <b> 도 같은 폰트로 매핑 (에러 방지)
            pdfmetrics.registerFontFamily("KR", normal="KR", bold="KR",
                                          italic="KR", boldItalic="KR")
            _registered = "KR"
            return _registered
        except Exception:
            continue
    _registered = "Helvetica"
    return _registered


def _styles(font: str) -> dict:
    base = dict(fontName=font, leading=13, alignment=TA_LEFT, textColor=INK)
    return {
        "h1": ParagraphStyle("h1", **{**base, "fontSize": 19, "leading": 23,
                                      "textColor": INK}),
        "meta": ParagraphStyle("meta", **{**base, "fontSize": 8.5, "leading": 12,
                                          "textColor": MUTED}),
        "h2": ParagraphStyle("h2", **{**base, "fontSize": 11.5, "leading": 15,
                                      "textColor": INK, "spaceBefore": 2}),
        "body": ParagraphStyle("body", **{**base, "fontSize": 9, "leading": 14}),
        "small": ParagraphStyle("small", **{**base, "fontSize": 7.8, "leading": 11,
                                            "textColor": MUTED}),
        "cardlab": ParagraphStyle("cardlab", **{**base, "fontSize": 8, "leading": 11,
                                                "textColor": MUTED}),
        "cardval": ParagraphStyle("cardval", **{**base, "fontSize": 12.5, "leading": 16}),
    }


def _fig_bytes(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _rl_image(buf: io.BytesIO, width: float):
    """가로폭을 맞추고 비율은 유지."""
    iw, ih = ImageReader(buf).getSize()
    buf.seek(0)
    return RLImage(buf, width=width, height=width * ih / iw)


def _summary_cards(preds, proc, st_) -> Table:
    """감지 결함(분홍) / 추정 공정(노랑) 요약 — 화면 카드와 같은 구성."""
    defect = (" + ".join(preds) + (" (복합)" if len(preds) > 1 else "")) \
        if preds else "없음 (정상 웨이퍼)"

    def cell(label, value, color):
        return Table([[Paragraph(label, st_["cardlab"])],
                      [Paragraph(f'<font color="{color}">{value}</font>', st_["cardval"])]],
                     colWidths=[80 * mm])

    left, right = cell("감지 결함", defect, RED_DK.hexval()), cell("추정 공정", proc, AMBER.hexval())
    t = Table([[left, right]], colWidths=[85 * mm, 85 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, 0), RED_BG),
        ("BACKGROUND", (1, 0), (1, 0), AMBER_BG),
        ("BOX", (0, 0), (0, 0), .6, RED),
        ("BOX", (1, 0), (1, 0), .6, colors.HexColor("#eab308")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def _bar(value: float, width: float = 34 * mm, height: float = 3.6):
    """0~1 값을 막대 그림으로 (패턴별 점수/신뢰도용)."""
    from reportlab.graphics.shapes import Drawing, Rect
    d = Drawing(width, height + 2)
    d.add(Rect(0, 1, width, height, fillColor=LINE, strokeColor=None))
    v = max(0.0, min(float(value), 1.0))
    if v > 0:
        d.add(Rect(0, 1, width * v, height,
                   fillColor=RED if v >= .5 else BLUE, strokeColor=None))
    return d


def _boxes_table(boxes, st_) -> Table:
    rows = [[Paragraph("<b>클래스</b>", st_["small"]), Paragraph("<b>신뢰도</b>", st_["small"]),
             Paragraph("<b>위치 (x0, y0) – (x1, y1)</b>", st_["small"])]]
    for name, x0, y0, x1, y1, conf in boxes[:8]:
        rows.append([Paragraph(str(name), st_["body"]),
                     Paragraph(f"{conf:.2f}", st_["body"]),
                     Paragraph(f"({x0:.0f}, {y0:.0f}) – ({x1:.0f}, {y1:.0f})", st_["small"])])
    if not boxes:
        rows.append([Paragraph("탐지된 박스 없음", st_["small"]), "", ""])
    t = Table(rows, colWidths=[24 * mm, 16 * mm, 42 * mm])
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), .6, LINE),
        ("LINEBELOW", (0, 1), (-1, -2), .25, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
    ]))
    return t


def _scores_table(scores: dict, preds, st_) -> Table:
    rows = []
    for label, v in sorted(scores.items(), key=lambda kv: -kv[1]):
        hit = "●" if label in preds else "○"
        rows.append([Paragraph(f'<font color="{RED_DK.hexval() if label in preds else "#94a3b8"}">'
                               f"{hit}</font> {label}", st_["body"]),
                     _bar(v), Paragraph(f"{v:.3f}", st_["small"])])
    t = Table(rows, colWidths=[36 * mm, 36 * mm, 14 * mm], hAlign="LEFT")
    t.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -2), .25, LINE),
    ]))
    return t


def _page_furniture(canvas, doc, font: str, src: str):
    """모든 페이지 상단 붉은 띠 + 하단 푸터."""
    canvas.saveState()
    W, H = A4
    canvas.setFillColor(RED)
    canvas.rect(0, H - 6 * mm, W, 6 * mm, stroke=0, fill=1)      # 상단 액센트 바
    canvas.setFillColor(MUTED)
    canvas.setFont(font, 7.5)
    canvas.drawString(20 * mm, 12 * mm, "웨이퍼 결함 분석 시스템 · MixedWM38 · 자동 생성 리포트")
    canvas.drawRightString(W - 20 * mm, 12 * mm, f"{doc.page} 페이지")
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 15 * mm, W - 20 * mm, 15 * mm)
    canvas.restoreState()


def build_pdf(img, boxes, report_text, proc, preds, scores, src):
    """리포트 PDF (bytes). 시그니처는 기존과 동일."""
    font = _register_font()
    st_ = _styles(font)
    buf = io.BytesIO()

    doc = BaseDocTemplate(buf, pagesize=A4,
                          leftMargin=20 * mm, rightMargin=20 * mm,
                          topMargin=16 * mm, bottomMargin=20 * mm,
                          title="웨이퍼 결함 분석 리포트", author="Wafer Defect Analyzer")
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="main")
    doc.addPageTemplates([PageTemplate(id="all", frames=[frame],
                                       onPage=lambda c, d: _page_furniture(c, d, font, src))])

    conf_vals = [scores[p] for p in preds if p in scores]
    top_conf = max(conf_vals) if conf_vals else None
    now = datetime.datetime.now()

    story = []
    # ---- 표지 헤더 ----
    story.append(Paragraph("웨이퍼 결함 분석 리포트", st_["h1"]))
    story.append(Paragraph(
        f"MixedWM38 기반 복합 결함 패턴 분석 &nbsp;|&nbsp; 생성 {now:%Y-%m-%d %H:%M} "
        f"&nbsp;|&nbsp; 탐지 엔진 {src}", st_["meta"]))
    story.append(Spacer(1, 7))

    # ---- 요약 카드 ----
    story.append(_summary_cards(preds, proc, st_))
    story.append(Spacer(1, 6))
    if top_conf is not None:
        story.append(Table([[Paragraph("분류 신뢰도", st_["small"]), _bar(top_conf, 100 * mm, 5),
                             Paragraph(f"{top_conf:.0%}", st_["body"])]],
                           colWidths=[24 * mm, 104 * mm, 16 * mm], hAlign="LEFT",
                           style=TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                             ("LEFTPADDING", (0, 0), (-1, -1), 0)])))
        story.append(Spacer(1, 8))

    # ---- 1. 결함 탐지 ----
    story.append(Paragraph("1. 결함 탐지", st_["h2"]))
    story.append(Spacer(1, 3))
    wafer_img = _rl_image(_fig_bytes(wafer_report_figure(img, boxes)), 76 * mm)
    right_col = [Paragraph("탐지된 Bounding Box", st_["small"]), Spacer(1, 3),
                 _boxes_table(boxes, st_)]
    story.append(Table([[wafer_img, right_col]], colWidths=[80 * mm, 90 * mm],
                       style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"),
                                         ("LEFTPADDING", (0, 0), (-1, -1), 0)])))
    story.append(Paragraph("웨이퍼 맵 52×52 격자 · 파랑=정상 다이, 빨강=불량 다이, 붉은 사각형=탐지 영역",
                           st_["small"]))
    story.append(Spacer(1, 7))

    # ---- 2. 주파수 분석 ----
    story.append(KeepTogether([
        Paragraph("2. 주파수 분석 (2D FFT → 3D Surface)", st_["h2"]),
        Spacer(1, 3),
        _rl_image(_fig_bytes(fft_report_figure(img)), 98 * mm),
        Paragraph("규칙적인 패턴일수록 특정 주파수 대역에 에너지가 집중됩니다. "
                  "값은 log|FFT| 를 표준화한 것입니다.", st_["small"]),
    ]))
    story.append(Spacer(1, 9))

    # ---- 3. 원인 역추적 ----
    body = "<br/>".join(line if line.strip() else "&nbsp;" for line in report_text.split("\n"))
    box = Table([[Paragraph(body, st_["body"])]], colWidths=[170 * mm])
    box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), .5, LINE),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(KeepTogether([Paragraph("3. 원인 역추적 (Graph-RAG)", st_["h2"]),
                               Spacer(1, 3), box]))
    story.append(Spacer(1, 9))

    # ---- 4. 패턴별 점수 ----
    story.append(KeepTogether([
        Paragraph("4. 패턴별 점수", st_["h2"]),
        Spacer(1, 3),
        _scores_table(scores, preds, st_),
        Spacer(1, 3),
        Paragraph("● = 임계값을 넘어 최종 예측에 포함된 패턴", st_["small"]),
    ]))

    doc.build(story)
    buf.seek(0)
    return buf.getvalue()
