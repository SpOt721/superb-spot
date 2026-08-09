"""
시각화.

화면용은 plotly(인터랙티브), PDF용은 matplotlib(정적) — plotly 정적 변환에 필요한
kaleido 없이도 리포트가 나오도록 두 벌을 유지한다.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

import models                            # noqa: E402
from .theme import C_BG, C_DEFECT, C_NORMAL, FFT_H, PLOT_H   # noqa: E402

GRID = 52                                # 웨이퍼 맵 격자


def _fft_map(img: Image.Image) -> np.ndarray:
    m = models.image_to_mask(img, GRID).astype(float)
    return np.log1p(np.fft.fftshift(np.abs(np.fft.fft2(m))))


# ---------------- 화면용 (plotly) ----------------
def wafer_figure(img: Image.Image, boxes=(), height: int | None = None):
    """F-02: 웨이퍼 맵(52 격자) + Bounding Box."""
    m = models.image_to_mask(img, GRID)
    W, H = img.size
    fig = go.Figure(go.Heatmap(
        z=m, zmin=0, zmax=2, showscale=False,
        colorscale=[[0.0, C_BG], [0.33, C_BG], [0.33, C_NORMAL],
                    [0.66, C_NORMAL], [0.66, C_DEFECT], [1.0, C_DEFECT]],
        hovertemplate="x=%{x} y=%{y}<extra></extra>"))
    sx, sy = GRID / max(W, 1), GRID / max(H, 1)          # 원본 px → 52 격자
    for name, x0, y0, x1, y1, conf in boxes:
        fig.add_shape(type="rect", x0=x0 * sx, y0=y0 * sy, x1=x1 * sx, y1=y1 * sy,
                      line=dict(color="#f87171", width=2))
        fig.add_annotation(x=x0 * sx, y=y1 * sy, text=f"{name} {conf:.2f}",
                           showarrow=False, xanchor="left", yanchor="bottom",
                           font=dict(size=11, color="#ffffff"),
                           bgcolor="rgba(239,68,68,.85)", borderpad=2)
    fig.update_layout(height=height or PLOT_H, margin=dict(l=0, r=0, t=6, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    # 마스크의 0행과 박스 좌표는 둘 다 '위 = 작은 값' 규약이다.
    # plotly 는 기본적으로 0행을 아래에 그리므로 뒤집어야 업로드 이미지와 방향이 같아진다.
    fig.update_yaxes(scaleanchor="x", scaleratio=1, autorange="reversed")
    return fig


def fft_surface_3d(img: Image.Image, height: int | None = None):
    """F-03: 2D FFT → 3D Surface."""
    F = _fft_map(img)
    Z = (F - F.mean()) / (F.std() + 1e-9)
    fig = go.Figure(go.Surface(z=Z, colorscale="Viridis",
                               colorbar=dict(thickness=14, len=.7)))
    fig.update_layout(height=height or FFT_H, margin=dict(l=0, r=0, t=6, b=0),
                      paper_bgcolor="rgba(0,0,0,0)",
                      scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z",
                                 aspectratio=dict(x=1, y=1, z=.55)))
    return fig


# ---------------- 리포트(PDF)용 — 화면과 같은 팔레트로 렌더 ----------------
def wafer_report_figure(img: Image.Image, boxes=()):
    """화면의 웨이퍼 맵과 동일한 모양을 정적 이미지로 (PDF 삽입용)."""
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as patches

    m = models.image_to_mask(img, GRID)
    W, H = img.size
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=200)
    # origin='upper' — 0행이 위. 박스 좌표도 '위 = 작은 값'이라 방향이 일치한다.
    ax.imshow(m, cmap=ListedColormap([C_BG, C_NORMAL, C_DEFECT]), vmin=0, vmax=2,
              origin="upper", interpolation="nearest")
    sx, sy = GRID / max(W, 1), GRID / max(H, 1)
    show_labels = len(boxes) <= 6            # 박스가 많으면 라벨이 서로 겹쳐 읽을 수 없다
    for name, x0, y0, x1, y1, conf in boxes:
        ax.add_patch(patches.Rectangle((x0 * sx, y0 * sy), (x1 - x0) * sx, (y1 - y0) * sy,
                                       fill=False, edgecolor="#f87171", lw=1.4))
        if show_labels:
            # 박스가 위쪽 끝에 붙어 있으면 라벨을 안쪽으로 넣어 잘리지 않게 한다
            top = y0 * sy
            inside = top < 3
            ax.text(x0 * sx, top + (0.6 if inside else -0.8), f"{name} {conf:.2f}",
                    color="white", fontsize=6, va="top" if inside else "bottom",
                    bbox=dict(facecolor="#ef4444", pad=1.2, edgecolor="none"))
    ax.set_xlim(-.5, GRID - .5); ax.set_ylim(GRID - .5, -.5)
    ax.tick_params(labelsize=6, colors="#64748b")
    for s in ax.spines.values():
        s.set_color("#cbd5e1")
    fig.tight_layout(pad=.3)
    return fig


def fft_report_figure(img: Image.Image):
    """화면의 3D 주파수 표면과 동일한 정규화/색으로 (PDF 삽입용)."""
    F = _fft_map(img)
    Z = (F - F.mean()) / (F.std() + 1e-9)
    fig = plt.figure(figsize=(4.6, 3.0), dpi=200)
    ax = fig.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid(np.arange(Z.shape[1]), np.arange(Z.shape[0]))
    surf = ax.plot_surface(xx, yy, Z, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("x", fontsize=6); ax.set_ylabel("y", fontsize=6); ax.set_zlabel("z", fontsize=6)
    ax.tick_params(labelsize=5, colors="#64748b")
    ax.set_box_aspect((1, 1, .55))
    fig.colorbar(surf, ax=ax, shrink=.55, aspect=14, pad=.02).ax.tick_params(labelsize=5)
    # 3D 축은 기본 여백이 커서 PDF 에 큰 빈 공간이 생긴다 — 바짝 붙인다
    fig.subplots_adjust(left=.02, right=.90, bottom=.02, top=.98)
    return fig


# ---------------- 하위 호환 (기존 시그니처 유지) ----------------
def draw_boxes(img: Image.Image, boxes):
    """F-02: 원본 위에 빨간 박스."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img); ax.axis("off")
    import matplotlib.patches as patches
    for name, x0, y0, x1, y1, conf in boxes:
        ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                     fill=False, edgecolor="#ef4444", lw=2.2))
        ax.text(x0, max(y0 - 4, 0), f"{name} {conf:.2f}", color="white",
                fontsize=8, bbox=dict(facecolor="#ef4444", pad=1, edgecolor="none"))
    fig.tight_layout()
    return fig


def freq_surface(img: Image.Image):
    """F-03: 2D FFT 크기 스펙트럼 3D surface (정적)."""
    F = _fft_map(img)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    xx, yy = np.meshgrid(np.arange(F.shape[1]), np.arange(F.shape[0]))
    ax.plot_surface(xx, yy, F, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_title("3D Frequency Spectrum (log|FFT|)", fontsize=9)
    ax.set_xlabel("u"); ax.set_ylabel("v"); ax.set_zlabel("mag")
    fig.tight_layout()
    return fig
