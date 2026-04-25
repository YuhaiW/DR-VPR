"""
Branch 1 (Appearance Encoder) detail diagram — BoQ aggregator expanded.

Structure mirrors Ali-Bey et al. (2024) Figure 2:
  Input -> ResNet-50 Backbone -> x^{i-1} -> Transformer Encoder -> x^i
                                                                    |
  Learned Bag of Queries -> Self-Attn -> Cross-Attn <----------------+
                                             |
                                             v
                                             o^i      (x L)
  o^1,...,o^L -> Concat -> Linear -> L2-norm -> d_1 in R^{16384}

Palette matches the main architecture figure (dr_vpr_architecture.png):
  muted blue-gray for the ResNet-50 backbone, mustard for BoQ, sage/gold
  for descriptor, serif font, rounded rectangles.

Output: figures/dr_vpr_branch1.{pdf,png}.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# Palette — matches the main figure (Image #3)
C_BACKBONE_FILL  = "#7A8CA8"
C_BACKBONE_EDGE  = "#42526E"

C_BOQ_FILL       = "#D5A740"    # mustard (matches BoQ block in main fig)
C_BOQ_EDGE       = "#8B6A1E"

C_BOQ_LIGHT_FILL = "#F3DFA5"    # lighter mustard for Self-/Cross-Attn sub-blocks
C_BOQ_LIGHT_EDGE = "#B88D2E"

C_QUERIES_FILL   = "#FCF6E5"    # very light cream for queries box
C_QUERIES_EDGE   = "#B88D2E"

C_ENC_FILL       = "#E8EEF5"    # light blue for Transformer Encoder
C_ENC_EDGE       = "#42526E"

C_DESC_FILL      = "#E5B948"    # gold (matches descriptor pills in main fig)
C_DESC_EDGE      = "#8B6A1E"

C_GROUP_EDGE     = "#8B6A1E"
C_TEXT           = "#222222"
C_ARROW          = "#2E2E2E"

# Muted pastel palette for learned-query dots
C_QUERY_DOTS = [
    "#E8A6B4", "#9EC0E3", "#F1CE7E",
    "#B7D4A7", "#CEB0DD", "#F1A683",
]


def box(ax, x, y, w, h, text, fill, edge, fontsize=10, fontweight="normal",
        text_color=None, lw=1.3, rounded=True):
    boxstyle = "round,pad=0.02,rounding_size=0.16" if rounded else "square,pad=0.02"
    p = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                       linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color or C_TEXT, zorder=4)


def arrow(ax, x1, y1, x2, y2, lw=1.3, color=None, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=13,
                        linewidth=lw, color=color or C_ARROW, zorder=2)
    ax.add_patch(a)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(16, 5.2))
    ax.set_xlim(0, 33)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Input + Backbone
    # ------------------------------------------------------------------
    box(ax, 0.3, 3.6, 2.6, 2.6, "Input\n(Image$_i$)",
        fill="#F4F4F4", edge="#555555", fontsize=10)

    box(ax, 3.5, 3.8, 3.4, 2.4, "ResNet-50\nBackbone",
        fill=C_BACKBONE_FILL, edge=C_BACKBONE_EDGE,
        fontsize=10.5, text_color="white", fontweight="bold")
    # freeze snowflake — placed above the backbone box
    import numpy as _np
    sx_c, sy_c, sr = 5.20, 6.70, 0.18
    for th_deg in (0, 60, 120):
        th = _np.deg2rad(th_deg)
        dx, dy = sr * _np.cos(th), sr * _np.sin(th)
        ax.plot([sx_c - dx, sx_c + dx], [sy_c - dy, sy_c + dy],
                color=C_BACKBONE_EDGE, linewidth=1.9, solid_capstyle="round",
                zorder=6)
    ax.text(sx_c + 0.45, sy_c, "frozen", fontsize=8.5, fontstyle="italic",
            color=C_BACKBONE_EDGE, ha="left", va="center", zorder=6)

    # ------------------------------------------------------------------
    # BoQ Aggregator group (x L)
    # ------------------------------------------------------------------
    gx, gy, gw, gh = 7.8, 1.0, 16.2, 8.2
    group = Rectangle((gx, gy), gw, gh, linewidth=1.4,
                      edgecolor=C_GROUP_EDGE, facecolor="none",
                      linestyle=(0, (4, 3)), zorder=1, alpha=0.85)
    ax.add_patch(group)
    ax.text(gx + 0.3, gy + gh - 0.45, "BoQ Aggregator",
            ha="left", va="center", fontsize=11, fontweight="bold",
            fontstyle="italic", color=C_BOQ_EDGE, zorder=5)
    ax.text(gx + gw - 0.7, gy + gh - 0.45, r"$\times\, L$",
            ha="right", va="center", fontsize=12.5, fontweight="bold",
            color=C_BOQ_EDGE, zorder=5)

    # Transformer Encoder (top path)
    ex, ey, ew, eh = gx + 2.4, 5.9, 4.3, 1.9
    box(ax, ex, ey, ew, eh, "Transformer\nEncoder",
        fill=C_ENC_FILL, edge=C_ENC_EDGE, fontsize=10.5, fontweight="bold")

    # x^{i-1}
    ax.text(ex - 0.35, ey + eh/2 + 0.25, r"$\mathbf{x}^{i-1}$",
            ha="right", va="center", fontsize=11.5,
            fontweight="bold", color=C_TEXT)
    # x^{i}
    ax.text(ex + ew + 0.35, ey + eh/2 + 0.25, r"$\mathbf{x}^{i}$",
            ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=C_TEXT)

    # Learned Bag of Queries — caption on top to avoid clipping by group box
    qx, qy, qw, qh = gx + 0.8, 2.1, 1.9, 1.7
    queries = Rectangle((qx, qy), qw, qh, linewidth=1.2,
                        edgecolor=C_QUERIES_EDGE, facecolor=C_QUERIES_FILL,
                        linestyle=(0, (3, 2)), zorder=3)
    ax.add_patch(queries)
    for i, col in enumerate(C_QUERY_DOTS):
        row = i // 3
        cm = i % 3
        cx = qx + 0.40 + cm * 0.52
        cy = qy + 0.40 + row * 0.60
        c = Circle((cx, cy), 0.17, facecolor=col, edgecolor="#555555",
                   linewidth=0.6, zorder=4)
        ax.add_patch(c)
    ax.text(qx + qw/2, qy + qh + 0.35, "Learned Bag of Queries",
            ha="center", va="center", fontsize=9, fontstyle="italic",
            fontweight="bold", color=C_BOQ_EDGE)

    # Self-Attn
    sx, sy, sw, sh = qx + qw + 0.9, 2.3, 2.3, 1.8
    box(ax, sx, sy, sw, sh, "Self-Attn",
        fill=C_BOQ_LIGHT_FILL, edge=C_BOQ_LIGHT_EDGE,
        fontsize=10.5, fontweight="bold")

    # Cross-Attn
    cx, cy, cw, ch = sx + sw + 1.0, 2.3, 2.3, 1.8
    box(ax, cx, cy, cw, ch, "Cross-Attn",
        fill=C_BOQ_LIGHT_FILL, edge=C_BOQ_LIGHT_EDGE,
        fontsize=10.5, fontweight="bold")

    # o^i
    ax.text(cx + cw + 0.35, cy + ch/2, r"$\mathbf{o}^{i}$",
            ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=C_TEXT)

    # ------------------------------------------------------------------
    # Arrows (inside group)
    # ------------------------------------------------------------------
    # Input -> backbone
    arrow(ax, 2.9, 5.0, 3.5, 5.0, lw=1.4)
    # backbone -> x^{i-1} (into group) -> Encoder
    arrow(ax, 6.9, 5.4, ex, ey + eh/2 + 0.25, lw=1.4)
    # queries -> Self-Attn
    arrow(ax, qx + qw, qy + qh/2 - 0.2, sx, sy + sh/2)
    # Self-Attn -> Cross-Attn (carries Q)
    arrow(ax, sx + sw, sy + sh/2, cx, cy + ch/2)
    # Encoder bottom -> Cross-Attn (KV feed, downward)
    arrow(ax, ex + ew - 0.8, ey, cx + cw/2, cy + ch, lw=1.2)
    # Cross-Attn -> o^i
    arrow(ax, cx + cw, cy + ch/2, cx + cw + 0.3, cy + ch/2)

    # small q / k / v labels on the arrows into Cross-Attn
    ax.text(sx + sw + 0.5, sy + sh/2 + 0.30, "q", fontsize=8.5,
            fontstyle="italic", color="#555555", ha="center")
    ax.text(cx + cw/2 + 0.55, cy + ch + 0.45, "k, v", fontsize=8.5,
            fontstyle="italic", color="#555555", ha="center")

    # ------------------------------------------------------------------
    # Outside group: Concat -> Linear -> L2-norm -> d_1
    # ------------------------------------------------------------------
    # bring o^i out of group
    arrow(ax, cx + cw + 0.5, cy + ch/2, gx + gw - 0.0, cy + ch/2 + 0.0, lw=1.4)

    # Concat
    kx, ky, kw, kh = gx + gw + 0.4, 3.4, 2.7, 2.2
    box(ax, kx, ky, kw, kh,
        "Concat\n$(\\mathbf{o}^{1},\\ldots,\\mathbf{o}^{L})$",
        fill=C_BOQ_FILL, edge=C_BOQ_EDGE, fontsize=9.5,
        text_color="white", fontweight="bold")

    # Linear
    lx, ly, lw2, lh = kx + kw + 0.4, 3.4, 1.7, 2.2
    box(ax, lx, ly, lw2, lh, "Linear",
        fill=C_BOQ_FILL, edge=C_BOQ_EDGE, fontsize=10.5,
        text_color="white", fontweight="bold")

    # L2-norm
    nx, ny, nw, nh = lx + lw2 + 0.4, 3.4, 1.9, 2.2
    box(ax, nx, ny, nw, nh, r"$L_2$-norm",
        fill=C_BOQ_FILL, edge=C_BOQ_EDGE, fontsize=10.5,
        text_color="white", fontweight="bold")

    arrow(ax, kx + kw, ky + kh/2, lx, ly + lh/2)
    arrow(ax, lx + lw2, ly + lh/2, nx, ny + nh/2)

    # d_1 descriptor
    arrow(ax, nx + nw, ny + nh/2, nx + nw + 0.8, ny + nh/2)
    ax.text(nx + nw + 0.95, ny + nh/2,
            r"$\mathbf{d}_{1} \in \mathbb{R}^{16\,384}$",
            ha="left", va="center", fontsize=11.5,
            fontweight="bold", color=C_TEXT)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"dr_vpr_branch1.{ext}"
        fig.savefig(out, dpi=280, bbox_inches="tight")
        print(f"[branch1] wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
