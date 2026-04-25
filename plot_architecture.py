"""
DR-VPR v2 architecture diagram — clean, publication-quality, single-panel.

Horizontal left-to-right flow matching the original submitted figure's layout:
  Input image → [Discriminative Branch: ResNet-50 → BoQ] → d_1
              → [Equivariant Branch:   E2ResNet C16 → MultiScale GeM] → d_2
                                                        → Joint scoring (β=0.10)
                                                        → top-1 match

Nature / Cell-style palette, minimalist, no "before/after" annotations.

Output: figures/dr_vpr_v2_architecture.{pdf,png}.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------------
# Nature/Cell-style palette (muted, professional)
# ------------------------------------------------------------------------
C_DISC_FILL  = "#DCE6F4"; C_DISC_EDGE  = "#2E5395"   # Branch 1 blue
C_EQUI_FILL  = "#F8DEC4"; C_EQUI_EDGE  = "#C25A2E"   # Branch 2 warm orange
C_BOQ_FILL   = "#FCEDC0"; C_BOQ_EDGE   = "#B8860B"   # BoQ aggregator mustard
C_MS_FILL    = "#DAEDD9"; C_MS_EDGE    = "#2E7D32"   # Multi-scale head sage
C_FUSE_FILL  = "#E5DAEF"; C_FUSE_EDGE  = "#6A4C9C"   # Joint scoring purple
C_DESC_FILL  = "#F4EDD8"; C_DESC_EDGE  = "#9A7F00"   # Descriptor gold
C_OUT_FILL   = "#DAEDD9"; C_OUT_EDGE   = "#2E7D32"   # Output sage
C_TEXT       = "#222222"
C_ARROW      = "#333333"


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------
def box(ax, x, y, w, h, text, fill, edge, fontsize=9.0, fontweight="normal",
        text_color=None, lw=1.1, rounded=True):
    boxstyle = "round,pad=0.03,rounding_size=0.10" if rounded else "square,pad=0.03"
    p = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color or C_TEXT, zorder=4)


def arrow(ax, x1, y1, x2, y2, lw=1.2, color=None):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle="-|>", mutation_scale=13,
                         linewidth=lw, color=color or C_ARROW, zorder=2)
    ax.add_patch(a)


def branch_label(ax, x, y, text, color):
    """Small italic branch label."""
    ax.text(x, y, text, ha="left", va="center", fontsize=8.5,
            fontstyle="italic", color=color, fontweight="bold", zorder=4)


def branch_group(ax, x, y, w, h, edge_color):
    """Dashed group box surrounding a branch."""
    p = Rectangle((x, y), w, h, linewidth=1.2, edgecolor=edge_color,
                  facecolor="none", linestyle=(0, (4, 3)), zorder=1, alpha=0.7)
    ax.add_patch(p)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(14.5, 5.5))
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 12)
    ax.set_aspect("equal")
    ax.axis("off")

    # ----- Input image box (placeholder glyph) -----
    box(ax, 0.5, 4.6, 3.0, 2.8, "Image$_i$",
        fill="#F2F2F2", edge="#444444", fontsize=11)

    # ----- Branch group boxes (dashed surround) -----
    branch_group(ax, 4.6, 7.6, 14.4, 3.6, C_DISC_EDGE)
    branch_group(ax, 4.6, 1.2, 14.4, 3.6, C_EQUI_EDGE)
    branch_label(ax, 5.0, 11.5, "Discriminative Branch", C_DISC_EDGE)
    branch_label(ax, 5.0, 0.7,  "Equivariant Branch",    C_EQUI_EDGE)

    # =============================
    # Discriminative Branch (top)
    # =============================
    box(ax, 5.3, 8.4, 5.8, 2.0,
        "ResNet-50\nBackbone",
        fill=C_DISC_FILL, edge=C_DISC_EDGE, fontsize=10)

    box(ax, 12.0, 8.4, 5.8, 2.0,
        "BoQ\nAggregator",
        fill=C_BOQ_FILL, edge=C_BOQ_EDGE, fontsize=10)

    # descriptor d_1 output
    box(ax, 19.5, 8.7, 3.2, 1.4,
        "d$_1 \\in \\mathbb{R}^{16384}$",
        fill=C_DESC_FILL, edge=C_DESC_EDGE, fontsize=10, fontweight="bold")

    # internal arrows Disc
    arrow(ax, 3.6, 6.0, 5.2, 9.4)
    arrow(ax, 11.2, 9.4, 11.95, 9.4)
    arrow(ax, 17.85, 9.4, 19.45, 9.4)

    # =============================
    # Equivariant Branch (bottom)
    # =============================
    box(ax, 5.3, 2.0, 5.8, 2.0,
        "E2ResNet\nC$_{16}$ Backbone",
        fill=C_EQUI_FILL, edge=C_EQUI_EDGE, fontsize=10)

    box(ax, 12.0, 2.0, 5.8, 2.0,
        "Multi-scale\nGroupPool + GeM",
        fill=C_MS_FILL, edge=C_MS_EDGE, fontsize=10)

    box(ax, 19.5, 2.3, 3.2, 1.4,
        "d$_2 \\in \\mathbb{R}^{1024}$",
        fill=C_DESC_FILL, edge=C_DESC_EDGE, fontsize=10, fontweight="bold")

    arrow(ax, 3.6, 6.0, 5.2, 3.0)
    arrow(ax, 11.2, 3.0, 11.95, 3.0)
    arrow(ax, 17.85, 3.0, 19.45, 3.0)

    # ----- Input → branches (single branching arrow) -----
    # (image → split point)
    # we already drew individual arrows from image to each backbone

    # =============================
    # Joint scoring + output
    # =============================
    # Joint scoring block (centre right)
    jx, jy, jw, jh = 24.0, 4.6, 6.0, 3.0
    box(ax, jx, jy, jw, jh,
        "Joint Scoring  (β = 0.10)\n\n"
        "s(q, c) = 0.9·⟨d$_1$(q), d$_1$(c)⟩\n"
        "        + 0.1·⟨d$_2$(q), d$_2$(c)⟩\n\n"
        "top-1 = argmax$_c$ s(q, c)",
        fill=C_FUSE_FILL, edge=C_FUSE_EDGE, fontsize=9)

    # arrows from descriptors into joint scoring
    arrow(ax, 22.75, 9.4, 24.0, 7.4)
    arrow(ax, 22.75, 3.0, 24.0, 4.8)

    # final arrow out of joint scoring → F(Image_i)
    arrow(ax, 30.0, 6.1, 31.2, 6.1)
    ax.text(31.3, 6.1, r"$\mathcal{F}($Image$_i)$",
            ha="left", va="center", fontsize=11, fontweight="bold",
            color=C_TEXT)

    # ----- Save -----
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"dr_vpr_v2_architecture.{ext}"
        fig.savefig(out, dpi=280, bbox_inches="tight")
        print(f"[arch] wrote {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
