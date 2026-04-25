"""
Branch 2 (Equivariant Encoder) detail diagram — C16 multi-scale,
compact layout:
  - Input photo ABOVE Stem (arrow down into Stem)
  - layer1 + layer2 merged into a single "layer 1-2" block
  - Linear + L2-norm merged into a single block
  - d_2 pill drops DOWN below the merged Linear+L2 block
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.image import imread
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "figures"
OUT_DIR.mkdir(exist_ok=True)

INPUT_IMG_PATH = Path(
    "/home/yuhai/project/DR-VPR/datasets/ConPR/"
    "20230628/Camera_matched/1687941979599136512.png"
)

# Palette — coordinated with main architecture figure (green = output);
# each colour family has intra-family tonal variation (lighter → deeper)
# Red / terracotta family (backbone)
C_BB_FILL    = "#A94E3C"   # brick terracotta (layer blocks)
C_BB_EDGE    = "#5A2014"
C_STEM_FILL  = "#C77766"   # lighter clay (stem)
C_STEM_EDGE  = "#5A2014"
# Gold / ochre family (GroupPool — deeper, more sophisticated mustard-brown)
C_GP_FILL    = "#AE7F22"   # deeper dusty ochre (GroupPool)
C_GP_EDGE    = "#5A3E0A"
# Olive-sage bridge between ochre GroupPool and green Concat
C_GEM_FILL   = "#8CA866"   # olive sage (GeM pool — bridges GroupPool → Concat)
C_GEM_EDGE   = "#3E4A22"
# Green / sage family — Concat → Linear+L2 → d_2
C_CAT_FILL   = "#519475"   # mid sage (Concat — fusion)
C_CAT_EDGE   = "#1C4434"
C_LIN_FILL   = "#64A681"   # medium sage (Linear + L2-norm)
C_LIN_EDGE   = "#1C4434"
C_OUT_FILL   = "#BED9BB"   # light sage (d_2 — matches main's output tone)
C_OUT_EDGE   = "#2E7D32"
C_TEXT       = "#1A1A1A"
C_ARROW      = "#2A2A2A"
C_DIM_TEXT   = "#4A4A4A"


def shadow(ax, x, y, w, h, offset_x=0.08, offset_y=-0.10, alpha=0.18,
           rounding=0.16):
    p = FancyBboxPatch((x + offset_x, y + offset_y), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                       linewidth=0, edgecolor="none",
                       facecolor=(0.15, 0.15, 0.15, alpha),
                       zorder=2.8)
    ax.add_patch(p)


def box(ax, x, y, w, h, text, fill, edge, fontsize=10, fontweight="normal",
        text_color=None, lw=1.3, rounded=True, with_shadow=True):
    if with_shadow:
        shadow(ax, x, y, w, h)
    boxstyle = "round,pad=0.02,rounding_size=0.16" if rounded else "square,pad=0.02"
    p = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                       linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight,
            color=text_color or C_TEXT, zorder=4)


def arrow(ax, x1, y1, x2, y2, lw=1.4, color=None, style="-|>",
          connection="arc3,rad=0"):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle=style, mutation_scale=14,
                        linewidth=lw, color=color or C_ARROW, zorder=2,
                        connectionstyle=connection)
    ax.add_patch(a)


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stix",
        "mathtext.rm": "serif",
    })

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.2, 25.6)
    ax.set_ylim(0.3, 14.8)
    ax.set_aspect("equal")
    ax.axis("off")

    # ------------------------------------------------------------------
    # Row y-levels (bottom → top)
    # ------------------------------------------------------------------
    Y_BB_BOT, Y_BB_TOP = 2.0, 4.6          # backbone row (bottom)
    Y_GP_BOT, Y_GP_TOP = 6.3, 8.6          # GroupPool row
    Y_GEM_BOT, Y_GEM_TOP = 10.5, 13.0      # top merge row (GeM + Concat + Linear+L2)
    Y_GEM_MID = (Y_GEM_BOT + Y_GEM_TOP) / 2

    # ------------------------------------------------------------------
    # Backbone chain (bottom row): Stem -> [layer 1-2] -> layer3 -> layer4
    # ------------------------------------------------------------------
    stem_x, stem_w = 1.0, 2.6
    box(ax, stem_x, Y_BB_BOT, stem_w, Y_BB_TOP - Y_BB_BOT,
        "Stem",
        fill=C_STEM_FILL, edge=C_STEM_EDGE, fontsize=15,
        text_color="white", fontweight="bold")

    # merged layer 1-2
    l12_x = stem_x + stem_w + 0.55
    l12_w = 3.0
    box(ax, l12_x, Y_BB_BOT, l12_w, Y_BB_TOP - Y_BB_BOT,
        "layer 1\u20132",
        fill=C_BB_FILL, edge=C_BB_EDGE, fontsize=15,
        text_color="white", fontweight="bold")

    # layer3
    l3_x = l12_x + l12_w + 0.55
    l3_w = 2.8
    box(ax, l3_x, Y_BB_BOT, l3_w, Y_BB_TOP - Y_BB_BOT,
        "layer3",
        fill=C_BB_FILL, edge=C_BB_EDGE, fontsize=15,
        text_color="white", fontweight="bold")

    # layer4 (wider gap before it so GP columns don't touch)
    l4_x = l3_x + l3_w + 1.3
    l4_w = 2.8
    box(ax, l4_x, Y_BB_BOT, l4_w, Y_BB_TOP - Y_BB_BOT,
        "layer4",
        fill=C_BB_FILL, edge=C_BB_EDGE, fontsize=15,
        text_color="white", fontweight="bold")

    l3_cx = l3_x + l3_w / 2
    l4_cx = l4_x + l4_w / 2

    # C_16 indicator below backbone (centred on layer3-layer4 area)
    ax.text((l3_x + l4_x + l4_w) / 2, Y_BB_BOT - 0.75,
            r"$C_{16}$: 16 discrete rotations at $22.5^{\circ}$ intervals",
            ha="center", va="center", fontsize=14, fontstyle="italic",
            fontweight="bold", color=C_BB_EDGE)

    # ------------------------------------------------------------------
    # Input image (ABOVE Stem, square) - arrow down into Stem top
    # ------------------------------------------------------------------
    img_side = 4.0
    img_cx = stem_x + stem_w / 2
    img_x0 = img_cx - img_side / 2
    img_x1 = img_cx + img_side / 2
    img_y0 = Y_BB_TOP + 1.1
    img_y1 = img_y0 + img_side
    shadow(ax, img_x0, img_y0, img_side, img_side, rounding=0.10)
    if INPUT_IMG_PATH.exists():
        input_img = imread(str(INPUT_IMG_PATH))
        ax.imshow(input_img,
                  extent=[img_x0, img_x1, img_y0, img_y1],
                  aspect='auto', zorder=3.2, origin='upper')
    border = FancyBboxPatch((img_x0, img_y0), img_side, img_side,
                             boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.4, edgecolor="#555555",
                             facecolor="none", zorder=3.5)
    ax.add_patch(border)
    ax.text(img_cx, img_y1 + 0.35,
            "Input",
            ha="center", va="bottom", fontsize=16, fontstyle="italic",
            fontweight="bold", color="#333333")

    # arrow down from image bottom into Stem top
    arrow(ax, img_cx, img_y0, img_cx, Y_BB_TOP, lw=1.6)

    # ------------------------------------------------------------------
    # GroupPool row (above layer3 / layer4)
    # ------------------------------------------------------------------
    gp_w = 3.6
    gp3_x = l3_cx - gp_w / 2
    gp4_x = l4_cx - gp_w / 2
    box(ax, gp3_x, Y_GP_BOT, gp_w, Y_GP_TOP - Y_GP_BOT,
        "GroupPool\n(max over $C_{16}$)",
        fill=C_GP_FILL, edge=C_GP_EDGE, fontsize=13,
        text_color="white", fontweight="bold")
    box(ax, gp4_x, Y_GP_BOT, gp_w, Y_GP_TOP - Y_GP_BOT,
        "GroupPool\n(max over $C_{16}$)",
        fill=C_GP_FILL, edge=C_GP_EDGE, fontsize=13,
        text_color="white", fontweight="bold")

    # ------------------------------------------------------------------
    # GeM row + Concat + (Linear + L2-norm) merged — top horizontal row
    # ------------------------------------------------------------------
    gem_w = gp_w
    gem3_x = gp3_x
    gem4_x = gp4_x
    box(ax, gem3_x, Y_GEM_BOT, gem_w, Y_GEM_TOP - Y_GEM_BOT,
        "GeM pool\n$\\rightarrow \\mathbb{R}^{16}$",
        fill=C_GEM_FILL, edge=C_GEM_EDGE, fontsize=14,
        text_color="white", fontweight="bold")
    box(ax, gem4_x, Y_GEM_BOT, gem_w, Y_GEM_TOP - Y_GEM_BOT,
        "GeM pool\n$\\rightarrow \\mathbb{R}^{32}$",
        fill=C_GEM_FILL, edge=C_GEM_EDGE, fontsize=14,
        text_color="white", fontweight="bold")

    # Tensor shape annotations next to each vertical arrow
    SH_FS = 16
    ax.text(l3_cx + 0.22, (Y_BB_TOP + Y_GP_BOT) / 2,
            r"$\mathbb{R}^{256 \times 20 \times 20}$",
            ha="left", va="center", fontsize=SH_FS, fontstyle="italic",
            fontweight="bold", color=C_DIM_TEXT)
    ax.text(l3_cx + 0.22, (Y_GP_TOP + Y_GEM_BOT) / 2,
            r"$\mathbb{R}^{16 \times 20 \times 20}$",
            ha="left", va="center", fontsize=SH_FS, fontstyle="italic",
            fontweight="bold", color=C_DIM_TEXT)
    ax.text(l4_cx + 0.22, (Y_BB_TOP + Y_GP_BOT) / 2,
            r"$\mathbb{R}^{512 \times 10 \times 10}$",
            ha="left", va="center", fontsize=SH_FS, fontstyle="italic",
            fontweight="bold", color=C_DIM_TEXT)
    ax.text(l4_cx + 0.22, (Y_GP_TOP + Y_GEM_BOT) / 2,
            r"$\mathbb{R}^{32 \times 10 \times 10}$",
            ha="left", va="center", fontsize=SH_FS, fontstyle="italic",
            fontweight="bold", color=C_DIM_TEXT)

    # Concat
    cat_x = gem4_x + gem_w + 1.5
    cat_w = 3.5
    box(ax, cat_x, Y_GEM_BOT, cat_w, Y_GEM_TOP - Y_GEM_BOT,
        "Concat\n$(16 + 32 = 48)$",
        fill=C_CAT_FILL, edge=C_CAT_EDGE, fontsize=13.5,
        text_color="white", fontweight="bold")

    # Merged Linear + L2-norm
    lin_x = cat_x + cat_w + 0.6
    lin_w = 4.2
    box(ax, lin_x, Y_GEM_BOT, lin_w, Y_GEM_TOP - Y_GEM_BOT,
        "Linear $48 \\rightarrow 1024$\n$+\\ L_2$-norm",
        fill=C_LIN_FILL, edge=C_LIN_EDGE, fontsize=13.5,
        text_color="white", fontweight="bold")

    # d_2 pill — BELOW the merged Linear+L2 block
    pill_w = 4.4
    pill_h = 1.6
    pill_x = lin_x + (lin_w - pill_w) / 2
    pill_y = Y_GEM_BOT - 2.2
    box(ax, pill_x, pill_y, pill_w, pill_h,
        r"$\mathbf{d}_{2} \in \mathbb{R}^{1024}$",
        fill=C_OUT_FILL, edge=C_OUT_EDGE, fontsize=16,
        fontweight="bold")

    # ------------------------------------------------------------------
    # Arrows
    # ------------------------------------------------------------------
    y_bb_mid = (Y_BB_BOT + Y_BB_TOP) / 2

    # Backbone horizontal flow: Stem -> layer 1-2 -> layer3 -> layer4
    arrow(ax, stem_x + stem_w, y_bb_mid, l12_x, y_bb_mid)
    arrow(ax, l12_x + l12_w, y_bb_mid, l3_x, y_bb_mid)
    arrow(ax, l3_x + l3_w, y_bb_mid, l4_x, y_bb_mid)

    # Vertical taps from layer3 / layer4 -> GroupPool
    arrow(ax, l3_cx, Y_BB_TOP, l3_cx, Y_GP_BOT)
    arrow(ax, l3_cx, Y_GP_TOP, l3_cx, Y_GEM_BOT)
    arrow(ax, l4_cx, Y_BB_TOP, l4_cx, Y_GP_BOT)
    arrow(ax, l4_cx, Y_GP_TOP, l4_cx, Y_GEM_BOT)

    # GeM3 -> Concat: Z-shape over the top
    br_y = Y_GEM_TOP + 1.0
    x_up = gem3_x + gem_w / 2
    x_down = cat_x + cat_w * 0.32
    ax.plot([x_up, x_up], [Y_GEM_TOP, br_y],
            color=C_ARROW, lw=1.4, zorder=2, solid_capstyle="round")
    ax.plot([x_up, x_down], [br_y, br_y],
            color=C_ARROW, lw=1.4, zorder=2, solid_capstyle="round")
    arrow(ax, x_down, br_y, x_down, Y_GEM_TOP + 0.05)

    # GeM4 -> Concat: perfectly horizontal
    arrow(ax, gem4_x + gem_w, Y_GEM_MID, cat_x, Y_GEM_MID)

    # Concat -> Linear+L2 (horizontal)
    arrow(ax, cat_x + cat_w, Y_GEM_MID, lin_x, Y_GEM_MID)

    # Linear+L2 -> d_2 (downward)
    arrow(ax, lin_x + lin_w / 2, Y_GEM_BOT,
          lin_x + lin_w / 2, pill_y + pill_h, lw=1.6)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    for ext in ("pdf", "png"):
        out = OUT_DIR / f"dr_vpr_branch2.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[branch2] wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
