"""
Model-size comparison figure — DR-VPR v2 vs. baselines.

Two-panel layout:
  (a) Horizontal bar chart of parameter count (M) for each method, sorted
      smallest → largest. DR-VPR highlighted in gold.
  (b) Scatter plot: params (x, log scale) vs R@1 (y) on both ConSLAM and
      ConPR. Marker size encodes per-image latency (ms). DR-VPR shown as a
      starred gold marker — a sweet-spot on the efficiency / accuracy trade-off.

Output: figures/size_comparison.{pdf,png}.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

PROJECT = Path(__file__).resolve().parent
OUT_DIR = PROJECT / "figures"
OUT_DIR.mkdir(exist_ok=True)

# (name, backbone_family, params_M, descriptor_dim, latency_ms, conslam_R1, conpr_R1)
METHODS = [
    ("MixVPR",            "ResNet50",  10.88,  4096, 1.38, 56.03, 78.55),
    ("CosPlace",          "ResNet50",  27.70,  2048, 1.38, 44.30, 73.48),
    ("BoQ (ResNet50)",    "ResNet50",  23.84, 16384, 2.12, 60.91, 79.30),
    ("BoQ (DINOv2)",      "DINOv2",    25.10, 12288, 3.98, 59.93, 84.61),
    ("DINOv2 ViT-B/14",   "DINOv2",    86.58,   768, 5.04, 39.74, 72.10),
    ("CricaVPR",          "DINOv2",   106.76, 10752, 7.71, 57.33, 80.30),
    ("SALAD",             "DINOv2",    87.99,  8448, 8.17, 58.96, 83.01),
    ("DR-VPR v2 (ours)",  "Hybrid",    25.19, 17408, 4.11, 61.89, 79.74),
]

# Colour scheme
C_RESNET  = "#4C72B0"   # blue
C_DINOV2  = "#55A868"   # green
C_OURS    = "#D8A200"   # gold
C_OURS_EDGE = "#8B6D00"

FAM_COLOR = {"ResNet50": C_RESNET, "DINOv2": C_DINOV2, "Hybrid": C_OURS}


def main():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, (ax_bar, ax_sc) = plt.subplots(
        1, 2, figsize=(14, 5.5),
        gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.30}
    )

    # ---------------------- Panel (a): horizontal bar --------------------
    # Sort by params ascending so smallest at top
    sorted_methods = sorted(METHODS, key=lambda m: m[2])
    names = [m[0] for m in sorted_methods]
    params = [m[2] for m in sorted_methods]
    colors = [FAM_COLOR[m[1]] for m in sorted_methods]
    edges = ["black" if m[1] != "Hybrid" else C_OURS_EDGE for m in sorted_methods]
    linewidths = [0.6 if m[1] != "Hybrid" else 1.8 for m in sorted_methods]

    y_pos = np.arange(len(sorted_methods))
    bars = ax_bar.barh(y_pos, params, color=colors, edgecolor=edges, linewidth=linewidths,
                       height=0.7, zorder=3)

    ax_bar.set_yticks(y_pos)
    # Make ours bold
    yticklabels = []
    for m in sorted_methods:
        label = m[0]
        if m[1] == "Hybrid":
            yticklabels.append(label)
        else:
            yticklabels.append(label)
    ax_bar.set_yticklabels(yticklabels, fontsize=10)
    for tick, m in zip(ax_bar.get_yticklabels(), sorted_methods):
        if m[1] == "Hybrid":
            tick.set_fontweight("bold")
            tick.set_color(C_OURS_EDGE)

    ax_bar.set_xlabel("Parameter count (M)", fontsize=10.5)
    ax_bar.set_title("(a) Model size comparison (↑ smaller is better)",
                      fontsize=11.5, fontweight="bold", pad=10)
    ax_bar.grid(axis="x", alpha=0.3, zorder=0)
    ax_bar.set_axisbelow(True)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["top"].set_visible(False)

    # Value labels on the right of each bar
    max_p = max(params)
    for i, (p, m) in enumerate(zip(params, sorted_methods)):
        label = f"{p:.2f} M"
        weight = "bold" if m[1] == "Hybrid" else "normal"
        color = C_OURS_EDGE if m[1] == "Hybrid" else "#333333"
        ax_bar.text(p + max_p * 0.012, i, label, va="center", ha="left",
                    fontsize=9, fontweight=weight, color=color)

    ax_bar.set_xlim(0, max_p * 1.18)

    # Legend for family
    legend_fam = [
        Patch(facecolor=C_RESNET, edgecolor="black", linewidth=0.6, label="ResNet50 backbone"),
        Patch(facecolor=C_DINOV2, edgecolor="black", linewidth=0.6, label="DINOv2 backbone"),
        Patch(facecolor=C_OURS,   edgecolor=C_OURS_EDGE, linewidth=1.8, label="DR-VPR v2 (ours, hybrid)"),
    ]
    ax_bar.legend(handles=legend_fam, loc="lower right", fontsize=9, framealpha=0.95)

    # ---------------------- Panel (b): scatter -------------------------
    # Two overlaid scatters:  ConSLAM R@1 (circles) and ConPR R@1 (squares)
    # x: params (log), y: R@1, marker size ∝ latency
    ax_sc.set_xscale("log")
    ax_sc.set_xlabel("Parameter count (M, log scale)", fontsize=10.5)
    ax_sc.set_ylabel("R@1 (%)", fontsize=10.5)
    ax_sc.set_title("(b) Accuracy vs. size  —  marker size ∝ latency",
                      fontsize=11.5, fontweight="bold", pad=10)
    ax_sc.grid(alpha=0.3, zorder=0)
    ax_sc.set_axisbelow(True)

    # Scale latency to marker size (visually)
    def lat_to_size(lat_ms):
        return 35 + lat_ms * 55

    for (name, fam, p, d, lat, conslam, conpr) in METHODS:
        c = FAM_COLOR[fam]
        # ConSLAM (circle, filled)
        is_ours = (fam == "Hybrid")
        ax_sc.scatter(p, conslam, s=lat_to_size(lat), marker="o" if not is_ours else "*",
                      color=c, edgecolor=C_OURS_EDGE if is_ours else "black",
                      linewidth=1.8 if is_ours else 0.6, alpha=0.85, zorder=4 if is_ours else 3)
        # ConPR (square, open/hatched) — same method, lighter look to distinguish
        ax_sc.scatter(p, conpr, s=lat_to_size(lat),
                      marker="s" if not is_ours else "P",
                      facecolor="none" if not is_ours else c,
                      edgecolor=C_OURS_EDGE if is_ours else c,
                      linewidth=1.8 if is_ours else 1.2, alpha=0.9, zorder=4 if is_ours else 3)

        # Annotate method name once (near ConSLAM point)
        # pick offset direction dynamically
        if name == "DR-VPR v2 (ours)":
            offset = (15, 10); fw = "bold"; fc = C_OURS_EDGE
        elif name == "SALAD":
            offset = (-20, -15); fw = "normal"; fc = "#333333"
        elif name == "CricaVPR":
            offset = (-18, 14); fw = "normal"; fc = "#333333"
        elif name == "BoQ (DINOv2)":
            offset = (10, -12); fw = "normal"; fc = "#333333"
        elif name == "MixVPR":
            offset = (-22, -10); fw = "normal"; fc = "#333333"
        elif name == "BoQ (ResNet50)":
            offset = (-30, 11); fw = "normal"; fc = "#333333"
        elif name == "DINOv2 ViT-B/14":
            offset = (6, -14); fw = "normal"; fc = "#333333"
        elif name == "CosPlace":
            offset = (6, -14); fw = "normal"; fc = "#333333"
        else:
            offset = (8, 8); fw = "normal"; fc = "#333333"

        ax_sc.annotate(name, xy=(p, conslam), xytext=offset, textcoords="offset points",
                       fontsize=8.5, fontweight=fw, color=fc,
                       arrowprops=None, zorder=6)

    ax_sc.set_ylim(35, 90)
    ax_sc.set_xlim(8, 200)
    ax_sc.set_xticks([10, 25, 50, 100, 200])
    ax_sc.set_xticklabels(["10", "25", "50", "100", "200"])

    # Custom legend
    legend_ds = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888888", markeredgecolor="black",
               markersize=10, label="ConSLAM R@1 (θ=15°)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="white", markeredgecolor="#888888",
               markersize=10, label="ConPR R@1 (θ=0°)"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor=C_OURS, markeredgecolor=C_OURS_EDGE,
               markersize=16, markeredgewidth=1.5, label="DR-VPR (ours) — ConSLAM"),
        Line2D([0], [0], marker="P", color="w", markerfacecolor=C_OURS, markeredgecolor=C_OURS_EDGE,
               markersize=14, markeredgewidth=1.5, label="DR-VPR (ours) — ConPR"),
    ]
    ax_sc.legend(handles=legend_ds, loc="lower right", fontsize=8.8, framealpha=0.95)

    # Marker-size legend (latency)
    ax_sc.text(9, 86, "Marker size ∝ latency (ms)", fontsize=8.5, style="italic", color="#555")
    for lat_ref, x_ref in zip([1.5, 4.0, 8.0], [10, 14, 20]):
        ax_sc.scatter(x_ref, 83, s=lat_to_size(lat_ref), marker="o",
                      facecolor="#cccccc", edgecolor="#555", linewidth=0.6, alpha=0.6)
        ax_sc.text(x_ref, 80.5, f"{lat_ref:.1f} ms", fontsize=7.8, ha="center", color="#555")

    fig.suptitle("DR-VPR v2 vs. baseline VPR methods — model size, accuracy, latency",
                  fontsize=13, fontweight="bold", y=1.02)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"size_comparison.{ext}"
        fig.savefig(out, dpi=260, bbox_inches="tight")
        print(f"[size] wrote {out}")

    plt.close(fig)


if __name__ == "__main__":
    main()
