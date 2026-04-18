"""
Publication-ready figures for DR-VPR vs baseline comparison.

Figures:
  figures/fig1_recall_comparison.pdf / .png   -- grouped bars on both datasets
  figures/fig2_efficiency.pdf / .png          -- descriptor dim vs R@1 (Pareto)
  figures/fig3_delta_conslam.pdf / .png       -- DR-VPR delta over baselines on ConSLAM
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# -------------------- Data --------------------
# name -> (dim, ConPR R@1/5/10, ConSLAM R@1/5/10)
DATA = {
    "DINOv2":   (768,   (72.10, 76.37, 78.36), (20.00, 41.13, 52.08)),
    "CosPlace": (2048,  (73.49, 78.34, 80.89), (28.30, 44.91, 56.23)),
    "MixVPR":   (4096,  (78.55, 81.52, 83.38), (36.98, 55.47, 60.38)),
    "CricaVPR": (10752, (79.37, 82.60, 84.64), (35.47, 53.96, 61.51)),
    "SALAD":    (8448,  (83.01, 85.92, 87.20), (34.72, 52.83, 62.26)),
    "BoQ":      (12288, (84.61, 86.92, 87.97), (33.96, 54.72, 62.64)),
    "DR-VPR":   (4096,  (79.28, 82.90, 84.46), (55.21, 73.26, 79.17)),   # ours
}

OURS = "DR-VPR"

# -------------------- Style --------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelweight": "regular",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.35,
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Muted baseline palette + bold accent for ours
# Adapted from ColorBrewer Set2 + a bright red for DR-VPR
BASELINE_COLORS = {
    "DINOv2":   "#B4B4B4",   # warm gray
    "CosPlace": "#8C8C8C",
    "MixVPR":   "#4C72B0",   # muted blue
    "CricaVPR": "#55A868",   # muted green
    "SALAD":    "#8172B3",   # muted purple
    "BoQ":      "#C44E52",   # muted red
    "DR-VPR":   "#E8A33D",   # amber — ours
}

# Recall@K shades for grouped bars
K_SHADE = {1: 1.0, 5: 0.72, 10: 0.48}  # alpha-ish modulation

os.makedirs("figures", exist_ok=True)


def shade(hex_color, factor):
    """Lighten a hex color by blending toward white."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * (1 - factor))
    g = int(g + (255 - g) * (1 - factor))
    b = int(b + (255 - b) * (1 - factor))
    return f"#{r:02x}{g:02x}{b:02x}"


# ==========================================================================
# FIGURE 1 — Grouped bar chart, both datasets
# ==========================================================================
# Sort by ConPR R@1 ascending so the trend is left-to-right; DR-VPR placed
# adjacent to MixVPR for direct ablation-style contrast.
order = ["DINOv2", "CosPlace", "MixVPR", "DR-VPR", "CricaVPR", "SALAD", "BoQ"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

bar_w = 0.26
x = np.arange(len(order))

for ax, idx, title, ylim in [
    (axes[0], 1, "ConPR (9 query sequences)", (55, 95)),
    (axes[1], 2, "ConSLAM (Sequence4)",       (10, 85)),
]:
    for j, k in enumerate([1, 5, 10]):
        heights = [DATA[m][idx][j] for m in order]
        colors  = [shade(BASELINE_COLORS[m], K_SHADE[k]) for m in order]
        edgecolors = ["#1a1a1a" if m == OURS else "none" for m in order]
        edgewidths = [1.4 if m == OURS else 0.0 for m in order]
        ax.bar(
            x + (j - 1) * bar_w, heights, bar_w,
            color=colors,
            edgecolor=edgecolors, linewidth=edgewidths,
            label=f"R@{k}",
        )
    # annotate R@1 values above each group
    for i, m in enumerate(order):
        v = DATA[m][idx][0]
        weight = "bold" if m == OURS else "regular"
        color  = "#8B5A00" if m == OURS else "#333333"
        ax.text(x[i] - bar_w, v + (ylim[1]-ylim[0]) * 0.012,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=8, weight=weight, color=color)

    # Highlight OURS background
    ours_idx = order.index(OURS)
    ax.axvspan(ours_idx - 0.5, ours_idx + 0.5, color="#FFE9B8", alpha=0.35,
               zorder=0)

    ax.set_xticks(x)
    labels = [f"$\\bf{{{m}}}$" if m == OURS else m for m in order]
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_ylim(ylim)
    ax.set_ylabel("Recall (%)")
    ax.set_title(title, pad=9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0)

# Build a clean unified legend (three Recall@K levels + ours annotation)
from matplotlib.patches import Patch
legend_handles = [
    Patch(facecolor=shade("#4C72B0", K_SHADE[1]),  edgecolor="none", label="R@1"),
    Patch(facecolor=shade("#4C72B0", K_SHADE[5]),  edgecolor="none", label="R@5"),
    Patch(facecolor=shade("#4C72B0", K_SHADE[10]), edgecolor="none", label="R@10"),
    Patch(facecolor=BASELINE_COLORS[OURS], edgecolor="#1a1a1a",
          linewidth=1.3, label=f"{OURS} (ours)"),
]
axes[0].legend(handles=legend_handles, loc="upper left",
               ncol=4, columnspacing=1.2, handlelength=1.2,
               bbox_to_anchor=(0.0, 1.02), frameon=False)

fig.tight_layout()
fig.savefig("figures/fig1_recall_comparison.pdf")
fig.savefig("figures/fig1_recall_comparison.png", dpi=300)
plt.close(fig)


# ==========================================================================
# FIGURE 2 — Efficiency: dim vs R@1 (two panels)
# ==========================================================================
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

for ax, idx, title, ylim in [
    (axes[0], 1, "ConPR",   (65, 90)),
    (axes[1], 2, "ConSLAM", (15, 60)),
]:
    for m in order:
        d = DATA[m][0]
        r1 = DATA[m][idx][0]
        is_ours = (m == OURS)
        marker = "*" if is_ours else "o"
        size = 380 if is_ours else 130
        edge = "#1a1a1a" if is_ours else "white"
        ew = 1.6 if is_ours else 1.0
        ax.scatter(d, r1, s=size, marker=marker,
                   color=BASELINE_COLORS[m],
                   edgecolor=edge, linewidth=ew,
                   zorder=4 if is_ours else 3, label=m)

    # Annotate each point with method name
    offsets = {
        "ConPR": {
            "DINOv2":   (1.08, 0.6, "left"),
            "CosPlace": (1.08, 0.6, "left"),
            "MixVPR":   (1.08, -1.4, "left"),
            "DR-VPR":   (1.10, 1.3, "left"),
            "CricaVPR": (0.92, -1.2, "right"),
            "SALAD":    (0.92, 0.9, "right"),
            "BoQ":      (0.92, 0.9, "right"),
        },
        "ConSLAM": {
            "DINOv2":   (1.08, -0.3, "left"),
            "CosPlace": (1.08, -0.3, "left"),
            "MixVPR":   (1.08, -0.3, "left"),
            "DR-VPR":   (1.08, 1.5, "left"),
            "CricaVPR": (0.92, -1.0, "right"),
            "SALAD":    (0.92, -1.0, "right"),
            "BoQ":      (1.08, 0.9, "left"),
        },
    }
    name = "ConPR" if idx == 1 else "ConSLAM"
    for m in order:
        d = DATA[m][0]
        r1 = DATA[m][idx][0]
        mx, my, ha = offsets[name][m]
        label = f"$\\bf{{{m}}}$" if m == OURS else m
        color = "#8B5A00" if m == OURS else "#333333"
        weight = "bold" if m == OURS else "regular"
        ax.annotate(label, (d * mx, r1 + my),
                    fontsize=8.5, ha=ha, color=color, weight=weight)

    # Pareto front (upper-left) computed greedily
    pts = sorted([(DATA[m][0], DATA[m][idx][0], m) for m in order])
    best = -np.inf
    pareto = []
    for d, r, m in pts:
        if r > best:
            pareto.append((d, r))
            best = r
    if len(pareto) >= 2:
        pd, pr = zip(*pareto)
        ax.plot(pd, pr, color="#333333", linestyle="--", linewidth=0.8,
                alpha=0.5, zorder=1, label="_nolegend_")

    ax.set_xscale("log")
    ax.set_xlabel("Descriptor dimension (log scale)")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title(title, pad=8)
    ax.set_xlim(500, 20000)
    ax.set_ylim(ylim)
    ax.grid(axis="both", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

fig.suptitle("Efficiency: descriptor size vs. retrieval accuracy",
             fontsize=12.5, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig("figures/fig2_efficiency.pdf")
fig.savefig("figures/fig2_efficiency.png", dpi=300)
plt.close(fig)


# ==========================================================================
# FIGURE 3 — DR-VPR delta over each baseline on ConSLAM (our headline result)
# ==========================================================================
fig, ax = plt.subplots(figsize=(7.2, 3.8))

ours_r1 = DATA[OURS][2][0]
baselines_sorted = sorted(
    [m for m in order if m != OURS],
    key=lambda m: DATA[m][2][0], reverse=True,
)
deltas = [ours_r1 - DATA[m][2][0] for m in baselines_sorted]
y = np.arange(len(baselines_sorted))

bars = ax.barh(
    y, deltas,
    color=[BASELINE_COLORS[m] for m in baselines_sorted],
    edgecolor="#1a1a1a", linewidth=0.7,
)
for yi, d, m in zip(y, deltas, baselines_sorted):
    bl = DATA[m][2][0]
    ax.text(d + 0.4, yi, f"+{d:.2f}  (vs {bl:.2f}%)",
            va="center", ha="left", fontsize=9.5, color="#222222")

ax.axvline(0, color="#1a1a1a", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(baselines_sorted)
ax.invert_yaxis()
ax.set_xlabel("R@1 improvement of DR-VPR over baseline (percentage points)")
ax.set_title(f"ConSLAM: DR-VPR ({ours_r1:.2f}% R@1) vs. baselines",
             pad=8)
ax.set_xlim(0, max(deltas) * 1.35)
ax.grid(axis="x", alpha=0.3, linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)

fig.tight_layout()
fig.savefig("figures/fig3_delta_conslam.pdf")
fig.savefig("figures/fig3_delta_conslam.png", dpi=300)
plt.close(fig)

print("Figures written:")
for f in sorted(os.listdir("figures")):
    if f.startswith("fig") and (f.endswith(".pdf") or f.endswith(".png")):
        print(f"  figures/{f}")
