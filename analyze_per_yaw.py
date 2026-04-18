"""
Per-yaw-bucket analysis of ConPR retrieval performance.

Reads the per-query diagnosis CSVs produced by evaluateResults() (one per
method, tagged with method_name) and computes R@1/R@5/R@10 within each
yaw-difficulty bucket. Produces publication-ready line + bar plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Config --------------------
YAW_THRESHOLD = 80.0
BUCKETS = [(0, 20, "Low\n(0–20°)"),
           (20, 45, "Mid\n(20–45°)"),
           (45, 80, "High\n(45–80°)")]
METHODS = ["DINOv2", "CosPlace", "MixVPR", "DR-VPR",
           "CricaVPR", "SALAD", "BoQ"]   # order left→right
OURS = "DR-VPR"

COLOR = {
    "DINOv2":   "#B4B4B4",
    "CosPlace": "#8C8C8C",
    "MixVPR":   "#4C72B0",
    "CricaVPR": "#55A868",
    "SALAD":    "#8172B3",
    "BoQ":      "#C44E52",
    "DR-VPR":   "#E8A33D",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "xtick.labelsize": 10,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Map method name -> CSV naming (evaluateResults writes lower-case of whatever
# we pass as method_name; eval_baselines passes upper-case SALAD etc.)
CSV_NAME = {
    "DINOv2":   f"diagnosis_matrix_conpr_yaw{YAW_THRESHOLD}_DINOV2.csv",
    "CosPlace": f"diagnosis_matrix_conpr_yaw{YAW_THRESHOLD}_COSPLACE.csv",
    "MixVPR":   f"diagnosis_matrix_conpr_yaw{YAW_THRESHOLD}_MIXVPR.csv",
    "CricaVPR": f"diagnosis_matrix_conpr_yaw{YAW_THRESHOLD}_CRICAVPR.csv",
    "SALAD":    f"diagnosis_matrix_conpr_yaw{YAW_THRESHOLD}_SALAD.csv",
    "BoQ":      f"diagnosis_matrix_conpr_yaw{YAW_THRESHOLD}_BOQ.csv",
    "DR-VPR":   f"diagnosis_matrix_conpr_yaw{YAW_THRESHOLD}_DR-VPR.csv",
}


def load(method):
    path = CSV_NAME[method]
    if not os.path.exists(path):
        print(f"  MISSING: {path}")
        return None
    df = pd.read_csv(path)
    # Only count queries that had at least one valid positive
    df = df[df["num_positives"] > 0].copy()
    return df


def bucket_of(yaw):
    if yaw < 20:   return 0
    if yaw < 45:   return 1
    return 2


def main():
    # Collect per-method, per-bucket R@1/5/10
    # rows = method, cols = (bucket, k) -> recall
    records = {}
    bucket_sizes = None

    for m in METHODS:
        df = load(m)
        if df is None:
            continue
        df["bucket"] = df["yaw_diff_min_to_pos"].apply(bucket_of)
        per = {}
        sizes = {}
        for b, (lo, hi, _) in enumerate(BUCKETS):
            sub = df[df["bucket"] == b]
            n = len(sub)
            sizes[b] = n
            if n == 0:
                per[b] = (np.nan, np.nan, np.nan)
                continue
            r1 = sub["success"].mean() * 100
            r5 = sub["success_5"].mean() * 100
            r10 = sub["success_10"].mean() * 100
            per[b] = (r1, r5, r10)
        records[m] = per
        if bucket_sizes is None:
            bucket_sizes = sizes
        print(f"  {m:10s}: "
              + "  ".join(
                  f"B{b} R@1={per[b][0]:5.1f}% R@5={per[b][1]:5.1f}% "
                  f"R@10={per[b][2]:5.1f}%"
                  for b in range(3)))

    if not records:
        print("No data loaded — nothing to plot.")
        return

    bucket_labels = [f"{lbl}\n(n={bucket_sizes[b]})"
                     for b, (_, _, lbl) in enumerate(BUCKETS)]

    os.makedirs("figures", exist_ok=True)

    # =====================================================
    # FIG 4 — Line plot of R@1 across buckets
    # =====================================================
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(BUCKETS))

    for m in METHODS:
        if m not in records:
            continue
        y = [records[m][b][0] for b in range(len(BUCKETS))]
        if m == OURS:
            ax.plot(x, y, marker="*", markersize=18, linewidth=2.6,
                    color=COLOR[m], markeredgecolor="#1a1a1a",
                    markeredgewidth=1.2, label=f"{m} (ours)", zorder=5)
        else:
            ax.plot(x, y, marker="o", markersize=7, linewidth=1.5,
                    color=COLOR[m], label=m, alpha=0.92, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("Query-to-positive yaw difference bucket")
    ax.set_ylabel("Recall@1 (%)")
    ax.set_title("ConPR: R@1 breakdown by rotation difficulty")
    ax.legend(loc="lower left", ncol=2, columnspacing=1.0, handlelength=1.8)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig("figures/fig4_per_yaw_r1.pdf")
    fig.savefig("figures/fig4_per_yaw_r1.png", dpi=300)
    plt.close(fig)

    # =====================================================
    # FIG 5 — Grouped bars: R@1/R@5/R@10 per bucket
    # (3 subplots, one per bucket)
    # =====================================================
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    n = len(METHODS)
    xx = np.arange(n)
    bar_w = 0.27

    for bi, (lo, hi, title) in enumerate(BUCKETS):
        ax = axes[bi]
        r1 = [records[m][bi][0] if m in records else np.nan for m in METHODS]
        r5 = [records[m][bi][1] if m in records else np.nan for m in METHODS]
        r10 = [records[m][bi][2] if m in records else np.nan for m in METHODS]

        def sh(h, f):
            hc = h.lstrip("#")
            r, g, b = int(hc[0:2], 16), int(hc[2:4], 16), int(hc[4:6], 16)
            r = int(r + (255 - r) * (1 - f))
            g = int(g + (255 - g) * (1 - f))
            b = int(b + (255 - b) * (1 - f))
            return f"#{r:02x}{g:02x}{b:02x}"

        colors_r1  = [sh(COLOR[m], 1.00) for m in METHODS]
        colors_r5  = [sh(COLOR[m], 0.72) for m in METHODS]
        colors_r10 = [sh(COLOR[m], 0.48) for m in METHODS]
        edges = ["#1a1a1a" if m == OURS else "none" for m in METHODS]
        widths = [1.3 if m == OURS else 0.0 for m in METHODS]

        ax.bar(xx - bar_w, r1, bar_w, color=colors_r1,
               edgecolor=edges, linewidth=widths, label="R@1")
        ax.bar(xx,          r5, bar_w, color=colors_r5,
               edgecolor=edges, linewidth=widths, label="R@5")
        ax.bar(xx + bar_w, r10, bar_w, color=colors_r10,
               edgecolor=edges, linewidth=widths, label="R@10")

        # Highlight ours background
        ours_idx = METHODS.index(OURS)
        ax.axvspan(ours_idx - 0.5, ours_idx + 0.5,
                   color="#FFE9B8", alpha=0.35, zorder=0)

        # R@1 numeric labels
        for i, m in enumerate(METHODS):
            v = r1[i]
            if np.isnan(v):
                continue
            wt = "bold" if m == OURS else "regular"
            ax.text(xx[i] - bar_w, v + 1.2, f"{v:.1f}",
                    ha="center", va="bottom",
                    fontsize=7.5, weight=wt,
                    color="#8B5A00" if m == OURS else "#333333")

        ax.set_xticks(xx)
        labels = [f"$\\bf{{{m}}}$" if m == OURS else m for m in METHODS]
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_title(f"{title}  —  n={bucket_sizes[bi]}")
        ax.set_ylim(0, 100)
        if bi == 0:
            ax.set_ylabel("Recall (%)")
        ax.tick_params(axis="x", length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.3)

    axes[0].legend(loc="upper right", ncol=3, columnspacing=0.9,
                   handlelength=1.2, fontsize=8.5)
    fig.suptitle("ConPR: per-method Recall broken down by yaw-difficulty bucket",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig("figures/fig5_per_yaw_bars.pdf")
    fig.savefig("figures/fig5_per_yaw_bars.png", dpi=300)
    plt.close(fig)

    # -----------------------------------------------
    # Dump a CSV summary
    # -----------------------------------------------
    rows = []
    for m in METHODS:
        if m not in records:
            continue
        for b in range(len(BUCKETS)):
            r1, r5, r10 = records[m][b]
            rows.append({
                "method": m,
                "bucket": BUCKETS[b][2].split("\n")[0],
                "yaw_range": f"{BUCKETS[b][0]}-{BUCKETS[b][1]}",
                "n": bucket_sizes[b],
                "R@1": r1, "R@5": r5, "R@10": r10,
            })
    summary = pd.DataFrame(rows)
    summary.to_csv("per_yaw_bucket_summary.csv", index=False)
    print("\nSaved: per_yaw_bucket_summary.csv")
    print("Figures: figures/fig4_per_yaw_r1.{pdf,png}")
    print("         figures/fig5_per_yaw_bars.{pdf,png}")


if __name__ == "__main__":
    main()
