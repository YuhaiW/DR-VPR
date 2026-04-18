"""
ConPR vs ConSLAM query-to-best-positive yaw-difference distribution.

Produces a publication-ready side-by-side histogram that visualizes *why*
rotation robustness matters far more on ConSLAM than on ConPR.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==================== plot style ====================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "legend.fontsize": 9,
    "legend.frameon": False,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,   # TrueType, editor-friendly
    "ps.fonttype": 42,
})

CONPR_COLOR    = "#4C72B0"   # cool blue  → benign
CONSLAM_COLOR  = "#C44E52"   # warm red   → challenging
ANNOT_COLOR    = "#333333"
THRESHOLD_LINE = "#8C8C8C"

# ==================== data: ConPR ====================
DATA_DIR = "./datasets/ConPR"
POSE_DIR = os.path.join(DATA_DIR, "poses")
DB_SEQ = "20230623"
QUERY_SEQS_CP = ["20230531", "20230611", "20230627", "20230628",
                 "20230706", "20230717", "20230803", "20230809", "20230818"]
POS_THRESH_CP = 5.0
YAW_THRESH_CP = 80.0


def yaw_from_pose(p): return np.arctan2(p[4], p[0])
def xy_from_pose(p):  return np.array([p[3], p[7]])


def load(seq, root):
    poses = np.loadtxt(os.path.join(root, f"{seq}.txt"))
    return (np.array([xy_from_pose(p) for p in poses]),
            np.array([yaw_from_pose(p) for p in poses]))


# ---- ConPR ----
db_xy, db_yaw = load(DB_SEQ, POSE_DIR)
conpr_yaw_diffs = []
for seq in QUERY_SEQS_CP:
    q_xy, q_yaw = load(seq, POSE_DIR)
    for i in range(len(q_xy)):
        dists = np.linalg.norm(db_xy - q_xy[i], axis=1)
        pos = np.where(dists <= POS_THRESH_CP)[0]
        if len(pos) == 0:
            continue
        diffs = np.degrees(np.abs(
            (q_yaw[i] - db_yaw[pos] + np.pi) % (2 * np.pi) - np.pi))
        valid = diffs <= YAW_THRESH_CP
        if not np.any(valid):
            continue
        conpr_yaw_diffs.append(diffs[valid].min())
conpr_arr = np.array(conpr_yaw_diffs)

# ---- ConSLAM ----
CS_POSE = "./datasets/ConSLAM/poses"
query_poses = np.loadtxt(os.path.join(CS_POSE, "Sequence4.txt"))
db_poses    = np.loadtxt(os.path.join(CS_POSE, "Sequence5.txt"))
q_xy  = np.array([xy_from_pose(p) for p in query_poses])
q_yaw = np.array([yaw_from_pose(p) for p in query_poses])
db_xy  = np.array([xy_from_pose(p) for p in db_poses])
db_yaw = np.array([yaw_from_pose(p) for p in db_poses])
POS_THRESH_CS = 3.0
conslam_yaw_diffs = []
for i in range(len(q_xy)):
    dists = np.linalg.norm(db_xy - q_xy[i], axis=1)
    pos = np.where(dists <= POS_THRESH_CS)[0]
    if len(pos) == 0:
        continue
    diffs = np.degrees(np.abs(
        (q_yaw[i] - db_yaw[pos] + np.pi) % (2 * np.pi) - np.pi))
    conslam_yaw_diffs.append(diffs.min())
conslam_arr = np.array(conslam_yaw_diffs)

print(f"ConPR   n={len(conpr_arr)}, median={np.median(conpr_arr):.1f}°, "
      f"mean={np.mean(conpr_arr):.1f}°, max={np.max(conpr_arr):.1f}°")
print(f"ConSLAM n={len(conslam_arr)}, median={np.median(conslam_arr):.1f}°, "
      f"mean={np.mean(conslam_arr):.1f}°, max={np.max(conslam_arr):.1f}°")


# ==================== figure ====================
fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=False)

BINS = np.arange(0, 181, 5)
BUCKET_EDGES = [0, 20, 45, 90, 180]
BUCKET_LABELS = ["low\n0-20°", "mid\n20-45°", "high\n45-90°", "extreme\n90-180°"]


def draw_panel(ax, arr, color, title, y_pad):
    counts, edges, patches = ax.hist(
        arr, bins=BINS, color=color, edgecolor="white",
        linewidth=0.7, alpha=0.92)
    ymax = counts.max() * 1.25
    ax.set_ylim(0, ymax)

    # Vertical threshold lines with dashed style
    for x, lbl in zip([20, 45, 90], ["20°", "45°", "90°"]):
        ax.axvline(x, color=THRESHOLD_LINE, linestyle="--",
                   linewidth=0.9, alpha=0.8, zorder=1)
        ax.text(x + 1.2, ymax * 0.965, lbl, color=THRESHOLD_LINE,
                fontsize=8.5, va="top")

    # Bucket percentages along top
    total = len(arr)
    for i in range(len(BUCKET_EDGES) - 1):
        lo, hi = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
        mask = (arr >= lo) & (arr < hi) if i < 3 else (arr >= lo) & (arr <= hi)
        pct = mask.sum() / total * 100
        x_c = (lo + hi) / 2
        ax.text(x_c, ymax * 0.82,
                f"{pct:.1f}%",
                ha="center", va="top", fontsize=10,
                fontweight="bold",
                color="#1a1a1a")
        ax.text(x_c, ymax * 0.74,
                BUCKET_LABELS[i].split("\n")[0],
                ha="center", va="top", fontsize=8.3,
                color="#555555")

    # Stats box (median/mean) top-right
    stats_text = (f"n = {total}\n"
                  f"median = {np.median(arr):.1f}°\n"
                  f"mean   = {np.mean(arr):.1f}°\n"
                  f"max    = {np.max(arr):.1f}°")
    ax.text(0.978, 0.615, stats_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8.8, family="monospace",
            color=ANNOT_COLOR,
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.4", linewidth=0.6))

    ax.set_xlabel("Query-to-best-positive yaw difference (°)")
    ax.set_ylabel("# queries")
    ax.set_title(title)
    ax.set_xlim(0, 180)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


draw_panel(axes[0], conpr_arr, CONPR_COLOR,
           "ConPR  (urban-style, temporal changes)", 1.2)
draw_panel(axes[1], conslam_arr, CONSLAM_COLOR,
           "ConSLAM  (handheld, rotation-heavy)", 1.2)

# A single figure caption-ish overline
fig.suptitle(
    "Yaw-difference distribution: ConPR is rotation-benign; "
    "ConSLAM has 25% of queries with ≥90° rotation",
    fontsize=11.5, y=1.02, fontweight="bold", color="#222222")

fig.tight_layout()
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/yaw_distribution_conpr_vs_conslam.pdf", bbox_inches="tight")
fig.savefig("figures/yaw_distribution_conpr_vs_conslam.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved: figures/yaw_distribution_conpr_vs_conslam.{pdf,png}")
