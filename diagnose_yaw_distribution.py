"""
Diagnostic: distribution of yaw_diff between each ConPR query and its
nearest (by 2D position) GT positive in the 20230623 database.

Gives us:
  - Histogram of yaw_diff across all queries
  - Bucket counts (0-20, 20-45, 45-80)
  - Per-sequence bucket counts
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "./datasets/ConPR"
POSE_DIR = os.path.join(DATA_DIR, "poses")
DB_SEQ = "20230623"
QUERY_SEQS = ["20230531", "20230611", "20230627", "20230628",
              "20230706", "20230717", "20230803", "20230809", "20230818"]
POS_THRESHOLD_M = 5.0
YAW_THRESHOLD_DEG = 80.0


def yaw_from_pose(pose):
    # pose row: 12 floats, R3x3 in row-major stored at [0:3, 4:7, 8:11], t at [3, 7, 11]
    return np.arctan2(pose[4], pose[0])   # atan2(R[1,0], R[0,0])


def position_from_pose(pose):
    return np.array([pose[3], pose[7]])   # (tx, ty), 2D


def load_seq(seq):
    f = os.path.join(POSE_DIR, f"{seq}.txt")
    poses = np.loadtxt(f)
    yaws = np.array([yaw_from_pose(p) for p in poses])     # radians
    xy = np.array([position_from_pose(p) for p in poses])  # (N, 2)
    return xy, yaws


def circ_diff_deg(a, b):
    # shortest angular difference, in degrees, in [0, 180]
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return np.abs(np.degrees(d))


def main():
    db_xy, db_yaw = load_seq(DB_SEQ)
    print(f"Database {DB_SEQ}: {len(db_xy)} frames")

    all_yaw_diffs = []
    per_seq_counts = {}

    for seq in QUERY_SEQS:
        q_xy, q_yaw = load_seq(seq)
        n = len(q_xy)

        # Vectorized 2D distance matrix (N_q x N_db)
        d2 = np.sum((q_xy[:, None, :] - db_xy[None, :, :]) ** 2, axis=2)
        dists = np.sqrt(d2)
        mask = dists <= POS_THRESHOLD_M   # positives gate

        # For each query, compute yaw_diff to each positive and keep the
        # smallest one (i.e. the best-aligned positive).
        # After that we further apply yaw_threshold to replicate eval protocol.
        yaw_d_all = np.degrees(np.abs(
            ((q_yaw[:, None] - db_yaw[None, :]) + np.pi) % (2 * np.pi) - np.pi
        ))  # (N_q, N_db) in [0, 180]

        kept = 0
        seq_counts = [0, 0, 0]  # [0-20), [20-45), [45-80]
        for i in range(n):
            pos_idx = np.where(mask[i])[0]
            if len(pos_idx) == 0:
                continue
            # Among position-positives, apply yaw threshold (same as eval)
            pos_yaw_d = yaw_d_all[i, pos_idx]
            valid = pos_yaw_d <= YAW_THRESHOLD_DEG
            if not np.any(valid):
                continue
            # Best-aligned positive (smallest yaw diff) — the "easiest" match
            best_y = np.min(pos_yaw_d[valid])
            all_yaw_diffs.append(best_y)
            kept += 1
            if best_y < 20:
                seq_counts[0] += 1
            elif best_y < 45:
                seq_counts[1] += 1
            else:
                seq_counts[2] += 1

        per_seq_counts[seq] = (n, kept, seq_counts)
        print(f"  {seq}: total={n:>4}  eligible={kept:>4}  "
              f"[0-20°={seq_counts[0]:>4}  20-45°={seq_counts[1]:>4}  "
              f"45-80°={seq_counts[2]:>4}]")

    arr = np.asarray(all_yaw_diffs)
    print(f"\nTotal eligible queries (all 9 seqs): {len(arr)}")
    print(f"  0-20°  : {np.sum(arr < 20):>5}  ({np.mean(arr < 20)*100:.1f}%)")
    print(f"  20-45° : {np.sum((arr >= 20) & (arr < 45)):>5}  "
          f"({np.mean((arr >= 20) & (arr < 45))*100:.1f}%)")
    print(f"  45-80° : {np.sum(arr >= 45):>5}  "
          f"({np.mean(arr >= 45)*100:.1f}%)")
    print(f"  median : {np.median(arr):.2f}°")
    print(f"  mean   : {np.mean(arr):.2f}°")
    print(f"  max    : {np.max(arr):.2f}°")

    # Quick plot
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    bins = np.arange(0, 85, 2.5)
    ax.hist(arr, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.6)
    for edge, label in [(20, "20°"), (45, "45°")]:
        ax.axvline(edge, color="#C44E52", linestyle="--", linewidth=1)
        ax.text(edge + 0.5, ax.get_ylim()[1] * 0.95, label,
                color="#C44E52", fontsize=9)
    ax.set_xlabel("Yaw difference to nearest positive (degrees)")
    ax.set_ylabel("# queries")
    ax.set_title("ConPR: distribution of query-to-best-positive yaw difference")
    fig.tight_layout()
    fig.savefig("figures/diag_yaw_distribution.png", dpi=200, bbox_inches="tight")
    fig.savefig("figures/diag_yaw_distribution.pdf", bbox_inches="tight")
    print("\nWrote figures/diag_yaw_distribution.{png,pdf}")


if __name__ == "__main__":
    main()
