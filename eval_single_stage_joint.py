"""
Single-stage joint scoring ablation  —  ConSLAM, BoQ(ResNet50) + P1 equi.

Compares two retrieve-rerank protocols at matched β values:

  (A) two-stage  (current main):
      stage-1: FAISS IndexFlatIP(desc_boq_DB).search(desc_boq_q, k=100)
      stage-2: score(c) = (1-β)·boq_sim(q,c) + β·equi_sim(q,c) over top-100
               top-1 = argmax over the 100 candidates
      Bottleneck: any true positive NOT in BoQ top-100 is unrecoverable
                  (Limitation L8 — stage-1 recall ceiling).

  (B) single-stage joint:
      No stage-1 filter. Compute score(c) = (1-β)·boq_sim(q,c) + β·equi_sim(q,c)
      over ALL N db descriptors. top-1 = argmax over the full db.
      Cost: O(N) instead of O(100) — fine for ConSLAM db=401 but matters
            for larger db.

Hypothesis: (B) ≥ (A) because (B) considers candidates that may sit outside
BoQ top-100 but win on the joint score (e.g., queries with > 30° yaw whose
true positive is excluded by stage-1).

Setup: ConSLAM Sequence5 db (401) vs Sequence4 query (307 valid), θ=15°,
yaw=80°. 3 P1 standalone seeds. β sweep {0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}.
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset, get_yaw_from_pose
from models.equi_multiscale import E2ResNetMultiScale

DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
THETA_DEGREES = 15.0
DATASET_PATH = './datasets/ConSLAM/'
SEQS = ['Sequence5', 'Sequence4']
BETA_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

CKPTS = {
    1:      'LOGS/equi_standalone_seed1_ms/lightning_logs/version_0/checkpoints/equi_ms_seed1_epoch(07)_R1[0.3359].ckpt',
    42:     'LOGS/equi_standalone_seed42_ms/lightning_logs/version_0/checkpoints/equi_ms_seed42_epoch(01)_R1[0.3157].ckpt',
    190223: 'LOGS/equi_standalone_seed190223_ms/lightning_logs/version_0/checkpoints/equi_ms_seed190223_epoch(08)_R1[0.3359].ckpt',
}


def load_official_boq():
    model = torch.hub.load(
        "amaralibey/bag-of-queries", "get_trained_boq",
        backbone_name="resnet50", output_dim=16384,
    )
    return model.eval().to(DEVICE)


def load_p1_equi(ckpt_path):
    model = E2ResNetMultiScale(
        orientation=8, layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512), out_dim=1024,
    )
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_boq(model, dl):
    descs = []
    for imgs, _ in dl:
        out = model(imgs.to(DEVICE))
        if isinstance(out, tuple):
            out = out[0]
        descs.append(F.normalize(out, p=2, dim=1).cpu().numpy())
    return np.vstack(descs)


@torch.no_grad()
def extract_equi(model, dl):
    descs = []
    for imgs, _ in dl:
        descs.append(model(imgs.to(DEVICE)).cpu().numpy())
    return np.vstack(descs)


def rotate_query_poses(q_poses_raw, theta_degrees):
    theta_rad = np.deg2rad(theta_degrees)
    q = q_poses_raw.copy()
    qx, qy = q[:, 3], q[:, 7]
    qx_rot = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    qy_rot = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    q[:, 3], q[:, 7] = qx_rot, qy_rot
    return q


def precompute_valid(q_poses, db_poses, gt_thres, yaw_threshold):
    """Return list of (q_idx, ypos_set) for queries with at least one valid positive."""
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    valid = []
    for q_idx in range(len(q_poses)):
        dist_sq = (q_poses[q_idx, 3] - db_x) ** 2 + (q_poses[q_idx, 7] - db_y) ** 2
        pp = np.where(dist_sq < gt_thres ** 2)[0]
        if len(pp) == 0:
            continue
        q_yaw = get_yaw_from_pose(q_poses[q_idx])
        ypos = set()
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180:
                d = 360 - d
            if d <= yaw_threshold:
                ypos.add(int(p))
        if ypos:
            valid.append((q_idx, ypos))
    return valid


def eval_two_stage(d_boq_db, d_boq_q, d_equi_db, d_equi_q, valid, beta, top_k):
    """Stage-1 BoQ FAISS top-K → Stage-2 weighted rerank within top-K."""
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k_idx = idx.search(d_boq_q, top_k)
    correct = 0
    for q_idx, ypos in valid:
        cands = top_k_idx[q_idx]
        if beta == 0.0:
            top1 = cands[0]
        else:
            boq_sim = d_boq_q[q_idx] @ d_boq_db[cands].T
            equi_sim = d_equi_q[q_idx] @ d_equi_db[cands].T
            score = (1 - beta) * boq_sim + beta * equi_sim
            top1 = cands[np.argmax(score)]
        if top1 in ypos:
            correct += 1
    return correct / len(valid) * 100


def eval_single_stage(d_boq_db, d_boq_q, d_equi_db, d_equi_q, valid, beta):
    """Single-stage joint scoring over ALL db descriptors (no FAISS pre-filter)."""
    # Pre-compute full sim matrices (small enough: 396 × 401 here)
    boq_sim_full = d_boq_q @ d_boq_db.T
    equi_sim_full = d_equi_q @ d_equi_db.T
    score_full = (1 - beta) * boq_sim_full + beta * equi_sim_full
    correct = 0
    for q_idx, ypos in valid:
        top1 = int(np.argmax(score_full[q_idx]))
        if top1 in ypos:
            correct += 1
    return correct / len(valid) * 100


def main():
    print("=" * 100)
    print("Single-stage joint scoring vs. two-stage retrieve-rerank ablation")
    print("ConSLAM Sequence5 db vs Sequence4 query, θ=15°, yaw=80°")
    print(f"3 seeds × β sweep {BETA_VALUES}")
    print("=" * 100)

    # Load BoQ once
    print("\nLoading BoQ(ResNet50)@320...")
    boq = load_official_boq()

    db_ds = InferDataset(SEQS[0], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
    q_ds  = InferDataset(SEQS[1], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    q_dl  = DataLoader(q_ds,  batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    print("Extracting BoQ descriptors...")
    d_boq_db = extract_boq(boq, db_dl)
    d_boq_q  = extract_boq(boq, q_dl)
    print(f"  db {d_boq_db.shape}  q {d_boq_q.shape}")

    # Pre-compute valid query set (rotation + yaw filter)
    q_poses = rotate_query_poses(q_ds.poses, THETA_DEGREES)
    db_poses = db_ds.poses
    valid = precompute_valid(q_poses, db_poses, GT_THRES, YAW_THRESHOLD)
    n_valid = len(valid)
    print(f"  {n_valid} valid queries (with at least one yaw-valid positive)")

    # Diagnostic: at how many of those queries is the true positive INSIDE BoQ top-100?
    # (This bounds two-stage performance: if not in top-100, two-stage cannot recover)
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k_idx = idx.search(d_boq_q, TOP_K)
    n_pos_in_top100 = sum(1 for q_idx, ypos in valid
                           if any(int(c) in ypos for c in top_k_idx[q_idx]))
    print(f"  diagnostic: BoQ top-{TOP_K} contains a true positive for "
          f"{n_pos_in_top100}/{n_valid} queries "
          f"({n_pos_in_top100 / n_valid * 100:.2f}% recall@{TOP_K})")
    print(f"  → two-stage R@1 ceiling: {n_pos_in_top100 / n_valid * 100:.2f}%")

    # Per-seed eval
    results_two = {b: [] for b in BETA_VALUES}     # β → list per seed
    results_single = {b: [] for b in BETA_VALUES}
    for seed, ckpt_path in CKPTS.items():
        if not os.path.exists(ckpt_path):
            print(f"\n  MISSING ckpt seed={seed}"); continue
        print(f"\n[Seed {seed}] extracting equi from {os.path.basename(ckpt_path)}...")
        eq = load_p1_equi(ckpt_path)
        d_eq_db = extract_equi(eq, db_dl)
        d_eq_q  = extract_equi(eq, q_dl)
        del eq; torch.cuda.empty_cache()

        for beta in BETA_VALUES:
            r_two = eval_two_stage(d_boq_db, d_boq_q, d_eq_db, d_eq_q, valid, beta, TOP_K)
            r_one = eval_single_stage(d_boq_db, d_boq_q, d_eq_db, d_eq_q, valid, beta)
            results_two[beta].append(r_two)
            results_single[beta].append(r_one)
            print(f"  β={beta:.2f}  two-stage={r_two:6.2f}%  single-stage={r_one:6.2f}%  "
                  f"Δ(single − two)={r_one - r_two:+.2f}")

    # Aggregate
    print(f"\n{'=' * 100}")
    print("3-seed mean ± std R@1 — Two-stage vs Single-stage joint scoring")
    print(f"{'=' * 100}")
    print(f"{'β':>5s}  {'two-stage mean':>16s}  {'std':>5s}  "
          f"{'single-stage mean':>18s}  {'std':>5s}  {'Δ(single − two)':>15s}")
    print('-' * 100)
    for beta in BETA_VALUES:
        v2 = results_two[beta]
        v1 = results_single[beta]
        if len(v2) < 2: continue
        m2, s2 = np.mean(v2), np.std(v2, ddof=1)
        m1, s1 = np.mean(v1), np.std(v1, ddof=1)
        d = m1 - m2
        print(f"{beta:5.2f}  {m2:16.2f}  {s2:5.2f}  {m1:18.2f}  {s1:5.2f}  {d:>+14.2f}")
    print('=' * 100)

    # Best β each protocol
    best_two_beta = max(BETA_VALUES, key=lambda b: np.mean(results_two[b]) if len(results_two[b]) >= 2 else -1)
    best_one_beta = max(BETA_VALUES, key=lambda b: np.mean(results_single[b]) if len(results_single[b]) >= 2 else -1)
    print(f"\nBest β per protocol:")
    print(f"  two-stage  : β={best_two_beta:.2f}  R@1={np.mean(results_two[best_two_beta]):.2f}")
    print(f"  single-stage : β={best_one_beta:.2f}  R@1={np.mean(results_single[best_one_beta]):.2f}")
    print(f"  ceiling (BoQ recall@100): {n_pos_in_top100 / n_valid * 100:.2f}%")


if __name__ == '__main__':
    main()
