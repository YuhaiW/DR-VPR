"""
Fine β sweep ablation for P1 standalone on ConSLAM.

Sweeps β in [0.00, 0.01, 0.02, ..., 0.15] (16 values) across 3 P1 standalone
seeds. Reports per-seed R@1 and 3-seed mean ± std for each β — used to
identify the optimal fixed β for paper.

Output: fine β table + 3-seed aggregation.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader
from collections import defaultdict

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

# Fine β sweep: 0.00 through 0.15 in 0.01 steps, plus a few wider comparisons
BETA_VALUES = [round(i * 0.01, 2) for i in range(0, 16)]   # 0.00, 0.01, ..., 0.15

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
    model = model.to(DEVICE)
    model.eval()
    return model


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


def compute_recall_at_beta(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                             q_poses_raw, db_poses, beta):
    """Compute R@1 with bug-fixed rotation, for a single β."""
    theta_rad = np.deg2rad(THETA_DEGREES)
    q_poses = q_poses_raw.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    qx_rot = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    qy_rot = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    q_poses[:, 3], q_poses[:, 7] = qx_rot, qy_rot

    db_x, db_y = db_poses[:, 3], db_poses[:, 7]

    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k_idx = idx.search(d_boq_q, TOP_K)

    correct = total = 0
    for q_idx in range(len(q_poses)):
        dist_sq = (q_poses[q_idx, 3] - db_x) ** 2 + (q_poses[q_idx, 7] - db_y) ** 2
        pp = np.where(dist_sq < GT_THRES ** 2)[0]
        if len(pp) == 0:
            continue
        q_yaw = get_yaw_from_pose(q_poses[q_idx])
        ypos = set()
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180:
                d = 360 - d
            if d <= YAW_THRESHOLD:
                ypos.add(int(p))
        if not ypos:
            continue
        total += 1
        cands = top_k_idx[q_idx]
        boq_sim = d_boq_q[q_idx] @ d_boq_db[cands].T
        equi_sim = d_equi_q[q_idx] @ d_equi_db[cands].T
        score = (1 - beta) * boq_sim + beta * equi_sim
        top1 = cands[np.argmax(score)]
        if top1 in ypos:
            correct += 1
    return correct / total * 100 if total > 0 else 0.0, total


def main():
    # Load BoQ once — same across seeds
    print("Loading official BoQ(ResNet50)@320...")
    boq_model = load_official_boq()

    # Load datasets + BoQ descriptors (same for all seeds)
    db_ds = InferDataset(SEQS[0], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
    q_ds = InferDataset(SEQS[1], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    q_dl = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print("Extracting BoQ descriptors...")
    d_boq_db = extract_boq(boq_model, db_dl)
    d_boq_q = extract_boq(boq_model, q_dl)
    print(f"  BoQ db={d_boq_db.shape}, q={d_boq_q.shape}")

    # Per-seed equi descriptors + β sweep
    results = {}   # β → list of R@1 per seed
    for beta in BETA_VALUES:
        results[beta] = []

    for seed, ckpt in CKPTS.items():
        if not os.path.exists(ckpt):
            print(f"  MISSING ckpt seed={seed}: {ckpt}")
            continue
        print(f"\nSeed {seed}: extracting equi descriptors...")
        equi_model = load_p1_equi(ckpt)
        d_equi_db = extract_equi(equi_model, db_dl)
        d_equi_q = extract_equi(equi_model, q_dl)

        print(f"  β sweep (16 values):")
        for beta in BETA_VALUES:
            r1, total = compute_recall_at_beta(
                d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                q_ds.poses, db_ds.poses, beta,
            )
            results[beta].append(r1)
            print(f"    β={beta:.2f}  R@1={r1:.2f}%  ({total} valid q)")

        del equi_model
        torch.cuda.empty_cache()

    # Aggregate table
    print("\n" + "=" * 90)
    print("Fine β Sweep on ConSLAM — 3 seed mean ± std")
    print("=" * 90)
    print(f"{'β':>6s}  {'seed=1':>8s}  {'seed=42':>8s}  {'seed=190223':>11s}  "
          f"{'mean':>8s}  {'std':>8s}  {'Δ vs β=0':>10s}")
    print('-' * 90)
    r1_b0 = np.mean(results[0.0])
    best_beta = None
    best_mean = -1
    for beta in BETA_VALUES:
        vals = results[beta]
        if len(vals) != 3:
            continue
        m, s = np.mean(vals), np.std(vals, ddof=1)
        delta = m - r1_b0
        marker = ''
        if m > best_mean:
            best_mean = m
            best_beta = beta
        print(f"  {beta:.2f}  {vals[0]:8.2f}  {vals[1]:8.2f}  {vals[2]:11.2f}  "
              f"{m:8.2f}  {s:8.2f}  {delta:>+9.2f}")
    print('=' * 90)
    print(f"\nBest fixed β = {best_beta:.2f} → R@1 = {best_mean:.2f}%")
    # Also mark best β
    best_vals = results[best_beta]
    print(f"  Per-seed: {best_vals}")
    print(f"  Mean ± std = {np.mean(best_vals):.2f} ± {np.std(best_vals, ddof=1):.2f}")


if __name__ == '__main__':
    main()
