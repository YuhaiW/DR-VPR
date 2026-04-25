"""
Full 10-sequence ConPR two-stage rerank evaluation for DR-VPR v2.

Iterates over all 10 ConPR sequences (Seq '20230623' as database, 9 others as queries),
computes BoQ top-100 retrieval + β-weighted rerank, then averages R@1 across all 9
query-db pairs. Runs per-seed (3 P1 standalone ckpts) and reports 3-seed mean ± std
at β=0.00 (pure BoQ) and β=0.10 (DR-VPR v2 paper setting).

Output: per-pair R@1 × 3 seeds × 2 β values + aggregated table.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from conpr_eval_dataset_rot import InferDataset, get_yaw_from_pose
from models.equi_multiscale import E2ResNetMultiScale

DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
THETA_DEGREES = 0.0  # ConPR: no query rotation
DATASET_PATH = './datasets/ConPR/'
SEQUENCES = ['20230623', '20230531', '20230611', '20230627', '20230628',
             '20230706', '20230717', '20230803', '20230809', '20230818']
BETA_VALUES = [0.00, 0.10]

CKPTS = {
    1:      'LOGS/equi_standalone_seed1_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed1_epoch(08)_R1[0.3510].ckpt',
    42:     'LOGS/equi_standalone_seed42_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed42_epoch(01)_R1[0.3283].ckpt',
    190223: 'LOGS/equi_standalone_seed190223_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed190223_epoch(04)_R1[0.3384].ckpt',
}


def load_official_boq():
    model = torch.hub.load(
        "amaralibey/bag-of-queries", "get_trained_boq",
        backbone_name="resnet50", output_dim=16384,
    )
    return model.eval().to(DEVICE)


def load_p1_equi(ckpt_path):
    model = E2ResNetMultiScale(
        orientation=16, layers=(2, 2, 2, 2),
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
                             q_poses_raw, db_poses, beta,
                             theta_degrees=0.0, gt_thres=5.0,
                             yaw_threshold=80.0, top_k=100):
    """ConPR-style rerank R@1 with bug-fixed rotation."""
    theta_rad = np.deg2rad(theta_degrees)
    q_poses = q_poses_raw.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    qx_rot = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    qy_rot = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    q_poses[:, 3], q_poses[:, 7] = qx_rot, qy_rot

    db_x, db_y = db_poses[:, 3], db_poses[:, 7]

    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k_idx = idx.search(d_boq_q, top_k)

    correct = total = 0
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
        if not ypos:
            continue
        total += 1
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
    return correct / total * 100 if total > 0 else 0.0, total


def main():
    print("=" * 90)
    print(f"Full 10-seq ConPR rerank eval — {len(SEQUENCES)} sequences, "
          f"db={SEQUENCES[0]}, {len(SEQUENCES)-1} query pairs")
    print(f"β={BETA_VALUES}, θ={THETA_DEGREES}°, yaw={YAW_THRESHOLD}°, top-{TOP_K}")
    print("=" * 90)

    # 1) Load BoQ once and extract descriptors for all 10 sequences (shared across seeds)
    print("\nLoading official BoQ(ResNet50)@320...")
    boq_model = load_official_boq()

    datasets = {}
    dataloaders = {}
    boq_descs = {}
    print("\nExtracting BoQ descriptors for all 10 ConPR sequences...")
    for seq in SEQUENCES:
        ds = InferDataset(seq, dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
        datasets[seq] = ds
        dataloaders[seq] = dl
        d = extract_boq(boq_model, dl)
        boq_descs[seq] = d
        print(f"  {seq}: {len(ds)} imgs → BoQ {d.shape}")

    del boq_model
    torch.cuda.empty_cache()

    # 2) For each seed, extract equi descriptors and compute rerank R@1 for each pair
    results = {}  # seed → {seq: {β: (r1, n)}}

    for seed, ckpt in CKPTS.items():
        if not os.path.exists(ckpt):
            print(f"\n  MISSING ckpt seed={seed}: {ckpt}")
            continue
        print(f"\n[Seed {seed}] Loading equi ckpt: {os.path.basename(ckpt)}")
        equi_model = load_p1_equi(ckpt)
        equi_descs = {}
        for seq in SEQUENCES:
            e = extract_equi(equi_model, dataloaders[seq])
            equi_descs[seq] = e
            print(f"  {seq}: equi {e.shape}")
        del equi_model
        torch.cuda.empty_cache()

        results[seed] = {}
        db_seq = SEQUENCES[0]
        d_boq_db = boq_descs[db_seq]
        d_equi_db = equi_descs[db_seq]
        db_poses = datasets[db_seq].poses

        for q_seq in SEQUENCES[1:]:
            d_boq_q = boq_descs[q_seq]
            d_equi_q = equi_descs[q_seq]
            q_poses = datasets[q_seq].poses
            per_beta = {}
            for beta in BETA_VALUES:
                r1, n = compute_recall_at_beta(
                    d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                    q_poses, db_poses, beta,
                    theta_degrees=THETA_DEGREES, gt_thres=GT_THRES,
                    yaw_threshold=YAW_THRESHOLD, top_k=TOP_K,
                )
                per_beta[beta] = (r1, n)
            results[seed][q_seq] = per_beta
            log = "  ".join(f"β={b:.2f}:{per_beta[b][0]:6.2f}% (n={per_beta[b][1]})"
                            for b in BETA_VALUES)
            print(f"  pair db={db_seq} vs q={q_seq}  |  {log}")

    # 3) Aggregate: for each β, per-seed mean across 9 pairs, then 3-seed mean ± std
    print("\n" + "=" * 90)
    print("Per-seed mean R@1 across 9 ConPR query pairs")
    print("=" * 90)
    header = f"{'seed':>8s}"
    for beta in BETA_VALUES:
        header += f"{'β=' + f'{beta:.2f}':>10s}"
    print(header)
    print('-' * 90)
    seed_means = {b: [] for b in BETA_VALUES}
    for seed, per_seq in results.items():
        row = f"{seed:>8d}"
        for beta in BETA_VALUES:
            m = np.mean([per_seq[q][beta][0] for q in SEQUENCES[1:] if q in per_seq])
            seed_means[beta].append(m)
            row += f"{m:>10.2f}"
        print(row)
    print('=' * 90)
    print(f"{'3-seed mean':>8s}")
    for beta in BETA_VALUES:
        vals = seed_means[beta]
        if len(vals) >= 2:
            m, s = np.mean(vals), np.std(vals, ddof=1)
            d = m - seed_means[BETA_VALUES[0]][0] if BETA_VALUES[0] == 0.00 else 0
            delta_str = f" (Δ={m - np.mean(seed_means[0.00]):+.2f})" if beta != 0.00 else ""
            print(f"  β={beta:.2f}:  {m:.2f} ± {s:.2f}  n_seeds={len(vals)}{delta_str}")

    # 4) Per-pair table (seed-averaged) for paper
    print("\n" + "=" * 90)
    print("Per-pair R@1 (3-seed averaged) — Table 2 detailed rows for ConPR")
    print("=" * 90)
    pair_header = f"{'pair':>24s}"
    for beta in BETA_VALUES:
        pair_header += f"{'β=' + f'{beta:.2f}':>12s}"
    pair_header += f"{'Δ@0.10':>10s}"
    print(pair_header)
    print('-' * 90)
    for q_seq in SEQUENCES[1:]:
        row = f"{'db=' + SEQUENCES[0] + ' vs q=' + q_seq:>24s}"
        per_beta_means = {}
        for beta in BETA_VALUES:
            vals = [results[s][q_seq][beta][0] for s in results if q_seq in results[s]]
            per_beta_means[beta] = np.mean(vals) if vals else 0.0
            row += f"{per_beta_means[beta]:>12.2f}"
        delta = per_beta_means.get(0.10, 0) - per_beta_means.get(0.00, 0)
        row += f"{delta:>+10.2f}"
        print(row)
    print('=' * 90)


if __name__ == '__main__':
    main()
