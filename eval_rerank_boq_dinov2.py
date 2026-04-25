"""
DR-VPR v2 with BoQ(DINOv2) stage-1  —  exploratory backbone swap.

Standard DR-VPR v2 uses BoQ(ResNet50) at 320×320 as stage-1 retriever.
Here we swap stage-1 to the official BoQ(DINOv2) at 322×322 (the DINOv2-native
input size), keeping the rest of the pipeline identical:
  - Stage-2 rerank descriptor = same P1 standalone E2ResNet(C8) multi-scale
    checkpoints (3 seeds, 1024-d).
  - Rerank rule = (1 − β)·boq_sim + β·equi_sim, per-query top-100.

Tests both ConSLAM (Sequence5 db vs Sequence4 query, θ=15°, 307 queries) and
ConPR full 10-sequence protocol (9 query seqs vs db=20230623, θ=0°).

β sweep: {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}

Output: per-dataset table of R@1 vs β, 3-seed mean ± std. Log to
`eval_rerank_boq_dinov2.log`.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset as ConslamInferDataset
from conpr_eval_dataset_rot import InferDataset as ConprInferDataset
from conpr_eval_dataset_rot import get_yaw_from_pose as cp_yaw
from Conslam_dataset_rot import get_yaw_from_pose as cs_yaw
from models.equi_multiscale import E2ResNetMultiScale


DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE_EQUI = (320, 320)
IMG_SIZE_BOQ  = (322, 322)   # DINOv2 native resolution (multiple of patch=14)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
BETA_VALUES = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]

CONSLAM_PATH = './datasets/ConSLAM/'
CONSLAM_SEQS = ['Sequence5', 'Sequence4']
CONSLAM_THETA = 15.0

CONPR_PATH = './datasets/ConPR/'
CONPR_SEQS = ['20230623', '20230531', '20230611', '20230627', '20230628',
              '20230706', '20230717', '20230803', '20230809', '20230818']
CONPR_THETA = 0.0

CKPTS = {
    1:      'LOGS/equi_standalone_seed1_ms/lightning_logs/version_0/checkpoints/equi_ms_seed1_epoch(07)_R1[0.3359].ckpt',
    42:     'LOGS/equi_standalone_seed42_ms/lightning_logs/version_0/checkpoints/equi_ms_seed42_epoch(01)_R1[0.3157].ckpt',
    190223: 'LOGS/equi_standalone_seed190223_ms/lightning_logs/version_0/checkpoints/equi_ms_seed190223_epoch(08)_R1[0.3359].ckpt',
}


def load_boq_dinov2():
    print("Loading BoQ(DINOv2) from torch.hub...")
    model = torch.hub.load(
        "amaralibey/bag-of-queries", "get_trained_boq",
        backbone_name="dinov2", output_dim=12288,
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


def compute_rerank_r1(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                       q_poses_raw, db_poses, beta,
                       theta_degrees, top_k, gt_thres, yaw_threshold,
                       yaw_fn):
    q_poses = rotate_query_poses(q_poses_raw, theta_degrees)
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
        q_yaw = yaw_fn(q_poses[q_idx])
        ypos = set()
        for p in pp:
            d = abs(q_yaw - yaw_fn(db_poses[p]))
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


def run_conslam(boq, ckpts):
    print(f"\n{'=' * 90}\nConSLAM — BoQ(DINOv2) stage-1 + P1 equi rerank (θ={CONSLAM_THETA}°)\n{'=' * 90}")

    db_ds_boq = ConslamInferDataset(CONSLAM_SEQS[0], dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_BOQ)
    q_ds_boq  = ConslamInferDataset(CONSLAM_SEQS[1], dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_BOQ)
    db_dl_boq = DataLoader(db_ds_boq, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    q_dl_boq  = DataLoader(q_ds_boq,  batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    db_ds_eq  = ConslamInferDataset(CONSLAM_SEQS[0], dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_EQUI)
    q_ds_eq   = ConslamInferDataset(CONSLAM_SEQS[1], dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_EQUI)
    db_dl_eq  = DataLoader(db_ds_eq,  batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)
    q_dl_eq   = DataLoader(q_ds_eq,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Extracting BoQ(DINOv2) desc at {IMG_SIZE_BOQ}...")
    d_boq_db = extract_boq(boq, db_dl_boq)
    d_boq_q  = extract_boq(boq, q_dl_boq)
    print(f"  db {d_boq_db.shape}  q {d_boq_q.shape}")

    per_seed = {b: [] for b in BETA_VALUES}
    for seed, ckpt_path in ckpts.items():
        if not os.path.exists(ckpt_path):
            print(f"  MISSING seed={seed}: {ckpt_path}"); continue
        print(f"\n[Seed {seed}] extracting equi...")
        eq = load_p1_equi(ckpt_path)
        d_eq_db = extract_equi(eq, db_dl_eq)
        d_eq_q  = extract_equi(eq, q_dl_eq)
        del eq; torch.cuda.empty_cache()

        print(f"  β sweep:")
        for beta in BETA_VALUES:
            r1, n = compute_rerank_r1(
                d_boq_db, d_boq_q, d_eq_db, d_eq_q,
                q_ds_boq.poses, db_ds_boq.poses, beta,
                CONSLAM_THETA, TOP_K, GT_THRES, YAW_THRESHOLD, cs_yaw)
            per_seed[beta].append(r1)
            print(f"    β={beta:.2f}  R@1={r1:6.2f}%  (n={n})")

    print(f"\n{'-' * 90}\nConSLAM — 3-seed mean ± std R@1\n{'-' * 90}")
    print(f"{'β':>5s}  {'seed=1':>8s}  {'seed=42':>8s}  {'seed=190223':>11s}  {'mean':>7s}  {'std':>7s}  {'Δ vs β=0':>10s}")
    print('-' * 90)
    ref = np.mean(per_seed[0.00]) if per_seed[0.00] else 0.0
    for beta in BETA_VALUES:
        vals = per_seed[beta]
        if len(vals) < 2: continue
        m, s = np.mean(vals), np.std(vals, ddof=1)
        delta = m - ref
        vals_str = "  ".join(f"{v:8.2f}" for v in vals)
        print(f"{beta:5.2f}  {vals_str}  {m:7.2f}  {s:7.2f}  {delta:>+9.2f}")


def run_conpr(boq, ckpts):
    print(f"\n{'=' * 90}\nConPR — BoQ(DINOv2) stage-1 + P1 equi rerank (θ={CONPR_THETA}°, full 10-seq)\n{'=' * 90}")

    # Build BoQ and equi datasets for each sequence
    datasets = {}; dataloaders_boq = {}; dataloaders_eq = {}
    for seq in CONPR_SEQS:
        ds_boq = ConprInferDataset(seq, dataset_path=CONPR_PATH, img_size=IMG_SIZE_BOQ)
        ds_eq  = ConprInferDataset(seq, dataset_path=CONPR_PATH, img_size=IMG_SIZE_EQUI)
        datasets[seq] = ds_boq   # poses are identical, use either
        dataloaders_boq[seq] = DataLoader(ds_boq, batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True)
        dataloaders_eq[seq]  = DataLoader(ds_eq,  batch_size=BATCH_SIZE, shuffle=False,
                                           num_workers=NUM_WORKERS, pin_memory=True)

    print(f"Extracting BoQ(DINOv2) desc at {IMG_SIZE_BOQ} for all {len(CONPR_SEQS)} sequences...")
    boq_descs = {}
    for seq in CONPR_SEQS:
        d = extract_boq(boq, dataloaders_boq[seq])
        boq_descs[seq] = d
        print(f"  {seq}: {d.shape}")

    # per seed, extract equi then compute β R@1 over 9 pairs
    per_seed_pair = {}  # seed → {seq: {β: r1}}
    for seed, ckpt_path in ckpts.items():
        if not os.path.exists(ckpt_path):
            print(f"  MISSING seed={seed}"); continue
        print(f"\n[Seed {seed}] extracting equi...")
        eq = load_p1_equi(ckpt_path)
        eq_descs = {seq: extract_equi(eq, dataloaders_eq[seq]) for seq in CONPR_SEQS}
        del eq; torch.cuda.empty_cache()

        per_seed_pair[seed] = {}
        db_seq = CONPR_SEQS[0]
        d_boq_db = boq_descs[db_seq]
        d_eq_db  = eq_descs[db_seq]
        db_poses = datasets[db_seq].poses

        for q_seq in CONPR_SEQS[1:]:
            d_boq_q = boq_descs[q_seq]
            d_eq_q  = eq_descs[q_seq]
            q_poses = datasets[q_seq].poses
            per_beta = {}
            for beta in BETA_VALUES:
                r1, n = compute_rerank_r1(
                    d_boq_db, d_boq_q, d_eq_db, d_eq_q,
                    q_poses, db_poses, beta,
                    CONPR_THETA, TOP_K, GT_THRES, YAW_THRESHOLD, cp_yaw)
                per_beta[beta] = r1
            per_seed_pair[seed][q_seq] = per_beta
            log = "  ".join(f"β={b:.2f}:{per_beta[b]:6.2f}" for b in BETA_VALUES)
            print(f"  pair db={db_seq} vs q={q_seq}  |  {log}")

    # Aggregate: seed mean across 9 pairs
    print(f"\n{'-' * 110}\nConPR — per-seed mean across 9 pairs, 3-seed mean ± std\n{'-' * 110}")
    header = f"{'β':>5s}"
    for s in per_seed_pair:
        header += f"  {'seed=' + str(s):>12s}"
    header += f"  {'mean':>8s}  {'std':>7s}  {'Δ vs β=0':>10s}"
    print(header)
    print('-' * 110)
    seed_means = {b: {} for b in BETA_VALUES}
    for beta in BETA_VALUES:
        for seed, per_seq in per_seed_pair.items():
            m_seed = np.mean([per_seq[q][beta] for q in CONPR_SEQS[1:]])
            seed_means[beta][seed] = m_seed
    ref = np.mean(list(seed_means[0.00].values()))
    for beta in BETA_VALUES:
        row = f"{beta:5.2f}"
        vals = list(seed_means[beta].values())
        for s in per_seed_pair:
            row += f"  {seed_means[beta][s]:>12.2f}"
        m, sd = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        delta = m - ref
        row += f"  {m:8.2f}  {sd:7.2f}  {delta:>+9.2f}"
        print(row)


def main():
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"BoQ(DINOv2) image size: {IMG_SIZE_BOQ}")
    print(f"E2ResNet image size: {IMG_SIZE_EQUI}")
    print(f"β sweep: {BETA_VALUES}")
    print(f"3 seeds: {list(CKPTS.keys())}")

    boq = load_boq_dinov2()
    # ConSLAM (fast)
    run_conslam(boq, CKPTS)
    # ConPR full 10-seq (~30 min)
    run_conpr(boq, CKPTS)


if __name__ == '__main__':
    main()
