"""
Union retrieve + β rerank for DR-VPR v2 (P1 standalone) on ConSLAM + ConPR.

Stage 1: candidate pool = BoQ top-K ∪ equi top-K (union of indices per query)
Stage 2: within union, rerank by (1-β)·boq_sim + β·equi_sim at β=0.10

Motivation: the standard two-stage rerank protocol uses BoQ-only top-K, so any
true positive not in BoQ's top-K is unrecoverable (confirmed by per-yaw analysis:
[30°+] buckets stuck at 0%). Union retrieve gives equi a chance to contribute
a CANDIDATE (not just a rerank score), potentially rescuing rotation-heavy
queries where equi should retrieve better than BoQ.

Control: reports 4 conditions per dataset:
  (a) BoQ-only top-K rerank      (current paper result, β=0.00 and β=0.10)
  (b) Union top-K rerank         (new proposal)
  (c) equi-only top-K            (stage-1 = equi FAISS, no BoQ)

Per-seed + 3-seed mean ± std, ConSLAM+ConPR.
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
from conpr_eval_dataset_rot import get_yaw_from_pose
from models.equi_multiscale import E2ResNetMultiScale

DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
TOP_K_BOQ = 100
TOP_K_EQUI = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0

CONSLAM_PATH = './datasets/ConSLAM/'
CONSLAM_SEQS = ['Sequence5', 'Sequence4']
CONSLAM_THETA = 15.0

CONPR_PATH = './datasets/ConPR/'
# Single representative pair for initial union test (speed). Extend after sanity.
CONPR_SEQS = ['20230623', '20230809']
CONPR_THETA = 0.0

BETA = 0.10

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


def rotate_query_poses(q_poses_raw, theta_degrees):
    theta_rad = np.deg2rad(theta_degrees)
    q_poses = q_poses_raw.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    qx_rot = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    qy_rot = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    q_poses[:, 3], q_poses[:, 7] = qx_rot, qy_rot
    return q_poses


def valid_positives(q_poses, db_poses, q_idx, gt_thres, yaw_threshold):
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    dist_sq = (q_poses[q_idx, 3] - db_x) ** 2 + (q_poses[q_idx, 7] - db_y) ** 2
    pp = np.where(dist_sq < gt_thres ** 2)[0]
    if len(pp) == 0:
        return None
    q_yaw = get_yaw_from_pose(q_poses[q_idx])
    ypos = set()
    for p in pp:
        d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
        if d > 180:
            d = 360 - d
        if d <= yaw_threshold:
            ypos.add(int(p))
    return ypos if ypos else None


def eval_four_conditions(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                          q_poses_raw, db_poses, theta_degrees,
                          top_k_boq=100, top_k_equi=100, beta=0.10,
                          gt_thres=5.0, yaw_threshold=80.0):
    """Return R@1 for 4 conditions.

    (a) boq-only β=0          → pure BoQ top-1 from top-K_boq
    (b) boq-only β=0.10       → current DR-VPR v2 rerank
    (c) union      β=0.10     → new proposal
    (d) equi-only              → stage-1 = equi FAISS top-1
    """
    q_poses = rotate_query_poses(q_poses_raw, theta_degrees)

    idx_boq = faiss.IndexFlatIP(d_boq_db.shape[1]); idx_boq.add(d_boq_db)
    _, top_boq = idx_boq.search(d_boq_q, top_k_boq)
    idx_equi = faiss.IndexFlatIP(d_equi_db.shape[1]); idx_equi.add(d_equi_db)
    _, top_equi = idx_equi.search(d_equi_q, top_k_equi)

    stats = {'a_boq_only_b0': [0, 0],
             'b_boq_only_b010': [0, 0],
             'c_union_b010': [0, 0],
             'd_equi_only': [0, 0]}

    for q_idx in range(len(q_poses)):
        ypos = valid_positives(q_poses, db_poses, q_idx, gt_thres, yaw_threshold)
        if ypos is None:
            continue
        # Increment totals for all 4 conditions
        for k in stats:
            stats[k][1] += 1

        # (a) pure BoQ top-1
        if top_boq[q_idx, 0] in ypos:
            stats['a_boq_only_b0'][0] += 1

        # (b) BoQ top-K_boq → rerank at β
        cands_b = top_boq[q_idx]
        boq_sim = d_boq_q[q_idx] @ d_boq_db[cands_b].T
        equi_sim = d_equi_q[q_idx] @ d_equi_db[cands_b].T
        score_b = (1 - beta) * boq_sim + beta * equi_sim
        if cands_b[np.argmax(score_b)] in ypos:
            stats['b_boq_only_b010'][0] += 1

        # (c) Union top-K_boq ∪ top-K_equi → rerank at β
        cands_u = np.unique(np.concatenate([top_boq[q_idx], top_equi[q_idx]]))
        boq_sim_u = d_boq_q[q_idx] @ d_boq_db[cands_u].T
        equi_sim_u = d_equi_q[q_idx] @ d_equi_db[cands_u].T
        score_u = (1 - beta) * boq_sim_u + beta * equi_sim_u
        if cands_u[np.argmax(score_u)] in ypos:
            stats['c_union_b010'][0] += 1

        # (d) pure equi top-1
        if top_equi[q_idx, 0] in ypos:
            stats['d_equi_only'][0] += 1

    out = {}
    for k, (c, t) in stats.items():
        out[k] = (c / t * 100 if t > 0 else 0.0, t)
    return out


def run_dataset(name, ds_cls, dataset_path, seqs, theta_degrees):
    print(f"\n{'=' * 90}\nUnion retrieve — {name}  (θ={theta_degrees}°)\n{'=' * 90}")

    print(f"Loading BoQ and extracting descriptors...")
    boq_model = load_official_boq()
    db_ds = ds_cls(seqs[0], dataset_path=dataset_path, img_size=IMG_SIZE)
    q_ds  = ds_cls(seqs[1], dataset_path=dataset_path, img_size=IMG_SIZE)
    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    q_dl  = DataLoader(q_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    d_boq_db = extract_boq(boq_model, db_dl)
    d_boq_q  = extract_boq(boq_model, q_dl)
    del boq_model; torch.cuda.empty_cache()
    print(f"  db boq {d_boq_db.shape}, q boq {d_boq_q.shape}")

    per_seed = {}
    for seed, ckpt in CKPTS.items():
        if not os.path.exists(ckpt):
            print(f"  MISSING seed={seed}"); continue
        print(f"\n[Seed {seed}] extracting equi...")
        equi_model = load_p1_equi(ckpt)
        d_equi_db = extract_equi(equi_model, db_dl)
        d_equi_q  = extract_equi(equi_model, q_dl)
        del equi_model; torch.cuda.empty_cache()

        stats = eval_four_conditions(
            d_boq_db, d_boq_q, d_equi_db, d_equi_q,
            q_ds.poses, db_ds.poses,
            theta_degrees=theta_degrees,
            top_k_boq=TOP_K_BOQ, top_k_equi=TOP_K_EQUI, beta=BETA,
            gt_thres=GT_THRES, yaw_threshold=YAW_THRESHOLD,
        )
        per_seed[seed] = stats
        row = f"  seed={seed}"
        for k, (r, t) in stats.items():
            row += f"  {k}:{r:6.2f}%"
        print(row + f"  (n={t})")

    # 3-seed aggregate
    print(f"\n{'-' * 90}\n{name} — 3-seed mean ± std R@1\n{'-' * 90}")
    print(f"{'condition':<28s}  {'seed=1':>8s}  {'seed=42':>8s}  {'seed=190223':>11s}  {'mean':>8s}  {'std':>8s}  {'Δ vs (a)':>10s}")
    ref = None
    for cond in ['a_boq_only_b0', 'b_boq_only_b010', 'c_union_b010', 'd_equi_only']:
        vals = [per_seed[s][cond][0] for s in per_seed]
        if not vals: continue
        m, sd = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        if cond == 'a_boq_only_b0':
            ref = m
        delta = m - ref if ref is not None else 0.0
        vals_str = "  ".join(f"{v:8.2f}" for v in vals)
        print(f"{cond:<28s}  {vals_str}  {m:8.2f}  {sd:8.2f}  {delta:>+9.2f}")


def main():
    run_dataset('ConSLAM',  ConslamInferDataset, CONSLAM_PATH, CONSLAM_SEQS, CONSLAM_THETA)
    run_dataset('ConPR',    ConprInferDataset,   CONPR_PATH,   CONPR_SEQS,   CONPR_THETA)


if __name__ == '__main__':
    main()
