"""
Test-Time Augmentation (TTA) eval for DR-VPR v2 P1 standalone on ConSLAM + ConPR.

Query-side TTA: rotate each query image N times, forward through equi model,
average the resulting descriptors (L2-re-normalized after averaging). Database
side stays single-orientation (standard VPR practice).

Tests two TTA modes:
  (A) TTA_on_grid  (8 rotations at k·45°, k=0..7) — all in C8 group.
      Equivariant net should give near-identical descriptors → tests exactness
      of learned equivariance; gains indicate bilinear-interpolation denoising.
  (B) TTA_off_grid (8 rotations at 22.5° + k·45°, k=0..7) — all outside C8.
      These break exact equivariance → descriptors genuinely differ → real
      TTA averaging benefit expected.

Reports 3-seed mean ± std at β=0.10 for:
  (a) no TTA                (current paper result)
  (b) TTA on-grid (C8)
  (c) TTA off-grid
  (d) TTA full (16 rotations = on + off grid combined)

Query-only TTA cost: 8× or 16× forward passes on query side only. Cheap.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torchvision.transforms.functional import rotate as tv_rotate
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset as ConslamInferDataset
from conpr_eval_dataset_rot import InferDataset as ConprInferDataset
from conpr_eval_dataset_rot import get_yaw_from_pose
from models.equi_multiscale import E2ResNetMultiScale

DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0

CONSLAM_PATH = './datasets/ConSLAM/'
CONSLAM_SEQS = ['Sequence5', 'Sequence4']
CONSLAM_THETA = 15.0

CONPR_PATH = './datasets/ConPR/'
CONPR_SEQS = ['20230623', '20230809']   # single representative pair for speed
CONPR_THETA = 0.0

BETA = 0.10

TTA_ANGLES_ONGRID = [k * 45.0 for k in range(8)]                   # 0, 45, ..., 315
TTA_ANGLES_OFFGRID = [22.5 + k * 45.0 for k in range(8)]            # 22.5, 67.5, ..., 337.5
TTA_ANGLES_FULL = TTA_ANGLES_ONGRID + TTA_ANGLES_OFFGRID            # 16 rotations

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


@torch.no_grad()
def extract_equi_tta(model, dl, angles):
    """Extract TTA-averaged equi desc.

    For each angle a: rotate input by a°, forward, collect desc.
    Average across angles (sum of L2-normed → re-L2).
    Returns (N, D) where N=len(dl.dataset), D=1024.
    """
    all_agg = []
    for imgs, _ in dl:
        imgs = imgs.to(DEVICE)
        # sum across rotations (bilinear interp; output is same HxW, zeros pad)
        desc_sum = torch.zeros(imgs.shape[0], 1024, device=DEVICE)
        for a in angles:
            if a == 0.0:
                imgs_a = imgs
            else:
                imgs_a = tv_rotate(imgs, angle=a, fill=0.0)
            d = model(imgs_a)    # already L2-normed
            desc_sum = desc_sum + d
        desc_avg = F.normalize(desc_sum, p=2, dim=1)
        all_agg.append(desc_avg.cpu().numpy())
    return np.vstack(all_agg)


def rotate_query_poses(q_poses_raw, theta_degrees):
    theta_rad = np.deg2rad(theta_degrees)
    q_poses = q_poses_raw.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    qx_rot = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    qy_rot = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    q_poses[:, 3], q_poses[:, 7] = qx_rot, qy_rot
    return q_poses


def eval_rerank_r1(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                    q_poses_raw, db_poses, beta,
                    theta_degrees, top_k=100, gt_thres=5.0, yaw_threshold=80.0):
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


def run_dataset(name, ds_cls, dataset_path, seqs, theta_degrees):
    print(f"\n{'=' * 100}\nTTA — {name}  (θ={theta_degrees}°)\n{'=' * 100}")

    print("Loading BoQ and extracting descriptors...")
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

        # db: always single-orientation (no TTA)
        d_equi_db = extract_equi(equi_model, db_dl)
        # query: 4 TTA modes
        print(f"  TTA extraction (query side)...")
        d_equi_q_none   = extract_equi(equi_model, q_dl)
        d_equi_q_on     = extract_equi_tta(equi_model, q_dl, TTA_ANGLES_ONGRID)
        d_equi_q_off    = extract_equi_tta(equi_model, q_dl, TTA_ANGLES_OFFGRID)
        d_equi_q_full   = extract_equi_tta(equi_model, q_dl, TTA_ANGLES_FULL)
        del equi_model; torch.cuda.empty_cache()

        # sanity: check how different on-grid TTA vs no-TTA descriptors are
        diff_on = np.linalg.norm(d_equi_q_on - d_equi_q_none, axis=1).mean()
        diff_off = np.linalg.norm(d_equi_q_off - d_equi_q_none, axis=1).mean()
        print(f"  sanity: mean ‖TTA_on - none‖ = {diff_on:.4f}, ‖TTA_off - none‖ = {diff_off:.4f}")

        stats = {}
        for mode_name, d_equi_q in [
            ('a_no_tta_b0',    d_equi_q_none),    # β=0 baseline
            ('a_no_tta_b010',  d_equi_q_none),    # standard rerank
            ('b_tta_on_b010',  d_equi_q_on),
            ('c_tta_off_b010', d_equi_q_off),
            ('d_tta_full_b010',d_equi_q_full),
        ]:
            beta_val = 0.0 if mode_name.endswith('b0') else BETA
            r1, n = eval_rerank_r1(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                                    q_ds.poses, db_ds.poses, beta_val,
                                    theta_degrees, top_k=TOP_K,
                                    gt_thres=GT_THRES, yaw_threshold=YAW_THRESHOLD)
            stats[mode_name] = (r1, n)
        per_seed[seed] = stats
        row = f"  seed={seed}  "
        for k, (r, _) in stats.items():
            row += f"{k}:{r:6.2f}%  "
        print(row)

    # 3-seed aggregate
    print(f"\n{'-' * 100}\n{name} — 3-seed mean ± std R@1 (β={BETA}, except a_no_tta_b0)\n{'-' * 100}")
    print(f"{'condition':<22s}  {'seed=1':>8s}  {'seed=42':>8s}  {'seed=190223':>11s}  {'mean':>8s}  {'std':>8s}  {'Δ vs (a_b0)':>12s}")
    ref = None
    for cond in ['a_no_tta_b0', 'a_no_tta_b010', 'b_tta_on_b010', 'c_tta_off_b010', 'd_tta_full_b010']:
        vals = [per_seed[s][cond][0] for s in per_seed]
        if not vals: continue
        m, sd = np.mean(vals), (np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        if cond == 'a_no_tta_b0':
            ref = m
        delta = m - ref if ref is not None else 0.0
        vals_str = "  ".join(f"{v:8.2f}" for v in vals)
        print(f"{cond:<22s}  {vals_str}  {m:8.2f}  {sd:8.2f}  {delta:>+11.2f}")


def main():
    run_dataset('ConSLAM',  ConslamInferDataset, CONSLAM_PATH, CONSLAM_SEQS, CONSLAM_THETA)
    run_dataset('ConPR',    ConprInferDataset,   CONPR_PATH,   CONPR_SEQS,   CONPR_THETA)


if __name__ == '__main__':
    main()
