"""
Single-stage joint scoring vs two-stage retrieve-rerank — ConPR full 10-seq.

Replicates the eval_single_stage_joint.py ablation but on ConPR's full
10-sequence protocol (db = 20230623, 9 query sequences, θ = 0°). Confirms
whether the two protocols are R@1-equivalent on the larger ConPR benchmark.

Output: per-pair, per-β table with both two-stage and single-stage R@1 +
3-seed aggregate. If the two are equivalent (or single-stage ≥ two-stage)
across the board, we can simplify the paper framing to drop stage-1.

Setup: ConPR 10 sequences, BoQ(ResNet50)@320, P1 standalone equi (3 seeds),
β sweep {0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50}.
"""
from __future__ import annotations
import os
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
THETA_DEGREES = 0.0
DATASET_PATH = './datasets/ConPR/'
SEQUENCES = ['20230623', '20230531', '20230611', '20230627', '20230628',
             '20230706', '20230717', '20230803', '20230809', '20230818']
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
    model = E2ResNetMultiScale(orientation=8, layers=(2, 2, 2, 2),
                                channels=(64, 128, 256, 512), out_dim=1024)
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
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    valid = []
    for q_idx in range(len(q_poses)):
        dist_sq = (q_poses[q_idx, 3] - db_x) ** 2 + (q_poses[q_idx, 7] - db_y) ** 2
        pp = np.where(dist_sq < gt_thres ** 2)[0]
        if len(pp) == 0: continue
        q_yaw = get_yaw_from_pose(q_poses[q_idx])
        ypos = set()
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180: d = 360 - d
            if d <= yaw_threshold: ypos.add(int(p))
        if ypos: valid.append((q_idx, ypos))
    return valid


def eval_pair(d_boq_db, d_boq_q, d_eq_db, d_eq_q, valid, betas, top_k):
    """Returns dict β → {'two': r1, 'single': r1, 'recall_at_K': float}."""
    # Pre-compute both full sim matrices once per (pair, seed)
    boq_sim_full = d_boq_q @ d_boq_db.T   # (N_q, N_db)
    eq_sim_full  = d_eq_q  @ d_eq_db.T

    # Stage-1 BoQ FAISS top-K (for two-stage)
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k_idx = idx.search(d_boq_q, top_k)

    # Recall@K diagnostic
    n_pos_in_topK = sum(1 for q_idx, ypos in valid
                        if any(int(c) in ypos for c in top_k_idx[q_idx]))
    recall_at_k = n_pos_in_topK / len(valid) * 100 if valid else 0.0

    out = {'recall_at_K': recall_at_k, 'n_valid': len(valid)}
    for beta in betas:
        # two-stage: rerank within top-K
        score_top = (1 - beta) * boq_sim_full + beta * eq_sim_full
        correct_two = correct_single = 0
        for q_idx, ypos in valid:
            cands = top_k_idx[q_idx]
            if beta == 0.0:
                top1_two = cands[0]
            else:
                # Pick best in top-K by combined score
                local = score_top[q_idx, cands]
                top1_two = int(cands[np.argmax(local)])
            if top1_two in ypos:
                correct_two += 1
            # single-stage: argmax over all db
            top1_single = int(np.argmax(score_top[q_idx]))
            if top1_single in ypos:
                correct_single += 1
        n = len(valid)
        out[beta] = {
            'two':    correct_two    / n * 100 if n else 0.0,
            'single': correct_single / n * 100 if n else 0.0,
        }
    return out


def main():
    print("=" * 110)
    print("Single-stage vs two-stage rerank — ConPR full 10-seq, BoQ(ResNet50)@320 + P1 equi")
    print("=" * 110)

    # 1) Load BoQ + extract once
    print("\nLoading BoQ(ResNet50)...")
    boq = load_official_boq()
    datasets = {}
    dataloaders = {}
    boq_descs = {}
    print("Extracting BoQ desc for 10 sequences...")
    for seq in SEQUENCES:
        ds = InferDataset(seq, dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
        datasets[seq] = ds; dataloaders[seq] = dl
        d = extract_boq(boq, dl)
        boq_descs[seq] = d
        print(f"  {seq}: {d.shape}")
    del boq; torch.cuda.empty_cache()

    # 2) Per seed, extract equi + run all pair evals
    db_seq = SEQUENCES[0]
    db_poses = datasets[db_seq].poses
    results = {}   # seed → {q_seq: {β: {'two':..., 'single':...}}}

    for seed, ckpt_path in CKPTS.items():
        if not os.path.exists(ckpt_path):
            print(f"\n  MISSING ckpt seed={seed}"); continue
        print(f"\n[Seed {seed}] extracting equi...")
        eq = load_p1_equi(ckpt_path)
        eq_descs = {}
        for seq in SEQUENCES:
            eq_descs[seq] = extract_equi(eq, dataloaders[seq])
            print(f"  {seq}: equi {eq_descs[seq].shape}")
        del eq; torch.cuda.empty_cache()

        results[seed] = {}
        d_boq_db = boq_descs[db_seq]
        d_eq_db  = eq_descs[db_seq]
        for q_seq in SEQUENCES[1:]:
            d_boq_q = boq_descs[q_seq]
            d_eq_q  = eq_descs[q_seq]
            q_poses = rotate_query_poses(datasets[q_seq].poses, THETA_DEGREES)
            valid = precompute_valid(q_poses, db_poses, GT_THRES, YAW_THRESHOLD)
            res = eval_pair(d_boq_db, d_boq_q, d_eq_db, d_eq_q, valid, BETA_VALUES, TOP_K)
            results[seed][q_seq] = res
            log = "  ".join(
                f"β={b:.2f} two={res[b]['two']:5.2f} sin={res[b]['single']:5.2f} Δ={res[b]['single']-res[b]['two']:+.2f}"
                for b in BETA_VALUES)
            print(f"  pair db={db_seq} vs q={q_seq}  recall@{TOP_K}={res['recall_at_K']:.2f}%  "
                  f"n={res['n_valid']}\n    {log}")

    # 3) Aggregate: per seed mean across 9 pairs, then 3-seed mean
    print(f"\n{'=' * 110}")
    print("3-seed mean R@1 across 9 pairs")
    print(f"{'=' * 110}")
    print(f"{'β':>5s}  {'two-stage mean':>16s}  {'std':>6s}  "
          f"{'single-stage mean':>18s}  {'std':>6s}  {'Δ(single − two)':>16s}")
    print('-' * 110)
    for beta in BETA_VALUES:
        per_seed_two    = []
        per_seed_single = []
        for seed in results:
            two_pair_means    = [results[seed][q][beta]['two']    for q in SEQUENCES[1:]]
            single_pair_means = [results[seed][q][beta]['single'] for q in SEQUENCES[1:]]
            per_seed_two.append(np.mean(two_pair_means))
            per_seed_single.append(np.mean(single_pair_means))
        if len(per_seed_two) < 2: continue
        m2, s2 = np.mean(per_seed_two), np.std(per_seed_two, ddof=1)
        m1, s1 = np.mean(per_seed_single), np.std(per_seed_single, ddof=1)
        d = m1 - m2
        print(f"{beta:5.2f}  {m2:16.2f}  {s2:6.2f}  {m1:18.2f}  {s1:6.2f}  {d:>+15.2f}")
    print('=' * 110)

    # Recall@100 ceiling per pair (averaged across seeds since BoQ is deterministic, but we average for safety)
    print("\nBoQ recall@100 ceiling per pair (deterministic — same across seeds):")
    seeds_list = list(results.keys())
    if seeds_list:
        s0 = seeds_list[0]
        for q_seq in SEQUENCES[1:]:
            ceil = results[s0][q_seq]['recall_at_K']
            n = results[s0][q_seq]['n_valid']
            print(f"  q={q_seq}  recall@100 = {ceil:5.2f}%  (n={n})")


if __name__ == '__main__':
    main()
