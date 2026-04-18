"""
Path A.1: Adaptive β based on BoQ confidence.

Hypothesis: Different yaw buckets need different β. We don't know yaw_diff at
test time, but BoQ's own confidence signals (top-1 absolute score, top-1-vs-top-5
gap) correlate with query difficulty:
  - High BoQ confidence → BoQ alone is right → use β≈0
  - Low BoQ confidence → BoQ struggles → let equi help → use β≈0.7

Variants compared:
  V0  fixed β=0.0  (BoQ baseline, no rerank)
  V1  fixed β=0.5  (our current method)
  V2  fixed β=0.7  (equi-heavy)
  V3  adaptive β = clip(α(gap_15→β), [0, 0.7]), gap-based
  V4  adaptive β = clip(score-based), top-1 score based
  V5  binary: β=0 if top1>thr else β=0.5
  V6  union retrieve + adaptive β
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
from train_fusion import VPRModel, load_boq_pretrained

os.environ.setdefault('GROUP_POOL_MODE', 'max')

DATASET_PATH = './datasets/ConSLAM/'
SEQS = ['Sequence5', 'Sequence4']
THETA_DEGREES = 15.0
YAW_THRESHOLD = 80.0
DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
GT_THRES = 5.0
TOP_K = 100

CKPTS = {
    1: 'LOGS/resnet50_DualBranch_freeze_boq_seed1/lightning_logs/version_0/checkpoints/resnet50_DualBranch_C8_concat_seed1_epoch(02)_R1[0.6417].ckpt',
    42: 'LOGS/resnet50_DualBranch_freeze_boq_seed42/lightning_logs/version_0/checkpoints/resnet50_DualBranch_C8_concat_seed42_epoch(03)_R1[0.6402].ckpt',
    190223: 'LOGS/resnet50_DualBranch_freeze_boq_seed190223/lightning_logs/version_0/checkpoints/resnet50_DualBranch_C8_concat_seed190223_epoch(08)_R1[0.6420].ckpt',
}


def build_model():
    return VPRModel(
        backbone_arch='resnet50', pretrained=True, layers_to_freeze=2, layers_to_crop=[4],
        agg_arch='boq', agg_config={'in_channels': 1024, 'proj_channels': 512,
                                    'num_queries': 64, 'num_layers': 2, 'row_dim': 32},
        use_dual_branch=True, equi_orientation=8, equi_layers=[2, 2, 2, 2],
        equi_channels=[64, 128, 256, 512], equi_out_dim=1024, fusion_method='concat',
        use_projection=False, lr=1e-3, optimizer='adamw', weight_decay=1e-4,
        momentum=0.9, warmpup_steps=300, milestones=[8, 14], lr_mult=0.3,
        loss_name='MultiSimilarityLoss', miner_name='MultiSimilarityMiner',
        miner_margin=0.1, faiss_gpu=False,
    )


@torch.no_grad()
def extract(model, dl):
    descs1, descs2 = [], []
    for imgs, _ in dl:
        imgs = imgs.to(DEVICE)
        f1 = model.backbone(imgs); f2 = model.backbone2(imgs)
        d1 = F.normalize(model.aggregator.branch1_aggregator(f1), p=2, dim=1)
        d2 = F.normalize(model.aggregator.branch2_aggregator(f2), p=2, dim=1)
        descs1.append(d1.cpu().numpy()); descs2.append(d2.cpu().numpy())
    return np.vstack(descs1), np.vstack(descs2)


def yaw_bucket(d):
    if d < 20: return '[ 0°, 20°)'
    if d < 40: return '[20°, 40°)'
    if d < 60: return '[40°, 60°)'
    return '[60°, 80°)'


def get_valid_queries(q_poses_raw, db_poses):
    theta_rad = np.deg2rad(THETA_DEGREES)
    q_poses = q_poses_raw.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    q_poses[:, 3] = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    q_poses[:, 7] = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    valid = []
    for q_idx in range(len(q_poses)):
        dist_sq = (q_poses[q_idx, 3] - db_x)**2 + (q_poses[q_idx, 7] - db_y)**2
        pp = np.where(dist_sq < GT_THRES**2)[0]
        if len(pp) == 0: continue
        q_yaw = get_yaw_from_pose(q_poses[q_idx])
        ypos, ydiffs = [], []
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180: d = 360 - d
            if d <= YAW_THRESHOLD:
                ypos.append(p); ydiffs.append(d)
        if not ypos: continue
        valid.append({
            'q_idx': q_idx, 'pos': set(ypos),
            'bucket': yaw_bucket(min(ydiffs)),
        })
    return valid


def adaptive_beta_gap(top_k_boq_sims):
    """β based on gap between top-1 and top-5 BoQ similarity.
       Larger gap → confident BoQ → small β.
       Smaller gap → uncertain → large β."""
    gap = float(top_k_boq_sims[0] - top_k_boq_sims[min(4, len(top_k_boq_sims)-1)])
    # Empirical mapping (will calibrate)
    if gap > 0.05: return 0.1
    if gap < 0.01: return 0.7
    return 0.7 - (gap - 0.01) / 0.04 * 0.6


def adaptive_beta_score(top_k_boq_sims):
    """β based on absolute top-1 BoQ score.
       High score → confident → small β."""
    s = float(top_k_boq_sims[0])
    if s > 0.85: return 0.1
    if s < 0.65: return 0.7
    return 0.7 - (s - 0.65) / 0.20 * 0.6


def adaptive_beta_binary(top_k_boq_sims, score_thr=0.8):
    """If BoQ top-1 score is high enough, β=0; else β=0.5."""
    return 0.0 if top_k_boq_sims[0] > score_thr else 0.5


def rerank_with_beta_per_query(d1_q, d1_db, d2_q, d2_db, top_k_idx, beta_fn):
    """beta_fn: takes sorted boq_sims (descending) for top-K, returns β."""
    n_q = d1_q.shape[0]
    top1 = np.zeros(n_q, dtype=np.int64)
    used_betas = []
    for i in range(n_q):
        cands = top_k_idx[i]
        boq_sim = d1_q[i] @ d1_db[cands].T
        equi_sim = d2_q[i] @ d2_db[cands].T
        # FAISS already returns sorted; sort by boq_sim descending to be safe
        order = np.argsort(-boq_sim)
        boq_sorted = boq_sim[order]
        beta = beta_fn(boq_sorted) if callable(beta_fn) else beta_fn
        used_betas.append(beta)
        score = (1 - beta) * boq_sim + beta * equi_sim
        top1[i] = cands[np.argmax(score)]
    return top1, np.array(used_betas)


def union_retrieve_with_adaptive(d1_q, d1_db, d2_q, d2_db, k_boq, k_equi, beta_fn):
    """Union BoQ top-k_boq + equi top-k_equi, then rerank with adaptive β."""
    n_q = d1_q.shape[0]
    idx_boq = faiss.IndexFlatIP(d1_db.shape[1]); idx_boq.add(d1_db)
    idx_equi = faiss.IndexFlatIP(d2_db.shape[1]); idx_equi.add(d2_db)
    _, tk_b = idx_boq.search(d1_q, k_boq)
    _, tk_e = idx_equi.search(d2_q, k_equi)

    top1 = np.zeros(n_q, dtype=np.int64)
    used_betas = []
    for i in range(n_q):
        seen = set(); merged = []
        for c in tk_b[i]:
            if c not in seen:
                merged.append(int(c)); seen.add(int(c))
        for c in tk_e[i]:
            if c not in seen:
                merged.append(int(c)); seen.add(int(c))
        cands = np.array(merged, dtype=np.int64)
        boq_sim = d1_q[i] @ d1_db[cands].T
        equi_sim = d2_q[i] @ d2_db[cands].T
        order = np.argsort(-boq_sim)
        boq_sorted = boq_sim[order]
        beta = beta_fn(boq_sorted) if callable(beta_fn) else beta_fn
        used_betas.append(beta)
        score = (1 - beta) * boq_sim + beta * equi_sim
        top1[i] = cands[np.argmax(score)]
    return top1, np.array(used_betas)


def main():
    # Aggregate per-variant per-bucket flags
    variants = [
        ('V0_β=0.0',          'topk', lambda s: 0.0),
        ('V1_β=0.5',          'topk', lambda s: 0.5),
        ('V2_β=0.7',          'topk', lambda s: 0.7),
        ('V3_adapt_gap',      'topk', adaptive_beta_gap),
        ('V4_adapt_score',    'topk', adaptive_beta_score),
        ('V5_binary_thr0.8',  'topk', lambda s: adaptive_beta_binary(s, 0.8)),
        ('V6_union_adapt',    'union', adaptive_beta_score),
    ]
    bucket_flags = {v[0]: defaultdict(list) for v in variants}
    beta_dist = {v[0]: defaultdict(list) for v in variants}  # (bucket → list of used β)
    score_stats = defaultdict(list)  # bucket → list of top-1 boq scores (for diagnostic)

    for seed, ckpt in CKPTS.items():
        print(f"\n--- seed={seed} ---")
        if not os.path.exists(ckpt): sys.exit(f"MISSING: {ckpt}")
        model = build_model()
        load_boq_pretrained(model)
        state = torch.load(ckpt, map_location='cpu')['state_dict']
        model.load_state_dict(state, strict=False)
        model = model.to(DEVICE).eval()

        db_ds = InferDataset(SEQS[0], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        q_ds = InferDataset(SEQS[1], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
        q_dl = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
        d1_db, d2_db = extract(model, db_dl)
        d1_q, d2_q = extract(model, q_dl)

        valid = get_valid_queries(q_ds.poses, db_ds.poses)
        idx_boq = faiss.IndexFlatIP(d1_db.shape[1]); idx_boq.add(d1_db)
        _, top_k_idx = idx_boq.search(d1_q, TOP_K)

        # Diagnostic: per-bucket top-1 BoQ score distribution
        for q in valid:
            i = q['q_idx']
            top_score = float(d1_q[i] @ d1_db[top_k_idx[i, 0]])
            score_stats[q['bucket']].append(top_score)

        # Run all variants
        for vname, mode, bfn in variants:
            if mode == 'topk':
                top1, betas = rerank_with_beta_per_query(d1_q, d1_db, d2_q, d2_db, top_k_idx, bfn)
            else:  # union
                top1, betas = union_retrieve_with_adaptive(d1_q, d1_db, d2_q, d2_db, 50, 50, bfn)
            for q in valid:
                bucket_flags[vname][q['bucket']].append(int(top1[q['q_idx']] in q['pos']))
                beta_dist[vname][q['bucket']].append(float(betas[q['q_idx']]))

        del model
        torch.cuda.empty_cache()

    # Diagnostic: BoQ top-1 score distribution per bucket
    print("\n" + "=" * 80)
    print("BoQ top-1 score distribution per bucket (calibration aid)")
    print("=" * 80)
    print(f"{'bucket':>13s}  {'N':>4s}  {'min':>6s}  {'mean':>6s}  {'median':>6s}  {'max':>6s}")
    for b in ['[ 0°, 20°)', '[20°, 40°)', '[40°, 60°)', '[60°, 80°)']:
        s = score_stats[b]
        if not s: continue
        print(f"  {b:>13s}  {len(s):>4d}  {min(s):>6.3f}  {np.mean(s):>6.3f}  "
              f"{np.median(s):>6.3f}  {max(s):>6.3f}")

    # Main results table
    print("\n" + "=" * 110)
    print("Path A.1: Adaptive β Variants (3 seeds × ConSLAM)")
    print("=" * 110)
    buckets = ['[ 0°, 20°)', '[20°, 40°)', '[40°, 60°)', '[60°, 80°)']
    print(f"{'variant':>20s}", end='')
    for b in buckets:
        print(f"  {b:>13s}", end='')
    print(f"  {'TOTAL':>10s}  {'avg β':>8s}")
    print('-' * 110)

    for vname, _, _ in variants:
        print(f"{vname:>20s}", end='')
        total_n = total_c = 0
        all_betas = []
        for b in buckets:
            flags = bucket_flags[vname][b]
            betas = beta_dist[vname][b]
            n = len(flags)
            if n == 0:
                print(f"  {'n/a':>13s}", end='')
            else:
                r1 = sum(flags) / n * 100
                print(f"  {r1:6.2f}% ({n:>3d})", end='')
                total_n += n
                total_c += sum(flags)
                all_betas.extend(betas)
        avg_beta = np.mean(all_betas) if all_betas else 0
        print(f"  {total_c/total_n*100:>9.2f}%  {avg_beta:>8.3f}")
    print('=' * 110)


if __name__ == '__main__':
    main()
