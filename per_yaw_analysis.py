"""
Per-yaw bucket analysis on ConSLAM.

For each query, compute the minimum |yaw_diff| to its nearest valid GT positive
(a "rotation difficulty" proxy), then bucket queries by this angle. In each bucket,
compare R@1 under:
  - BoQ(ResNet50) baseline (= β=0 rerank, i.e. pure BoQ top-1)
  - DR-VPR (= β=0.5 fixed rerank)

Tells us WHERE the +0.54 gain comes from.

Usage:
    python per_yaw_analysis.py

Uses the 3 yesterday freeze_boq seeds' val-best ckpts (hardcoded list below).
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset, get_yaw_from_pose
from train_fusion import VPRModel, load_boq_pretrained

# ---- config ----
os.environ.setdefault('GROUP_POOL_MODE', 'max')  # yesterday freeze_boq used max pool

DATASET_PATH = './datasets/ConSLAM/'
SEQS = ['Sequence5', 'Sequence4']
THETA_DEGREES = 15.0
YAW_THRESHOLD = 80.0
TOP_K = 100
BETA_VALUES = [0.0, 0.5]
DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
GT_THRES = 5.0

# Val-best epoch per seed (from freeze_boq training log)
CKPTS = {
    1: 'LOGS/resnet50_DualBranch_freeze_boq_seed1/lightning_logs/version_0/checkpoints/resnet50_DualBranch_C8_concat_seed1_epoch(02)_R1[0.6417].ckpt',
    42: 'LOGS/resnet50_DualBranch_freeze_boq_seed42/lightning_logs/version_0/checkpoints/resnet50_DualBranch_C8_concat_seed42_epoch(03)_R1[0.6402].ckpt',
    190223: 'LOGS/resnet50_DualBranch_freeze_boq_seed190223/lightning_logs/version_0/checkpoints/resnet50_DualBranch_C8_concat_seed190223_epoch(08)_R1[0.6420].ckpt',
}


def build_model():
    return VPRModel(
        backbone_arch='resnet50', pretrained=True, layers_to_freeze=2, layers_to_crop=[4],
        agg_arch='boq',
        agg_config={'in_channels': 1024, 'proj_channels': 512, 'num_queries': 64,
                     'num_layers': 2, 'row_dim': 32},
        use_dual_branch=True, equi_orientation=8,
        equi_layers=[2, 2, 2, 2], equi_channels=[64, 128, 256, 512],
        equi_out_dim=1024, fusion_method='concat', use_projection=False,
        lr=1e-3, optimizer='adamw', weight_decay=1e-4, momentum=0.9,
        warmpup_steps=300, milestones=[8, 14], lr_mult=0.3,
        loss_name='MultiSimilarityLoss', miner_name='MultiSimilarityMiner',
        miner_margin=0.1, faiss_gpu=False,
    )


@torch.no_grad()
def extract_branch_descs(model, dataloader):
    descs1, descs2 = [], []
    for imgs, _ in dataloader:
        imgs = imgs.to(DEVICE)
        feat1 = model.backbone(imgs)
        feat2 = model.backbone2(imgs)
        d1 = model.aggregator.branch1_aggregator(feat1)
        d2 = model.aggregator.branch2_aggregator(feat2)
        d1 = F.normalize(d1, p=2, dim=1)
        d2 = F.normalize(d2, p=2, dim=1)
        descs1.append(d1.cpu().numpy())
        descs2.append(d2.cpu().numpy())
    return np.vstack(descs1), np.vstack(descs2)


def load_ckpt(ckpt_path):
    model = build_model()
    load_boq_pretrained(model)
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith('.filter')]
    if real_missing or unexpected:
        print(f"  [warn] load: {len(real_missing)} real missing, {len(unexpected)} unexpected")
    return model.to(DEVICE).eval()


def per_query_analysis(desc_boq_db, desc_boq_q, desc_equi_db, desc_equi_q,
                        db_poses, query_poses):
    """Return per-query records:
       [(yaw_diff_min_to_positive, boq_correct_top1, drvpr_correct_top1), ...]
       for each VALID query (has at least one yaw positive in top-K).
    """
    # Rotate query trajectory by θ (same as recall eval)
    theta_rad = np.deg2rad(THETA_DEGREES)
    q_poses = query_poses.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    # BUGFIX: compute both rotated coords FIRST then assign — qx is a view, in-place
    # update of q_poses[:, 3] would corrupt qx before y rotation reads it.
    qx_rot = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    qy_rot = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    q_poses[:, 3], q_poses[:, 7] = qx_rot, qy_rot

    db_x, db_y = db_poses[:, 3], db_poses[:, 7]

    # Stage-1 FAISS on desc_boq
    index = faiss.IndexFlatIP(desc_boq_db.shape[1])
    index.add(desc_boq_db)
    _, top_k_idx = index.search(desc_boq_q, TOP_K)

    records = []
    for q_idx in range(len(q_poses)):
        # Position positives (within 5m)
        dist_sq = (q_poses[q_idx, 3] - db_x)**2 + (q_poses[q_idx, 7] - db_y)**2
        pos_positives = np.where(dist_sq < GT_THRES**2)[0]
        if len(pos_positives) == 0:
            continue

        # Yaw filter
        q_yaw = get_yaw_from_pose(q_poses[q_idx])
        yaw_positives = []
        yaw_diffs = []
        for pp in pos_positives:
            db_yaw = get_yaw_from_pose(db_poses[pp])
            diff = abs(q_yaw - db_yaw)
            if diff > 180:
                diff = 360 - diff
            if diff <= YAW_THRESHOLD:
                yaw_positives.append(pp)
                yaw_diffs.append(diff)
        if not yaw_positives:
            continue

        yaw_positive_set = set(yaw_positives)
        # "Rotation difficulty" = min yaw_diff among valid positives
        min_yaw_diff = min(yaw_diffs)

        # Top-1 under β=0 (pure BoQ) and β=0.5
        cands = top_k_idx[q_idx]
        boq_sim = desc_boq_q[q_idx] @ desc_boq_db[cands].T
        equi_sim = desc_equi_q[q_idx] @ desc_equi_db[cands].T

        top1_b0 = cands[np.argmax(boq_sim)]
        top1_b5 = cands[np.argmax(0.5 * boq_sim + 0.5 * equi_sim)]

        records.append({
            'yaw_diff': min_yaw_diff,
            'boq_correct': int(top1_b0 in yaw_positive_set),
            'drvpr_correct': int(top1_b5 in yaw_positive_set),
            'n_positives': len(yaw_positives),
        })

    return records


def main():
    all_records = []
    for seed, ckpt in CKPTS.items():
        print(f"\n--- seed={seed} ---")
        print(f"ckpt: {ckpt}")
        if not os.path.exists(ckpt):
            sys.exit(f"MISSING CKPT: {ckpt}")

        model = load_ckpt(ckpt)
        # Load datasets
        db_ds = InferDataset(SEQS[0], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        q_ds = InferDataset(SEQS[1], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
        q_dl = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

        d1_db, d2_db = extract_branch_descs(model, db_dl)
        d1_q, d2_q = extract_branch_descs(model, q_dl)
        print(f"  desc_boq db={d1_db.shape} q={d1_q.shape}")
        print(f"  desc_equi db={d2_db.shape} q={d2_q.shape}")

        recs = per_query_analysis(d1_db, d1_q, d2_db, d2_q, db_ds.poses, q_ds.poses)
        for r in recs:
            r['seed'] = seed
        all_records.extend(recs)
        print(f"  {len(recs)} valid queries")

        del model
        torch.cuda.empty_cache()

    # Aggregate by bucket
    buckets = [(0, 20), (20, 40), (40, 60), (60, 80)]
    print("\n" + "=" * 80)
    print("Per-Yaw Bucket Analysis (aggregated across 3 seeds)")
    print("=" * 80)
    print(f"{'bucket':>13s}  {'n_q':>5s}  {'BoQ R@1':>9s}  {'DR-VPR R@1':>11s}  {'Δ':>7s}  "
          f"{'flip→✓':>7s}  {'flip→✗':>7s}")
    print("-" * 80)

    total_n = total_boq = total_drvpr = 0
    for lo, hi in buckets:
        bucket = [r for r in all_records if lo <= r['yaw_diff'] < hi]
        n = len(bucket)
        if n == 0:
            print(f"  [{lo:2d}°, {hi:2d}°)  {n:5d}  {'n/a':>9s}  {'n/a':>11s}  {'n/a':>7s}")
            continue
        boq_r1 = sum(r['boq_correct'] for r in bucket) / n * 100
        drvpr_r1 = sum(r['drvpr_correct'] for r in bucket) / n * 100
        # Count flips: BoQ wrong → DR-VPR correct (✓); BoQ correct → DR-VPR wrong (✗)
        flip_pos = sum(1 for r in bucket if r['boq_correct'] == 0 and r['drvpr_correct'] == 1)
        flip_neg = sum(1 for r in bucket if r['boq_correct'] == 1 and r['drvpr_correct'] == 0)
        print(f"  [{lo:2d}°, {hi:2d}°)  {n:5d}  {boq_r1:8.2f}%  {drvpr_r1:10.2f}%  "
              f"{drvpr_r1-boq_r1:+6.2f}  {flip_pos:7d}  {flip_neg:7d}")
        total_n += n
        total_boq += sum(r['boq_correct'] for r in bucket)
        total_drvpr += sum(r['drvpr_correct'] for r in bucket)

    print("-" * 80)
    print(f"{'TOTAL':>13s}  {total_n:5d}  {total_boq/total_n*100:8.2f}%  "
          f"{total_drvpr/total_n*100:10.2f}%  "
          f"{(total_drvpr-total_boq)/total_n*100:+6.2f}")

    # Also compute bucket distribution (sanity check)
    print("\nBucket distribution (query counts):")
    for lo, hi in buckets:
        n = sum(1 for r in all_records if lo <= r['yaw_diff'] < hi)
        print(f"  [{lo:2d}°, {hi:2d}°): {n} queries")

    # Save raw records for further analysis
    import json
    with open('per_yaw_records.json', 'w') as f:
        json.dump(all_records, f, indent=2)
    print("\nRaw records saved to per_yaw_records.json")


if __name__ == '__main__':
    main()
