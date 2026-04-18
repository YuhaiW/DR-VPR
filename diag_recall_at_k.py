"""
Diagnostic: per-yaw-bucket recall@K for BoQ-only, equi-only, and union retrieval.

Tells us whether equi has the positive in its top-K when BoQ doesn't.
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

CKPT = 'LOGS/resnet50_DualBranch_freeze_boq_seed1/lightning_logs/version_0/checkpoints/resnet50_DualBranch_C8_concat_seed1_epoch(02)_R1[0.6417].ckpt'

K_VALUES = [1, 5, 10, 50, 100, 200, 401]   # 401 = full DB


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


def main():
    print(f"Loading {CKPT}")
    model = build_model()
    load_boq_pretrained(model)
    state = torch.load(CKPT, map_location='cpu')['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith('.filter')]
    if real_missing or unexpected:
        print(f"  [warn] {len(real_missing)} real missing, {len(unexpected)} unexpected")
    model = model.to(DEVICE).eval()

    db_ds = InferDataset(SEQS[0], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
    q_ds = InferDataset(SEQS[1], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)
    q_dl = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)
    d1_db, d2_db = extract(model, db_dl)
    d1_q, d2_q = extract(model, q_dl)

    # Build FAISS index for both
    idx_boq = faiss.IndexFlatIP(d1_db.shape[1]); idx_boq.add(d1_db)
    idx_equi = faiss.IndexFlatIP(d2_db.shape[1]); idx_equi.add(d2_db)
    K_max = max(K_VALUES)
    _, top_boq = idx_boq.search(d1_q, K_max)    # (n_q, K_max)
    _, top_equi = idx_equi.search(d2_q, K_max)

    # Compute per-query yaw_positives + bucket
    theta_rad = np.deg2rad(THETA_DEGREES)
    q_poses = q_ds.poses.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    q_poses[:, 3] = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    q_poses[:, 7] = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    db_poses = db_ds.poses
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]

    # Per query: yaw_positives (set), min_yaw_diff
    valid_q = []
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
        valid_q.append({
            'q_idx': q_idx, 'pos': set(ypos),
            'bucket': yaw_bucket(min(ydiffs)),
        })

    print(f"\n{len(valid_q)} valid queries (seed=1 only for diag speed)")
    print(f"Bucket distribution:")
    bucket_counts = defaultdict(int)
    for q in valid_q:
        bucket_counts[q['bucket']] += 1
    for b in ['[ 0°, 20°)', '[20°, 40°)', '[40°, 60°)', '[60°, 80°)']:
        print(f"  {b}: {bucket_counts[b]}")

    # Per-bucket recall@K for BoQ-only, equi-only, union
    print("\n" + "=" * 110)
    print("Per-bucket Recall@K — % of queries whose GT positive is in top-K")
    print("=" * 110)
    print(f"{'bucket':>13s}  {'method':>10s}", end='')
    for k in K_VALUES:
        print(f"  {'R@'+str(k):>7s}", end='')
    print()
    print('-' * 110)

    for b in ['[ 0°, 20°)', '[20°, 40°)', '[40°, 60°)', '[60°, 80°)']:
        bq = [q for q in valid_q if q['bucket'] == b]
        n = len(bq)
        if n == 0: continue
        for method in ['boq', 'equi', 'union']:
            print(f"  {b:>13s}  {method:>10s}", end='')
            for k in K_VALUES:
                hit = 0
                for q in bq:
                    if method == 'boq':
                        cand = set(top_boq[q['q_idx'], :k])
                    elif method == 'equi':
                        cand = set(top_equi[q['q_idx'], :k])
                    else:  # union
                        # Half from each
                        kh = k // 2
                        cand = set(top_boq[q['q_idx'], :kh]) | set(top_equi[q['q_idx'], :kh])
                    if cand & q['pos']:
                        hit += 1
                print(f"  {hit/n*100:>6.1f}%", end='')
            print()
        print()


if __name__ == '__main__':
    main()
