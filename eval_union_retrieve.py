"""
Path A: Union retrieve experiment.

Hypothesis: In high-yaw buckets ([60°, 80°)), BoQ stage-1 retrieval fails to
include the true positive in the top-100. equi-based retrieval (which is
rotation-invariant by structure) might find it. By taking the UNION of
BoQ top-K and equi top-K candidates, then reranking, we should rescue some
high-yaw queries.

Compared variants:
  Baseline:   Stage-1 BoQ top-100 → rerank β=0.5
  Union50/50: Stage-1 (BoQ top-50 ∪ equi top-50) → rerank β=0.5
  Union100/100: Stage-1 (BoQ top-100 ∪ equi top-100) → rerank β=0.5

Reports both overall R@1 and per-yaw bucket breakdown.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

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
BETA = 0.5

VARIANTS = [
    ('baseline_K100',     'boq_only',  100, None),   # BoQ top-100, no union
    ('union_50_50',       'union',      50,   50),   # 50 from each, union
    ('union_100_100',     'union',     100,  100),   # 100 from each, union (~200 cands)
    ('union_25_75',       'union',      25,   75),   # bias towards equi recall
    ('union_75_25',       'union',      75,   25),   # bias towards BoQ recall
]

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


def get_candidates(d1_db, d1_q, d2_db, d2_q, mode, k_boq, k_equi):
    """Return per-query candidate set indices."""
    n_q = d1_q.shape[0]
    if mode == 'boq_only':
        idx_boq = faiss.IndexFlatIP(d1_db.shape[1]); idx_boq.add(d1_db)
        _, top_k = idx_boq.search(d1_q, k_boq)
        return [list(top_k[i]) for i in range(n_q)]
    elif mode == 'union':
        idx_boq = faiss.IndexFlatIP(d1_db.shape[1]); idx_boq.add(d1_db)
        idx_equi = faiss.IndexFlatIP(d2_db.shape[1]); idx_equi.add(d2_db)
        _, tk_b = idx_boq.search(d1_q, k_boq)
        _, tk_e = idx_equi.search(d2_q, k_equi)
        cand = []
        for i in range(n_q):
            seen = set()
            merged = []
            for c in tk_b[i]:
                if c not in seen:
                    merged.append(int(c)); seen.add(int(c))
            for c in tk_e[i]:
                if c not in seen:
                    merged.append(int(c)); seen.add(int(c))
            cand.append(merged)
        return cand
    raise ValueError(mode)


def per_query_top1(d1_db, d1_q, d2_db, d2_q, candidates, beta):
    """Return per-query top-1 db_idx using rerank with weighted score."""
    n_q = d1_q.shape[0]
    top1 = np.zeros(n_q, dtype=np.int64)
    for i in range(n_q):
        cands = np.array(candidates[i], dtype=np.int64)
        boq_sim = d1_q[i] @ d1_db[cands].T
        equi_sim = d2_q[i] @ d2_db[cands].T
        score = (1 - beta) * boq_sim + beta * equi_sim
        top1[i] = cands[np.argmax(score)]
    return top1


def yaw_bucket(diff):
    if diff < 20: return '[ 0°, 20°)'
    if diff < 40: return '[20°, 40°)'
    if diff < 60: return '[40°, 60°)'
    return '[60°, 80°)'


def evaluate(top1_per_query, query_poses, db_poses):
    """Return list of records [{yaw_diff, correct, bucket}, ...]"""
    theta_rad = np.deg2rad(THETA_DEGREES)
    q_poses = query_poses.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    q_poses[:, 3] = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    q_poses[:, 7] = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]

    records = []
    for q_idx in range(len(q_poses)):
        dist_sq = (q_poses[q_idx, 3] - db_x)**2 + (q_poses[q_idx, 7] - db_y)**2
        pp = np.where(dist_sq < GT_THRES**2)[0]
        if len(pp) == 0: continue

        q_yaw = get_yaw_from_pose(q_poses[q_idx])
        yaw_pos, yaw_diffs = [], []
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180: d = 360 - d
            if d <= YAW_THRESHOLD:
                yaw_pos.append(p); yaw_diffs.append(d)
        if not yaw_pos: continue

        min_diff = min(yaw_diffs)
        records.append({
            'yaw_diff': min_diff,
            'bucket': yaw_bucket(min_diff),
            'correct': int(top1_per_query[q_idx] in set(yaw_pos)),
        })
    return records


def main():
    # Aggregate: variant -> bucket -> [0,1,...] correct flags
    from collections import defaultdict
    results = {v[0]: defaultdict(list) for v in VARIANTS}
    cand_sizes = {v[0]: [] for v in VARIANTS}

    for seed, ckpt in CKPTS.items():
        print(f"\n--- seed={seed} ---")
        if not os.path.exists(ckpt): sys.exit(f"MISSING: {ckpt}")
        model = load_ckpt(ckpt)
        db_ds = InferDataset(SEQS[0], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        q_ds = InferDataset(SEQS[1], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
        q_dl = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

        d1_db, d2_db = extract_branch_descs(model, db_dl)
        d1_q, d2_q = extract_branch_descs(model, q_dl)
        print(f"  desc shapes: boq db {d1_db.shape}, equi db {d2_db.shape}")

        for vname, mode, kb, ke in VARIANTS:
            cands = get_candidates(d1_db, d1_q, d2_db, d2_q, mode, kb, ke)
            avg_size = np.mean([len(c) for c in cands])
            cand_sizes[vname].append(avg_size)
            top1 = per_query_top1(d1_db, d1_q, d2_db, d2_q, cands, BETA)
            recs = evaluate(top1, q_ds.poses, db_ds.poses)
            for r in recs:
                results[vname][r['bucket']].append(r['correct'])
            tot_correct = sum(r['correct'] for r in recs)
            tot_n = len(recs)
            print(f"  {vname:18s}: avg_cands={avg_size:5.1f}  R@1={tot_correct/tot_n*100:.2f}% ({tot_correct}/{tot_n})")

        del model
        torch.cuda.empty_cache()

    # Aggregated table
    print("\n" + "=" * 100)
    print(f"Path A: Union Retrieve Comparison (3 seeds × ConSLAM, β={BETA})")
    print("=" * 100)
    buckets = ['[ 0°, 20°)', '[20°, 40°)', '[40°, 60°)', '[60°, 80°)']
    print(f"{'variant':>20s}  {'avg cands':>10s}", end='')
    for b in buckets:
        print(f"  {b:>13s}", end='')
    print(f"  {'TOTAL':>10s}")
    print('-' * 100)

    for vname, _, kb, ke in VARIANTS:
        print(f"{vname:>20s}  {np.mean(cand_sizes[vname]):>10.1f}", end='')
        total_n = total_c = 0
        for b in buckets:
            flags = results[vname][b]
            n = len(flags)
            if n == 0:
                print(f"  {'n/a':>13s}", end='')
            else:
                r1 = sum(flags) / n * 100
                print(f"  {r1:6.2f}% ({n:>3d})", end='')
                total_n += n
                total_c += sum(flags)
        print(f"  {total_c/total_n*100:>9.2f}%")

    print('=' * 100)
    print("Notes:")
    print("  - Lower avg_cands = fewer rerank candidates (faster, but may miss positives)")
    print("  - bucket counts in parens; '[XX°, YY°)' means yaw_diff to nearest GT positive")
    print("  - β=0.5 fixed throughout; rerank score = 0.5·boq_sim + 0.5·equi_sim")


if __name__ == '__main__':
    main()
