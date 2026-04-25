"""
Two-stage retrieve-then-rerank evaluation for DR-VPR.

Stage 1: Retrieve top-K candidates using Branch 1 (BoQ) descriptors via FAISS.
Stage 2: Rerank candidates using Branch 2 (equivariant) cosine similarity.

Also sweeps a weighted rerank: score = (1-β)*boq_sim + β*equi_sim

Usage:
    FUSION_METHOD=concat DRVPR_CKPT=<path> mamba run -n drvpr python eval_rerank.py
"""
from __future__ import annotations

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset, evaluateResults
from train_fusion import VPRModel, load_boq_pretrained

# ---------------------- config ------------------------------------------
CHECKPOINT_PATH = os.environ.get(
    'DRVPR_CKPT',
    './LOGS/resnet50_DualBranch_freeze_boq_seed190223/lightning_logs/version_0/checkpoints/'
)
DATASET_PATH = './datasets/ConSLAM/'
SEQS = ['Sequence5', 'Sequence4']
THETA_DEGREES = 15.0
YAW_THRESHOLD = 80.0
TOP_K = 100
BETA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)


def build_model():
    os.environ.setdefault('FUSION_METHOD', 'concat')
    os.environ.setdefault('GROUP_POOL_MODE', 'max')
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
def extract_branch_descriptors(model, dataloader):
    """Extract separate Branch 1 and Branch 2 descriptors for all images."""
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


def rerank_with_beta(boq_db, boq_q, equi_db, equi_q, top_k_indices, beta):
    """
    Rerank top-K candidates using weighted score.
    score = (1-β) * boq_cosine_sim + β * equi_cosine_sim
    Returns reranked indices for each query.
    """
    n_queries = boq_q.shape[0]
    reranked = np.zeros((n_queries, top_k_indices.shape[1]), dtype=np.int64)

    for q_idx in range(n_queries):
        candidates = top_k_indices[q_idx]

        boq_sim = boq_q[q_idx] @ boq_db[candidates].T
        equi_sim = equi_q[q_idx] @ equi_db[candidates].T

        combined = (1 - beta) * boq_sim + beta * equi_sim
        order = np.argsort(-combined)
        reranked[q_idx] = candidates[order]

    return reranked


def compute_recall_from_reranked(reranked_indices, query_poses, db_poses,
                                  gt_thres=5.0, yaw_threshold=80.0,
                                  theta_degrees=15.0, offset=(0.0, 0.0)):
    """Compute R@1, R@5, R@10 from reranked indices using position + yaw matching."""
    from Conslam_dataset_rot import get_yaw_from_pose

    theta_rad = np.deg2rad(theta_degrees)
    q_poses = query_poses.copy()
    q_poses[:, 3] += offset[0]
    q_poses[:, 7] += offset[1]
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    qx_rot = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    qy_rot = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    q_poses[:, 3], q_poses[:, 7] = qx_rot, qy_rot

    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    tp1 = tp5 = tp10 = total = 0

    for q_idx in range(len(q_poses)):
        dist_sq = (q_poses[q_idx, 3] - db_x)**2 + (q_poses[q_idx, 7] - db_y)**2
        position_positives = set(np.where(dist_sq < gt_thres**2)[0])
        if not position_positives:
            continue

        query_yaw = get_yaw_from_pose(q_poses[q_idx])
        yaw_positives = set()
        for pp in position_positives:
            db_yaw = get_yaw_from_pose(db_poses[pp])
            if abs(query_yaw - db_yaw) <= yaw_threshold or abs(query_yaw - db_yaw) >= (360 - yaw_threshold):
                yaw_positives.add(pp)

        if not yaw_positives:
            continue

        total += 1
        preds = reranked_indices[q_idx]
        if preds[0] in yaw_positives:
            tp1 += 1
        if any(p in yaw_positives for p in preds[:5]):
            tp5 += 1
        if any(p in yaw_positives for p in preds[:10]):
            tp10 += 1

    if total == 0:
        return 0.0, 0.0, 0.0
    return tp1 / total, tp5 / total, tp10 / total


def find_best_checkpoint(ckpt_dir):
    """Find the checkpoint with highest val R@1 in the directory."""
    import glob
    ckpts = glob.glob(os.path.join(ckpt_dir, '**/*.ckpt'), recursive=True)
    if not ckpts:
        return None
    best = max(ckpts, key=lambda f: float(f.split('[')[1].split(']')[0]))
    return best


def main():
    ckpt = CHECKPOINT_PATH
    if os.path.isdir(ckpt):
        ckpt = find_best_checkpoint(ckpt)
        if not ckpt:
            sys.exit(f"No checkpoint found in {CHECKPOINT_PATH}")
    print(f"Checkpoint: {ckpt}")

    model = build_model()
    load_boq_pretrained(model)
    state = torch.load(ckpt, map_location='cpu')['state_dict']
    # Explicit missing/unexpected key report: previously used strict=False which
    # silently dropped shape-mismatched weights (e.g. when ckpt was trained with a
    # different GROUP_POOL_MODE than build_model() creates — tier2 ckpt has
    # branch2_aggregator Linear weight shape (1024, 320) while a max-mode model
    # creates (1024, 64), causing silent weight skip and nonsensical results).
    missing, unexpected = model.load_state_dict(state, strict=False)
    # e2cnn's R2Conv registers a lazily-computed `.filter` buffer that is NOT saved in
    # state_dict (the trained weights are in `.weights`; `.filter` is recomputed from
    # them on first forward). So these are always benign "missing" — whitelist them.
    real_missing = [k for k in missing if not k.endswith('.filter')]
    print(f"load_state_dict: {len(missing)} missing ({len(real_missing)} real, "
          f"{len(missing) - len(real_missing)} benign .filter buffers), "
          f"{len(unexpected)} unexpected")
    if real_missing:
        print(f"  REAL MISSING (sample): {real_missing[:5]}")
    if unexpected:
        print(f"  UNEXPECTED (sample): {list(unexpected)[:5]}")
    # Treat real mismatch as fatal (likely GROUP_POOL_MODE vs ckpt disagreement).
    # Set DRVPR_EVAL_ALLOW_MISMATCH=1 to bypass for debug.
    if (real_missing or unexpected) and os.environ.get('DRVPR_EVAL_ALLOW_MISMATCH', '0') != '1':
        sys.exit("Aborting: state_dict mismatch. Check GROUP_POOL_MODE matches the ckpt's "
                 "training config. Set DRVPR_EVAL_ALLOW_MISMATCH=1 to proceed anyway.")
    model = model.to(DEVICE).eval()
    print(f"Model built with GROUP_POOL_MODE={os.environ.get('GROUP_POOL_MODE', 'max')}, "
          f"branch2 invariant channels={model.backbone2.out_channels}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Params: {total_p:,} total, {trainable:,} trainable")

    # Load datasets
    datasets, dataloaders = [], []
    for seq in SEQS:
        ds = InferDataset(seq, dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
        datasets.append(ds)
        dataloaders.append(dl)
        print(f"Loaded {seq}: {len(ds)} images")

    # Extract separate branch descriptors
    print("\nExtracting Branch 1 (BoQ) + Branch 2 (equi) descriptors...")
    boq_descs, equi_descs = [], []
    for i, dl in enumerate(dataloaders):
        d1, d2 = extract_branch_descriptors(model, dl)
        boq_descs.append(d1)
        equi_descs.append(d2)
        print(f"  {SEQS[i]}: boq={d1.shape}, equi={d2.shape}")

    # Stage 1: FAISS retrieval using BoQ descriptors
    print(f"\nStage 1: FAISS top-{TOP_K} retrieval using BoQ descriptors...")
    db_boq = boq_descs[0]
    index = faiss.IndexFlatIP(db_boq.shape[1])
    index.add(db_boq)

    results_per_beta = {}
    for i in range(1, len(datasets)):
        q_boq = boq_descs[i]
        q_equi = equi_descs[i]
        db_equi = equi_descs[0]

        sims, top_k_idx = index.search(q_boq, TOP_K)

        db_poses = datasets[0].poses
        q_poses = datasets[i].poses

        print(f"\nQuery sequence: {SEQS[i]} ({len(q_boq)} queries)")
        print(f"{'beta':>5s}  {'R@1':>8s}  {'R@5':>8s}  {'R@10':>8s}")
        print("-" * 36)

        for beta in BETA_VALUES:
            if beta == 0.0:
                reranked = top_k_idx
            else:
                reranked = rerank_with_beta(db_boq, q_boq, db_equi, q_equi, top_k_idx, beta)

            r1, r5, r10 = compute_recall_from_reranked(
                reranked, q_poses, db_poses,
                gt_thres=5.0, yaw_threshold=YAW_THRESHOLD,
                theta_degrees=THETA_DEGREES
            )
            print(f"{beta:5.1f}  {r1*100:7.2f}%  {r5*100:7.2f}%  {r10*100:7.2f}%")

            if beta not in results_per_beta:
                results_per_beta[beta] = []
            results_per_beta[beta].append({'R1': r1, 'R5': r5, 'R10': r10})

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY (averaged over query sequences)")
    print("=" * 50)
    best_beta = None
    best_r1 = -1
    for beta in BETA_VALUES:
        entries = results_per_beta[beta]
        avg_r1 = np.mean([e['R1'] for e in entries])
        avg_r5 = np.mean([e['R5'] for e in entries])
        avg_r10 = np.mean([e['R10'] for e in entries])
        marker = ""
        if avg_r1 > best_r1:
            best_r1 = avg_r1
            best_beta = beta
            marker = " <-- best"
        print(f"β={beta:.1f}  R@1={avg_r1*100:.2f}%  R@5={avg_r5*100:.2f}%  R@10={avg_r10*100:.2f}%{marker}")

    print(f"\nBest β={best_beta:.1f} → R@1={best_r1*100:.2f}%")
    print(f"BoQ standalone baseline: R@1=62.21% (for reference)")


if __name__ == "__main__":
    main()
