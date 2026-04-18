"""
Two-stage rerank eval with STANDALONE equi model + frozen official BoQ.

  Stage 1: official BoQ(ResNet50, 16384-d) FAISS top-K
  Stage 2: rerank with weighted score (1-β)·boq_sim + β·equi_sim
           where desc_equi comes from standalone E2ResNetMultiScale ckpt.

Usage:
    EQUI_CKPT=<path/to/equi.ckpt> mamba run -n drvpr python eval_rerank_standalone.py
"""
from __future__ import annotations
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset
from models.equi_multiscale import E2ResNetMultiScale

# ---- config ----
EQUI_CKPT_DEFAULT = './LOGS/equi_standalone_seed1_equi_standalone_ms/lightning_logs/version_0/checkpoints/'
EQUI_CKPT = os.environ.get('EQUI_CKPT', EQUI_CKPT_DEFAULT)
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
BOQ_IMG_SIZE = int(os.environ.get('BOQ_IMG_SIZE', '320'))


def load_official_boq(device):
    """Load BoQ(ResNet50) from official torch hub release."""
    print("Loading BoQ(ResNet50) from torch.hub ...")
    model = torch.hub.load(
        "amaralibey/bag-of-queries", "get_trained_boq",
        backbone_name="resnet50", output_dim=16384,
    )
    model = model.eval().to(device)
    print(f"  BoQ(ResNet50) loaded, will run at {BOQ_IMG_SIZE}x{BOQ_IMG_SIZE}")
    return model


def load_equi_standalone(ckpt_path, device):
    """Load standalone equi model from Lightning ckpt."""
    print(f"Loading standalone equi from: {ckpt_path}")
    model = E2ResNetMultiScale(
        orientation=8, layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512), out_dim=1024,
    )
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    # Strip "model." prefix added by Lightning module wrapper
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith('.filter')]
    if real_missing:
        print(f"  WARN: {len(real_missing)} real missing keys: {real_missing[:5]}")
    if unexpected:
        print(f"  WARN: {len(unexpected)} unexpected keys: {list(unexpected)[:5]}")
    return model.eval().to(device)


def find_best_ckpt(path):
    if os.path.isfile(path):
        return path
    ckpts = glob.glob(os.path.join(path, '**/*.ckpt'), recursive=True)
    if not ckpts:
        sys.exit(f"No .ckpt found under {path}")
    # Pick highest R@1 from filename pattern R1[X.XXXX]
    def get_r1(p):
        try:
            return float(p.split('R1[')[1].split(']')[0])
        except Exception:
            return -1
    best = max(ckpts, key=get_r1)
    print(f"  selected best ckpt: {best}")
    return best


@torch.no_grad()
def extract_boq(model, imgs):
    """BoQ forward; handles tuple return of (desc, attentions)."""
    out = model(imgs)
    if isinstance(out, tuple):
        out = out[0]
    return F.normalize(out, p=2, dim=1)


@torch.no_grad()
def extract_equi(model, imgs):
    return model(imgs)   # already L2-normalized in E2ResNetMultiScale.forward


def make_dataloader(seq, img_size):
    ds = InferDataset(seq, dataset_path=DATASET_PATH, img_size=img_size)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)
    return ds, dl


def extract_all(boq_model, equi_model, seq):
    """Extract both desc_boq (BoQ resolution) and desc_equi (320x320) for one sequence."""
    # BoQ: native resolution
    ds_boq, dl_boq = make_dataloader(seq, (BOQ_IMG_SIZE, BOQ_IMG_SIZE))
    descs_boq = []
    for imgs, _ in dl_boq:
        imgs = imgs.to(DEVICE)
        descs_boq.append(extract_boq(boq_model, imgs).cpu().numpy())
    descs_boq = np.vstack(descs_boq)

    # Equi: 320x320
    ds_equi, dl_equi = make_dataloader(seq, IMG_SIZE)
    descs_equi = []
    for imgs, _ in dl_equi:
        imgs = imgs.to(DEVICE)
        descs_equi.append(extract_equi(equi_model, imgs).cpu().numpy())
    descs_equi = np.vstack(descs_equi)

    # Both datasets are the same sequence (same poses) — return one
    return descs_boq, descs_equi, ds_boq.poses


def rerank_with_beta(boq_db, boq_q, equi_db, equi_q, top_k_indices, beta):
    n_q = boq_q.shape[0]
    reranked = np.zeros((n_q, top_k_indices.shape[1]), dtype=np.int64)
    for q_idx in range(n_q):
        cands = top_k_indices[q_idx]
        bs = boq_q[q_idx] @ boq_db[cands].T
        es = equi_q[q_idx] @ equi_db[cands].T
        score = (1 - beta) * bs + beta * es
        order = np.argsort(-score)
        reranked[q_idx] = cands[order]
    return reranked


def compute_recall_from_reranked(reranked, q_poses_raw, db_poses,
                                  gt_thres=5.0, yaw_threshold=80.0,
                                  theta_degrees=15.0, offset=(0.0, 0.0)):
    from Conslam_dataset_rot import get_yaw_from_pose
    theta = np.deg2rad(theta_degrees)
    q = q_poses_raw.copy()
    q[:, 3] += offset[0]; q[:, 7] += offset[1]
    qx, qy = q[:, 3], q[:, 7]
    q[:, 3] = qx * np.cos(theta) - qy * np.sin(theta)
    q[:, 7] = qx * np.sin(theta) + qy * np.cos(theta)
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    tp1 = tp5 = tp10 = total = 0
    for q_idx in range(len(q)):
        ds = (q[q_idx, 3] - db_x) ** 2 + (q[q_idx, 7] - db_y) ** 2
        pp = set(np.where(ds < gt_thres ** 2)[0])
        if not pp: continue
        q_yaw = get_yaw_from_pose(q[q_idx])
        ypos = set()
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180: d = 360 - d
            if d <= yaw_threshold: ypos.add(p)
        if not ypos: continue
        total += 1
        preds = reranked[q_idx]
        if preds[0] in ypos: tp1 += 1
        if any(p in ypos for p in preds[:5]): tp5 += 1
        if any(p in ypos for p in preds[:10]): tp10 += 1
    if total == 0:
        return 0.0, 0.0, 0.0
    return tp1 / total, tp5 / total, tp10 / total


def main():
    ckpt = find_best_ckpt(EQUI_CKPT)
    boq_model = load_official_boq(DEVICE)
    equi_model = load_equi_standalone(ckpt, DEVICE)

    print(f"\nExtracting features (BoQ@{BOQ_IMG_SIZE}, Equi@{IMG_SIZE[0]})...")
    db_boq, db_equi, db_poses = extract_all(boq_model, equi_model, SEQS[0])
    print(f"  DB ({SEQS[0]}): boq={db_boq.shape}, equi={db_equi.shape}")
    q_boq, q_equi, q_poses = extract_all(boq_model, equi_model, SEQS[1])
    print(f"  Query ({SEQS[1]}): boq={q_boq.shape}, equi={q_equi.shape}")

    print(f"\nStage-1: BoQ FAISS top-{TOP_K}")
    index = faiss.IndexFlatIP(db_boq.shape[1])
    index.add(db_boq)
    _, top_k_idx = index.search(q_boq, TOP_K)

    print(f"\nStage-2: rerank β sweep")
    print(f"{'beta':>5s}  {'R@1':>8s}  {'R@5':>8s}  {'R@10':>8s}")
    print("-" * 40)
    results = {}
    for beta in BETA_VALUES:
        if beta == 0.0:
            reranked = top_k_idx
        else:
            reranked = rerank_with_beta(db_boq, q_boq, db_equi, q_equi, top_k_idx, beta)
        r1, r5, r10 = compute_recall_from_reranked(reranked, q_poses, db_poses,
                                                    yaw_threshold=YAW_THRESHOLD,
                                                    theta_degrees=THETA_DEGREES)
        results[beta] = (r1, r5, r10)
        print(f"{beta:5.1f}  {r1*100:7.2f}%  {r5*100:7.2f}%  {r10*100:7.2f}%")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    best_beta, best_r1 = max(results.items(), key=lambda kv: kv[1][0])
    for b in BETA_VALUES:
        r1, r5, r10 = results[b]
        marker = " <-- best" if b == best_beta else ""
        print(f"β={b:.1f}  R@1={r1*100:.2f}%  R@5={r5*100:.2f}%  R@10={r10*100:.2f}%{marker}")
    print(f"\nBest β={best_beta:.1f} → R@1={best_r1[0]*100:.2f}%")
    print(f"Reference: BoQ(ResNet50)@{BOQ_IMG_SIZE} alone → R@1=60.91%")
    print(f"Reference: freeze_boq + max + β=0.5 (current best) → R@1=61.45 ± 0.18%")


if __name__ == '__main__':
    main()
