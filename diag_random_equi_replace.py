"""
Stringent test: replace desc_equi with random unit vectors, see if rerank still gives gain.

If random vectors also give +0.5 R@1 → equi is providing ensemble noise, not real signal.
If random vectors give 0 or negative → equi's +0.54 is structural rotation signal.

Tests 3 conditions across 3 seeds:
  (A) Real desc_equi (our actual model)
  (B) Random unit vectors of same shape (1024-d, L2-normed)
  (C) Same random with different RNG seed (sanity, should give similar to B)
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
TOP_K = 100
BETA = 0.5
DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
GT_THRES = 5.0

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


def make_random_unit(shape, seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(shape).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def rerank_top1(d1_db, d1_q, d2_db, d2_q, beta):
    """Stage-1 BoQ FAISS top-K, stage-2 weighted rerank, return top-1 db idx per query."""
    n_q = d1_q.shape[0]
    idx_boq = faiss.IndexFlatIP(d1_db.shape[1]); idx_boq.add(d1_db)
    _, top_k = idx_boq.search(d1_q, TOP_K)
    top1 = np.zeros(n_q, dtype=np.int64)
    for i in range(n_q):
        cands = top_k[i]
        bs = d1_q[i] @ d1_db[cands].T
        es = d2_q[i] @ d2_db[cands].T
        score = (1 - beta) * bs + beta * es
        top1[i] = cands[np.argmax(score)]
    return top1


def evaluate(top1, q_poses_raw, db_poses):
    theta = np.deg2rad(THETA_DEGREES)
    q = q_poses_raw.copy()
    qx, qy = q[:, 3], q[:, 7]
    # BUGFIX: temp vars
    qx_rot = qx * np.cos(theta) - qy * np.sin(theta)
    qy_rot = qx * np.sin(theta) + qy * np.cos(theta)
    q[:, 3], q[:, 7] = qx_rot, qy_rot
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    correct = total = 0
    for i in range(len(q)):
        ds = (q[i, 3] - db_x) ** 2 + (q[i, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0: continue
        q_yaw = get_yaw_from_pose(q[i])
        ypos = []
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180: d = 360 - d
            if d <= YAW_THRESHOLD: ypos.append(int(p))
        if not ypos: continue
        total += 1
        if top1[i] in set(ypos): correct += 1
    return correct, total


def main():
    print("=" * 100)
    print("Stringent test: real desc_equi vs random unit vectors")
    print(f"β={BETA}, top-K={TOP_K}, theta=15°, yaw_threshold=80°")
    print("=" * 100)

    print(f"\n{'seed':>7s}  {'condition':>20s}  {'desc_equi source':>25s}  {'R@1 (%)':>9s}  {'Δ vs β=0':>10s}")
    print('-' * 100)

    for seed, ckpt in CKPTS.items():
        if not os.path.exists(ckpt):
            print(f"MISSING ckpt for seed={seed}: {ckpt}")
            continue
        model = build_model()
        load_boq_pretrained(model)
        model.load_state_dict(torch.load(ckpt, map_location='cpu')['state_dict'], strict=False)
        model = model.to(DEVICE).eval()

        db_ds = InferDataset(SEQS[0], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        q_ds = InferDataset(SEQS[1], dataset_path=DATASET_PATH, img_size=IMG_SIZE)
        db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
        q_dl = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
        d1_db, d2_db_real = extract(model, db_dl)
        d1_q, d2_q_real = extract(model, q_dl)

        # Baseline β=0 (no equi)
        # Use a dummy zero equi (since β=0 ignores equi anyway)
        zero_equi_db = np.zeros_like(d2_db_real)
        zero_equi_q = np.zeros_like(d2_q_real)
        top1_b0 = rerank_top1(d1_db, d1_q, zero_equi_db, zero_equi_q, beta=0.0)
        c0, t = evaluate(top1_b0, q_ds.poses, db_ds.poses)
        r1_b0 = c0 / t * 100

        # (A) Real equi @ β=0.5
        top1_real = rerank_top1(d1_db, d1_q, d2_db_real, d2_q_real, beta=BETA)
        c_real, _ = evaluate(top1_real, q_ds.poses, db_ds.poses)
        r1_real = c_real / t * 100

        # (B) Random unit vectors @ β=0.5 (seed = 0)
        d2_db_rand_b = make_random_unit(d2_db_real.shape, seed=0)
        d2_q_rand_b = make_random_unit(d2_q_real.shape, seed=1)
        top1_b = rerank_top1(d1_db, d1_q, d2_db_rand_b, d2_q_rand_b, beta=BETA)
        c_b, _ = evaluate(top1_b, q_ds.poses, db_ds.poses)
        r1_b = c_b / t * 100

        # (C) Random unit vectors @ β=0.5 (seed = 2 for sanity)
        d2_db_rand_c = make_random_unit(d2_db_real.shape, seed=2)
        d2_q_rand_c = make_random_unit(d2_q_real.shape, seed=3)
        top1_c = rerank_top1(d1_db, d1_q, d2_db_rand_c, d2_q_rand_c, beta=BETA)
        c_c, _ = evaluate(top1_c, q_ds.poses, db_ds.poses)
        r1_c = c_c / t * 100

        print(f"  seed={seed}: β=0 baseline = {r1_b0:.2f}% ({t} valid q)")
        print(f"  {seed:>7d}  {'(A) real desc_equi':>20s}  {'trained E2ResNet pool':>25s}  {r1_real:>9.2f}  {r1_real-r1_b0:>+10.2f}")
        print(f"  {seed:>7d}  {'(B) random vec [0,1]':>20s}  {'random unit, RNG seed 0/1':>25s}  {r1_b:>9.2f}  {r1_b-r1_b0:>+10.2f}")
        print(f"  {seed:>7d}  {'(C) random vec [2,3]':>20s}  {'random unit, RNG seed 2/3':>25s}  {r1_c:>9.2f}  {r1_c-r1_b0:>+10.2f}")
        print()

        del model
        torch.cuda.empty_cache()

    print("=" * 100)
    print("INTERPRETATION:")
    print("  If (A) gives clearly positive Δ AND (B)/(C) give 0 or negative Δ:")
    print("    → Real desc_equi carries STRUCTURAL signal beyond random noise.")
    print("  If (B)/(C) also give similar positive Δ as (A):")
    print("    → Our +0.54 'gain' is just ensemble noise effect, not equi-specific.")


if __name__ == '__main__':
    main()
