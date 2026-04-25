"""
Per-yaw bucket analysis for P1 Standalone on BOTH ConSLAM and ConPR.

- Uses official BoQ(ResNet50)@320 as stage-1 descriptor source.
- Uses P1 standalone E2ResNetMultiScale (3 seeds) as stage-2 rerank descriptor.
- Compares β=0 (pure BoQ retrieve) vs β=0.1 (P1 best fixed β).
- Bucket width: 10° (finer than v1's 20° buckets).
- Bug-fixed rotation (temp vars).

Outputs to stdout + saves per-dataset + per-bucket tables.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader
from collections import defaultdict

# ConSLAM / ConPR dataset imports
from Conslam_dataset_rot import InferDataset as ConslamInferDataset
from Conslam_dataset_rot import get_yaw_from_pose
from conpr_eval_dataset_rot import InferDataset as ConPRInferDataset
from models.equi_multiscale import E2ResNetMultiScale

# ---- config ----
DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
BOQ_IMG_SIZE = 320       # match BoQ training resolution
EQUI_IMG_SIZE = 320
TOP_K = 100
FIXED_BETA = 0.1          # P1 best fixed β (3-seed agreement)
GT_THRES = 5.0
BUCKET_WIDTH = 10
BUCKETS = [(b, b + BUCKET_WIDTH) for b in range(0, 80, BUCKET_WIDTH)]   # 8 buckets

# P1 standalone ckpts — val-best epoch per seed
CKPTS = {
    1:      'LOGS/equi_standalone_seed1_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed1_epoch(08)_R1[0.3510].ckpt',
    42:     'LOGS/equi_standalone_seed42_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed42_epoch(01)_R1[0.3283].ckpt',
    190223: 'LOGS/equi_standalone_seed190223_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed190223_epoch(04)_R1[0.3384].ckpt',
}

# Dataset configs
DATASETS = {
    'conslam': {
        'path': './datasets/ConSLAM/',
        'db_seq': 'Sequence5',
        'query_seq': 'Sequence4',
        'theta_degrees': 15.0,
        'yaw_threshold': 80.0,
        'dataset_cls': ConslamInferDataset,
    },
    'conpr': {
        'path': './datasets/ConPR/',
        'db_seq': '20230623',
        'query_seq': '20230809',
        'theta_degrees': 0.0,   # ConPR doesn't use query rotation
        'yaw_threshold': 80.0,
        'dataset_cls': ConPRInferDataset,
    },
}


# ---- model helpers ----

def load_official_boq():
    print("Loading official BoQ(ResNet50)@320 from torch.hub...")
    model = torch.hub.load(
        "amaralibey/bag-of-queries", "get_trained_boq",
        backbone_name="resnet50", output_dim=16384,
    )
    return model.eval().to(DEVICE)


def load_p1_equi(ckpt_path):
    model = E2ResNetMultiScale(
        orientation=16, layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512), out_dim=1024,
    )
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith('.filter')]
    if real_missing:
        print(f"  [warn] {len(real_missing)} real missing: {real_missing[:5]}")
    # .to(DEVICE) BEFORE .eval() — e2cnn R2Conv's expand_parameters runs inside
    # .eval() and needs weights + sampled_basis on same device; moving to CUDA
    # first keeps them consistent.
    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def extract_boq(boq_model, dl):
    descs = []
    for imgs, _ in dl:
        out = boq_model(imgs.to(DEVICE))
        if isinstance(out, tuple):
            out = out[0]
        descs.append(F.normalize(out, p=2, dim=1).cpu().numpy())
    return np.vstack(descs)


@torch.no_grad()
def extract_equi(equi_model, dl):
    descs = []
    for imgs, _ in dl:
        descs.append(equi_model(imgs.to(DEVICE)).cpu().numpy())   # already L2-normalized
    return np.vstack(descs)


# ---- per-query analysis with bug-fixed rotation ----

def yaw_bucket(diff):
    b = int(diff // BUCKET_WIDTH) * BUCKET_WIDTH
    return f'[{b:>2d}°, {b + BUCKET_WIDTH:>2d}°)'


def per_query_records(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                       q_poses_raw, db_poses, theta_degrees, yaw_threshold):
    """Return list of per-query records: (bucket, boq_correct, drvpr_correct)."""
    # Rotation (BUG-FIXED: temp vars!)
    theta_rad = np.deg2rad(theta_degrees)
    q_poses = q_poses_raw.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    qx_rot = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
    qy_rot = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)
    q_poses[:, 3], q_poses[:, 7] = qx_rot, qy_rot

    db_x, db_y = db_poses[:, 3], db_poses[:, 7]

    # Stage 1 FAISS top-K on BoQ
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k_idx = idx.search(d_boq_q, TOP_K)

    records = []
    for q_idx in range(len(q_poses)):
        dist_sq = (q_poses[q_idx, 3] - db_x) ** 2 + (q_poses[q_idx, 7] - db_y) ** 2
        pp = np.where(dist_sq < GT_THRES ** 2)[0]
        if len(pp) == 0:
            continue
        q_yaw = get_yaw_from_pose(q_poses[q_idx])
        ypos = []
        ydiffs = []
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180:
                d = 360 - d
            if d <= yaw_threshold:
                ypos.append(int(p))
                ydiffs.append(d)
        if not ypos:
            continue
        yset = set(ypos)
        min_diff = min(ydiffs)

        # Top-1 under β=0 (pure BoQ) and β=FIXED_BETA
        cands = top_k_idx[q_idx]
        boq_sim = d_boq_q[q_idx] @ d_boq_db[cands].T
        equi_sim = d_equi_q[q_idx] @ d_equi_db[cands].T

        top1_b0 = cands[np.argmax(boq_sim)]
        top1_bfx = cands[np.argmax((1 - FIXED_BETA) * boq_sim + FIXED_BETA * equi_sim)]

        records.append({
            'yaw_diff': min_diff,
            'bucket': yaw_bucket(min_diff),
            'boq_correct': int(top1_b0 in yset),
            'drvpr_correct': int(top1_bfx in yset),
        })
    return records


# ---- main loop ----

def run_dataset(ds_name, ds_cfg, boq_model):
    print(f"\n{'='*78}\nDataset: {ds_name.upper()} "
          f"({ds_cfg['db_seq']} db vs {ds_cfg['query_seq']} query, "
          f"θ={ds_cfg['theta_degrees']}°, yaw={ds_cfg['yaw_threshold']}°)\n{'='*78}")

    # Load datasets
    db_ds = ds_cfg['dataset_cls'](ds_cfg['db_seq'],
                                    dataset_path=ds_cfg['path'],
                                    img_size=(BOQ_IMG_SIZE, BOQ_IMG_SIZE))
    q_ds = ds_cfg['dataset_cls'](ds_cfg['query_seq'],
                                   dataset_path=ds_cfg['path'],
                                   img_size=(BOQ_IMG_SIZE, BOQ_IMG_SIZE))

    # Separate dataset instances for equi @ 320 if different (here same, 320 for both)
    db_ds_equi = db_ds
    q_ds_equi = q_ds
    if BOQ_IMG_SIZE != EQUI_IMG_SIZE:
        db_ds_equi = ds_cfg['dataset_cls'](ds_cfg['db_seq'],
                                             dataset_path=ds_cfg['path'],
                                             img_size=(EQUI_IMG_SIZE, EQUI_IMG_SIZE))
        q_ds_equi = ds_cfg['dataset_cls'](ds_cfg['query_seq'],
                                            dataset_path=ds_cfg['path'],
                                            img_size=(EQUI_IMG_SIZE, EQUI_IMG_SIZE))

    db_dl_boq = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    q_dl_boq = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # Extract BoQ descriptors ONCE (same across seeds)
    print("Extracting BoQ descriptors...")
    d_boq_db = extract_boq(boq_model, db_dl_boq)
    d_boq_q = extract_boq(boq_model, q_dl_boq)
    print(f"  BoQ db={d_boq_db.shape}, q={d_boq_q.shape}")

    # Accumulate per-seed records
    all_records = []
    for seed, ckpt in CKPTS.items():
        if not os.path.exists(ckpt):
            print(f"  [warn] missing ckpt seed={seed}: {ckpt}"); continue

        equi_model = load_p1_equi(ckpt)
        db_dl_equi = DataLoader(db_ds_equi, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        q_dl_equi = DataLoader(q_ds_equi, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        d_equi_db = extract_equi(equi_model, db_dl_equi)
        d_equi_q = extract_equi(equi_model, q_dl_equi)

        recs = per_query_records(
            d_boq_db, d_boq_q, d_equi_db, d_equi_q,
            q_ds.poses, db_ds.poses,
            theta_degrees=ds_cfg['theta_degrees'],
            yaw_threshold=ds_cfg['yaw_threshold'],
        )
        for r in recs:
            r['seed'] = seed
        all_records.extend(recs)
        print(f"  seed={seed}: {len(recs)} valid queries")

        del equi_model
        torch.cuda.empty_cache()

    # Aggregate per bucket
    print(f"\n{'Bucket':>14s}  {'N':>5s}  {'BoQ R@1':>9s}  {'DR-VPR R@1':>11s}  "
          f"{'Δ':>7s}  {'flip→✓':>7s}  {'flip→✗':>7s}")
    print('-' * 78)
    total_n = total_boq = total_drvpr = 0
    for lo, hi in BUCKETS:
        bucket_label = f'[{lo:>2d}°, {hi:>2d}°)'
        bucket = [r for r in all_records if r['bucket'] == bucket_label]
        n = len(bucket)
        if n == 0:
            print(f"  {bucket_label:>12s}  {n:>5d}  {'—':>9s}  {'—':>11s}  {'—':>7s}")
            continue
        boq_r1 = sum(r['boq_correct'] for r in bucket) / n * 100
        drvpr_r1 = sum(r['drvpr_correct'] for r in bucket) / n * 100
        flip_pos = sum(1 for r in bucket if r['boq_correct'] == 0 and r['drvpr_correct'] == 1)
        flip_neg = sum(1 for r in bucket if r['boq_correct'] == 1 and r['drvpr_correct'] == 0)
        print(f"  {bucket_label:>12s}  {n:>5d}  {boq_r1:8.2f}%  {drvpr_r1:10.2f}%  "
              f"{drvpr_r1 - boq_r1:+6.2f}  {flip_pos:>7d}  {flip_neg:>7d}")
        total_n += n
        total_boq += sum(r['boq_correct'] for r in bucket)
        total_drvpr += sum(r['drvpr_correct'] for r in bucket)

    print('-' * 78)
    if total_n > 0:
        print(f"  {'TOTAL':>12s}  {total_n:>5d}  {total_boq/total_n*100:8.2f}%  "
              f"{total_drvpr/total_n*100:10.2f}%  "
              f"{(total_drvpr - total_boq)/total_n*100:+6.2f}")
    return all_records


def main():
    boq_model = load_official_boq()

    for ds_name, ds_cfg in DATASETS.items():
        records = run_dataset(ds_name, ds_cfg, boq_model)
        # Save raw records per dataset
        import json
        out = f'per_yaw_p1_{ds_name}_records.json'
        with open(out, 'w') as f:
            json.dump(records, f, indent=2)
        print(f"Saved {out}")


if __name__ == '__main__':
    main()
