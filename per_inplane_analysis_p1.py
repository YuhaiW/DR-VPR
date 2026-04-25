"""
Per-INPLANE-ROTATION bucket decomposition of R@1 for P1 standalone (C16 multi-scale)
on BOTH ConSLAM and ConPR.

Difference from per_yaw_analysis_p1.py:
  - Buckets by TRUE in-plane image rotation between query and the NEAREST-YAW GT
    positive (the same quantity computed by plot_inplane_distribution_*.py),
    NOT by Δyaw / world heading.
  - 5° bucket width as requested. Last bucket "25°+" catches the tail.
  - Robotics convention (X = optical axis), Rodrigues axis alignment, then
    arctan2(R[2,1], R[2,2]) for the residual rotation about X.

This is the table that tests whether the C_16 branch's R@1 gains correlate
with the rotation magnitude it is *supposed* to handle.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset as ConslamInferDataset
from conpr_eval_dataset_rot import InferDataset as ConPRInferDataset
from models.equi_multiscale import E2ResNetMultiScale


DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = 320
TOP_K = 100
FIXED_BETA = 0.10
GT_THRES = 5.0
YAW_THRESHOLD = 80.0

# 5° buckets up to 25°, then a single tail bin
BUCKET_EDGES = [0, 5, 10, 15, 20, 25, 90]
BUCKET_LABELS = ['[ 0°,  5°)', '[ 5°, 10°)', '[10°, 15°)',
                 '[15°, 20°)', '[20°, 25°)', '   25°+   ']

CKPTS = {
    1:      'LOGS/equi_standalone_seed1_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed1_epoch(08)_R1[0.3510].ckpt',
    42:     'LOGS/equi_standalone_seed42_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed42_epoch(01)_R1[0.3283].ckpt',
    190223: 'LOGS/equi_standalone_seed190223_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed190223_epoch(04)_R1[0.3384].ckpt',
}

DATASETS = {
    'conslam': {
        'path': './datasets/ConSLAM/',
        'db_seq': 'Sequence5',
        'query_seq': 'Sequence4',
        'theta_degrees': 15.0,
        'dataset_cls': ConslamInferDataset,
    },
    'conpr': {
        'path': './datasets/ConPR/',
        'db_seq': '20230623',
        'query_seq': '20230809',
        'theta_degrees': 0.0,
        'dataset_cls': ConPRInferDataset,
    },
}


# ---- pose / rotation helpers (mirror plot_inplane_distribution_*.py) ----

def get_R(pose):
    return np.array([[pose[0], pose[1], pose[2]],
                      [pose[4], pose[5], pose[6]],
                      [pose[8], pose[9], pose[10]]])


def get_yaw(pose):
    R = get_R(pose)
    return np.degrees(np.arctan2(R[1, 0], R[0, 0]))


def angdiff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def inplane_rotation_q_to_db(R_q, R_db):
    """Robotics convention: align db's optical axis (X) to q's, then
    extract residual rotation about X via arctan2(R[2,1], R[2,2])."""
    R_rel = R_q.T @ R_db
    v = R_rel[:, 0]
    cos_theta = float(np.clip(v[0], -1.0, 1.0))
    if cos_theta > 0.9999:
        return float(np.degrees(np.arctan2(R_rel[2, 1], R_rel[2, 2])))
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    if sin_theta < 1e-9:
        return 0.0
    ax = np.array([0.0, -v[2], v[1]]) / sin_theta
    K = np.array([[0, -ax[2], ax[1]],
                   [ax[2], 0, -ax[0]],
                   [-ax[1], ax[0], 0]])
    R_align = np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
    R_inplane = R_align.T @ R_rel
    return float(np.degrees(np.arctan2(R_inplane[2, 1], R_inplane[2, 2])))


def inplane_to_bucket(angle_abs):
    for i, hi in enumerate(BUCKET_EDGES[1:]):
        if angle_abs < hi:
            return BUCKET_LABELS[i]
    return BUCKET_LABELS[-1]


# ---- model loading ----

def load_official_boq():
    print('Loading official BoQ(ResNet50)@320 from torch.hub...')
    m = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq",
                        backbone_name="resnet50", output_dim=16384)
    return m.eval().to(DEVICE)


def load_p1_equi(ckpt_path):
    m = E2ResNetMultiScale(orientation=16, layers=(2, 2, 2, 2),
                            channels=(64, 128, 256, 512), out_dim=1024)
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    m.load_state_dict(state, strict=False)
    return m.to(DEVICE).eval()


@torch.no_grad()
def extract_boq(model, dl):
    out = []
    for imgs, _ in dl:
        o = model(imgs.to(DEVICE))
        if isinstance(o, tuple):
            o = o[0]
        out.append(F.normalize(o, p=2, dim=1).cpu().numpy())
    return np.vstack(out)


@torch.no_grad()
def extract_equi(model, dl):
    out = []
    for imgs, _ in dl:
        out.append(model(imgs.to(DEVICE)).cpu().numpy())
    return np.vstack(out)


# ---- per-query records ----

def per_query_records(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                       q_poses_raw, db_poses, theta_degrees):
    """For each valid query, find nearest-yaw GT positive, compute |in-plane|
    to that positive, then evaluate top-1 under β=0 vs β=FIXED_BETA."""
    # Bug-fixed XY rotation for query (temp vars)
    th = np.deg2rad(theta_degrees)
    q_poses = q_poses_raw.copy()
    qx, qy = q_poses[:, 3], q_poses[:, 7]
    qx_rot = qx * np.cos(th) - qy * np.sin(th)
    qy_rot = qx * np.sin(th) + qy * np.cos(th)
    q_poses[:, 3], q_poses[:, 7] = qx_rot, qy_rot

    db_x, db_y = db_poses[:, 3], db_poses[:, 7]

    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k_idx = idx.search(d_boq_q, TOP_K)

    records = []
    for qi in range(len(q_poses)):
        ds = (q_poses[qi, 3] - db_x) ** 2 + (q_poses[qi, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0:
            continue
        q_yaw = get_yaw(q_poses[qi])
        valid_pos = []
        for p in pp:
            d_yaw = angdiff(q_yaw, get_yaw(db_poses[p]))
            if d_yaw <= YAW_THRESHOLD:
                valid_pos.append((int(p), d_yaw))
        if not valid_pos:
            continue
        ypos_set = {p for p, _ in valid_pos}
        # nearest-yaw positive — same definition as the distribution figure
        valid_pos.sort(key=lambda x: x[1])
        nearest_p = valid_pos[0][0]
        R_q, R_db = get_R(q_poses[qi]), get_R(db_poses[nearest_p])
        ip = abs(inplane_rotation_q_to_db(R_q, R_db))
        if ip > 180:
            ip = 360 - ip

        cands = top_k_idx[qi]
        boq_sim = d_boq_q[qi] @ d_boq_db[cands].T
        equi_sim = d_equi_q[qi] @ d_equi_db[cands].T
        # BoQ top-1 = FAISS-sorted #1 (matches Table 1 / eval_rerank.py exactly,
        # avoiding the 1-query float32 tie-break drift between FAISS and numpy
        # argmax that put per_yaw_analysis at 61.24 vs Table 1's 60.91).
        top1_b0 = int(cands[0])
        top1_bfx = int(cands[np.argmax((1 - FIXED_BETA) * boq_sim
                                        + FIXED_BETA * equi_sim)])

        records.append({
            'inplane_abs': ip,
            'bucket': inplane_to_bucket(ip),
            'boq_correct': int(top1_b0 in ypos_set),
            'drvpr_correct': int(top1_bfx in ypos_set),
        })
    return records


# ---- driver ----

def run_dataset(ds_name, ds_cfg, boq_model):
    print(f"\n{'='*82}\nDataset: {ds_name.upper()}  "
          f"({ds_cfg['db_seq']} db vs {ds_cfg['query_seq']} q,  "
          f"θ={ds_cfg['theta_degrees']}°)\n{'='*82}")

    db_ds = ds_cfg['dataset_cls'](ds_cfg['db_seq'],
                                    dataset_path=ds_cfg['path'],
                                    img_size=(IMG_SIZE, IMG_SIZE))
    q_ds = ds_cfg['dataset_cls'](ds_cfg['query_seq'],
                                   dataset_path=ds_cfg['path'],
                                   img_size=(IMG_SIZE, IMG_SIZE))
    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    q_dl = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    print('Extracting BoQ...')
    d_boq_db = extract_boq(boq_model, db_dl)
    d_boq_q  = extract_boq(boq_model, q_dl)
    print(f'  BoQ db={d_boq_db.shape}  q={d_boq_q.shape}')

    all_records = []
    for seed, ckpt in CKPTS.items():
        if not os.path.exists(ckpt):
            print(f'  [warn] missing ckpt seed={seed}: {ckpt}')
            continue
        equi = load_p1_equi(ckpt)
        d_equi_db = extract_equi(equi, db_dl)
        d_equi_q  = extract_equi(equi, q_dl)
        recs = per_query_records(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                                  q_ds.poses, db_ds.poses,
                                  ds_cfg['theta_degrees'])
        for r in recs:
            r['seed'] = seed
            r['dataset'] = ds_name
        all_records.extend(recs)
        print(f'  seed={seed}: {len(recs)} valid queries')
        del equi
        torch.cuda.empty_cache()

    # Aggregate per bucket (pooled across 3 seeds)
    print(f"\n{'Bucket':>14s}  {'N':>6s}  {'BoQ R@1':>9s}  "
          f"{'DR-VPR R@1':>11s}  {'Δ':>7s}  {'flip→✓':>7s}  {'flip→✗':>7s}")
    print('-' * 82)
    total_n = total_boq = total_drvpr = 0
    for lbl in BUCKET_LABELS:
        bucket = [r for r in all_records if r['bucket'] == lbl]
        n = len(bucket)
        if n == 0:
            print(f"  {lbl:>14s}  {n:>6d}   {'—':>8s}   {'—':>10s}   "
                  f"{'—':>6s}   {'—':>6s}   {'—':>6s}")
            continue
        boq_r1 = sum(r['boq_correct'] for r in bucket) / n * 100
        drvpr_r1 = sum(r['drvpr_correct'] for r in bucket) / n * 100
        flip_pos = sum(1 for r in bucket
                        if r['boq_correct'] == 0 and r['drvpr_correct'] == 1)
        flip_neg = sum(1 for r in bucket
                        if r['boq_correct'] == 1 and r['drvpr_correct'] == 0)
        print(f"  {lbl:>14s}  {n:>6d}   {boq_r1:7.2f}%   {drvpr_r1:9.2f}%   "
              f"{drvpr_r1 - boq_r1:+6.2f}   {flip_pos:>6d}   {flip_neg:>6d}")
        total_n += n
        total_boq += sum(r['boq_correct'] for r in bucket)
        total_drvpr += sum(r['drvpr_correct'] for r in bucket)
    print('-' * 82)
    if total_n > 0:
        print(f"  {'TOTAL':>14s}  {total_n:>6d}   "
              f"{total_boq/total_n*100:7.2f}%   "
              f"{total_drvpr/total_n*100:9.2f}%   "
              f"{(total_drvpr - total_boq)/total_n*100:+6.2f}")
    return all_records


def main():
    boq = load_official_boq()
    for ds_name, ds_cfg in DATASETS.items():
        recs = run_dataset(ds_name, ds_cfg, boq)
        out = f'per_inplane_p1_{ds_name}_records.json'
        with open(out, 'w') as f:
            json.dump(recs, f, indent=2)
        print(f'Saved {out}')


if __name__ == '__main__':
    main()
