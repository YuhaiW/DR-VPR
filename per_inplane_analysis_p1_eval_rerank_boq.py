"""
Verification: re-run per-inplane bucket analysis on ConSLAM, but extract
BoQ via the EXACT same path as eval_rerank.py (the path that produces
the 60.91 baseline in the paper Table 1):

    VPRModel(use_dual_branch=True, equi_orientation=8, fusion='concat')
    load_boq_pretrained(model)            # downloads from GitHub release
    state_dict ← DualBranch C8 freeze_boq seed190223 ckpt
    BoQ desc = F.normalize(
        model.aggregator.branch1_aggregator(model.backbone(imgs)) )

Equi descriptor still comes from the C16 standalone ckpt (paper main),
so the rerank @ β=0.10 still gives DR-VPR. The point is to verify whether
just the BoQ-loading path swap moves the TOTAL BoQ R@1 from 61.24 → 60.91.
"""
import os
os.environ.setdefault('FUSION_METHOD', 'concat')
os.environ.setdefault('GROUP_POOL_MODE', 'max')

import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset as ConslamInferDataset
from train_fusion import VPRModel, load_boq_pretrained
from models.equi_multiscale import E2ResNetMultiScale


DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = 320
TOP_K = 100
FIXED_BETA = 0.10
GT_THRES = 5.0
YAW_THRESHOLD = 80.0

# Same DualBranch C8 freeze_boq ckpt that eval_rerank.py defaults to
DUALBRANCH_CKPT_DIR = (
    './LOGS/resnet50_DualBranch_freeze_boq_seed190223/'
    'lightning_logs/version_0/checkpoints/'
)

C16_CKPTS = {
    1:      'LOGS/equi_standalone_seed1_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed1_epoch(08)_R1[0.3510].ckpt',
    42:     'LOGS/equi_standalone_seed42_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed42_epoch(01)_R1[0.3283].ckpt',
    190223: 'LOGS/equi_standalone_seed190223_ms_C16/lightning_logs/version_0/checkpoints/equi_ms_seed190223_epoch(04)_R1[0.3384].ckpt',
}

CONSLAM_PATH = './datasets/ConSLAM/'
DB_SEQ = 'Sequence5'
Q_SEQ  = 'Sequence4'
THETA  = 15.0

BUCKET_EDGES = [0, 5, 10, 15, 20, 25, 90]
BUCKET_LABELS = ['[ 0°,  5°)', '[ 5°, 10°)', '[10°, 15°)',
                 '[15°, 20°)', '[20°, 25°)', '   25°+   ']


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


# --- BoQ loading via eval_rerank.py path ---

def find_best_ckpt_in_dir(d):
    import glob
    ckpts = glob.glob(os.path.join(d, '**/*.ckpt'), recursive=True)
    return max(ckpts, key=lambda f: float(f.split('[')[1].split(']')[0]))


def build_dualbranch_model_for_boq():
    """Mirror eval_rerank.py build_model() exactly."""
    return VPRModel(
        backbone_arch='resnet50', pretrained=True,
        layers_to_freeze=2, layers_to_crop=[4],
        agg_arch='boq',
        agg_config={'in_channels': 1024, 'proj_channels': 512,
                     'num_queries': 64, 'num_layers': 2, 'row_dim': 32},
        use_dual_branch=True, equi_orientation=8,
        equi_layers=[2, 2, 2, 2], equi_channels=[64, 128, 256, 512],
        equi_out_dim=1024, fusion_method='concat', use_projection=False,
        lr=1e-3, optimizer='adamw', weight_decay=1e-4, momentum=0.9,
        warmpup_steps=300, milestones=[8, 14], lr_mult=0.3,
        loss_name='MultiSimilarityLoss', miner_name='MultiSimilarityMiner',
        miner_margin=0.1, faiss_gpu=False,
    )


@torch.no_grad()
def extract_boq_via_dualbranch(model, dl):
    """Match eval_rerank.py extract_branch_descriptors() exactly for d1."""
    out = []
    for imgs, _ in dl:
        imgs = imgs.to(DEVICE)
        feat1 = model.backbone(imgs)
        d1 = model.aggregator.branch1_aggregator(feat1)
        d1 = F.normalize(d1, p=2, dim=1)
        out.append(d1.cpu().numpy())
    return np.vstack(out)


# --- Equi loading (same as per_inplane_analysis_p1.py) ---

def load_p1_equi(ckpt_path):
    m = E2ResNetMultiScale(orientation=16, layers=(2, 2, 2, 2),
                            channels=(64, 128, 256, 512), out_dim=1024)
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    m.load_state_dict(state, strict=False)
    return m.to(DEVICE).eval()


@torch.no_grad()
def extract_equi(model, dl):
    out = []
    for imgs, _ in dl:
        out.append(model(imgs.to(DEVICE)).cpu().numpy())
    return np.vstack(out)


def per_query_records(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                       q_poses_raw, db_poses, theta_degrees):
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
        valid_pos.sort(key=lambda x: x[1])
        nearest_p = valid_pos[0][0]
        R_q, R_db = get_R(q_poses[qi]), get_R(db_poses[nearest_p])
        ip = abs(inplane_rotation_q_to_db(R_q, R_db))
        if ip > 180:
            ip = 360 - ip

        cands = top_k_idx[qi]
        boq_sim = d_boq_q[qi] @ d_boq_db[cands].T
        equi_sim = d_equi_q[qi] @ d_equi_db[cands].T
        # Match eval_rerank.py: at β=0 use FAISS top-1 (= cands[0]).
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


def report(records):
    print(f"\n{'Bucket':>14s}  {'N':>6s}  {'BoQ R@1':>9s}  "
          f"{'DR-VPR R@1':>11s}  {'Δ':>7s}  {'flip→✓':>7s}  {'flip→✗':>7s}")
    print('-' * 82)
    total_n = total_boq = total_drvpr = 0
    for lbl in BUCKET_LABELS:
        bucket = [r for r in records if r['bucket'] == lbl]
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


def main():
    print("=" * 82)
    print("Per-inplane analysis on ConSLAM")
    print("BoQ extracted via eval_rerank.py path "
          "(VPRModel + load_boq_pretrained + DualBranch C8 ckpt)")
    print("Equi: C16 standalone (paper main, 3 seeds)")
    print("=" * 82)

    # --- BoQ via eval_rerank.py path ---
    print("\n[BoQ pipeline] build VPRModel(C8 dual_branch, fusion=concat)")
    boq_model = build_dualbranch_model_for_boq()
    print("[BoQ pipeline] load_boq_pretrained(model) "
          "→ download GitHub release weights")
    load_boq_pretrained(boq_model)
    ckpt_path = find_best_ckpt_in_dir(DUALBRANCH_CKPT_DIR)
    print(f"[BoQ pipeline] state_dict ← {ckpt_path}")
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    missing, unexpected = boq_model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith('.filter')]
    print(f"  load_state_dict: {len(missing)} missing "
          f"({len(real_missing)} real), {len(unexpected)} unexpected")
    boq_model = boq_model.to(DEVICE).eval()

    # --- Datasets ---
    db_ds = ConslamInferDataset(DB_SEQ, dataset_path=CONSLAM_PATH,
                                 img_size=(IMG_SIZE, IMG_SIZE))
    q_ds  = ConslamInferDataset(Q_SEQ,  dataset_path=CONSLAM_PATH,
                                 img_size=(IMG_SIZE, IMG_SIZE))
    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)
    q_dl  = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

    # Extract BoQ once (deterministic across seeds)
    print("\n[BoQ pipeline] extracting descriptors...")
    d_boq_db = extract_boq_via_dualbranch(boq_model, db_dl)
    d_boq_q  = extract_boq_via_dualbranch(boq_model, q_dl)
    print(f"  BoQ db={d_boq_db.shape}  q={d_boq_q.shape}")

    # Per-seed records
    all_records = []
    for seed, ckpt in C16_CKPTS.items():
        if not os.path.exists(ckpt):
            print(f"  [warn] missing C16 ckpt seed={seed}: {ckpt}")
            continue
        equi = load_p1_equi(ckpt)
        d_equi_db = extract_equi(equi, db_dl)
        d_equi_q  = extract_equi(equi, q_dl)
        recs = per_query_records(d_boq_db, d_boq_q, d_equi_db, d_equi_q,
                                  q_ds.poses, db_ds.poses, THETA)
        for r in recs:
            r['seed'] = seed
        all_records.extend(recs)
        print(f"  seed={seed}: {len(recs)} valid queries")
        del equi
        torch.cuda.empty_cache()

    report(all_records)
    out = 'per_inplane_p1_conslam_records_eval_rerank_boq.json'
    with open(out, 'w') as f:
        json.dump(all_records, f, indent=2)
    print(f"\nSaved → {out}")

    # Final verdict line
    n = len(all_records)
    boq_correct = sum(r['boq_correct'] for r in all_records)
    print(f"\n>>> TOTAL BoQ R@1 (via eval_rerank.py path) = "
          f"{boq_correct / n * 100:.2f}%   ({boq_correct}/{n})")
    print(f">>> Compare: torch.hub path gave 61.24% (564/921). "
          f"Paper Table 1 quotes 60.91%.")


if __name__ == '__main__':
    main()
