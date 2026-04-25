"""
Cross-product evaluation: {C4, C8, C16} × {BoQ-R50, BoQ-DINOv2} × {ConSLAM, ConPR full}.

For each (group_order, seed) checkpoint, compute joint-scoring R@1 at β=0.10
under both BoQ-ResNet50 and BoQ-DINOv2 stage-1 backbones, on both ConSLAM
(Sequence5 db vs Sequence4 query, θ=15°) and ConPR (full 10-sequence
protocol, θ=0°).

Output: 3-seed mean ± std table for each (group, backbone, dataset) cell —
populates the paper's group-order ablation table (paper Table 5).
"""
from __future__ import annotations
import os
import re
import glob
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset as ConslamInferDataset
from conpr_eval_dataset_rot import InferDataset as ConprInferDataset
from conpr_eval_dataset_rot import get_yaw_from_pose as cp_yaw
from Conslam_dataset_rot import get_yaw_from_pose as cs_yaw
from models.equi_multiscale import E2ResNetMultiScale


DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE_EQUI = (320, 320)
IMG_SIZE_BOQ_R50 = (320, 320)
IMG_SIZE_BOQ_DINOV2 = (322, 322)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
BETA = 0.10

CONSLAM_PATH = './datasets/ConSLAM/'
CONSLAM_SEQS = ['Sequence5', 'Sequence4']
CONSLAM_THETA = 15.0
CONPR_PATH = './datasets/ConPR/'
CONPR_SEQS = ['20230623', '20230531', '20230611', '20230627', '20230628',
              '20230706', '20230717', '20230803', '20230809', '20230818']
CONPR_THETA = 0.0

GROUP_ORDERS = [4, 8, 16, 32]
SEEDS = [1, 42, 190223]


def find_ckpt(orientation, seed):
    """Pick val-best ckpt by R1 in filename for the given (orientation, seed).

    C8 ckpts use the original tag 'ms', C4/C16 use 'ms_C4'/'ms_C16'.
    """
    if orientation == 8:
        tag_dir = f"LOGS/equi_standalone_seed{seed}_ms"
    else:
        tag_dir = f"LOGS/equi_standalone_seed{seed}_ms_C{orientation}"
    pattern = f"{tag_dir}/lightning_logs/version_*/checkpoints/equi_ms_seed{seed}_epoch*.ckpt"
    ckpts = glob.glob(pattern)
    if not ckpts:
        return None
    # Pick highest R1 in filename
    def get_r1(p):
        m = re.search(r'R1\[([0-9.]+)\]', p)
        return float(m.group(1)) if m else -1
    best = max(ckpts, key=get_r1)
    return best


def load_boq_resnet50():
    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq",
                            backbone_name="resnet50", output_dim=16384)
    return model.eval().to(DEVICE)


def load_boq_dinov2():
    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq",
                            backbone_name="dinov2", output_dim=12288)
    return model.eval().to(DEVICE)


def load_p1_equi(orientation, ckpt_path):
    model = E2ResNetMultiScale(
        orientation=orientation, layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512), out_dim=1024,
    )
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


@torch.no_grad()
def extract_boq(model, dl):
    descs = []
    for imgs, _ in dl:
        out = model(imgs.to(DEVICE))
        if isinstance(out, tuple):
            out = out[0]
        descs.append(F.normalize(out, p=2, dim=1).cpu().numpy())
    return np.vstack(descs)


@torch.no_grad()
def extract_equi(model, dl):
    descs = []
    for imgs, _ in dl:
        descs.append(model(imgs.to(DEVICE)).cpu().numpy())
    return np.vstack(descs)


def rotate_query_poses(q_poses_raw, theta_degrees):
    th = np.deg2rad(theta_degrees)
    q = q_poses_raw.copy()
    qx, qy = q[:, 3], q[:, 7]
    qx_rot = qx * np.cos(th) - qy * np.sin(th)
    qy_rot = qx * np.sin(th) + qy * np.cos(th)
    q[:, 3], q[:, 7] = qx_rot, qy_rot
    return q


def eval_pair(d_boq_db, d_boq_q, d_eq_db, d_eq_q, q_poses_raw, db_poses,
              theta_degrees, yaw_fn):
    q = rotate_query_poses(q_poses_raw, theta_degrees)
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k = idx.search(d_boq_q, TOP_K)
    correct = total = 0
    for q_idx in range(len(q)):
        ds = (q[q_idx, 3] - db_x) ** 2 + (q[q_idx, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0: continue
        q_yaw = yaw_fn(q[q_idx])
        ypos = set()
        for p in pp:
            d = abs(q_yaw - yaw_fn(db_poses[p]))
            if d > 180: d = 360 - d
            if d <= YAW_THRESHOLD: ypos.add(int(p))
        if not ypos: continue
        total += 1
        cands = top_k[q_idx]
        bs = d_boq_q[q_idx] @ d_boq_db[cands].T
        es = d_eq_q[q_idx] @ d_eq_db[cands].T
        score = (1 - BETA) * bs + BETA * es
        if cands[np.argmax(score)] in ypos:
            correct += 1
    return correct / total * 100 if total > 0 else 0.0, total


def eval_conslam(boq_descs_cache, equi_model, equi_dl_cache, ds_cache):
    """Returns single R@1 number using cached BoQ desc + fresh equi desc."""
    db_dl = equi_dl_cache['db']
    q_dl = equi_dl_cache['q']
    d_eq_db = extract_equi(equi_model, db_dl)
    d_eq_q = extract_equi(equi_model, q_dl)
    d_boq_db, d_boq_q = boq_descs_cache['db'], boq_descs_cache['q']
    db_ds, q_ds = ds_cache['db'], ds_cache['q']
    r1, _ = eval_pair(d_boq_db, d_boq_q, d_eq_db, d_eq_q,
                       q_ds.poses, db_ds.poses, CONSLAM_THETA, cs_yaw)
    return r1


def eval_conpr_full(boq_descs_cache, equi_model, equi_dl_cache, ds_cache):
    """Returns mean R@1 across 9 query pairs."""
    eq_descs = {seq: extract_equi(equi_model, equi_dl_cache[seq]) for seq in CONPR_SEQS}
    db_seq = CONPR_SEQS[0]
    db_poses = ds_cache[db_seq].poses
    pair_r1s = []
    for q_seq in CONPR_SEQS[1:]:
        r1, _ = eval_pair(
            boq_descs_cache[db_seq], boq_descs_cache[q_seq],
            eq_descs[db_seq], eq_descs[q_seq],
            ds_cache[q_seq].poses, db_poses, CONPR_THETA, cp_yaw)
        pair_r1s.append(r1)
    return float(np.mean(pair_r1s))


def precompute_boq_caches(boq_model, boq_size):
    """Build BoQ desc caches for ConSLAM (1 pair) and ConPR (10 seqs)."""
    cs_db_ds = ConslamInferDataset(CONSLAM_SEQS[0], dataset_path=CONSLAM_PATH, img_size=boq_size)
    cs_q_ds  = ConslamInferDataset(CONSLAM_SEQS[1], dataset_path=CONSLAM_PATH, img_size=boq_size)
    cs_db_dl = DataLoader(cs_db_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    cs_q_dl  = DataLoader(cs_q_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"  Extracting BoQ ConSLAM at {boq_size}...")
    cs_boq = {'db': extract_boq(boq_model, cs_db_dl), 'q': extract_boq(boq_model, cs_q_dl)}

    cp_ds_cache = {}
    cp_dl_cache = {}
    cp_boq = {}
    print(f"  Extracting BoQ ConPR (10 seqs) at {boq_size}...")
    for seq in CONPR_SEQS:
        ds = ConprInferDataset(seq, dataset_path=CONPR_PATH, img_size=boq_size)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
        cp_ds_cache[seq] = ds
        cp_dl_cache[seq] = dl
        cp_boq[seq] = extract_boq(boq_model, dl)
    return {
        'conslam_boq':  cs_boq,
        'conslam_ds':   {'db': cs_db_ds, 'q': cs_q_ds},
        'conpr_boq':    cp_boq,
        'conpr_ds':     cp_ds_cache,
        'conpr_dl':     cp_dl_cache,
    }


def precompute_equi_dataloaders():
    """Equi DataLoaders at IMG_SIZE_EQUI = (320, 320). BoQ-DINOv2 cache uses 322,
    so equi DataLoaders are independent and shared across both backbone runs."""
    cs_db_ds = ConslamInferDataset(CONSLAM_SEQS[0], dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_EQUI)
    cs_q_ds  = ConslamInferDataset(CONSLAM_SEQS[1], dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_EQUI)
    return {
        'conslam_dl': {
            'db': DataLoader(cs_db_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
            'q':  DataLoader(cs_q_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True),
        },
        'conpr_dl': {seq: DataLoader(
            ConprInferDataset(seq, dataset_path=CONPR_PATH, img_size=IMG_SIZE_EQUI),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
        ) for seq in CONPR_SEQS},
    }


def main():
    print("=" * 100)
    print("Group-order × backbone × dataset cross-product eval (β = 0.10)")
    print("=" * 100)

    # 1) Discover all ckpts
    print("\n[1] Discovering ckpts...")
    ckpt_table = {}  # (orientation, seed) → ckpt_path
    for ori in GROUP_ORDERS:
        for seed in SEEDS:
            p = find_ckpt(ori, seed)
            ckpt_table[(ori, seed)] = p
            status = "OK" if p else "MISSING"
            print(f"  C{ori} seed={seed}: {status}  {p or ''}")
    n_have = sum(1 for v in ckpt_table.values() if v is not None)
    print(f"  → {n_have}/9 ckpts available")

    # 2) Precompute equi DataLoaders (shared across backbone runs)
    print("\n[2] Precomputing equi dataloaders...")
    equi_loaders = precompute_equi_dataloaders()

    # 3) For each backbone: load BoQ, precompute BoQ caches, sweep all (ori, seed)
    backbones = [
        ('BoQ-ResNet50', load_boq_resnet50, IMG_SIZE_BOQ_R50),
        ('BoQ-DINOv2',   load_boq_dinov2,   IMG_SIZE_BOQ_DINOV2),
    ]
    results = {}   # backbone_name → {(ori, seed): {'conslam': r1, 'conpr': r1}}

    for backbone_name, loader_fn, boq_size in backbones:
        print(f"\n{'=' * 100}\n[{backbone_name}] at {boq_size}\n{'=' * 100}")
        boq_model = loader_fn()
        cache = precompute_boq_caches(boq_model, boq_size)
        del boq_model; torch.cuda.empty_cache()

        results[backbone_name] = {}
        for (ori, seed), ckpt_path in ckpt_table.items():
            if ckpt_path is None:
                continue
            print(f"\n  -- C{ori} seed={seed} --")
            equi_model = load_p1_equi(ori, ckpt_path)
            r1_cs = eval_conslam(cache['conslam_boq'], equi_model,
                                  equi_loaders['conslam_dl'],
                                  cache['conslam_ds'])
            r1_cp = eval_conpr_full(cache['conpr_boq'], equi_model,
                                     {k: equi_loaders['conpr_dl'][k] for k in CONPR_SEQS},
                                     cache['conpr_ds'])
            results[backbone_name][(ori, seed)] = {'conslam': r1_cs, 'conpr': r1_cp}
            print(f"    ConSLAM R@1={r1_cs:.2f}  ConPR full R@1={r1_cp:.2f}")
            del equi_model; torch.cuda.empty_cache()

    # 4) Aggregate per-(backbone, group): 3-seed mean ± std
    print(f"\n{'=' * 100}")
    print("Aggregated 3-seed mean ± std R@1 (β = 0.10)")
    print(f"{'=' * 100}")
    print(f"{'Backbone':<14s}  {'Group':<5s}  {'ConSLAM R@1':>16s}  {'ConPR R@1':>14s}")
    print('-' * 100)
    for backbone_name in [b[0] for b in backbones]:
        for ori in GROUP_ORDERS:
            cs_vals = [results[backbone_name].get((ori, s), {}).get('conslam') for s in SEEDS]
            cp_vals = [results[backbone_name].get((ori, s), {}).get('conpr')   for s in SEEDS]
            cs_vals = [v for v in cs_vals if v is not None]
            cp_vals = [v for v in cp_vals if v is not None]
            if len(cs_vals) < 2:
                print(f"{backbone_name:<14s}  C{ori:<4d}  (insufficient seeds: {len(cs_vals)})")
                continue
            m_cs, s_cs = np.mean(cs_vals), np.std(cs_vals, ddof=1)
            m_cp, s_cp = np.mean(cp_vals), np.std(cp_vals, ddof=1)
            print(f"{backbone_name:<14s}  C{ori:<4d}  {m_cs:6.2f} ± {s_cs:5.2f}  {m_cp:6.2f} ± {s_cp:5.2f}")
    print('=' * 100)


if __name__ == '__main__':
    main()
