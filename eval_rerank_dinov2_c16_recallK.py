"""
DR-VPR with BoQ(DINOv2) stage-1 × C16 equi — R@1/R@5/R@10 @ β=0.10.

Fills the paper's missing cells: DR-VPR (DINOv2 backbone, C16 equi) for
top-5 and top-10 recall on ConSLAM and ConPR-full.

Rerank rule: score(q, c) = (1 - β) · BoQ(q)·BoQ(c) + β · equi(q)·equi(c).
Top-1 = argmax_c score; R@K = positive ∈ candidates sorted by score.
"""
from __future__ import annotations
import glob
import re
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
IMG_SIZE_BOQ_DINOV2 = (322, 322)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
BETA = 0.10
K_LIST = [1, 5, 10]

CONSLAM_PATH = './datasets/ConSLAM/'
CONSLAM_SEQS = ['Sequence5', 'Sequence4']
CONSLAM_THETA = 15.0
CONPR_PATH = './datasets/ConPR/'
CONPR_SEQS = ['20230623', '20230531', '20230611', '20230627', '20230628',
              '20230706', '20230717', '20230803', '20230809', '20230818']
CONPR_THETA = 0.0
SEEDS = [1, 42, 190223]


def find_c16_ckpt(seed):
    tag_dir = f'LOGS/equi_standalone_seed{seed}_ms_C16'
    pattern = f"{tag_dir}/lightning_logs/version_*/checkpoints/equi_ms_seed{seed}_epoch*.ckpt"
    ckpts = glob.glob(pattern)

    def get_r1(p):
        m = re.search(r'R1\[([\d.]+)\]', p)
        return float(m.group(1)) if m else 0.0
    return max(ckpts, key=get_r1)


def load_boq_dinov2():
    print('Loading BoQ(DINOv2) from torch.hub...')
    model = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq",
                            backbone_name="dinov2", output_dim=12288)
    return model.eval().to(DEVICE)


def load_c16_equi(ckpt_path):
    model = E2ResNetMultiScale(orientation=16, layers=(2, 2, 2, 2),
                                channels=(64, 128, 256, 512), out_dim=1024)
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


def eval_recall_k(d_boq_db, d_boq_q, d_eq_db, d_eq_q, q_poses_raw, db_poses,
                   theta_degrees, yaw_fn, beta, k_list):
    """Return dict k → R@k%, total valid queries."""
    q = rotate_query_poses(q_poses_raw, theta_degrees)
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k = idx.search(d_boq_q, TOP_K)
    hits = {k: 0 for k in k_list}
    total = 0
    for qi in range(len(q)):
        ds = (q[qi, 3] - db_x) ** 2 + (q[qi, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0:
            continue
        qy_val = yaw_fn(q[qi])
        ypos = set()
        for p in pp:
            d = abs(qy_val - yaw_fn(db_poses[p]))
            if d > 180:
                d = 360 - d
            if d <= YAW_THRESHOLD:
                ypos.add(int(p))
        if not ypos:
            continue
        total += 1
        cands = top_k[qi]
        if beta == 0.0 or d_eq_db is None:
            ranked = cands
        else:
            bs = d_boq_q[qi] @ d_boq_db[cands].T
            es = d_eq_q[qi] @ d_eq_db[cands].T
            score = (1 - beta) * bs + beta * es
            order = np.argsort(-score)
            ranked = cands[order]
        for k in k_list:
            if any(int(r) in ypos for r in ranked[:k]):
                hits[k] += 1
    out = {k: hits[k] / total * 100 if total > 0 else 0.0 for k in k_list}
    return out, total


def print_mean_std(label, rows, k_list):
    print(f'\n--- {label} 3-seed mean ± std @ β=0.10 ---')
    for k in k_list:
        vals = np.array([r[k] for r in rows])
        print(f'  R@{k:<2} = {vals.mean():.2f} ± {vals.std(ddof=1):.2f}')


def main():
    print('=' * 90)
    print(f'DR-VPR × BoQ-DINOv2 × C16  —  R@1/R@5/R@10 @ β={BETA}, 3 seeds')
    print(f'gt_thres={GT_THRES} m, yaw_threshold={YAW_THRESHOLD}°')
    print('=' * 90)

    # ---- Load BoQ-DINOv2 + ckpt table ----
    boq = load_boq_dinov2()
    ckpts = {s: find_c16_ckpt(s) for s in SEEDS}
    print('Selected C16 checkpoints (val-best R@1):')
    for s, p in ckpts.items():
        print(f'  seed={s}  {p}')

    # ================= ConSLAM =================
    print('\n' + '=' * 90)
    print('ConSLAM  (Seq5 db, Seq4 q, θ=15°)')
    print('=' * 90)

    cs_db_ds_boq = ConslamInferDataset(CONSLAM_SEQS[0], dataset_path=CONSLAM_PATH,
                                        img_size=IMG_SIZE_BOQ_DINOV2)
    cs_q_ds_boq  = ConslamInferDataset(CONSLAM_SEQS[1], dataset_path=CONSLAM_PATH,
                                        img_size=IMG_SIZE_BOQ_DINOV2)
    cs_db_ds_eq  = ConslamInferDataset(CONSLAM_SEQS[0], dataset_path=CONSLAM_PATH,
                                        img_size=IMG_SIZE_EQUI)
    cs_q_ds_eq   = ConslamInferDataset(CONSLAM_SEQS[1], dataset_path=CONSLAM_PATH,
                                        img_size=IMG_SIZE_EQUI)
    cs_db_dl_boq = DataLoader(cs_db_ds_boq, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    cs_q_dl_boq  = DataLoader(cs_q_ds_boq,  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    cs_db_dl_eq  = DataLoader(cs_db_ds_eq,  batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)
    cs_q_dl_eq   = DataLoader(cs_q_ds_eq,   batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

    print('Extracting BoQ-DINOv2 ConSLAM...')
    cs_boq_db = extract_boq(boq, cs_db_dl_boq)
    cs_boq_q  = extract_boq(boq, cs_q_dl_boq)
    print(f'  db={cs_boq_db.shape}  q={cs_boq_q.shape}')

    # β=0 reference (BoQ-DINOv2 alone)
    res0, n_valid_cs = eval_recall_k(cs_boq_db, cs_boq_q, None, None,
                                       cs_q_ds_boq.poses, cs_db_ds_boq.poses,
                                       CONSLAM_THETA, cs_yaw, beta=0.0,
                                       k_list=K_LIST)
    print(f'\n[β=0 baseline = BoQ-DINOv2 alone]  '
          f'R@1={res0[1]:.2f}  R@5={res0[5]:.2f}  R@10={res0[10]:.2f}  '
          f'(n={n_valid_cs})')

    cs_rows = []
    for s in SEEDS:
        print(f'\n[seed={s}] extracting C16 equi...')
        eq_model = load_c16_equi(ckpts[s])
        cs_eq_db = extract_equi(eq_model, cs_db_dl_eq)
        cs_eq_q  = extract_equi(eq_model, cs_q_dl_eq)
        res, _ = eval_recall_k(cs_boq_db, cs_boq_q, cs_eq_db, cs_eq_q,
                                cs_q_ds_boq.poses, cs_db_ds_boq.poses,
                                CONSLAM_THETA, cs_yaw, beta=BETA,
                                k_list=K_LIST)
        print(f'  R@1={res[1]:.2f}  R@5={res[5]:.2f}  R@10={res[10]:.2f}')
        cs_rows.append(res)
        del eq_model
        torch.cuda.empty_cache()

    print_mean_std('ConSLAM', cs_rows, K_LIST)

    # ================= ConPR full =================
    print('\n' + '=' * 90)
    print('ConPR full 10-sequence protocol')
    print('=' * 90)

    print('Extracting BoQ-DINOv2 for all 10 ConPR seqs...')
    cp_boq = {}
    cp_ds = {}
    for seq in CONPR_SEQS:
        ds = ConprInferDataset(seq, dataset_path=CONPR_PATH,
                                img_size=IMG_SIZE_BOQ_DINOV2)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)
        cp_boq[seq] = extract_boq(boq, dl)
        cp_ds[seq] = ds
        print(f'  {seq}: {cp_boq[seq].shape}')

    cp_rows = []
    for s in SEEDS:
        print(f'\n[seed={s}] extracting C16 equi for all 10 seqs...')
        eq_model = load_c16_equi(ckpts[s])
        cp_eq = {}
        for seq in CONPR_SEQS:
            ds = ConprInferDataset(seq, dataset_path=CONPR_PATH,
                                    img_size=IMG_SIZE_EQUI)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=True)
            cp_eq[seq] = extract_equi(eq_model, dl)
        db_seq = CONPR_SEQS[0]
        db_poses = cp_ds[db_seq].poses
        pair_results = []
        for q_seq in CONPR_SEQS[1:]:
            res, _ = eval_recall_k(
                cp_boq[db_seq], cp_boq[q_seq], cp_eq[db_seq], cp_eq[q_seq],
                cp_ds[q_seq].poses, db_poses, CONPR_THETA, cp_yaw,
                beta=BETA, k_list=K_LIST)
            pair_results.append(res)
            print(f'  pair={q_seq}  R@1={res[1]:.2f}  R@5={res[5]:.2f}  R@10={res[10]:.2f}')
        seed_mean = {k: float(np.mean([r[k] for r in pair_results]))
                     for k in K_LIST}
        print(f'  [seed={s} mean-of-9-pairs]  '
              f'R@1={seed_mean[1]:.2f}  R@5={seed_mean[5]:.2f}  R@10={seed_mean[10]:.2f}')
        cp_rows.append(seed_mean)
        del eq_model
        torch.cuda.empty_cache()

    print_mean_std('ConPR (mean-of-9-pairs)', cp_rows, K_LIST)


if __name__ == '__main__':
    main()
