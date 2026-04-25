"""
Two-panel trajectory comparison on ConSLAM (Seq5 db vs Seq4 q, θ=15°).

Panel (a): BoQ(ResNet50) baseline — green dots = correct, red X = failure.
Panel (b): DR-VPR (R50+C16) main method (β=0.10) — same markers, PLUS
          yellow circles on queries that BoQ failed but DR-VPR succeeded.

Output: figures/conslam_trajectory_boq_vs_drvpr.{pdf,png}
"""
from __future__ import annotations
import glob
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset, get_yaw_from_pose
from models.equi_multiscale import E2ResNetMultiScale


DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE_BOQ = (320, 320)
IMG_SIZE_EQUI = (320, 320)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
BETA = 0.10
THETA = 15.0
CONSLAM_PATH = './datasets/ConSLAM/'
SEQ_DB, SEQ_Q = 'Sequence5', 'Sequence4'
SEED_FOR_PLOT = 42  # seed 42 ConSLAM R@1 = 62.54, closest to 3-seed mean 62.65

OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)


def find_c16_ckpt(seed):
    tag_dir = f'LOGS/equi_standalone_seed{seed}_ms_C16'
    pattern = f"{tag_dir}/lightning_logs/version_*/checkpoints/equi_ms_seed{seed}_epoch*.ckpt"
    ckpts = glob.glob(pattern)
    def get_r1(p):
        m = re.search(r'R1\[([\d.]+)\]', p)
        return float(m.group(1)) if m else 0.0
    return max(ckpts, key=get_r1)


def load_boq_r50():
    print('Loading BoQ(ResNet50)...')
    m = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq",
                        backbone_name="resnet50", output_dim=16384)
    return m.eval().to(DEVICE)


def load_c16_equi(ckpt_path):
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


def rotate_query_poses(q_poses_raw, theta_degrees):
    th = np.deg2rad(theta_degrees)
    q = q_poses_raw.copy()
    qx, qy = q[:, 3], q[:, 7]
    qx_rot = qx * np.cos(th) - qy * np.sin(th)
    qy_rot = qx * np.sin(th) + qy * np.cos(th)
    q[:, 3], q[:, 7] = qx_rot, qy_rot
    return q


def compute_per_query_outcomes(d_boq_db, d_boq_q, d_eq_db, d_eq_q,
                                q_poses_rot, db_poses):
    """For each query index, determine:
       - valid (has GT positive after position+yaw filter)
       - boq_top1_correct (bool)
       - drvpr_top1_correct (bool)
    Returns: list of dicts per query_idx.
    """
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k = idx.search(d_boq_q, TOP_K)

    outcomes = []
    for qi in range(len(q_poses_rot)):
        entry = {
            'qi': qi,
            'qx': q_poses_rot[qi, 3],
            'qy': q_poses_rot[qi, 7],
            'valid': False,
            'boq_correct': False,
            'drvpr_correct': False,
        }
        ds = (q_poses_rot[qi, 3] - db_x) ** 2 + (q_poses_rot[qi, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0:
            outcomes.append(entry)
            continue
        q_yaw = get_yaw_from_pose(q_poses_rot[qi])
        ypos = set()
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180:
                d = 360 - d
            if d <= YAW_THRESHOLD:
                ypos.add(int(p))
        if not ypos:
            outcomes.append(entry)
            continue
        entry['valid'] = True
        cands = top_k[qi]
        # BoQ top-1 is cands[0]
        entry['boq_correct'] = int(cands[0]) in ypos
        # DR-VPR: rerank with β
        bs = d_boq_q[qi] @ d_boq_db[cands].T
        es = d_eq_q[qi] @ d_eq_db[cands].T
        score = (1 - BETA) * bs + BETA * es
        drvpr_top1 = int(cands[np.argmax(score)])
        entry['drvpr_correct'] = drvpr_top1 in ypos
        outcomes.append(entry)
    return outcomes


def plot_panel(ax, title, db_x, db_y, q_x, q_y, outcomes,
                highlight_yellow=False, recall_pct=None):
    # Database trajectory
    ax.plot(db_x, db_y, '-', color="#068ef0", linewidth=2.0,
            label='Database', alpha=0.85, zorder=1)
    # Query trajectory
    ax.plot(q_x, q_y, '--', color='#888888', linewidth=1.2,
            label='Query trajectory', alpha=0.6, zorder=1)
    # Origin
    ax.scatter(0, 0, color='black', s=120, marker='*',
                edgecolors='white', linewidths=1.2, zorder=6, label='Origin')

    valid = [o for o in outcomes if o['valid']]
    correct = [o for o in valid if o[title_key(title)]]
    failed = [o for o in valid if not o[title_key(title)]]

    if correct:
        cx, cy = zip(*[(o['qx'], o['qy']) for o in correct])
        ax.scatter(cx, cy, color='#2ca02c', s=34, marker='o',
                    edgecolors='white', linewidths=0.5, alpha=0.82,
                    zorder=4, label='Correct')
    if failed:
        fx, fy = zip(*[(o['qx'], o['qy']) for o in failed])
        # Muted red, smaller, lower alpha so failures read as pattern, not clutter
        ax.scatter(fx, fy, color='#c85a5a', s=22, marker='x',
                    linewidths=1.4, alpha=0.55, zorder=5,
                    label='Failed')

    # Yellow circles: queries where BoQ failed but DR-VPR succeeded
    if highlight_yellow:
        flips = [o for o in valid
                 if (not o['boq_correct']) and o['drvpr_correct']]
        if flips:
            yx, yy = zip(*[(o['qx'], o['qy']) for o in flips])
            # Soft yellow halo (underlay)
            ax.scatter(yx, yy, s=620, marker='o',
                        facecolors='#ffe08a', edgecolors='none',
                        alpha=0.35, zorder=6)
            # Bold ring (on top) — no label: large in-figure size would
            # blow up the legend marker too
            ax.scatter(yx, yy, s=360, marker='o',
                        facecolors='none', edgecolors='#e89000',
                        linewidths=3.2, zorder=8)
            # Proxy legend entry with normal size
            ax.scatter([], [], s=80, marker='o',
                        facecolors='none', edgecolors='#e89000',
                        linewidths=2.0, label='DR-VPR fix')

    ax.set_xlabel('X (m)', fontsize=11)
    ax.set_ylabel('Y (m)', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.legend(loc='lower right', frameon=True, framealpha=0.95, fontsize=9)
    ax.set_aspect('equal')
    ax.grid(alpha=0.25, linestyle=':')


def title_key(title):
    # The ordered 'correct' flag key used in outcomes[] — we use 'boq_correct'
    # for panel (a) and 'drvpr_correct' for panel (b). This helper maps
    # a panel title to its outcome key.
    t = title.lower()
    if 'dr-vpr' in t:
        return 'drvpr_correct'
    return 'boq_correct'


def main():
    # --- Load models + data ---
    boq = load_boq_r50()
    ckpt = find_c16_ckpt(SEED_FOR_PLOT)
    print(f'C16 ckpt (seed={SEED_FOR_PLOT}): {ckpt}')
    equi = load_c16_equi(ckpt)

    db_ds_boq = InferDataset(SEQ_DB, dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_BOQ)
    q_ds_boq  = InferDataset(SEQ_Q,  dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_BOQ)
    db_ds_eq  = InferDataset(SEQ_DB, dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_EQUI)
    q_ds_eq   = InferDataset(SEQ_Q,  dataset_path=CONSLAM_PATH, img_size=IMG_SIZE_EQUI)
    db_dl_boq = DataLoader(db_ds_boq, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    q_dl_boq  = DataLoader(q_ds_boq,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    db_dl_eq  = DataLoader(db_ds_eq,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    q_dl_eq   = DataLoader(q_ds_eq,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    print('Extracting BoQ-R50 descriptors...')
    d_boq_db = extract_boq(boq, db_dl_boq)
    d_boq_q  = extract_boq(boq, q_dl_boq)
    print(f'  db={d_boq_db.shape} q={d_boq_q.shape}')
    print('Extracting equi-C16 descriptors...')
    d_eq_db  = extract_equi(equi, db_dl_eq)
    d_eq_q   = extract_equi(equi, q_dl_eq)
    print(f'  db={d_eq_db.shape} q={d_eq_q.shape}')

    # Rotate query poses once
    q_rot = rotate_query_poses(q_ds_boq.poses, THETA)

    outcomes = compute_per_query_outcomes(
        d_boq_db, d_boq_q, d_eq_db, d_eq_q, q_rot, db_ds_boq.poses)

    valid = [o for o in outcomes if o['valid']]
    boq_r1 = sum(o['boq_correct'] for o in valid) / len(valid) * 100
    drvpr_r1 = sum(o['drvpr_correct'] for o in valid) / len(valid) * 100
    flip_to_correct = sum(1 for o in valid
                           if (not o['boq_correct']) and o['drvpr_correct'])
    flip_to_wrong  = sum(1 for o in valid
                          if o['boq_correct'] and (not o['drvpr_correct']))
    print(f'\nValid queries: {len(valid)}/{len(outcomes)}')
    print(f'BoQ-R50     R@1 = {boq_r1:.2f}%')
    print(f'DR-VPR      R@1 = {drvpr_r1:.2f}%')
    print(f'  Flip to correct (yellow circles): {flip_to_correct}')
    print(f'  Flip to wrong   : {flip_to_wrong}')

    # --- Plot ---
    plt.rcParams.update({'font.family': 'serif', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=300)
    db_poses = db_ds_boq.poses
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    q_x, q_y = q_rot[:, 3], q_rot[:, 7]

    plot_panel(ax1, '(a) BoQ (ResNet50)', db_x, db_y, q_x, q_y, outcomes,
                highlight_yellow=False, recall_pct=boq_r1)
    plot_panel(ax2, '(b) DR-VPR (ResNet50)', db_x, db_y, q_x, q_y, outcomes,
                highlight_yellow=True, recall_pct=drvpr_r1)

    plt.tight_layout()
    out_pdf = OUT_DIR / 'conslam_trajectory_boq_vs_drvpr.pdf'
    out_png = OUT_DIR / 'conslam_trajectory_boq_vs_drvpr.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'\nSaved:\n  {out_pdf}\n  {out_png}')


if __name__ == '__main__':
    main()
