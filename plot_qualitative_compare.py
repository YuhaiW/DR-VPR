"""
Qualitative retrieval comparison: BoQ(R50) vs DR-VPR (R50+C16, main).

For ConSLAM (Seq5 db, Seq4 q, θ=15°), select query examples where
DR-VPR's top-1 is correct but BoQ's top-1 is wrong, then build a
N-row × 3-col figure:
    [query]  [DR-VPR top-1 (green border)]  [BoQ top-1 (red border)]

Output: figures/qualitative_compare_boq_vs_drvpr.{pdf,png}
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
from matplotlib.patches import Rectangle
from PIL import Image
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset, get_yaw_from_pose
from models.equi_multiscale import E2ResNetMultiScale


def _get_R(pose):
    return np.array([[pose[0], pose[1], pose[2]],
                      [pose[4], pose[5], pose[6]],
                      [pose[8], pose[9], pose[10]]])


def _inplane_rotation_deg(R_q, R_db):
    """TRUE in-plane image rotation (roll) between q and db under the
    robotics convention used by ConSLAM (X = optical axis). Align db's
    X to q's, then extract residual rotation about X."""
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


DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
BETA = 0.10
THETA = 15.0
SEED_FOR_PLOT = 42

CONSLAM_PATH = './datasets/ConSLAM/'
SEQ_DB, SEQ_Q = 'Sequence5', 'Sequence4'
N_ROWS = 4

OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)


def find_c16_ckpt(seed):
    pat = (f'LOGS/equi_standalone_seed{seed}_ms_C16/lightning_logs/'
            f'version_*/checkpoints/equi_ms_seed{seed}_epoch*.ckpt')
    ckpts = glob.glob(pat)
    def r1(p):
        m = re.search(r'R1\[([\d.]+)\]', p)
        return float(m.group(1)) if m else 0.0
    return max(ckpts, key=r1)


def load_boq_r50():
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


def compute_outcomes(d_boq_db, d_boq_q, d_eq_db, d_eq_q,
                      q_poses_rot, db_poses):
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k = idx.search(d_boq_q, TOP_K)

    rows = []
    for qi in range(len(q_poses_rot)):
        ds = (q_poses_rot[qi, 3] - db_x) ** 2 + (q_poses_rot[qi, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0:
            continue
        q_yaw = get_yaw_from_pose(q_poses_rot[qi])
        ypos = []
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180:
                d = 360 - d
            if d <= YAW_THRESHOLD:
                ypos.append((int(p), d))
        if not ypos:
            continue
        ypos_set = {p for p, _ in ypos}
        ypos_sorted = sorted(ypos, key=lambda x: x[1])
        nearest_p, nearest_pos_yaw = ypos_sorted[0]
        # TRUE in-plane image rotation (roll) between q and the nearest-yaw
        # GT positive — the quantity the C_16 equivariant branch actually
        # handles (robotics convention, X = optical axis).
        R_q = _get_R(q_poses_rot[qi])
        R_db_np = _get_R(db_poses[nearest_p])
        inplane = abs(_inplane_rotation_deg(R_q, R_db_np))
        if inplane > 180:
            inplane = 360 - inplane
        cands = top_k[qi]
        boq_top1 = int(cands[0])
        bs = d_boq_q[qi] @ d_boq_db[cands].T
        es = d_eq_q[qi] @ d_eq_db[cands].T
        score = (1 - BETA) * bs + BETA * es
        drvpr_top1 = int(cands[np.argmax(score)])
        rows.append({
            'qi': qi,
            'boq_top1': boq_top1,
            'drvpr_top1': drvpr_top1,
            'boq_correct': boq_top1 in ypos_set,
            'drvpr_correct': drvpr_top1 in ypos_set,
            'nearest_pos_yaw': nearest_pos_yaw,
            'nearest_pos_inplane': inplane,
            'gt_positives': ypos_set,
        })
    return rows


def add_border(ax, color, width=8):
    """Add a thick coloured border around an imshow axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(width)
        spine.set_edgecolor(color)


def main():
    boq = load_boq_r50()
    ckpt = find_c16_ckpt(SEED_FOR_PLOT)
    print(f'Using C16 ckpt seed={SEED_FOR_PLOT}: {ckpt}')
    equi = load_c16_equi(ckpt)

    db_ds = InferDataset(SEQ_DB, dataset_path=CONSLAM_PATH, img_size=IMG_SIZE)
    q_ds  = InferDataset(SEQ_Q,  dataset_path=CONSLAM_PATH, img_size=IMG_SIZE)
    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    q_dl  = DataLoader(q_ds,  batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)

    print('Extracting descriptors...')
    d_boq_db = extract_boq(boq, db_dl)
    d_boq_q  = extract_boq(boq, q_dl)
    d_eq_db  = extract_equi(equi, db_dl)
    d_eq_q   = extract_equi(equi, q_dl)
    print(f'  BoQ db={d_boq_db.shape} q={d_boq_q.shape}')
    print(f'  Equi db={d_eq_db.shape} q={d_eq_q.shape}')

    q_rot = rotate_query_poses(q_ds.poses, THETA)
    rows = compute_outcomes(d_boq_db, d_boq_q, d_eq_db, d_eq_q,
                              q_rot, db_ds.poses)
    flips = [r for r in rows if (not r['boq_correct']) and r['drvpr_correct']]
    print(f'\n{len(flips)} queries: BoQ wrong → DR-VPR correct')
    # Sort by descending TRUE in-plane rotation (roll) — the quantity
    # the C_16 equivariant branch actually handles.
    flips.sort(key=lambda r: -r['nearest_pos_inplane'])
    for r in flips:
        q_name = Path(q_ds.imgs_path[r['qi']]).name
        boq_name = Path(db_ds.imgs_path[r['boq_top1']]).name
        drvpr_name = Path(db_ds.imgs_path[r['drvpr_top1']]).name
        print(f"  qi={r['qi']:3d}  roll={r['nearest_pos_inplane']:5.1f}°  "
              f"yaw={r['nearest_pos_yaw']:5.1f}°  "
              f"q={q_name}  drvpr={drvpr_name}  boq={boq_name}")

    # Also show all DR-VPR-correct queries with high in-plane rotation,
    # for picking rotation-heavy examples beyond the BoQ-flip set.
    print('\nTop 15 DR-VPR-correct queries by nearest-positive in-plane (roll):')
    high_inplane = [r for r in rows if r['drvpr_correct']]
    high_inplane.sort(key=lambda r: -r['nearest_pos_inplane'])
    for r in high_inplane[:15]:
        q_name = Path(q_ds.imgs_path[r['qi']]).name
        boq_name = Path(db_ds.imgs_path[r['boq_top1']]).name
        drvpr_name = Path(db_ds.imgs_path[r['drvpr_top1']]).name
        boq_ok = '✓' if r['boq_correct'] else '✗'
        print(f"  qi={r['qi']:3d}  roll={r['nearest_pos_inplane']:5.1f}°  "
              f"yaw={r['nearest_pos_yaw']:5.1f}°  "
              f"BoQ={boq_ok}  q={q_name}  drvpr={drvpr_name}  boq={boq_name}")

    # Curated rows: each showcases a different failure mode of the
    # discriminative baseline that DR-VPR resolves.
    #   Row 1 = highest-in-plane-rotation (roll) flip, the quantity
    #           the C_16 equivariant branch is designed to handle.
    #   Rows 2-4 = manually-picked exemplars for the other 3 failure
    #              modes (low-light, distance variation, structural
    #              change). All four qi's are BoQ wrong -> DR-VPR
    #              correct flips at this seed.
    flips_by_qi = {r['qi']: r for r in flips}
    SELECTED_QIS = [
        flips[0]['qi'],   # highest roll
        372,              # low-light
        250,              # distance variation
        194,              # structural change
    ]
    selected = [flips_by_qi[qi] for qi in SELECTED_QIS]

    # ---- Build figure ----
    # Row 1 label uses the TRUE in-plane rotation (roll, robotics
    # convention), not world-frame yaw — because that is what the
    # C_16 equivariant branch actually handles.
    ROW_LABELS = [
        "In-plane rotation",
        "Low light",
        "Distance variation",
        "Structural change",
    ]

    # ConSLAM images are ~4:3; size figure so subplots match → no internal
    # whitespace, and hspace truly controls inter-row gap.
    fig, axes = plt.subplots(N_ROWS, 3, figsize=(11, 2.55 * N_ROWS),
                              dpi=200, gridspec_kw={'wspace': 0.04, 'hspace': 0.04})
    if N_ROWS == 1:
        axes = np.array([axes])

    col_titles = ['Query', 'DR-VPR (R50+C16) top-1', 'BoQ (R50) top-1']
    for c in range(3):
        axes[0, c].set_title(col_titles[c], fontsize=13, fontweight='bold', pad=8)

    GREEN = '#2ca02c'
    RED   = '#d62728'

    for row_i, r in enumerate(selected):
        qi = r['qi']
        q_img = Image.open(q_ds.imgs_path[qi])
        drvpr_img = Image.open(db_ds.imgs_path[r['drvpr_top1']])
        boq_img = Image.open(db_ds.imgs_path[r['boq_top1']])
        axes[row_i, 0].imshow(q_img, aspect='auto')
        axes[row_i, 0].set_xticks([])
        axes[row_i, 0].set_yticks([])
        for s in axes[row_i, 0].spines.values():
            s.set_visible(False)
        # Vertical issue-type label on the left of the query column
        axes[row_i, 0].text(
            -0.045, 0.5, ROW_LABELS[row_i],
            transform=axes[row_i, 0].transAxes,
            rotation=90, ha='center', va='center',
            fontsize=12, fontweight='bold', color='#333333')
        axes[row_i, 1].imshow(drvpr_img, aspect='auto')
        axes[row_i, 1].set_xticks([])
        axes[row_i, 1].set_yticks([])
        add_border(axes[row_i, 1], GREEN, width=6)
        axes[row_i, 2].imshow(boq_img, aspect='auto')
        axes[row_i, 2].set_xticks([])
        axes[row_i, 2].set_yticks([])
        add_border(axes[row_i, 2], RED, width=6)

    plt.subplots_adjust(left=0.04, right=0.995, top=0.95, bottom=0.01)
    out_pdf = OUT_DIR / 'qualitative_compare_boq_vs_drvpr.pdf'
    out_png = OUT_DIR / 'qualitative_compare_boq_vs_drvpr.png'
    fig.savefig(out_pdf, bbox_inches='tight', facecolor='white')
    fig.savefig(out_png, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'\nSaved:\n  {out_pdf}\n  {out_png}')


if __name__ == '__main__':
    main()
