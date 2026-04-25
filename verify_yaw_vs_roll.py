"""Verify yaw vs roll on ConSLAM.

For every valid query-positive pair on ConSLAM (Seq5 db vs Seq4 q,
θ=15°), extract the camera's three Tait-Bryan ZYX angles
(yaw=world-Z, pitch=Y, roll=X = camera optical-axis rotation in this
convention). Then:

(A) Correlation between Δyaw and Δroll across all (q, nearest-pos)
    pairs. If high, the per-yaw bucket analysis is implicitly capturing
    roll variation too.
(B) Dominant rotation axis of the relative rotation R_q^T · R_db.
    If the axis is mostly +/-Z, the rotation is yaw. If mostly +/-X,
    it is roll (image-plane rotation in OpenCV camera convention).
(C) Re-bucket BoQ-R50 and DR-VPR (R50+C16) R@1 by Δroll instead of
    Δyaw. Compare per-bucket gains to the yaw-bucketed numbers.
"""
from __future__ import annotations
import glob
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from Conslam_dataset_rot import InferDataset, get_yaw_from_pose
from models.equi_multiscale import E2ResNetMultiScale


DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
BETA = 0.10
THETA = 15.0
SEED = 42

CONSLAM_PATH = './datasets/ConSLAM/'
SEQ_DB, SEQ_Q = 'Sequence5', 'Sequence4'


def get_R(pose):
    return np.array([[pose[0], pose[1], pose[2]],
                      [pose[4], pose[5], pose[6]],
                      [pose[8], pose[9], pose[10]]])


def euler_zyx_deg(R):
    """Tait-Bryan ZYX intrinsic.  In OpenCV camera convention
    (X right, Y down, Z forward) and world Z up:
      yaw  = rotation about world Z (camera heading)
      pitch = rotation about camera Y (tilt up/down)
      roll = rotation about camera X (in-plane image rotation around
             camera Z's projection in some conventions; see below).

    NB: which axis is the optical axis depends on the convention; we
    print all three and look at the axis-angle decomposition separately.
    """
    yaw   = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0],
                                    np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    roll  = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def angdiff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def relative_axis_angle(R_q, R_db):
    """Return (angle_deg, unit_axis) of R_q^T @ R_db."""
    R_rel = R_q.T @ R_db
    cos_a = np.clip((np.trace(R_rel) - 1) / 2.0, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_a))
    if angle < 1e-3:
        return 0.0, np.array([0.0, 0.0, 1.0])
    axis = np.array([R_rel[2, 1] - R_rel[1, 2],
                      R_rel[0, 2] - R_rel[2, 0],
                      R_rel[1, 0] - R_rel[0, 1]])
    axis = axis / (2.0 * np.sin(np.radians(angle)))
    return angle, axis


def inplane_rotation_q_to_db(R_q, R_db):
    """TRUE in-plane image rotation between q and db, after aligning
    db's optical axis to q's. Robotics convention: device X = optical
    axis (we verified this from ConSLAM pose 0 having R = I and Z up).

    Algorithm:
      1. R_rel = R_q^T @ R_db (db basis in q frame)
      2. v = R_rel[:, 0] = db's optical axis in q frame
      3. R_align = rotation that takes (1,0,0) to v WITHOUT any
         rotation about (1,0,0) (Rodrigues with axis (1,0,0) × v)
      4. R_inplane = R_align^T @ R_rel  (residual = rotation about X)
      5. in_plane_angle = arctan2(R_inplane[2,1], R_inplane[2,2])
    """
    R_rel = R_q.T @ R_db
    v = R_rel[:, 0]
    cos_theta = float(np.clip(v[0], -1.0, 1.0))
    if cos_theta > 0.9999:
        return float(np.degrees(np.arctan2(R_rel[2, 1], R_rel[2, 2])))
    sin_theta = float(np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta)))
    if sin_theta < 1e-9:
        return 0.0
    # axis = (1,0,0) × v, normalized
    ax = np.array([0.0, -v[2], v[1]]) / sin_theta
    K = np.array([[0, -ax[2], ax[1]],
                   [ax[2], 0, -ax[0]],
                   [-ax[1], ax[0], 0]])
    R_align = np.eye(3) + sin_theta * K + (1.0 - cos_theta) * (K @ K)
    R_inplane = R_align.T @ R_rel
    return float(np.degrees(np.arctan2(R_inplane[2, 1], R_inplane[2, 2])))


def optical_axis_misalignment_deg(R_q, R_db):
    """Angle between query optical axis (X col, robotics) and db's."""
    x_q = R_q[:, 0]
    x_db = R_db[:, 0]
    cos_a = np.clip(np.dot(x_q, x_db), -1.0, 1.0)
    return np.degrees(np.arccos(cos_a))


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
def extract(model, dl, kind='boq'):
    out = []
    for imgs, _ in dl:
        o = model(imgs.to(DEVICE))
        if kind == 'boq':
            if isinstance(o, tuple):
                o = o[0]
            o = F.normalize(o, p=2, dim=1)
        out.append(o.cpu().numpy())
    return np.vstack(out)


def rotate_query_poses(q_poses_raw, theta_degrees):
    th = np.deg2rad(theta_degrees)
    q = q_poses_raw.copy()
    qx, qy = q[:, 3], q[:, 7]
    qx_rot = qx * np.cos(th) - qy * np.sin(th)
    qy_rot = qx * np.sin(th) + qy * np.cos(th)
    q[:, 3], q[:, 7] = qx_rot, qy_rot
    return q


def main():
    print('=' * 90)
    print('Yaw-vs-roll verification on ConSLAM (Seq5 db, Seq4 q, θ=15°)')
    print('=' * 90)

    db_ds = InferDataset(SEQ_DB, dataset_path=CONSLAM_PATH, img_size=IMG_SIZE)
    q_ds  = InferDataset(SEQ_Q,  dataset_path=CONSLAM_PATH, img_size=IMG_SIZE)

    # --- Pose-only analysis first (no neural net needed) ---
    db_poses = db_ds.poses
    q_poses = rotate_query_poses(q_ds.poses, THETA)

    # 1. Sanity-check: distribution of camera angles in db & q
    db_eulers = np.array([euler_zyx_deg(get_R(p)) for p in db_poses])
    q_eulers = np.array([euler_zyx_deg(get_R(p)) for p in q_poses])
    print('\n[1] Per-image absolute angles (deg)')
    for name, arr in [('  DB yaw  ', db_eulers[:, 0]), ('  DB pitch', db_eulers[:, 1]),
                       ('  DB roll ', db_eulers[:, 2]), ('  Q  yaw  ', q_eulers[:, 0]),
                       ('  Q  pitch', q_eulers[:, 1]), ('  Q  roll ', q_eulers[:, 2])]:
        print(f"  {name}  min={arr.min():7.2f}  median={np.median(arr):7.2f}  "
              f"max={arr.max():7.2f}  std={arr.std():6.2f}")

    # 2. For each valid query, find nearest-pos GT and compute Δyaw, Δpitch, Δroll, axis
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    pairs = []  # list of dicts with {qi, p_idx, dyaw, dpitch, droll, rel_angle, axis_norm}
    for qi in range(len(q_poses)):
        ds = (q_poses[qi, 3] - db_x) ** 2 + (q_poses[qi, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0:
            continue
        q_yaw_val = q_eulers[qi, 0]
        valid = []
        for p in pp:
            db_yaw_val = db_eulers[p, 0]
            d = angdiff(q_yaw_val, db_yaw_val)
            if d <= YAW_THRESHOLD:
                valid.append((int(p), d))
        if not valid:
            continue
        # Pick nearest by yaw (matches paper's "nearest_pos_yaw")
        valid.sort(key=lambda x: x[1])
        p, dy = valid[0]
        R_q, R_db = get_R(q_poses[qi]), get_R(db_poses[p])
        yaw_q, pitch_q, roll_q = q_eulers[qi]
        yaw_db, pitch_db, roll_db = db_eulers[p]
        rel_angle, axis = relative_axis_angle(R_q, R_db)
        # Proper "in-plane image rotation" between query and db, in q's frame
        inplane_rot = abs(inplane_rotation_q_to_db(R_q, R_db))
        if inplane_rot > 180:
            inplane_rot = 360 - inplane_rot
        opt_axis_misalign = optical_axis_misalignment_deg(R_q, R_db)
        pairs.append({
            'qi': qi, 'p': p,
            'dyaw': angdiff(yaw_q, yaw_db),
            'dpitch': angdiff(pitch_q, pitch_db),
            'droll': angdiff(roll_q, roll_db),
            'inplane': inplane_rot,
            'opt_misalign': opt_axis_misalign,
            'rel_angle': rel_angle,
            'axis': axis,
        })

    print(f'\n[2] {len(pairs)} valid (query, nearest-pos) pairs')

    # 3. Correlation Δyaw vs Δroll vs Δpitch vs in-plane-rot
    dyaws    = np.array([p['dyaw']    for p in pairs])
    dpitch   = np.array([p['dpitch']  for p in pairs])
    drolls   = np.array([p['droll']   for p in pairs])
    inplanes = np.array([p['inplane'] for p in pairs])
    optmis   = np.array([p['opt_misalign'] for p in pairs])
    rel_a    = np.array([p['rel_angle'] for p in pairs])
    print('\n[3] Δ-angle distributions (across nearest-pos pairs)')
    for name, arr in [('Δyaw  ', dyaws), ('Δpitch', dpitch),
                       ('Δroll-Tait', drolls),
                       ('In-plane (true image rot)', inplanes),
                       ('Optical axis misalign', optmis),
                       ('Rel-rot total angle', rel_a)]:
        print(f"  {name:<30s}  median={np.median(arr):6.2f}°  "
              f"mean={arr.mean():6.2f}°  max={arr.max():6.2f}°  std={arr.std():5.2f}")

    # Pearson correlations
    print('\n[4] Pearson correlation between Δ angles')
    for n1, a1 in [('Δyaw', dyaws), ('Δyaw', dyaws), ('Δpitch', dpitch)]:
        for n2, a2 in [('Δroll', drolls), ('Δpitch', dpitch), ('Δroll', drolls)]:
            if n1 == n2:
                continue
            r = np.corrcoef(a1, a2)[0, 1]
            print(f"  ρ({n1}, {n2}) = {r:+.3f}")
            break

    # Quick non-redundant set — focus on in-plane vs yaw
    print('  --- non-redundant ---')
    print(f"  ρ(Δyaw,         in-plane)         = {np.corrcoef(dyaws,    inplanes)[0, 1]:+.3f}")
    print(f"  ρ(Δyaw,         opt-axis-misalign) = {np.corrcoef(dyaws,    optmis)[0, 1]:+.3f}")
    print(f"  ρ(in-plane,     opt-axis-misalign) = {np.corrcoef(inplanes, optmis)[0, 1]:+.3f}")
    print(f"  ρ(Δyaw,         Δroll-Tait)        = {np.corrcoef(dyaws,    drolls)[0, 1]:+.3f}")
    print(f"  ρ(in-plane,     Δroll-Tait)        = {np.corrcoef(inplanes, drolls)[0, 1]:+.3f}")

    # 5. Dominant rotation axis of the relative rotation
    axes = np.array([p['axis'] for p in pairs])  # (N, 3)
    abs_axes = np.abs(axes)
    print('\n[5] Dominant axis of relative rotation R_q^T · R_db')
    print('    axis components in WORLD frame: [X, Y, Z] (Z assumed up)')
    print(f"    mean |axis_X| = {abs_axes[:, 0].mean():.3f}")
    print(f"    mean |axis_Y| = {abs_axes[:, 1].mean():.3f}")
    print(f"    mean |axis_Z| = {abs_axes[:, 2].mean():.3f}")
    print('    → if |axis_Z| ≫ |axis_X|, |axis_Y|, the rotation is dominantly YAW (about world up)')
    print('    → if |axis_X| or |axis_Y| dominate, rotation is pitch/roll-like')

    # 6. Bucket comparison: Δyaw vs Δroll vs rel_angle
    # Need DR-VPR/BoQ correctness per query — load nets
    print('\n' + '=' * 90)
    print('[6] Re-bucket BoQ-R50 vs DR-VPR (R50+C16) R@1 by Δyaw / Δroll / rel-angle')
    print('=' * 90)

    boq = load_boq_r50()
    ckpt = find_c16_ckpt(SEED)
    print(f'C16 ckpt: {ckpt}')
    equi = load_c16_equi(ckpt)

    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    q_dl  = DataLoader(q_ds,  batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    print('Extracting descriptors...')
    d_boq_db = extract(boq, db_dl, 'boq')
    d_boq_q  = extract(boq, q_dl,  'boq')
    d_eq_db  = extract(equi, db_dl, 'equi')
    d_eq_q   = extract(equi, q_dl,  'equi')

    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k = idx.search(d_boq_q, TOP_K)

    # For each pair (each valid query → nearest-pos), compute correctness
    boq_correct = []
    drvpr_correct = []
    for p in pairs:
        qi = p['qi']
        cands = top_k[qi]
        # Find ALL valid GT positives for this query (not just nearest-pos)
        ds = (q_poses[qi, 3] - db_x) ** 2 + (q_poses[qi, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        ypos_set = set()
        q_yaw_val = q_eulers[qi, 0]
        for q in pp:
            d = angdiff(q_yaw_val, db_eulers[q, 0])
            if d <= YAW_THRESHOLD:
                ypos_set.add(int(q))
        boq_top1 = int(cands[0])
        bs = d_boq_q[qi] @ d_boq_db[cands].T
        es = d_eq_q[qi] @ d_eq_db[cands].T
        score = (1 - BETA) * bs + BETA * es
        drvpr_top1 = int(cands[np.argmax(score)])
        boq_correct.append(boq_top1 in ypos_set)
        drvpr_correct.append(drvpr_top1 in ypos_set)
    boq_correct = np.array(boq_correct)
    drvpr_correct = np.array(drvpr_correct)

    print(f'\nOverall:  BoQ R@1 = {boq_correct.mean()*100:.2f}%  '
          f'DR-VPR R@1 = {drvpr_correct.mean()*100:.2f}%  '
          f'Δ = +{(drvpr_correct.mean() - boq_correct.mean())*100:.2f}')

    def buckets(label, vals, edges):
        print(f'\n--- bucket by {label} ---')
        print(f"{'bucket':>14s}  {'n':>4s}  {'BoQ R@1':>8s}  {'DR R@1':>8s}  {'Δ':>6s}")
        for i in range(len(edges) - 1):
            mask = (vals >= edges[i]) & (vals < edges[i + 1])
            n = mask.sum()
            if n == 0:
                continue
            b = boq_correct[mask].mean() * 100
            d = drvpr_correct[mask].mean() * 100
            print(f"  [{edges[i]:5.1f}, {edges[i+1]:5.1f})°  {n:4d}  "
                  f"{b:7.2f}%  {d:7.2f}%  {d-b:+6.2f}")

    edges_yaw = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    edges_inplane = [0, 5, 10, 15, 22.5, 30, 45, 90, 180]
    edges_optmis = [0, 10, 20, 30, 45, 60, 90, 180]
    edges_relangle = [0, 10, 20, 30, 40, 50, 60, 80, 180]

    buckets('Δyaw (current paper)', dyaws, edges_yaw)
    buckets('TRUE in-plane image rot (what equi handles)',
            inplanes, edges_inplane)
    buckets('Optical axis misalignment', optmis, edges_optmis)
    buckets('Total relative rotation', rel_a, edges_relangle)


if __name__ == '__main__':
    main()
