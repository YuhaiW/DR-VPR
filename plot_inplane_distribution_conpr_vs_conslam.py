"""TRUE in-plane image rotation distribution of queries:
ConPR (rotation-benign) vs ConSLAM (rotation-heavy).

Companion to the existing yaw-distribution figure but on the
quantity our C_16 equivariant branch ACTUALLY handles (rotation
about the camera's optical axis = device X under robotics
convention used by ConSLAM/ConPR).

For each valid query (within 5 m of some db image AND within 80°
yaw of some db image), we pick the nearest-yaw GT positive and
compute the in-plane image rotation between query and that
positive (axis-aligned residual rotation about q's optical axis).

Output: figures/inplane_distribution_conpr_vs_conslam.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from Conslam_dataset_rot import InferDataset as ConslamDS
from conpr_eval_dataset_rot import InferDataset as ConprDS


GT_THRES = 5.0
YAW_THRESHOLD = 80.0
CONSLAM_PATH = './datasets/ConSLAM/'
CONSLAM_THETA = 15.0
CONPR_PATH = './datasets/ConPR/'
CONPR_SEQS = ['20230623', '20230531', '20230611', '20230627', '20230628',
              '20230706', '20230717', '20230803', '20230809', '20230818']
CONPR_THETA = 0.0

OUT_DIR = Path('figures')
OUT_DIR.mkdir(exist_ok=True)


def get_R(pose):
    return np.array([[pose[0], pose[1], pose[2]],
                      [pose[4], pose[5], pose[6]],
                      [pose[8], pose[9], pose[10]]])


def euler_zyx_deg(R):
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0],
                                    np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2)))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def angdiff(a, b):
    d = abs(a - b) % 360
    return d if d <= 180 else 360 - d


def inplane_rotation_q_to_db(R_q, R_db):
    """In-plane image rotation between q and db, robotics convention
    (X = optical axis). Aligns db's X to q's X, then extracts residual
    rotation about X."""
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


def rotate_query_xy(q_poses_raw, theta_deg):
    th = np.deg2rad(theta_deg)
    q = q_poses_raw.copy()
    qx, qy = q[:, 3], q[:, 7]
    qx_rot = qx * np.cos(th) - qy * np.sin(th)
    qy_rot = qx * np.sin(th) + qy * np.cos(th)
    q[:, 3], q[:, 7] = qx_rot, qy_rot
    return q


def collect_inplane_and_yaw(db_poses, q_poses):
    """For each valid q, find nearest-yaw GT positive then compute Δyaw +
    abs in-plane rotation. Returns two parallel lists."""
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    db_eul = np.array([euler_zyx_deg(get_R(p)) for p in db_poses])
    q_eul  = np.array([euler_zyx_deg(get_R(p)) for p in q_poses])
    inplanes, yaws = [], []
    for qi in range(len(q_poses)):
        ds = (q_poses[qi, 3] - db_x) ** 2 + (q_poses[qi, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0:
            continue
        valid = []
        q_yaw = q_eul[qi, 0]
        for p in pp:
            d = angdiff(q_yaw, db_eul[p, 0])
            if d <= YAW_THRESHOLD:
                valid.append((int(p), d))
        if not valid:
            continue
        valid.sort(key=lambda x: x[1])
        p, dy = valid[0]
        R_q, R_db = get_R(q_poses[qi]), get_R(db_poses[p])
        ip = abs(inplane_rotation_q_to_db(R_q, R_db))
        if ip > 180:
            ip = 360 - ip
        inplanes.append(ip)
        yaws.append(dy)
    return np.array(inplanes), np.array(yaws)


def compute_or_load_cache():
    cache = OUT_DIR / 'inplane_yaw_cache.npz'
    if cache.exists():
        print(f'Loading cached arrays from {cache}')
        z = np.load(cache)
        return z['cs_ip'], z['cs_yaw'], z['cp_ip'], z['cp_yaw']

    print('ConSLAM:')
    db_ds = ConslamDS('Sequence5', dataset_path=CONSLAM_PATH)
    q_ds  = ConslamDS('Sequence4', dataset_path=CONSLAM_PATH)
    q_poses = rotate_query_xy(q_ds.poses, CONSLAM_THETA)
    cs_ip, cs_yaw = collect_inplane_and_yaw(db_ds.poses, q_poses)

    print('\nConPR (full 10-seq):')
    db = ConprDS(CONPR_SEQS[0], dataset_path=CONPR_PATH)
    cp_ip_all, cp_yaw_all = [], []
    for q_seq in CONPR_SEQS[1:]:
        q = ConprDS(q_seq, dataset_path=CONPR_PATH)
        q_poses_rot = rotate_query_xy(q.poses, CONPR_THETA)
        ip, yaw = collect_inplane_and_yaw(db.poses, q_poses_rot)
        cp_ip_all.append(ip)
        cp_yaw_all.append(yaw)
        print(f'  q={q_seq}: {len(ip)} valid')
    cp_ip = np.concatenate(cp_ip_all)
    cp_yaw = np.concatenate(cp_yaw_all)

    np.savez(cache, cs_ip=cs_ip, cs_yaw=cs_yaw, cp_ip=cp_ip, cp_yaw=cp_yaw)
    print(f'Cached → {cache}')
    return cs_ip, cs_yaw, cp_ip, cp_yaw


def print_stats(name, ip, yaw):
    print(f'{name}:  n={len(ip)}')
    print(f'  in-plane: median={np.median(ip):.2f}°  '
          f'90%={np.percentile(ip,90):.2f}°  max={ip.max():.2f}°')
    print(f'  Δyaw    : median={np.median(yaw):.2f}°  '
          f'90%={np.percentile(yaw,90):.2f}°  max={yaw.max():.2f}°')


def main():
    cs_ip, cs_yaw, cp_ip, cp_yaw = compute_or_load_cache()
    print_stats('ConSLAM', cs_ip, cs_yaw)
    print_stats('ConPR  ', cp_ip, cp_yaw)

    # ---- Plot ----
    plt.rcParams.update({
        'font.family': 'serif', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'stix',
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=200)

    bins = np.arange(0, 32.5, 2.5)  # 2.5° wide; cover up to 30°+
    palette = {'conpr': '#2E5395', 'conslam': '#C25A2E'}

    # Left: in-plane image rotation (the thing C_16 actually handles)
    ax = axes[0]
    ax.hist(cp_ip, bins=bins, density=True, alpha=0.6,
            color=palette['conpr'], edgecolor='white',
            label=f'ConPR  (n={len(cp_ip):,})')
    ax.hist(cs_ip, bins=bins, density=True, alpha=0.6,
            color=palette['conslam'], edgecolor='white',
            label=f'ConSLAM (n={len(cs_ip):,})')
    ax.axvline(22.5, ls='--', color='#444', lw=1.0)
    ax.text(22.6, ax.get_ylim()[1] * 0.92,
            r'$C_{16}$ step (22.5$^\circ$)', fontsize=9,
            color='#444', va='top')
    ax.set_xlabel('In-plane image rotation between query and matched-DB ($^\\circ$)',
                   fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(a) TRUE in-plane image rotation\n'
                  '(what $C_{16}$ equivariance handles)',
                  fontsize=11.5, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=':')

    # Right: Δyaw (the previous figure's metric, for reference)
    bins_yaw = np.arange(0, 90, 5)
    ax = axes[1]
    ax.hist(cp_yaw, bins=bins_yaw, density=True, alpha=0.6,
            color=palette['conpr'], edgecolor='white',
            label=f'ConPR  (n={len(cp_yaw):,})')
    ax.hist(cs_yaw, bins=bins_yaw, density=True, alpha=0.6,
            color=palette['conslam'], edgecolor='white',
            label=f'ConSLAM (n={len(cs_yaw):,})')
    ax.set_xlabel(r'$\Delta$yaw (camera heading) between query and matched-DB ($^\circ$)',
                   fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(b) $\\Delta$yaw / camera heading\n(previous figure metric)',
                  fontsize=11.5, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, framealpha=0.95)
    ax.grid(alpha=0.25, linestyle=':')

    plt.tight_layout()
    out_pdf = OUT_DIR / 'inplane_distribution_conpr_vs_conslam.pdf'
    out_png = OUT_DIR / 'inplane_distribution_conpr_vs_conslam.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved:\n  {out_pdf}\n  {out_png}')

    # ---------- Bin chart with explicit % labels ----------
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4.6), dpi=200)

    # In-plane bins: dense up to C_16 step (22.5°), then coarser
    ip_edges = [0, 5, 10, 15, 22.5, 30, 45, 90]
    ip_labels = ['0–5', '5–10', '10–15', '15–22.5', '22.5–30',
                  '30–45', '45+']

    # Δyaw bins: 10° steps up to 80°
    yaw_edges = [0, 10, 20, 30, 40, 50, 60, 70, 80]
    yaw_labels = ['0–10', '10–20', '20–30', '30–40', '40–50',
                   '50–60', '60–70', '70–80']

    def pct_per_bin(arr, edges):
        counts, _ = np.histogram(arr, bins=edges)
        return 100.0 * counts / max(len(arr), 1)

    cp_ip_pct = pct_per_bin(cp_ip, ip_edges)
    cs_ip_pct = pct_per_bin(cs_ip, ip_edges)
    cp_yaw_pct = pct_per_bin(cp_yaw, yaw_edges)
    cs_yaw_pct = pct_per_bin(cs_yaw, yaw_edges)

    def bar_pair(ax, labels, pct_a, pct_b, title, xlabel,
                  c_a, c_b, label_a, label_b, vline_idx=None):
        x = np.arange(len(labels))
        w = 0.4
        ymax = max(pct_a.max(), pct_b.max())
        ax.bar(x - w/2, pct_a, w, color=c_a, edgecolor='white',
                label=label_a)
        ax.bar(x + w/2, pct_b, w, color=c_b, edgecolor='white',
                label=label_b)
        for i, v in enumerate(pct_a):
            ax.text(i - w/2, v + ymax * 0.012, f'{v:.1f}',
                     ha='center', va='bottom', fontsize=8.5,
                     color=c_a, fontweight='bold')
        for i, v in enumerate(pct_b):
            ax.text(i + w/2, v + ymax * 0.012, f'{v:.1f}',
                     ha='center', va='bottom', fontsize=8.5,
                     color=c_b, fontweight='bold')
        if vline_idx is not None:
            ax.axvline(vline_idx - 0.5, ls='--', color='#444', lw=1.2)
            ax.text(vline_idx - 0.5, ymax * 1.05,
                     r'$C_{16}$ step', fontsize=9, color='#444',
                     ha='center')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, fontsize=9.5)
        ax.set_ylim(0, ymax * 1.18)
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel('Percentage of queries (%)', fontsize=11)
        ax.set_title(title, fontsize=11.5, fontweight='bold')
        ax.legend(loc='upper right', frameon=True, framealpha=0.95)
        ax.grid(alpha=0.25, linestyle=':', axis='y')

    # vline_idx = index of first bin to the RIGHT of 22.5° (= idx 4)
    bar_pair(axes2[0], ip_labels, cp_ip_pct, cs_ip_pct,
              '(a) TRUE in-plane image rotation\n(what $C_{16}$ equivariance handles)',
              'In-plane image rotation ($^\\circ$)',
              palette['conpr'], palette['conslam'],
              f'ConPR  (n={len(cp_ip):,})',
              f'ConSLAM (n={len(cs_ip):,})',
              vline_idx=4)
    bar_pair(axes2[1], yaw_labels, cp_yaw_pct, cs_yaw_pct,
              '(b) $\\Delta$yaw / camera heading\n(previous figure metric)',
              r'$\Delta$yaw between query and matched-DB ($^\circ$)',
              palette['conpr'], palette['conslam'],
              f'ConPR  (n={len(cp_yaw):,})',
              f'ConSLAM (n={len(cs_yaw):,})')

    plt.tight_layout()
    out_pdf2 = OUT_DIR / 'inplane_bin_breakdown_conpr_vs_conslam.pdf'
    out_png2 = OUT_DIR / 'inplane_bin_breakdown_conpr_vs_conslam.png'
    fig2.savefig(out_pdf2, bbox_inches='tight')
    fig2.savefig(out_png2, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f'\nSaved bin chart:\n  {out_pdf2}\n  {out_png2}')

    # Print bin tables for the user to reference
    print('\nIn-plane bin %  (ConPR / ConSLAM):')
    for lbl, a, b in zip(ip_labels, cp_ip_pct, cs_ip_pct):
        print(f'  {lbl:>8}°   ConPR={a:5.2f}   ConSLAM={b:5.2f}')
    print('\nΔyaw bin %  (ConPR / ConSLAM):')
    for lbl, a, b in zip(yaw_labels, cp_yaw_pct, cs_yaw_pct):
        print(f'  {lbl:>6}°   ConPR={a:5.2f}   ConSLAM={b:5.2f}')


if __name__ == '__main__':
    main()
