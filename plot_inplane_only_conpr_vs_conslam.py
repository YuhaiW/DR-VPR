"""Single-panel: in-plane image rotation distribution, ConPR vs ConSLAM.

Bars = % of queries per bin; overlaid lines = cumulative %.
Loads cached arrays from figures/inplane_yaw_cache.npz (run
plot_inplane_distribution_conpr_vs_conslam.py first if missing).

Output: figures/inplane_only_conpr_vs_conslam.{pdf,png}
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


OUT_DIR = Path('figures')
CACHE = OUT_DIR / 'inplane_yaw_cache.npz'

# Bins: dense up to C_16 step, coarser after
EDGES = [0, 5, 10, 15, 22.5, 30, 45, 90]
LABELS = ['0–5', '5–10', '10–15', '15–22.5', '22.5–30',
          '30–45', '45+']
PALETTE = {'conpr': '#2E5395', 'conslam': '#C25A2E'}


def pct_per_bin(arr, edges):
    counts, _ = np.histogram(arr, bins=edges)
    return 100.0 * counts / max(len(arr), 1)


def main():
    if not CACHE.exists():
        raise SystemExit(
            f'Cache {CACHE} not found. Run '
            'plot_inplane_distribution_conpr_vs_conslam.py first.')
    z = np.load(CACHE)
    cs_ip, cp_ip = z['cs_ip'], z['cp_ip']
    print(f'ConSLAM n={len(cs_ip)}   ConPR n={len(cp_ip):,}')

    cp_pct = pct_per_bin(cp_ip, EDGES)
    cs_pct = pct_per_bin(cs_ip, EDGES)
    cp_cum = np.cumsum(cp_pct)
    cs_cum = np.cumsum(cs_pct)

    plt.rcParams.update({
        'font.family': 'serif', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'mathtext.fontset': 'stix',
    })
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=200)

    x = np.arange(len(LABELS))
    w = 0.4
    ymax_bar = max(cp_pct.max(), cs_pct.max())

    ax.bar(x - w/2, cp_pct, w, color=PALETTE['conpr'],
           edgecolor='white',
           label=f'ConPR  (n={len(cp_ip):,})')
    ax.bar(x + w/2, cs_pct, w, color=PALETTE['conslam'],
           edgecolor='white',
           label=f'ConSLAM (n={len(cs_ip):,})')

    for i, v in enumerate(cp_pct):
        ax.text(i - w/2, v + ymax_bar * 0.012, f'{v:.1f}',
                ha='center', va='bottom', fontsize=9,
                color=PALETTE['conpr'], fontweight='bold')
    for i, v in enumerate(cs_pct):
        ax.text(i + w/2, v + ymax_bar * 0.012, f'{v:.1f}',
                ha='center', va='bottom', fontsize=9,
                color=PALETTE['conslam'], fontweight='bold')

    # C_16 step boundary (22.5° = between bin idx 3 and 4)
    ax.axvline(3.5, ls='--', color='#444', lw=1.3)
    ax.text(3.5, ymax_bar * 1.10,
            r'$C_{16}$ step (22.5$^\circ$)',
            fontsize=10, color='#444', ha='center',
            fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_xlabel(r'In-plane image rotation between query and matched-DB ($^\circ$)',
                  fontsize=11.5)
    ax.set_ylabel('Percentage of queries (%)', fontsize=11.5)
    ax.set_ylim(0, ymax_bar * 1.22)
    ax.grid(alpha=0.25, linestyle=':', axis='y')

    # Right-axis: cumulative %
    ax2 = ax.twinx()
    ax2.plot(x, cp_cum, '-o', color=PALETTE['conpr'],
             markersize=5, lw=1.6, alpha=0.85,
             label='ConPR cum.')
    ax2.plot(x, cs_cum, '-s', color=PALETTE['conslam'],
             markersize=5, lw=1.6, alpha=0.85,
             label='ConSLAM cum.')
    for i in (3, 4):  # annotate at C_16 boundary and just after
        ax2.text(i + 0.18, cp_cum[i] - 3.5, f'{cp_cum[i]:.1f}',
                 fontsize=8.5, color=PALETTE['conpr'])
        ax2.text(i + 0.18, cs_cum[i] + 1.0, f'{cs_cum[i]:.1f}',
                 fontsize=8.5, color=PALETTE['conslam'])
    ax2.set_ylabel('Cumulative % (≤ bin upper edge)',
                   fontsize=11, color='#555')
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis='y', colors='#555')
    ax2.grid(False)

    # Combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right',
              frameon=True, framealpha=0.95, fontsize=9.5)

    ax.set_title('In-plane image rotation: ConPR vs. ConSLAM',
                 fontsize=12.5, fontweight='bold', pad=10)

    plt.tight_layout()
    out_pdf = OUT_DIR / 'inplane_only_conpr_vs_conslam.pdf'
    out_png = OUT_DIR / 'inplane_only_conpr_vs_conslam.png'
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'\nSaved:\n  {out_pdf}\n  {out_png}')

    # Print summary lines for the user
    print('\nIn-plane bin %  (ConPR / ConSLAM)  + cumulative')
    for lbl, a, b, ca, cb in zip(LABELS, cp_pct, cs_pct, cp_cum, cs_cum):
        print(f'  {lbl:>8}°   ConPR={a:5.2f} ({ca:5.1f}%)   '
              f'ConSLAM={b:5.2f} ({cb:5.1f}%)')


if __name__ == '__main__':
    main()
