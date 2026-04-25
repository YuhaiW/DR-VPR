"""Read per_inplane_p1_{conslam,conpr}_records.json and produce
a ranked-by-gain table per dataset. Also compute the same for ConPR
across the FULL 10-seq protocol if those records exist.

Output: per_inplane_summary.md  +  stdout
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np


BUCKET_LABELS = ['[ 0°,  5°)', '[ 5°, 10°)', '[10°, 15°)',
                 '[15°, 20°)', '[20°, 25°)', '   25°+   ']
DATASETS = ['conslam', 'conpr']

OUT_MD = Path('per_inplane_summary.md')


def aggregate(records):
    """Pool across all seeds in records, compute per-bucket stats."""
    rows = []
    for lbl in BUCKET_LABELS:
        bucket = [r for r in records if r['bucket'] == lbl]
        n = len(bucket)
        if n == 0:
            rows.append({'bucket': lbl, 'n': 0, 'boq': None,
                          'drvpr': None, 'delta': None,
                          'flip_pos': 0, 'flip_neg': 0})
            continue
        boq_r1 = sum(r['boq_correct'] for r in bucket) / n * 100
        drvpr_r1 = sum(r['drvpr_correct'] for r in bucket) / n * 100
        flip_pos = sum(1 for r in bucket
                        if r['boq_correct'] == 0 and r['drvpr_correct'] == 1)
        flip_neg = sum(1 for r in bucket
                        if r['boq_correct'] == 1 and r['drvpr_correct'] == 0)
        rows.append({'bucket': lbl, 'n': n, 'boq': boq_r1,
                      'drvpr': drvpr_r1, 'delta': drvpr_r1 - boq_r1,
                      'flip_pos': flip_pos, 'flip_neg': flip_neg})
    total_n = sum(r['n'] for r in rows)
    total_boq = sum(r['n'] * r['boq'] / 100 for r in rows if r['boq'] is not None)
    total_drvpr = sum(r['n'] * r['drvpr'] / 100 for r in rows if r['drvpr'] is not None)
    total_row = {'bucket': 'TOTAL', 'n': total_n,
                 'boq': total_boq / total_n * 100 if total_n else None,
                 'drvpr': total_drvpr / total_n * 100 if total_n else None,
                 'delta': (total_drvpr - total_boq) / total_n * 100
                          if total_n else None,
                 'flip_pos': sum(r['flip_pos'] for r in rows),
                 'flip_neg': sum(r['flip_neg'] for r in rows)}
    return rows, total_row


def fmt_row(r):
    if r['boq'] is None:
        return (f"  {r['bucket']:>14s}  {r['n']:>6d}   "
                f"{'—':>8s}   {'—':>10s}   {'—':>7s}   "
                f"{'—':>6s}   {'—':>6s}")
    return (f"  {r['bucket']:>14s}  {r['n']:>6d}   "
            f"{r['boq']:7.2f}%   {r['drvpr']:9.2f}%   "
            f"{r['delta']:+6.2f}   {r['flip_pos']:>6d}   {r['flip_neg']:>6d}")


def fmt_md_row(r):
    if r['boq'] is None:
        return (f"| {r['bucket'].strip()} | {r['n']} | — | — | — | — | — |")
    return (f"| {r['bucket'].strip()} | {r['n']} | {r['boq']:.2f}% | "
            f"{r['drvpr']:.2f}% | **{r['delta']:+.2f}** | "
            f"{r['flip_pos']} | {r['flip_neg']} |")


def report_dataset(name, json_path, md_lines):
    if not json_path.exists():
        msg = f'\n[{name}] {json_path} missing — skipping.\n'
        print(msg)
        md_lines.append(msg)
        return

    with open(json_path) as f:
        records = json.load(f)
    rows, total = aggregate(records)

    seeds = sorted({r.get('seed', '?') for r in records})

    print(f"\n{'='*82}")
    print(f"{name.upper()}  ·  {json_path.name}  ·  3-seed pooled")
    print(f"  seeds = {seeds}   total queries (sum across seeds) = {total['n']}")
    print('='*82)

    print(f"\n{'Bucket':>14s}  {'N':>6s}  {'BoQ R@1':>9s}  "
          f"{'DR-VPR R@1':>11s}  {'Δ':>7s}  {'flip→✓':>7s}  {'flip→✗':>7s}")
    print('-'*82 + '\n--- Order: by bucket (low → high in-plane rotation) ---')
    for r in rows:
        print(fmt_row(r))
    print('-'*82)
    print(fmt_row(total))

    rows_with_data = [r for r in rows if r['delta'] is not None]
    sorted_by_gain = sorted(rows_with_data, key=lambda x: -x['delta'])
    print('\n--- Order: by ΔR@1 (largest gain first) ---')
    for r in sorted_by_gain:
        print(fmt_row(r))

    md_lines.append(f"\n## {name.upper()}\n")
    md_lines.append(f"_Pooled across 3 seeds; total queries (with multiplicity) = {total['n']}._\n")
    md_lines.append('### Buckets ordered by in-plane rotation\n')
    md_lines.append('| Bucket (in-plane °) | N | BoQ R@1 | DR-VPR R@1 | Δ R@1 | flip→✓ | flip→✗ |')
    md_lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for r in rows:
        md_lines.append(fmt_md_row(r))
    md_lines.append(fmt_md_row(total))

    md_lines.append('\n### Buckets ranked by ΔR@1 (largest gain first)\n')
    md_lines.append('| Rank | Bucket (in-plane °) | N | BoQ R@1 | DR-VPR R@1 | Δ R@1 |')
    md_lines.append('|---:|---|---:|---:|---:|---:|')
    for i, r in enumerate(sorted_by_gain, 1):
        md_lines.append(f"| {i} | {r['bucket'].strip()} | {r['n']} | "
                        f"{r['boq']:.2f}% | {r['drvpr']:.2f}% | "
                        f"**{r['delta']:+.2f}** |")


def main():
    md_lines = ['# Per-INPLANE-ROTATION bucket decomposition of R@1\n',
                'Buckets are 5° wide (with a single tail bucket "25°+").  ',
                'In-plane rotation = TRUE residual rotation about the optical axis '
                'between query and the nearest-yaw GT positive '
                '(robotics convention, X = optical axis).  ',
                'BoQ baseline = official BoQ(ResNet50)@320, β=0.  ',
                'DR-VPR = BoQ ⊕ C16 multi-scale equi rerank, β=0.10.\n']

    for ds in DATASETS:
        json_path = Path(f'per_inplane_p1_{ds}_records.json')
        report_dataset(ds, json_path, md_lines)

    OUT_MD.write_text('\n'.join(md_lines))
    print(f'\nSaved markdown report → {OUT_MD}')


if __name__ == '__main__':
    main()
