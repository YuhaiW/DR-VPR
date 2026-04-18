"""
Aggregate β sweep results across ckpts.

Usage:
    python aggregate_rerank_sweep.py <dir>
        dir contains: rerank_s{seed}_ep{epoch}.log OR rerank_ep{epoch}.log

Parses each log's SUMMARY block to extract β → R@1 mapping, then:
  - per-ckpt: finds best β (and its R@1, R@5, R@10)
  - per-seed: finds best epoch
  - across seeds: computes mean ± std of best combinations
"""
import sys
import re
import os
from glob import glob
from collections import defaultdict

import numpy as np


def parse_log(path):
    """Return dict {β: (R@1, R@5, R@10)} from SUMMARY block."""
    with open(path) as f:
        text = f.read()
    # Find SUMMARY block
    m = re.search(r'SUMMARY.*?={10,}\n(.*?)(?:\n\n|\nBest β|\Z)', text, re.DOTALL)
    if not m:
        return {}
    block = m.group(1)
    results = {}
    for line in block.splitlines():
        mm = re.match(
            r'\s*β=([\d.]+)\s+R@1=([\d.]+)%\s+R@5=([\d.]+)%\s+R@10=([\d.]+)%', line)
        if mm:
            beta = float(mm.group(1))
            r1, r5, r10 = map(float, (mm.group(2), mm.group(3), mm.group(4)))
            results[beta] = (r1, r5, r10)
    return results


def parse_filename(path):
    """Extract seed and epoch from filename. Returns (seed, epoch) or (None, epoch)."""
    name = os.path.basename(path)
    m_seed = re.search(r's(\d+)_ep(\d+)', name)
    if m_seed:
        return int(m_seed.group(1)), int(m_seed.group(2))
    m_ep = re.search(r'ep(\d+)', name)
    if m_ep:
        return None, int(m_ep.group(1))
    return None, None


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    directory = sys.argv[1]
    logs = sorted(glob(os.path.join(directory, 'rerank_*.log')))
    if not logs:
        print(f"No rerank_*.log files in {directory}"); sys.exit(1)

    # Parse all
    per_ckpt = {}   # key: (seed, epoch), value: {β: (R1, R5, R10)}
    for path in logs:
        seed, epoch = parse_filename(path)
        data = parse_log(path)
        if not data:
            print(f"[warn] could not parse {path}")
            continue
        per_ckpt[(seed, epoch)] = data

    if not per_ckpt:
        print("No parseable data."); sys.exit(1)

    # Group by seed
    by_seed = defaultdict(dict)   # seed → {epoch: {β: (R1, R5, R10)}}
    for (seed, ep), data in per_ckpt.items():
        by_seed[seed][ep] = data

    # ---------- Per-ckpt β sweep best ----------
    print("=" * 80)
    print("Per-checkpoint best β")
    print("=" * 80)
    print(f"{'seed':>7s}  {'epoch':>5s}  {'β=0 R1':>7s}  {'best β':>7s}  {'best R1':>8s}  "
          f"{'Δ':>6s}  {'R5@best':>8s}  {'R10@best':>8s}")
    print("-" * 80)

    best_per_seed = {}   # seed → (epoch, best_β, R1, R5, R10, Δ)
    for seed in sorted(by_seed):
        for epoch in sorted(by_seed[seed]):
            data = by_seed[seed][epoch]
            r0 = data.get(0.0, (0,))[0]
            best_beta = max(data, key=lambda b: data[b][0])
            best_r1, best_r5, best_r10 = data[best_beta]
            delta = best_r1 - r0
            seed_str = f"s{seed}" if seed is not None else "—"
            print(f"{seed_str:>7s}  {epoch:>5d}  {r0:7.2f}  {best_beta:7.1f}  "
                  f"{best_r1:8.2f}  {delta:+6.2f}  {best_r5:8.2f}  {best_r10:8.2f}")
            prev = best_per_seed.get(seed)
            if prev is None or best_r1 > prev[2]:
                best_per_seed[seed] = (epoch, best_beta, best_r1, best_r5, best_r10, delta)

    # ---------- Best-epoch-per-seed summary ----------
    print()
    print("=" * 80)
    print("Best epoch per seed (max R@1 over 11 β values)")
    print("=" * 80)
    print(f"{'seed':>7s}  {'epoch':>5s}  {'β':>4s}  {'R@1':>7s}  "
          f"{'R@5':>7s}  {'R@10':>7s}  {'Δvs β=0':>8s}")
    print("-" * 80)
    seeds = sorted(best_per_seed)
    r1s, r5s, r10s, deltas = [], [], [], []
    for seed in seeds:
        ep, beta, r1, r5, r10, delta = best_per_seed[seed]
        seed_str = f"s{seed}" if seed is not None else "—"
        print(f"{seed_str:>7s}  {ep:>5d}  {beta:>4.1f}  {r1:>7.2f}  "
              f"{r5:>7.2f}  {r10:>7.2f}  {delta:>+8.2f}")
        r1s.append(r1); r5s.append(r5); r10s.append(r10); deltas.append(delta)

    if len(r1s) >= 2:
        print()
        print("=" * 80)
        print(f"Aggregated over {len(r1s)} seeds (mean ± std)")
        print("=" * 80)
        # sample std (ddof=1)
        print(f"  R@1  = {np.mean(r1s):.2f} ± {np.std(r1s, ddof=1):.2f}")
        print(f"  R@5  = {np.mean(r5s):.2f} ± {np.std(r5s, ddof=1):.2f}")
        print(f"  R@10 = {np.mean(r10s):.2f} ± {np.std(r10s, ddof=1):.2f}")
        print(f"  Rerank gain vs β=0 = {np.mean(deltas):+.2f} ± {np.std(deltas, ddof=1):.2f}")


if __name__ == '__main__':
    main()
