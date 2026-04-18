"""
Aggregate per-epoch ConPR + ConSLAM evaluation results across 3 seeds.

Reads eval_seed{S}_ep{E}_{conpr,conslam}.log files, builds a full table,
finds the best epoch per seed (by ConPR R@1), and reports mean ± std.
"""
import glob
import os
import re
from collections import defaultdict

import numpy as np

ROOT = "/home/yuhai/project/DR-VPR"
SEEDS = [1, 42, 190223]


def parse_conpr(path):
    """Return dict R1/R5/R10 in [0,1]."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        t = f.read()
    m1 = re.search(r"平均 Recall@1:\s+([\d.]+)", t)
    m5 = re.search(r"平均 Recall@5:\s+([\d.]+)", t)
    m10 = re.search(r"平均 Recall@10:\s+([\d.]+)", t)
    if not m1:
        return None
    out = {"R1": float(m1.group(1))}
    if m5:
        out["R5"] = float(m5.group(1))
    if m10:
        out["R10"] = float(m10.group(1))
    return out


def parse_conslam(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        t = f.read()
    m1 = re.search(r"Average Recall@1:\s+([\d.]+)", t)
    m5 = re.search(r"Average Recall@5:\s+([\d.]+)", t)
    m10 = re.search(r"Average Recall@10:\s+([\d.]+)", t)
    if not m1:
        return None
    out = {"R1": float(m1.group(1))}
    if m5:
        out["R5"] = float(m5.group(1))
    if m10:
        out["R10"] = float(m10.group(1))
    return out


# (seed, epoch) → {conpr: ..., conslam: ...}
results = defaultdict(dict)
for seed in SEEDS:
    files = glob.glob(os.path.join(ROOT, f"eval_seed{seed}_ep*_conpr.log"))
    for f in files:
        m = re.search(r"ep(\d+)_conpr", f)
        if not m:
            continue
        ep = int(m.group(1))
        cp = parse_conpr(f)
        cs = parse_conslam(f.replace("_conpr.log", "_conslam.log"))
        if cp:
            results[(seed, ep)]["conpr"] = cp
        if cs:
            results[(seed, ep)]["conslam"] = cs

# Print full table
print("=" * 84)
print(f"{'Seed':>8}  {'Ep':>3}  | {'CP-R@1':>7} {'CP-R@5':>7} {'CP-R@10':>7}  "
      f"| {'CS-R@1':>7} {'CS-R@5':>7} {'CS-R@10':>7}")
print("-" * 84)
for (seed, ep) in sorted(results.keys()):
    r = results[(seed, ep)]
    cp = r.get("conpr", {})
    cs = r.get("conslam", {})
    print(f"{seed:>8}  {ep:>3}  | "
          f"{100*cp.get('R1', 0):>6.2f}% {100*cp.get('R5', 0):>6.2f}% {100*cp.get('R10', 0):>6.2f}%  "
          f"| {100*cs.get('R1', 0):>6.2f}% {100*cs.get('R5', 0):>6.2f}% {100*cs.get('R10', 0):>6.2f}%")

# Per-seed best (by ConPR R@1)
print()
print("=" * 84)
print("Per-seed best (selected by ConPR R@1)")
print("=" * 84)
print(f"{'Seed':>8}  {'Best Ep':>7}  | {'CP-R@1':>7} {'CP-R@5':>7} {'CP-R@10':>7}  "
      f"| {'CS-R@1':>7} {'CS-R@5':>7} {'CS-R@10':>7}")
print("-" * 84)
best_per_seed = {}
for seed in SEEDS:
    rows = [(ep, results[(seed, ep)]) for ep in range(20)
            if (seed, ep) in results and "conpr" in results[(seed, ep)]]
    if not rows:
        continue
    rows.sort(key=lambda x: -x[1]["conpr"]["R1"])
    ep, r = rows[0]
    cp, cs = r.get("conpr", {}), r.get("conslam", {})
    print(f"{seed:>8}  {ep:>7}  | "
          f"{100*cp.get('R1', 0):>6.2f}% {100*cp.get('R5', 0):>6.2f}% {100*cp.get('R10', 0):>6.2f}%  "
          f"| {100*cs.get('R1', 0):>6.2f}% {100*cs.get('R5', 0):>6.2f}% {100*cs.get('R10', 0):>6.2f}%")
    best_per_seed[seed] = r

# Aggregate
if len(best_per_seed) >= 2:
    print()
    print("=" * 84)
    print(f"Mean ± std across {len(best_per_seed)} seeds")
    print("=" * 84)

    def collect(key_ds, key_r):
        vals = []
        for s, r in best_per_seed.items():
            if key_ds in r and key_r in r[key_ds]:
                vals.append(r[key_ds][key_r])
        return np.array(vals) * 100

    for ds in ("conpr", "conslam"):
        parts = []
        for k in ("R1", "R5", "R10"):
            vals = collect(ds, k)
            if len(vals):
                parts.append(f"{k}={vals.mean():.2f}±{vals.std(ddof=1):.2f}%")
        print(f"  {ds.upper():>7}:  " + "  ".join(parts))
