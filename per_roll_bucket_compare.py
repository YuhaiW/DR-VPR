"""
Rigorous per-roll-bucket comparison of DR-VPR vs BoQ.

Loads per_inplane_p1_{conslam,conpr}_records.json (produced by
per_inplane_analysis_p1.py), which contains per-query records keyed by
true in-plane (roll) rotation between q and its nearest-yaw GT positive
(robotics convention, X = optical axis).

For each 5° bucket this script reports:
  - N (pooled across 3 seeds)
  - BoQ R@1, DR-VPR R@1, Δ
  - flip→✓ / flip→✗ (paired outcomes)
  - Binomial two-tailed p-value on the flips (k = min(✓, ✗),
    n = ✓+✗, p0 = 0.5) — this tests H0: DR-VPR and BoQ equally likely
    to win a flip within this bucket
  - Per-seed Δ (consistency check across 3 seeds)

Also prints a side-by-side ConSLAM / ConPR table and exports markdown.
"""
from __future__ import annotations
import json
from math import comb
from pathlib import Path


BUCKETS = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25),
           (25, 30), (30, 45), (45, 90)]
SEEDS = [1, 42, 190223]

OUT_MD = Path('per_roll_bucket_compare.md')


def bucket_of(angle):
    for lo, hi in BUCKETS:
        if lo <= angle < hi:
            return (lo, hi)
    return BUCKETS[-1]


def binom_two_sided_p(k, n, p=0.5):
    """Exact two-sided binomial p-value. k = min(successes, failures)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
               for i in range(k + 1))
    return min(1.0, 2.0 * tail)


def aggregate(records, bucket):
    lo, hi = bucket
    sub = [r for r in records if lo <= r['inplane_abs'] < hi]
    n = len(sub)
    if n == 0:
        return {'n': 0}
    boq_c = sum(r['boq_correct'] for r in sub)
    drvpr_c = sum(r['drvpr_correct'] for r in sub)
    flip_pos = sum(1 for r in sub
                   if r['boq_correct'] == 0 and r['drvpr_correct'] == 1)
    flip_neg = sum(1 for r in sub
                   if r['boq_correct'] == 1 and r['drvpr_correct'] == 0)
    boq_r1 = 100 * boq_c / n
    drvpr_r1 = 100 * drvpr_c / n
    p = binom_two_sided_p(min(flip_pos, flip_neg), flip_pos + flip_neg)
    # Per-seed Δ
    per_seed = {}
    for s in SEEDS:
        seed_sub = [r for r in sub if r.get('seed') == s]
        if not seed_sub:
            continue
        b = sum(r['boq_correct'] for r in seed_sub) / len(seed_sub) * 100
        d = sum(r['drvpr_correct'] for r in seed_sub) / len(seed_sub) * 100
        per_seed[s] = d - b
    return {'n': n, 'boq_r1': boq_r1, 'drvpr_r1': drvpr_r1,
            'delta': drvpr_r1 - boq_r1,
            'flip_pos': flip_pos, 'flip_neg': flip_neg,
            'p_value': p, 'per_seed': per_seed}


def load(name):
    path = Path(f'per_inplane_p1_{name}_records.json')
    if not path.exists():
        raise SystemExit(f'{path} missing. Run per_inplane_analysis_p1.py.')
    with open(path) as f:
        return json.load(f)


def fmt_delta(d):
    return f'{d:+.2f}'


def fmt_p(p):
    if p < 0.001:
        return '<0.001 ***'
    if p < 0.01:
        return f'{p:.3f} **'
    if p < 0.05:
        return f'{p:.3f} *'
    if p < 0.10:
        return f'{p:.3f} .'
    return f'{p:.3f}'


def report_side_by_side(cs, cp):
    """Print ConSLAM vs ConPR side by side, per bucket."""
    print('=' * 110)
    print('PER-ROLL-BUCKET COMPARISON (5° gradient)')
    print('Roll = true in-plane image rotation between q and nearest-yaw GT '
          'positive (robotics conv., X = optical axis)')
    print('3-seed pooled. BoQ baseline = 61.24 on ConSLAM (torch.hub path, '
          'matches Table 1 after revision).')
    print('=' * 110)

    header = (f"{'Bucket':>11s}  "
              f"| {'N':>5s} {'BoQ':>7s} {'DR-VPR':>7s} {'Δ':>7s}"
              f" {'✓':>4s} {'✗':>4s} {'p':>10s} "
              f"| {'N':>5s} {'BoQ':>7s} {'DR-VPR':>7s} {'Δ':>7s}"
              f" {'✓':>4s} {'✗':>4s} {'p':>10s}")
    print(f'{"":>11s}  | {"":*^58s} | {"":*^58s}'.replace('*', '─'))
    print(f'{"":>11s}  | {"ConSLAM (pooled N=921)":^58s} '
          f'| {"ConPR (pooled N=7401)":^58s}')
    print(header)
    print('-' * 144)

    for lo, hi in BUCKETS:
        lbl = f'[{lo:>2d}°,{hi:>3d}°)'
        a = aggregate(cs, (lo, hi))
        b = aggregate(cp, (lo, hi))

        def row(a):
            if a['n'] == 0:
                return (f" {0:>5d} {'—':>7s} {'—':>7s} {'—':>7s}"
                        f" {'—':>4s} {'—':>4s} {'—':>10s} ")
            return (f" {a['n']:>5d} {a['boq_r1']:>6.2f}% "
                    f"{a['drvpr_r1']:>6.2f}% {fmt_delta(a['delta']):>7s}"
                    f" {a['flip_pos']:>4d} {a['flip_neg']:>4d} "
                    f"{fmt_p(a['p_value']):>10s}")

        print(f'{lbl:>11s}  |{row(a)}|{row(b)}')
    print('=' * 144)


def report_per_seed(records, name):
    print(f'\n--- {name.upper()} per-seed ΔR@1 (consistency check) ---')
    print(f"{'Bucket':>11s}  | {'s1':>6s}  {'s42':>6s}  {'s190223':>9s}  "
          f"| {'range':>7s}")
    print('-' * 56)
    for lo, hi in BUCKETS:
        a = aggregate(records, (lo, hi))
        if a['n'] == 0:
            continue
        seeds = a['per_seed']
        if len(seeds) < 2:
            continue
        s1 = seeds.get(1, float('nan'))
        s42 = seeds.get(42, float('nan'))
        s190 = seeds.get(190223, float('nan'))
        rng = max(seeds.values()) - min(seeds.values())
        print(f"[{lo:>2d}°,{hi:>3d}°)  | {s1:>+6.2f}  {s42:>+6.2f}  "
              f"{s190:>+9.2f}  | {rng:>7.2f}")


def interpretation(cs_agg, cp_agg):
    print('\n' + '=' * 110)
    print('INTERPRETATION')
    print('=' * 110)
    print("""\
Professor-level reading of the table:

1) CORRELATION BETWEEN ROLL AND GAIN
   If the C_16 equivariant branch's contribution came from its inductive
   bias against in-plane rotation, the per-bucket Δ should monotonically
   INCREASE with the roll bucket. It does not:
     - ConSLAM: 100% of the +1.41 total comes from the [5°,10°) bucket
       (+3.48). All buckets ≥15° show zero gain because BoQ gets 16.67% /
       0% there — these are essentially unsolvable under the current
       pipeline (texture-poor / out-of-distribution queries).
     - ConPR: gain is concentrated in the LOWEST bucket [0°,5°) (+0.84),
       and the model REGRESSES in [10°,15°) (−1.27) and [15°,20°) (−3.60)
       — the opposite of what equivariance would predict.

2) WHERE IS THE EFFECT SIZE REAL?
   The only bucket with a statistically significant paired outcome
   (binomial two-sided p<0.01) is ConSLAM's [5°,10°) (flips 15:1). On
   ConPR, [0°,5°) is the only bucket with p<0.05 (flips 71:32 against a
   much larger N — small effect in %, high stat-power).

3) WHAT ARE WE REALLY OBSERVING?
   The +1.41 R@1 on ConSLAM is statistically significant (paired flips
   are directionally consistent) but it does NOT track in-plane rotation
   magnitude. The C_16 branch is contributing COMPLEMENTARY features
   (different receptive field / filter bank from BoQ-trained ResNet50),
   not strict rotation equivariance that activates on the subset of
   large-roll queries.

4) IMPLICATIONS FOR THE PAPER
   The "C_16 equivariance handles in-plane rotation" framing in §5.4 is
   not supported by this data. A more honest framing: "C_16 group-
   equivariant features complement appearance features on a narrow range
   of queries with moderate in-plane rotation". The large gains in high-
   yaw buckets of Table 6 come from yaw ≠ roll — the dataset positives
   are filtered by yaw (camera heading), not by roll (image plane).
""")


def save_markdown(cs, cp):
    lines = ['# Per-roll-bucket R@1 comparison\n',
             f'Buckets: 5° wide up to 30°, then coarser tails. '
             f'Roll = true in-plane rotation between query and '
             f'nearest-yaw GT positive (robotics convention, X = optical axis). '
             f'Pooled across 3 seeds; `*` ≡ p<0.05, `**` ≡ p<0.01, `***` ≡ p<0.001, `.` ≡ p<0.10 '
             f'(two-sided binomial on paired flips).\n',
             '## ConSLAM (pooled N=921)\n',
             '| Bucket (roll °) | N | BoQ R@1 | DR-VPR R@1 | Δ R@1 | flip→✓ | flip→✗ | p |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for lo, hi in BUCKETS:
        a = aggregate(cs, (lo, hi))
        if a['n'] == 0:
            lines.append(f'| [{lo}, {hi}) | 0 | — | — | — | — | — | — |')
            continue
        lines.append(
            f"| [{lo}, {hi}) | {a['n']} | {a['boq_r1']:.2f}% | "
            f"{a['drvpr_r1']:.2f}% | **{a['delta']:+.2f}** | "
            f"{a['flip_pos']} | {a['flip_neg']} | {fmt_p(a['p_value'])} |")

    lines.append('\n## ConPR (pooled N=7401)\n')
    lines.append('| Bucket (roll °) | N | BoQ R@1 | DR-VPR R@1 | Δ R@1 | flip→✓ | flip→✗ | p |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
    for lo, hi in BUCKETS:
        a = aggregate(cp, (lo, hi))
        if a['n'] == 0:
            lines.append(f'| [{lo}, {hi}) | 0 | — | — | — | — | — | — |')
            continue
        lines.append(
            f"| [{lo}, {hi}) | {a['n']} | {a['boq_r1']:.2f}% | "
            f"{a['drvpr_r1']:.2f}% | **{a['delta']:+.2f}** | "
            f"{a['flip_pos']} | {a['flip_neg']} | {fmt_p(a['p_value'])} |")

    OUT_MD.write_text('\n'.join(lines))
    print(f'\nSaved markdown → {OUT_MD}')


def main():
    cs = load('conslam')
    cp = load('conpr')
    report_side_by_side(cs, cp)
    report_per_seed(cs, 'ConSLAM')
    report_per_seed(cp, 'ConPR')
    interpretation(cs, cp)
    save_markdown(cs, cp)


if __name__ == '__main__':
    main()
