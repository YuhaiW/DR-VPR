"""
Supplementary Table S2 — Per-epoch GroupPooling ablation (max vs. mean) on seed 190223.

Reads:
  - eval_seed190223_ep{EP}_{conpr,conslam}.log        (main method = max-pool, concat)
  - eval_concat_meanpool_s190223_ep{EP}_{conpr,conslam}.log   (mean-pool variant)

Checkpoint filenames encode val R@1 on GSV-Cities, used for val-best selection
highlighting.

Emits three artifacts:
  - doc/supp_table_s2.md    (readable review)
  - doc/supp_table_s2.tex   (drop-in LaTeX for the supplementary PDF)
  - doc/supp_table_s2.csv   (raw numbers)
"""
from __future__ import annotations

import re
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
SEED = 190223
EPOCHS = list(range(10))

CKPT_MAX_DIR = (
    PROJECT
    / "LOGS"
    / f"resnet50_DualBranch_seed{SEED}"
    / "lightning_logs"
    / "version_0"
    / "checkpoints"
)
CKPT_MEAN_DIR = (
    PROJECT
    / "LOGS"
    / f"resnet50_DualBranch_concat_meanpool_seed{SEED}"
    / "lightning_logs"
    / "version_0"
    / "checkpoints"
)

OUT_DIR = PROJECT / "doc"
OUT_DIR.mkdir(exist_ok=True)


RECALL_RE = re.compile(r"(?:平均|Average) Recall@1:\s*([0-9.]+)")
CKPT_VAL_RE = re.compile(r"epoch\((\d+)\)_R1\[([0-9.]+)\]")


def read_recall(path: Path) -> float | None:
    if not path.is_file():
        return None
    for line in path.read_text(errors="ignore").splitlines():
        m = RECALL_RE.search(line)
        if m:
            return float(m.group(1))
    return None


def read_val_r1_map(ckpt_dir: Path) -> dict[int, float]:
    mapping: dict[int, float] = {}
    if not ckpt_dir.is_dir():
        return mapping
    for ckpt in ckpt_dir.glob("*.ckpt"):
        m = CKPT_VAL_RE.search(ckpt.name)
        if m:
            mapping[int(m.group(1))] = float(m.group(2))
    return mapping


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.2f}"


def fmt_val(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:.2f}"


def build_rows():
    val_max = read_val_r1_map(CKPT_MAX_DIR)
    val_mean = read_val_r1_map(CKPT_MEAN_DIR)

    rows = []
    for ep in EPOCHS:
        max_val = val_max.get(ep)
        mean_val = val_mean.get(ep)
        max_cp = read_recall(PROJECT / f"eval_seed{SEED}_ep{ep:02d}_conpr.log")
        max_cs = read_recall(PROJECT / f"eval_seed{SEED}_ep{ep:02d}_conslam.log")
        mean_cp = read_recall(PROJECT / f"eval_concat_meanpool_s{SEED}_ep{ep:02d}_conpr.log")
        mean_cs = read_recall(PROJECT / f"eval_concat_meanpool_s{SEED}_ep{ep:02d}_conslam.log")
        rows.append({
            "epoch": ep,
            "max_val": max_val, "mean_val": mean_val,
            "max_conpr": max_cp, "max_conslam": max_cs,
            "mean_conpr": mean_cp, "mean_conslam": mean_cs,
        })
    return rows


def find_val_best_epoch(rows, key) -> int | None:
    best = None
    for r in rows:
        if r[key] is None:
            continue
        if best is None or r[key] > best[key]:
            best = r
    return None if best is None else best["epoch"]


def render_markdown(rows, max_best_ep, mean_best_ep) -> str:
    out = []
    out.append("# Supplementary Table S2")
    out.append("")
    out.append("Per-epoch GroupPooling ablation on seed 190223: max-pool (main method) vs. "
               "mean-pool. All R@1 values are percentages on the ConPR and ConSLAM test "
               "sets, evaluated from the per-epoch checkpoint with no test-set selection. "
               "Val R@1 is measured on our GSV-Cities validation split. "
               f"★ marks the val-best epoch used for reporting (max-pool: epoch {max_best_ep}; "
               f"mean-pool: epoch {mean_best_ep}).")
    out.append("")
    out.append("| Epoch | Val R@1 (max) | Val R@1 (mean) | ConPR (max) | ConPR (mean) | ΔConPR | ConSLAM (max) | ConSLAM (mean) | ΔConSLAM |")
    out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        ep_mark = r["epoch"]
        tag = []
        if ep_mark == max_best_ep:
            tag.append("max★")
        if ep_mark == mean_best_ep:
            tag.append("mean★")
        epoch_str = f"{ep_mark:02d}" + (f" ({'/'.join(tag)})" if tag else "")
        dcp = None if r["max_conpr"] is None or r["mean_conpr"] is None else (r["mean_conpr"] - r["max_conpr"]) * 100
        dcs = None if r["max_conslam"] is None or r["mean_conslam"] is None else (r["mean_conslam"] - r["max_conslam"]) * 100
        out.append(
            f"| {epoch_str} | {fmt_val(r['max_val'])} | {fmt_val(r['mean_val'])} "
            f"| {fmt_pct(r['max_conpr'])} | {fmt_pct(r['mean_conpr'])} | {'—' if dcp is None else f'{dcp:+.2f}'} "
            f"| {fmt_pct(r['max_conslam'])} | {fmt_pct(r['mean_conslam'])} | {'—' if dcs is None else f'{dcs:+.2f}'} |"
        )

    # Val-best summary line
    max_row = next(r for r in rows if r["epoch"] == max_best_ep)
    mean_row = next(r for r in rows if r["epoch"] == mean_best_ep)
    out.append("")
    out.append(f"**Val-best summary** (seed 190223):")
    out.append(f"- max-pool @ epoch {max_best_ep}: ConPR {fmt_pct(max_row['max_conpr'])}% / ConSLAM {fmt_pct(max_row['max_conslam'])}%")
    out.append(f"- mean-pool @ epoch {mean_best_ep}: ConPR {fmt_pct(mean_row['mean_conpr'])}% / ConSLAM {fmt_pct(mean_row['mean_conslam'])}%")
    # If val-best epochs match, we can report a single Δ; otherwise both
    if max_best_ep == mean_best_ep:
        d_cp = (mean_row['mean_conpr'] - max_row['max_conpr']) * 100
        d_cs = (mean_row['mean_conslam'] - max_row['max_conslam']) * 100
        out.append(f"- Δ(mean − max) at common val-best epoch {max_best_ep}: "
                   f"**ConPR {d_cp:+.2f}, ConSLAM {d_cs:+.2f}** R@1 (pp)")
    return "\n".join(out) + "\n"


def render_latex(rows, max_best_ep, mean_best_ep) -> str:
    # booktabs-style supplementary table
    lines = [
        "% Supplementary Table S2 — per-epoch max-pool vs mean-pool GroupPooling ablation (seed 190223).",
        "\\begin{table}[h]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\caption{Per-epoch GroupPooling ablation on seed 190223 (main-method seed): max-pool (the configuration used by the main architecture) vs.\\ mean-pool (orbit averaging). All R@1 values are percentages; Val R@1 is measured on our GSV-Cities validation split. $\\bigstar$ marks the per-configuration val-best epoch used for reporting.}",
        "\\label{tab:supp_s2_grouppool}",
        "\\begin{tabular}{c|rr|rrr|rrr}",
        "\\toprule",
        "Epoch & \\multicolumn{2}{c|}{Val R@1 (\\%)} & \\multicolumn{3}{c|}{ConPR R@1 (\\%)} & \\multicolumn{3}{c}{ConSLAM R@1 (\\%)} \\\\",
        " & max & mean & max & mean & $\\Delta$ & max & mean & $\\Delta$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        ep_mark = r["epoch"]
        marker = ""
        if ep_mark == max_best_ep and ep_mark == mean_best_ep:
            marker = "$\\bigstar\\bigstar$"
        elif ep_mark == max_best_ep:
            marker = "$\\bigstar_{\\text{max}}$"
        elif ep_mark == mean_best_ep:
            marker = "$\\bigstar_{\\text{mean}}$"
        ep_cell = f"{ep_mark:02d}\\,{marker}" if marker else f"{ep_mark:02d}"
        dcp = None if r["max_conpr"] is None or r["mean_conpr"] is None else (r["mean_conpr"] - r["max_conpr"]) * 100
        dcs = None if r["max_conslam"] is None or r["mean_conslam"] is None else (r["mean_conslam"] - r["max_conslam"]) * 100
        lines.append(
            f"{ep_cell} & {fmt_val(r['max_val'])} & {fmt_val(r['mean_val'])} "
            f"& {fmt_pct(r['max_conpr'])} & {fmt_pct(r['mean_conpr'])} & {'—' if dcp is None else f'{dcp:+.2f}'} "
            f"& {fmt_pct(r['max_conslam'])} & {fmt_pct(r['mean_conslam'])} & {'—' if dcs is None else f'{dcs:+.2f}'} \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def render_csv(rows) -> str:
    lines = ["epoch,val_max,val_mean,conpr_max,conpr_mean,delta_conpr_pp,conslam_max,conslam_mean,delta_conslam_pp"]
    for r in rows:
        dcp = None if r["max_conpr"] is None or r["mean_conpr"] is None else (r["mean_conpr"] - r["max_conpr"]) * 100
        dcs = None if r["max_conslam"] is None or r["mean_conslam"] is None else (r["mean_conslam"] - r["max_conslam"]) * 100
        def _v(x, fn=lambda v: f"{v * 100:.4f}"):
            return "" if x is None else fn(x)
        lines.append(
            f"{r['epoch']:02d},{_v(r['max_val'])},{_v(r['mean_val'])},"
            f"{_v(r['max_conpr'])},{_v(r['mean_conpr'])},{'' if dcp is None else f'{dcp:.4f}'},"
            f"{_v(r['max_conslam'])},{_v(r['mean_conslam'])},{'' if dcs is None else f'{dcs:.4f}'}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    rows = build_rows()
    max_best_ep = find_val_best_epoch(rows, "max_val")
    mean_best_ep = find_val_best_epoch(rows, "mean_val")

    md = render_markdown(rows, max_best_ep, mean_best_ep)
    tex = render_latex(rows, max_best_ep, mean_best_ep)
    csv = render_csv(rows)

    (OUT_DIR / "supp_table_s2.md").write_text(md)
    (OUT_DIR / "supp_table_s2.tex").write_text(tex)
    (OUT_DIR / "supp_table_s2.csv").write_text(csv)

    print(f"max-pool val-best:  epoch {max_best_ep}")
    print(f"mean-pool val-best: epoch {mean_best_ep}")
    print()
    print(md)
    print(f"Wrote: {OUT_DIR/'supp_table_s2.md'}")
    print(f"Wrote: {OUT_DIR/'supp_table_s2.tex'}")
    print(f"Wrote: {OUT_DIR/'supp_table_s2.csv'}")


if __name__ == "__main__":
    main()
