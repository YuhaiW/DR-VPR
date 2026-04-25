"""
Fig. 6 — Rotation feature response curve (DR-VPR v2: BoQ + P1 standalone equi).

For DR-VPR v2 main method:
  - Branch 1 = official BoQ(ResNet50)@320 (loaded from torch.hub)
  - Branch 2 = E2ResNet(C8) multi-scale standalone (P1 ckpt seed=1)

Sweeps a batch of ConSLAM Sequence4 query images through input rotations in
[0°, 360°] at 5° steps (73 angles). For each rotation θ, computes the cosine
similarity of each branch's descriptor against the same image's θ=0° reference.
Reports mean ± std across the sample of N=24 images, separately for Branch 1
and Branch 2.

Expected behavior:
  - Branch 2 (C8 equivariant + GroupPool max): near-1.0 with sharp peaks at
    multiples of 45° (the C8 group elements) and graceful approximate
    invariance in between (bilinear-interpolation-induced perturbation).
  - Branch 1 (BoQ appearance): monotonic degradation past ~30° rotation
    descending toward ≈ 0.13 near 180°; no architectural invariance.

Output: figures/fig6_rotation_response.{pdf,png} + raw .npz data.

Usage:
    mamba run -n drvpr python plot_rotation_response.py
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import rotate as tvf_rotate
from torchvision.transforms.functional import InterpolationMode

# ---------------------- config ------------------------------------------
PROJECT = Path(__file__).resolve().parent
P1_CKPT = (
    PROJECT / "LOGS" / "equi_standalone_seed1_ms_C16" / "lightning_logs"
    / "version_0" / "checkpoints"
    / "equi_ms_seed1_epoch(08)_R1[0.3510].ckpt"
)
IMG_SOURCES = [
    PROJECT / "datasets" / "ConSLAM" / "Sequence4" / "selected_images",
    PROJECT / "datasets" / "ConSLAM" / "Sequence5" / "selected_images",
]
N_IMAGES = 24
IMG_SIZE = 320
ANGLES_DEG = np.arange(0, 361, 5)  # 0, 5, 10, …, 360 (73 angles, includes 360 for visual closure)

OUT_DIR = PROJECT / "figures"
OUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda:0"


# ---------------------- model loading -----------------------------------
def load_official_boq():
    print("[fig6] loading BoQ(ResNet50) from torch.hub ...")
    model = torch.hub.load(
        "amaralibey/bag-of-queries", "get_trained_boq",
        backbone_name="resnet50", output_dim=16384,
    )
    return model.eval().to(DEVICE)


def load_p1_equi(ckpt_path):
    from models.equi_multiscale import E2ResNetMultiScale
    print(f"[fig6] loading P1 standalone equi: {ckpt_path}")
    model = E2ResNetMultiScale(
        orientation=16, layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512), out_dim=1024,
    )
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    state = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


# ---------------------- image loading -----------------------------------
def gather_image_paths():
    paths = []
    for root in IMG_SOURCES:
        if root.is_dir():
            paths.extend(sorted(root.glob("*.png")))
            paths.extend(sorted(root.glob("*.jpg")))
    if not paths:
        raise FileNotFoundError(f"No images found in {IMG_SOURCES}")
    return paths[:N_IMAGES]


def load_image_batch(paths):
    """Load grayscale → 3-channel → resize → stack [N, 3, H, W] in [0, 1]."""
    loader = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    tensors = []
    for p in paths:
        img = Image.open(p).convert("L")
        img_rgb = Image.merge("RGB", (img, img, img))
        tensors.append(loader(img_rgb))
    return torch.stack(tensors).to(DEVICE)   # (N, 3, H, W) in [0, 1]


def imagenet_normalize(x):
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


# ---------------------- forward -----------------------------------------
@torch.no_grad()
def extract_boq_batch(model, x_norm):
    out = model(x_norm)
    if isinstance(out, tuple):
        out = out[0]
    return F.normalize(out, p=2, dim=1)   # (N, 16384)


@torch.no_grad()
def extract_equi_batch(model, x_norm):
    return model(x_norm)   # (N, 1024), already L2-normalized in forward


# ---------------------- main --------------------------------------------
def main():
    if not P1_CKPT.is_file():
        sys.exit(f"P1 ckpt not found: {P1_CKPT}")
    paths = gather_image_paths()
    print(f"[fig6] using {len(paths)} ConSLAM images (first: {paths[0].name})")

    boq = load_official_boq()
    equi = load_p1_equi(P1_CKPT)

    # Load all images once (raw [0,1])
    base = load_image_batch(paths)
    print(f"[fig6] image batch shape: {tuple(base.shape)}")

    # Reference descriptors at θ=0
    ref_norm = imagenet_normalize(base)
    ref_boq = extract_boq_batch(boq, ref_norm)
    ref_equi = extract_equi_batch(equi, ref_norm)

    # Sweep rotations
    sims_boq = np.zeros((len(paths), len(ANGLES_DEG)))
    sims_equi = np.zeros((len(paths), len(ANGLES_DEG)))
    for k, ang in enumerate(ANGLES_DEG):
        if ang == 0 or ang == 360:
            x_rot = base
        else:
            x_rot = tvf_rotate(base, float(ang),
                                interpolation=InterpolationMode.BILINEAR, fill=0.0)
        x_norm = imagenet_normalize(x_rot)
        d_boq = extract_boq_batch(boq, x_norm)
        d_equi = extract_equi_batch(equi, x_norm)
        sims_boq[:, k]  = (d_boq  * ref_boq ).sum(dim=1).cpu().numpy()
        sims_equi[:, k] = (d_equi * ref_equi).sum(dim=1).cpu().numpy()
        if k % 12 == 0 or k == len(ANGLES_DEG) - 1:
            print(f"[fig6]  θ={ang:3d}°  BoQ={sims_boq[:, k].mean():+.4f}  "
                  f"Equi={sims_equi[:, k].mean():+.4f}")

    # ---------------------- plot ----------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    b2_mean, b2_std = sims_equi.mean(axis=0), sims_equi.std(axis=0)
    b1_mean, b1_std = sims_boq.mean(axis=0),  sims_boq.std(axis=0)

    # C8 group elements as faint vertical guides (behind curves)
    for theta in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
        ax.axvline(theta, color="#999999", alpha=0.25, linewidth=0.7,
                   linestyle="--", zorder=0)

    # Branch 2 (equivariant) — blue
    ax.plot(ANGLES_DEG, b2_mean, color="#1F77B4", linewidth=2.0,
            label="Branch 2 (E2ResNet C8 — equivariant)", zorder=3)
    ax.fill_between(ANGLES_DEG, b2_mean - b2_std, b2_mean + b2_std,
                    color="#1F77B4", alpha=0.18, zorder=2)

    # Branch 1 (BoQ appearance) — red
    ax.plot(ANGLES_DEG, b1_mean, color="#D62728", linewidth=2.0,
            label="Branch 1 (BoQ-ResNet50 — non-equivariant)", zorder=3)
    ax.fill_between(ANGLES_DEG, b1_mean - b1_std, b1_mean + b1_std,
                    color="#D62728", alpha=0.18, zorder=2)

    ax.set_xlim(0, 360)
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.set_xlabel(r"Input rotation angle $\theta$ (°)")
    ax.set_ylabel(r"Cosine similarity to $\theta = 0°$ descriptor")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.axhline(0, color="gray", linewidth=0.5, linestyle=":")
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower center", ncol=1, frameon=True, framealpha=0.9, fontsize=9.5)

    ax.text(180, 1.03,
            "Dashed verticals: C8 group elements (multiples of 45°)",
            ha="center", va="top", fontsize=8, color="#555555")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig6_rotation_response.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[fig6] wrote {out}")

    np.savez(OUT_DIR / "fig6_rotation_response_data.npz",
             angles_deg=ANGLES_DEG,
             branch1_sim_mean=b1_mean, branch1_sim_std=b1_std,
             branch2_sim_mean=b2_mean, branch2_sim_std=b2_std,
             n_images=len(paths),
             ckpt=str(P1_CKPT))
    print(f"[fig6] wrote {OUT_DIR / 'fig6_rotation_response_data.npz'}")

    print("\n[fig6] summary:")
    print(f"  Branch 2 (equi):   mean={b2_mean.mean():+.4f}  "
          f"min={b2_mean.min():+.4f}  range={b2_mean.max() - b2_mean.min():.4f}")
    print(f"  Branch 1 (BoQ):    mean={b1_mean.mean():+.4f}  "
          f"min={b1_mean.min():+.4f}  range={b1_mean.max() - b1_mean.min():.4f}")
    # C8 elements: pick exact angles
    c8_idx = [int(np.where(ANGLES_DEG == a)[0][0]) for a in [0, 45, 90, 135, 180, 225, 270, 315]]
    print(f"  Branch 2 @ C8 elements: {[f'{b2_mean[i]:.4f}' for i in c8_idx]}")
    print(f"  Branch 1 @ θ=180°: {b1_mean[ANGLES_DEG == 180][0]:.4f} (worst case)")


if __name__ == "__main__":
    main()
