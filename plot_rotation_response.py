"""
Fig. 6 — Rotation feature response curve.

For the main-method checkpoint (Equi-BoQ + gated concat, seed 190223 val-best = ep00),
we sweep a test image through input rotations in [0°, 360°] in 5° steps and measure
the cosine similarity of each branch's descriptor against its 0° reference.

Expected behavior:
  - Branch 2 (C8 equivariant, GroupPooling=max): near-1.0 at multiples of 45°
    (C8 group elements) and smooth approximate invariance in between.
  - Branch 1 (BoQ appearance): monotonic degradation past ~30° rotation; no
    architectural invariance.

Saves to: figures/fig6_rotation_response.{pdf,png}

Run with:
    GROUP_POOL_MODE=max FUSION_METHOD=concat \
        mamba run -n drvpr python plot_rotation_response.py
"""
from __future__ import annotations

import glob
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

from train_fusion import VPRModel, load_boq_pretrained

# ---------------------- config ------------------------------------------
PROJECT = Path(__file__).resolve().parent
CKPT = (
    PROJECT
    / "LOGS"
    / "resnet50_DualBranch_seed190223"
    / "lightning_logs"
    / "version_0"
    / "checkpoints"
    / "resnet50_DualBranch_C8_seed190223_epoch(00)_R1[0.6506].ckpt"
)
# Using ConSLAM Sequence4 imagery: rotation-heavy construction domain, thematically
# aligned with the paper's main gain. Fallback to GSV-Cities if ConSLAM is absent.
IMG_SOURCES = [
    PROJECT / "datasets" / "ConSLAM" / "Sequence4" / "selected_images",
    PROJECT / "datasets" / "ConSLAM" / "Sequence5" / "selected_images",
]
GSV_FALLBACK = PROJECT / "datasets" / "GSV-Cities" / "Images" / "Boston"
N_IMAGES = 24
IMG_SIZE = 320
ANGLES_DEG = np.arange(0, 361, 5)  # 0, 5, 10, ..., 360

OUT_DIR = PROJECT / "figures"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------- helpers -----------------------------------------
def build_main_method_model() -> VPRModel:
    m = VPRModel(
        backbone_arch="resnet50",
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4],
        agg_arch="boq",
        agg_config={
            "in_channels": 1024,
            "proj_channels": 512,
            "num_queries": 64,
            "num_layers": 2,
            "row_dim": 32,
        },
        use_dual_branch=True,
        equi_orientation=8,
        equi_layers=[2, 2, 2, 2],
        equi_channels=[64, 128, 256, 512],
        equi_out_dim=1024,
        fusion_method="concat",
        use_projection=False,
        lr=1e-3,
        optimizer="adamw",
        weight_decay=1e-4,
        momentum=0.9,
        warmpup_steps=300,
        milestones=[8, 14],
        lr_mult=0.3,
        loss_name="MultiSimilarityLoss",
        miner_name="MultiSimilarityMiner",
        miner_margin=0.1,
        faiss_gpu=False,
    )
    load_boq_pretrained(m)
    state = torch.load(CKPT, map_location="cuda")["state_dict"]
    m.load_state_dict(state, strict=False)
    return m


def load_image_tensor(paths: list[Path]) -> torch.Tensor:
    """Load and stack a batch of images resized to IMG_SIZE × IMG_SIZE (no normalization yet)."""
    loader = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(loader(img))
    return torch.stack(tensors)  # (N, 3, H, W) in [0, 1]


def imagenet_normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def gather_image_paths() -> list[Path]:
    paths: list[Path] = []
    for root in IMG_SOURCES:
        if root.is_dir():
            paths.extend(sorted(root.glob("*.png")))
            paths.extend(sorted(root.glob("*.jpg")))
    if len(paths) < N_IMAGES and GSV_FALLBACK.is_dir():
        paths.extend(sorted(GSV_FALLBACK.glob("*.jpg")))
    if not paths:
        raise FileNotFoundError("No images found for rotation response plot.")
    return paths[:N_IMAGES]


@torch.no_grad()
def extract_branch_descriptors(model: VPRModel, imgs_norm: torch.Tensor):
    """imgs_norm: (N, 3, H, W) pre-normalized. Returns (desc1, desc2) per image."""
    feat1 = model.backbone(imgs_norm)
    feat2 = model.backbone2(imgs_norm)
    desc1 = model.aggregator.branch1_aggregator(feat1)  # (N, 16384)
    desc2 = model.aggregator.branch2_aggregator(feat2)  # (N, 1024)
    desc1 = F.normalize(desc1, p=2, dim=1)
    desc2 = F.normalize(desc2, p=2, dim=1)
    return desc1, desc2


# ---------------------- main --------------------------------------------
def main() -> None:
    if not CKPT.is_file():
        sys.exit(f"Checkpoint not found: {CKPT}")
    os.environ.setdefault("FUSION_METHOD", "concat")
    os.environ.setdefault("GROUP_POOL_MODE", "max")

    paths = gather_image_paths()
    print(f"[fig6] using {len(paths)} images (first: {paths[0]})")

    model = build_main_method_model().cuda().eval()

    # Load and resize all images on CPU, stack, send to GPU
    base = load_image_tensor(paths).cuda()  # (N, 3, H, W) in [0, 1]

    # Reference descriptors at 0°
    ref_norm = imagenet_normalize(base)
    ref1, ref2 = extract_branch_descriptors(model, ref_norm)

    # Sweep rotations
    sims1 = np.zeros((len(paths), len(ANGLES_DEG)))
    sims2 = np.zeros((len(paths), len(ANGLES_DEG)))
    for k, angle in enumerate(ANGLES_DEG):
        # Rotate raw [0,1] image. Bilinear interp for non-multiples of 90°.
        rotated = tvf_rotate(base, float(angle), interpolation=InterpolationMode.BILINEAR, fill=0.0)
        rot_norm = imagenet_normalize(rotated)
        d1, d2 = extract_branch_descriptors(model, rot_norm)
        sims1[:, k] = (d1 * ref1).sum(dim=1).cpu().numpy()
        sims2[:, k] = (d2 * ref2).sum(dim=1).cpu().numpy()
        print(f"[fig6] angle={angle:3d}°  b1_mean={sims1[:, k].mean():.4f}  b2_mean={sims2[:, k].mean():.4f}")

    # ---------------------- plot ----------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    b2_mean = sims2.mean(axis=0)
    b2_std = sims2.std(axis=0)
    b1_mean = sims1.mean(axis=0)
    b1_std = sims1.std(axis=0)

    # C8 group-element vertical markers (behind curves)
    for theta in [0, 45, 90, 135, 180, 225, 270, 315, 360]:
        ax.axvline(theta, color="#999999", alpha=0.25, linewidth=0.8, linestyle="--", zorder=0)

    # Branch 2 (equivariant)
    ax.plot(ANGLES_DEG, b2_mean,
            color="#1F77B4", linewidth=2.0,
            label="Branch 2 (C8 equivariant)", zorder=3)
    ax.fill_between(ANGLES_DEG, b2_mean - b2_std, b2_mean + b2_std,
                    color="#1F77B4", alpha=0.18, zorder=2)

    # Branch 1 (appearance / BoQ)
    ax.plot(ANGLES_DEG, b1_mean,
            color="#D62728", linewidth=2.0,
            label="Branch 1 (BoQ appearance)", zorder=3)
    ax.fill_between(ANGLES_DEG, b1_mean - b1_std, b1_mean + b1_std,
                    color="#D62728", alpha=0.18, zorder=2)

    ax.set_xlim(0, 360)
    ax.set_xticks([0, 45, 90, 135, 180, 225, 270, 315, 360])
    ax.set_xlabel(r"Input rotation angle $\theta$ (°)")
    ax.set_ylabel(r"Cosine similarity with $\theta=0°$ descriptor")
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower center", ncol=2, frameon=False, fontsize=10)

    # Annotate C8 tick region
    ax.text(180, 1.035, "Dashed lines: C8 group elements (multiples of 45°)",
            ha="center", va="top", fontsize=8, color="#555555")

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig6_rotation_response.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"[fig6] wrote {out}")

    # Also dump the raw numbers for reproducibility / supplementary inclusion
    np.savez(
        OUT_DIR / "fig6_rotation_response_data.npz",
        angles_deg=ANGLES_DEG,
        branch1_sim_mean=b1_mean, branch1_sim_std=b1_std,
        branch2_sim_mean=b2_mean, branch2_sim_std=b2_std,
        n_images=len(paths),
    )
    print(f"[fig6] wrote {OUT_DIR / 'fig6_rotation_response_data.npz'}")


if __name__ == "__main__":
    main()
