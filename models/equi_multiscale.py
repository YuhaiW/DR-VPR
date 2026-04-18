"""
Standalone multi-scale equivariant network for VPR rerank.

Path P1 architecture:
  - E2ResNet C8 backbone (reuses EquivariantBasicBlock from e2resnet_backbone)
  - Separate GroupPool(max) at layer3 + layer4 (multi-scale invariant features)
  - Per-scale GeM with learnable p
  - concat (32 + 64 = 96 dim) → Linear(96 → 1024) → L2-norm
  - Output: 1024-d L2-normalized desc_equi

This module is standalone — no BoQ branch, no fusion, no gate. Used as the
stage-2 rerank descriptor in two-stage rerank protocol (eval_rerank_standalone.py).

Trained via train_equi_standalone.py with vanilla MS loss on GSV-Cities.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import e2cnn.nn as enn
from e2cnn import gspaces

from models.backbones.e2resnet_backbone import (
    EquivariantBasicBlock,
    regular_feature_type,
)


class E2ResNetMultiScale(nn.Module):
    """C8-equivariant ResNet with multi-scale invariant pooling for VPR.

    Args:
        orientation: rotation group order (default 8 for C8)
        layers: ResNet block counts per stage (default ResNet18-style)
        channels: total channel counts per stage (regular rep, divided by orientation
            internally to get field counts)
        out_dim: final descriptor dimension (default 1024)
        gem_p_init: initial GeM exponent (learnable per scale)

    Forward:
        x: (B, 3, H, W) RGB image tensor (mean/std normalized to ImageNet stats)
        Returns: (B, out_dim) L2-normalized descriptor
    """

    def __init__(
        self,
        orientation=8,
        layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512),
        out_dim=1024,
        gem_p_init=3.0,
    ):
        super().__init__()
        self.orientation = orientation
        self.gspace = gspaces.Rot2dOnR2(orientation)
        self.in_type = enn.FieldType(self.gspace, 3 * [self.gspace.trivial_repr])

        # Stem: trivial → regular field type
        out_type = regular_feature_type(self.gspace, channels[0])
        self.conv1 = enn.R2Conv(
            self.in_type, out_type, 7,
            stride=2, padding=3, bias=False,
            sigma=None, frequencies_cutoff=lambda r: 3 * r,
        )
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu = enn.ReLU(out_type, inplace=True)
        self.maxpool = enn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)

        # Residual stages (mirrors E2ResNetBackbone)
        self.layer1 = self._make_layer(channels[0], channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], layers[3], stride=2)

        # Multi-scale invariant mapping: separate GroupPool(max) per scale
        self.l3_type = regular_feature_type(self.gspace, channels[2])
        self.l4_type = regular_feature_type(self.gspace, channels[3])
        self.pool_l3 = enn.GroupPooling(self.l3_type)
        self.pool_l4 = enn.GroupPooling(self.l4_type)

        # Invariant channels per scale = (total channels) // orientation
        l3_inv_ch = channels[2] // orientation   # e.g. 256 // 8 = 32
        l4_inv_ch = channels[3] // orientation   # e.g. 512 // 8 = 64

        # Per-scale learnable GeM exponent
        self.gem_p_l3 = nn.Parameter(torch.tensor(float(gem_p_init)))
        self.gem_p_l4 = nn.Parameter(torch.tensor(float(gem_p_init)))

        # Concat → Linear projection
        self.proj = nn.Linear(l3_inv_ch + l4_inv_ch, out_dim)

        self.l3_inv_ch = l3_inv_ch
        self.l4_inv_ch = l4_inv_ch
        self.out_dim = out_dim

        print(f"✓ E2ResNetMultiScale: C{orientation}, layers={list(layers)}, "
              f"channels={list(channels)}, l3_inv={l3_inv_ch}+l4_inv={l4_inv_ch}={l3_inv_ch+l4_inv_ch} → {out_dim}")

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            in_type = regular_feature_type(self.gspace, in_channels)
            out_type = regular_feature_type(self.gspace, out_channels)
            downsample_layers = []
            if stride != 1:
                downsample_layers.append(
                    enn.PointwiseAvgPool(in_type, kernel_size=stride, stride=stride))
            downsample_layers.extend([
                enn.R2Conv(in_type, out_type, 1, stride=1, bias=False,
                          sigma=None, frequencies_cutoff=lambda r: 3 * r),
                enn.InnerBatchNorm(out_type),
            ])
            downsample = enn.SequentialModule(*downsample_layers)
        layers = [EquivariantBasicBlock(self.gspace, in_channels, out_channels,
                                          stride, downsample)]
        for _ in range(1, blocks):
            layers.append(EquivariantBasicBlock(self.gspace, out_channels, out_channels))
        return enn.SequentialModule(*layers)

    @staticmethod
    def _gem(x, p, eps=1e-6):
        """GeM pool over spatial dims. x: (B, C, H, W), p: scalar param. Returns (B, C)."""
        return F.adaptive_avg_pool2d(x.clamp(min=eps).pow(p), 1).pow(1.0 / p).flatten(1)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.in_type)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        # H/4, W/4
        x = self.layer1(x)          # H/4, W/4
        x = self.layer2(x)          # H/8, W/8
        x_l3 = self.layer3(x)       # H/16, W/16  (320×320 → 20×20)
        x_l4 = self.layer4(x_l3)    # H/32, W/32  (320×320 → 10×10)

        # Invariant pool at each scale (GroupPool max over orientation)
        inv_l3 = self.pool_l3(x_l3).tensor   # (B, l3_inv_ch=32, 20, 20)
        inv_l4 = self.pool_l4(x_l4).tensor   # (B, l4_inv_ch=64, 10, 10)

        # GeM per scale → vector
        d_l3 = self._gem(inv_l3, self.gem_p_l3)   # (B, 32)
        d_l4 = self._gem(inv_l4, self.gem_p_l4)   # (B, 64)

        # Concat + project + L2-norm
        d = torch.cat([d_l3, d_l4], dim=1)         # (B, 96)
        d = self.proj(d)                            # (B, 1024)
        d = F.normalize(d, p=2, dim=1)
        return d


# Quick test on CPU
if __name__ == '__main__':
    import sys
    print("Testing E2ResNetMultiScale...")
    model = E2ResNetMultiScale(orientation=8, layers=(1, 1, 1, 1),
                                 channels=(32, 64, 128, 256), out_dim=512).eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (2, 512), f"Expected (2, 512), got {tuple(out.shape)}"
    # Verify L2-normalized
    norms = out.norm(dim=1)
    print(f"Output norms: min={norms.min().item():.4f}, max={norms.max().item():.4f} (expected ~1)")
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Output not L2-normalized"

    # Quick rotation test (90° = C8 group element)
    with torch.no_grad():
        x_rot = torch.rot90(x, k=1, dims=[2, 3])
        out_rot = model(x_rot)
    diff = (out - out_rot).abs().max().item()
    print(f"90° rot max diff: {diff:.4e} (expected small, structurally invariant up to spatial perm)")
    print("✓ E2ResNetMultiScale test passed!")
    print(f"  total params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
