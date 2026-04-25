"""
Simplified E2ResNet backbone for dual-branch VPR
Based on e2cnn library for rotation equivariance
"""
import torch
import torch.nn as nn
import e2cnn.nn as enn
from e2cnn import gspaces
import math


def regular_feature_type(gspace, planes, fixparams=False):
    """Build regular feature map with specified channels"""
    N = gspace.fibergroup.order()
    if fixparams:
        planes *= math.sqrt(N)
    planes = int(planes / N)
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)


def trivial_feature_type(gspace, planes):
    """Build trivial feature map (rotation-insensitive)"""
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)


class EquivariantBasicBlock(enn.EquivariantModule):
    """Rotation-equivariant Basic Block"""
    def __init__(self, gspace, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.in_type = regular_feature_type(gspace, in_channels)
        self.out_type = regular_feature_type(gspace, out_channels)
        
        # 3x3 conv
        self.conv1 = enn.R2Conv(
            self.in_type, self.out_type, 3,
            stride=stride, padding=1, bias=False,
            sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn1 = enn.InnerBatchNorm(self.out_type)
        self.relu = enn.ReLU(self.out_type, inplace=True)
        
        # 3x3 conv
        self.conv2 = enn.R2Conv(
            self.out_type, self.out_type, 3,
            stride=1, padding=1, bias=False,
            sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn2 = enn.InnerBatchNorm(self.out_type)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out
    
    def evaluate_output_shape(self, input_shape):
        """Required method for EquivariantModule"""
        assert len(input_shape) == 4
        assert input_shape[1] == self.in_type.size
        
        if self.downsample is not None:
            return self.downsample.evaluate_output_shape(input_shape)
        else:
            # Output shape considering stride
            b, c, h, w = input_shape
            out_h = (h - 1) // self.stride + 1
            out_w = (w - 1) // self.stride + 1
            return (b, self.out_type.size, out_h, out_w)


class E2ResNetBackbone(nn.Module):
    """
    Rotation-Equivariant ResNet for VPR (second branch)
    Simpler than full E2ResNet, optimized for feature extraction
    """
    def __init__(self,
                 orientation=8,      # C8 rotation group
                 layers=[2, 2, 2, 2], # ResNet18-like structure
                 channels=[64, 128, 256, 512],
                 pretrained=False,   # Note: equivariant models usually need custom pretraining
                 group_pool_mode='max',  # 'max' (default, enn.GroupPooling), 'mean' (orbit-average ablation), 'norm' (B2: L2 over orbit, preserves total orientation energy), 'fourier' (Tier-2: preserves |F_0|..|F_{N/2}| per field, 5x invariant capacity for C8)
                 ):
        super().__init__()

        self.orientation = orientation
        self.group_pool_mode = group_pool_mode
        self.gspace = gspaces.Rot2dOnR2(orientation)  # C8 or C16 group
        
        # Input: RGB image (trivial representation)
        self.in_type = enn.FieldType(
            self.gspace, 3 * [self.gspace.trivial_repr]
        )
        
        # Initial conv: trivial -> regular
        out_type = regular_feature_type(self.gspace, channels[0])
        self.conv1 = enn.R2Conv(
            self.in_type, out_type, 7,
            stride=2, padding=3, bias=False,
            sigma=None, frequencies_cutoff=lambda r: 3*r
        )
        self.bn1 = enn.InnerBatchNorm(out_type)
        self.relu = enn.ReLU(out_type, inplace=True)
        self.maxpool = enn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)
        
        # Build layers
        self.layer1 = self._make_layer(channels[0], channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], layers[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], layers[3], stride=2)
        
        # Output type for group pooling
        self.final_type = regular_feature_type(self.gspace, channels[3])

        # Number of regular fields at layer4: regular_feature_type divides channels[3] by
        # orientation (see top-level helper). So the actual tensor going into the pool is
        # (B, F_ * G, H, W) where F_ = channels[3] // G.
        num_fields = channels[3] // orientation  # 64 for channels[3]=512, C8

        if group_pool_mode == 'max':
            self.group_pool = enn.GroupPooling(self.final_type)
            self.invariant_multiplier = 1                       # one scalar per field
        elif group_pool_mode == 'mean':
            # Mean-over-group alternative (ablation row for R2-Q4.2 GroupPool ablation).
            # Implemented manually: reshape raw tensor to separate field/orientation axes,
            # mean over orientation. Still produces an invariant descriptor (mean over
            # the group is an orbit average, which is invariant to group action).
            self.group_pool = None   # will use manual reshape in forward
            self.invariant_multiplier = 1
        elif group_pool_mode == 'norm':
            # B2 NormPool: L2 norm over the C_N orbit per field. Preserves total
            # orientation energy rather than just the strongest orientation (max)
            # or per-orient mean. Used by ReDet-style equivariant detection. Same
            # output dim as max/mean (1 scalar per field).
            self.group_pool = None
            self.invariant_multiplier = 1
        elif group_pool_mode == 'fourier':
            # Tier-2 Fourier-basis invariant mapping: for regular rep of C_N, the
            # irreducible representations are Fourier modes k = 0, 1, ..., N/2.
            # |F_k| is invariant under cyclic shift (image rotation by 2π/N).
            # For even N, independent invariants are {|F_0|, |F_1|, ..., |F_{N/2}|},
            # total N/2 + 1 real numbers per field. rfft returns exactly these bins.
            # Reference: Cohen & Welling, "Steerable CNNs" (ICLR 2017);
            # Weiler & Cesa, "General E(2)-Equivariant Steerable CNNs" (NeurIPS 2019).
            self.group_pool = None
            self.invariant_multiplier = orientation // 2 + 1    # 5 for C8
        else:
            raise ValueError(f"Unknown group_pool_mode: {group_pool_mode}")

        # Actual invariant channels exposed to downstream aggregator. Replaces the
        # previous (misleading) `self.out_channels = channels[3]` which claimed 512 but
        # the real post-pool width is 64 (max/mean) or 320 (fourier). Upstream should
        # read this attribute rather than computing `channels[-1] // orientation`.
        self.out_channels = num_fields * self.invariant_multiplier

        print(f"✓ E2ResNet initialized: C{orientation}, layers={layers}, channels={channels}, "
              f"pool={group_pool_mode}, invariant out_channels={self.out_channels}")
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Build a layer with multiple blocks"""
        downsample = None
        if stride != 1 or in_channels != out_channels:
            in_type = regular_feature_type(self.gspace, in_channels)
            out_type = regular_feature_type(self.gspace, out_channels)
            
            downsample_layers = []
            if stride != 1:
                downsample_layers.append(
                    enn.PointwiseAvgPool(in_type, kernel_size=stride, stride=stride)
                )
            downsample_layers.extend([
                enn.R2Conv(in_type, out_type, 1, stride=1, bias=False,
                          sigma=None, frequencies_cutoff=lambda r: 3*r),
                enn.InnerBatchNorm(out_type)
            ])
            downsample = enn.SequentialModule(*downsample_layers)
        
        layers = []
        layers.append(EquivariantBasicBlock(
            self.gspace, in_channels, out_channels, stride, downsample
        ))
        for _ in range(1, blocks):
            layers.append(EquivariantBasicBlock(
                self.gspace, out_channels, out_channels
            ))
        
        return enn.SequentialModule(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) standard tensor
        Returns:
            x: (B, C, H/32, W/32) rotation-invariant feature map
        """
        # Wrap input as GeometricTensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # H/4, W/4
        
        # ResNet stages
        x = self.layer1(x)  # H/4, W/4
        x = self.layer2(x)  # H/8, W/8
        x = self.layer3(x)  # H/16, W/16
        x = self.layer4(x)  # H/32, W/32
        
        # Group pooling: rotation-equivariant -> rotation-invariant
        if self.group_pool_mode == 'max':
            x = self.group_pool(x)
            return x.tensor

        # Shared reshape for 'mean' and 'fourier': e2cnn's regular FieldType with F
        # fields of regular rep (order G) lays out channels as
        #   [f0_g0, f0_g1, ..., f0_g{G-1}, f1_g0, f1_g1, ...].
        # So `.view(B, F_, G, H, W)` correctly puts orientation on dim=2.
        raw = x.tensor
        B, C, H, W = raw.shape
        G = self.orientation
        assert C % G == 0, f"channels {C} not divisible by group order {G}"
        F_ = C // G
        raw = raw.view(B, F_, G, H, W)

        if self.group_pool_mode == 'mean':
            return raw.mean(dim=2)  # (B, F_, H, W), invariant to C_G rotation

        if self.group_pool_mode == 'norm':
            # B2 NormPool: ||field_orbit||_2. Preserves total orientation energy.
            # Cast to fp32 to avoid fp16 underflow on small squared values; cast back
            # for downstream AMP compatibility.
            dtype_in = raw.dtype
            return raw.float().pow(2).sum(dim=2).clamp_min(1e-12).sqrt().to(dtype_in)

        # 'fourier': discrete Fourier transform over the orientation dim, keep |F_k|
        # for k = 0..G//2 (5 bins for C8). Cast to fp32 before FFT — PyTorch's
        # complex fp16 path is unstable under AMP.
        dtype_in = raw.dtype
        modes = torch.fft.rfft(raw.float(), dim=2)      # (B, F_, G//2+1, H, W) complex
        mags = modes.abs()                               # (B, F_, G//2+1, H, W) real
        out = mags.reshape(B, F_ * (G // 2 + 1), H, W)   # (B, F_*(G//2+1), H, W)
        return out.to(dtype_in)                          # back to fp16 if AMP


# Quick test
if __name__ == '__main__':
    import torch

    print("=" * 70)
    print("Testing E2ResNetBackbone with all three pool modes")
    print("=" * 70)

    def _test_mode(mode, expected_ch, use_torch_rot90=True):
        print(f"\n--- Mode: {mode} ---")
        model = E2ResNetBackbone(
            orientation=8,
            layers=[1, 1, 1, 1],
            channels=[32, 64, 128, 256],
            group_pool_mode=mode,
        ).eval()

        x = torch.randn(2, 3, 128, 128)
        with torch.no_grad():
            out = model(x)
        print(f"Input  shape: {tuple(x.shape)}")
        print(f"Output shape: {tuple(out.shape)}  (expected channel dim {expected_ch})")
        assert out.shape[1] == expected_ch, \
            f"Expected {expected_ch} channels, got {out.shape[1]}"
        assert model.out_channels == expected_ch, \
            f"out_channels attr {model.out_channels} != actual {expected_ch}"

        # Rotation invariance check — for C8, 90° rotation (torch.rot90 k=1) is a
        # group element, so invariance should hold up to spatial permutation.
        # We compare spatially-pooled outputs (mean over H,W) which are fully invariant.
        with torch.no_grad():
            x_rot = torch.rot90(x, k=1, dims=[2, 3])
            out_rot = model(x_rot)
        g0 = out.mean(dim=(2, 3))        # (B, C)
        g90 = out_rot.mean(dim=(2, 3))   # (B, C)
        diff = (g0 - g90).abs().max().item()
        print(f"90° rotation: max|g0 - g90| on globally-pooled desc = {diff:.2e}")
        # For 'fourier' mode this should be numerically tiny (just interpolation +
        # floating-point roundoff); for 'max' mode it is typically small but not 0
        # because max-pool is strictly invariant only at the per-field-per-location
        # level, not after further spatial averaging of non-shift-aligned pixels.
        return diff

    d_max = _test_mode('max',     expected_ch=32)    # 256//8 = 32 invariant channels
    d_mean = _test_mode('mean',   expected_ch=32)
    d_norm = _test_mode('norm',   expected_ch=32)    # B2 NormPool, same dim as max/mean
    d_fourier = _test_mode('fourier', expected_ch=32 * (8 // 2 + 1))  # 32 * 5 = 160

    print("\n" + "=" * 70)
    print("Summary (lower = better rotation invariance on globally-pooled desc):")
    print(f"  max     : {d_max:.2e}  (expected ≤ ~1e-3 due to spatial averaging)")
    print(f"  mean    : {d_mean:.2e}")
    print(f"  norm    : {d_norm:.2e}  (B2 NormPool, expected similar order to mean/max)")
    print(f"  fourier : {d_fourier:.2e}  (should be smallest — Fourier magnitudes are"
          " strictly C8-invariant up to spatial interpolation error)")
    print("✓ E2ResNetBackbone test run complete.")