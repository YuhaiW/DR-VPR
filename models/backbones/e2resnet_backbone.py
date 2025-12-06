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
                 pretrained=False):   # Note: equivariant models usually need custom pretraining
        super().__init__()
        
        self.orientation = orientation
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
        self.group_pool = enn.GroupPooling(self.final_type)
        
        self.out_channels = channels[3]  # Output channels after group pooling
        
        print(f"✓ E2ResNet initialized: C{orientation}, layers={layers}, channels={channels}")
        
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
        x = self.group_pool(x)
        
        # Return standard tensor
        return x.tensor


# Quick test
if __name__ == '__main__':
    import torch
    
    print("Testing E2ResNetBackbone...")
    
    # Test with small model
    model = E2ResNetBackbone(
        orientation=4,  # C4 group
        layers=[1, 1, 1, 1],
        channels=[32, 64, 128, 256]
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 128, 128)
    print(f"Input shape: {x.shape}")
    
    out = model(x)
    print(f"Output shape: {out.shape}")
    print(f"Expected: (2, 256, 4, 4)")
    
    # Test rotation equivariance
    import torch.nn.functional as F
    x_rot = torch.rot90(x, k=1, dims=[2, 3])
    out_rot = model(x_rot)
    
    print(f"\nRotation test:")
    print(f"Original output norm: {out.norm():.4f}")
    print(f"Rotated output norm: {out_rot.norm():.4f}")
    print(f"Difference: {(out.norm() - out_rot.norm()).abs():.6f}")
    
    print("\n✓ E2ResNetBackbone test passed!")