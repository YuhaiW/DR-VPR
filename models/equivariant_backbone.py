import torch
import torch.nn as nn
import numpy as np
from escnn import gspaces
from escnn import nn as enn


class EquivariantResNet(nn.Module):
    """
    Rotation-Equivariant ResNet backbone for Visual Place Recognition.
    Uses C8 (8-fold rotational symmetry) for better rotation invariance.
    
    This backbone maintains equivariance to rotations, making it more robust
    to viewpoint changes in place recognition tasks.
    """
    
    def __init__(self,
                 rotation_order=8,  # C8: 8-fold rotational symmetry (45° increments)
                 pretrained=False,  # Note: equivariant networks typically trained from scratch
                 layers_to_freeze=0,
                 initial_channels=64,
                 block_channels=[64, 128, 256, 512],
                 blocks_per_layer=[2, 2, 2, 2],
                 ):
        """
        Args:
            rotation_order (int): Order of cyclic group (e.g., 8 for C8, 4 for C4)
            pretrained (bool): Pretrained weights not supported for equivariant nets
            layers_to_freeze (int): Number of layers to freeze (0-4)
            initial_channels (int): Channels after initial conv
            block_channels (list): Output channels for each layer
            blocks_per_layer (list): Number of residual blocks per layer
        """
        super().__init__()
        
        self.rotation_order = rotation_order
        self.layers_to_freeze = layers_to_freeze
        
        # Define the group action space (C_n rotations)
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # Input: regular field (RGB image)
        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * 3)
        
        # Build network layers
        self.conv1, out_type = self._make_conv_block(
            self.in_type, 
            initial_channels, 
            kernel_size=7, 
            stride=2, 
            padding=3
        )
        self.relu1 = enn.ReLU(out_type, inplace=True)
        self.pool1 = enn.PointwiseMaxPool(out_type, kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        in_type = out_type
        self.layer1, in_type = self._make_layer(in_type, block_channels[0], blocks_per_layer[0])
        self.layer2, in_type = self._make_layer(in_type, block_channels[1], blocks_per_layer[1], stride=2)
        self.layer3, in_type = self._make_layer(in_type, block_channels[2], blocks_per_layer[2], stride=2)
        self.layer4, in_type = self._make_layer(in_type, block_channels[3], blocks_per_layer[3], stride=2)
        
        self.out_type = in_type
        # 等变网络的实际输出通道数 = block_channels[-1] * rotation_order
        self.out_channels = block_channels[3] * rotation_order
        
        # Freeze layers if requested
        if layers_to_freeze > 0:
            self._freeze_layers()
    
    def _make_conv_block(self, in_type, out_channels, kernel_size=3, stride=1, padding=1):
        """Create an equivariant convolutional block"""
        # Output field type: regular representation
        out_type = enn.FieldType(
            self.r2_act, 
            [self.r2_act.regular_repr] * out_channels
        )
        
        conv = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=kernel_size, 
                      stride=stride, padding=padding, bias=False),
            enn.InnerBatchNorm(out_type),
        )
        
        return conv, out_type
    
    def _make_residual_block(self, in_type, out_channels, stride=1):
        """Create an equivariant residual block - FIXED for escnn 1.0"""
        # Middle type
        mid_type = enn.FieldType(
            self.r2_act, 
            [self.r2_act.regular_repr] * out_channels
        )
        
        # Output type
        out_type = enn.FieldType(
            self.r2_act, 
            [self.r2_act.regular_repr] * out_channels
        )
        
        # Main path
        main_path = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=3, stride=stride, padding=1, bias=False),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type, inplace=True),
            enn.R2Conv(mid_type, out_type, kernel_size=3, stride=1, padding=1, bias=False),
            enn.InnerBatchNorm(out_type),
        )
        
        # Skip connection
        if stride != 1 or in_type != out_type:
            skip = enn.SequentialModule(
                enn.R2Conv(in_type, out_type, kernel_size=1, stride=stride, padding=0, bias=False),
                enn.InnerBatchNorm(out_type),
            )
        else:
            skip = enn.IdentityModule(in_type)
        
        # 简化的残差连接实现
        class ResidualBlock(torch.nn.Module):
            def __init__(self, main, skip, in_field_type, out_field_type):
                super().__init__()
                self.main = main
                self.skip = skip
                self.in_field_type = in_field_type
                self.out_field_type = out_field_type
                self.relu = enn.ReLU(out_field_type, inplace=True)
                
            def forward(self, x):
                # x is a GeometricTensor
                out = self.main(x)
                identity = self.skip(x)
                # Add tensors and wrap back
                out = enn.GeometricTensor(
                    out.tensor + identity.tensor, 
                    self.out_field_type
                )
                out = self.relu(out)
                return out
        
        return ResidualBlock(main_path, skip, in_type, out_type), out_type
    
    def _make_layer(self, in_type, out_channels, num_blocks, stride=1):
        """Create a layer with multiple residual blocks"""
        layers = []
        
        # First block (may downsample)
        block, in_type = self._make_residual_block(in_type, out_channels, stride)
        layers.append(block)
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            block, in_type = self._make_residual_block(in_type, out_channels, stride=1)
            layers.append(block)
        
        return enn.SequentialModule(*layers), in_type
    
    def _freeze_layers(self):
        """Freeze initial layers"""
        if self.layers_to_freeze >= 0:
            self.conv1.requires_grad_(False)
        if self.layers_to_freeze >= 1:
            self.layer1.requires_grad_(False)
        if self.layers_to_freeze >= 2:
            self.layer2.requires_grad_(False)
        if self.layers_to_freeze >= 3:
            self.layer3.requires_grad_(False)
        if self.layers_to_freeze >= 4:
            self.layer4.requires_grad_(False)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: torch.Tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor of shape (B, C', H', W')
        """
        # Wrap input as geometric tensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # Forward through network
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Extract regular torch tensor
        x = x.tensor
        
        return x


class EquivariantEfficientNet(nn.Module):
    """
    Lightweight equivariant network inspired by EfficientNet architecture.
    Uses depthwise separable convolutions with group equivariance.
    """
    
    def __init__(self,
                 rotation_order=8,
                 pretrained=False,
                 layers_to_freeze=0,
                 width_mult=1.0,
                 ):
        """
        Args:
            rotation_order (int): Order of cyclic group
            pretrained (bool): Not supported for equivariant networks
            layers_to_freeze (int): Number of initial blocks to freeze
            width_mult (float): Width multiplier for channels
        """
        super().__init__()
        
        self.rotation_order = rotation_order
        self.layers_to_freeze = layers_to_freeze
        
        # Define group action
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        self.in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr] * 3)
        
        # Channel configuration (scaled by width_mult)
        channels = [int(c * width_mult) for c in [32, 64, 128, 256, 512]]
        
        # Build network
        self.stem, out_type = self._make_conv_block(self.in_type, channels[0], stride=2)
        
        blocks = []
        in_type = out_type
        
        # Build efficient blocks
        for i, ch in enumerate(channels[1:]):
            stride = 2 if i > 0 else 1
            block, in_type = self._make_efficient_block(in_type, ch, stride)
            blocks.append(block)
        
        self.blocks = enn.SequentialModule(*blocks)
        
        # Final conv
        self.head, out_type = self._make_conv_block(in_type, channels[-1], kernel_size=1, padding=0)
        
        self.out_type = out_type
        # 等变网络的实际输出通道数 = channels[-1] * rotation_order
        self.out_channels = channels[-1] * rotation_order
        
        if layers_to_freeze > 0:
            self._freeze_layers()
    
    def _make_conv_block(self, in_type, out_channels, kernel_size=3, stride=1, padding=1):
        """Standard conv block"""
        out_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * out_channels
        )
        
        conv = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=False),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type, inplace=True)
        )
        
        return conv, out_type
    
    def _make_efficient_block(self, in_type, out_channels, stride=1):
        """Depthwise separable convolution block (equivariant version)"""
        # Depthwise convolution
        dw_type = in_type
        depthwise = enn.SequentialModule(
            enn.R2Conv(in_type, dw_type, kernel_size=3, stride=stride, 
                      padding=1, groups=len(in_type), bias=False),
            enn.InnerBatchNorm(dw_type),
            enn.ReLU(dw_type, inplace=True)
        )
        
        # Pointwise convolution
        out_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * out_channels
        )
        
        pointwise = enn.SequentialModule(
            enn.R2Conv(dw_type, out_type, kernel_size=1, stride=1, padding=0, bias=False),
            enn.InnerBatchNorm(out_type),
        )
        
        return enn.SequentialModule(depthwise, pointwise), out_type
    
    def _freeze_layers(self):
        """Freeze layers"""
        if self.layers_to_freeze >= 0:
            self.stem.requires_grad_(False)
        for i in range(min(self.layers_to_freeze, len(self.blocks))):
            self.blocks[i].requires_grad_(False)
    
    def forward(self, x):
        """Forward pass"""
        x = enn.GeometricTensor(x, self.in_type)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x.tensor


def print_nb_params(m):
    """Print number of trainable parameters"""
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3f}M')


if __name__ == '__main__':
    # Test EquivariantResNet
    print("Testing EquivariantResNet:")
    x = torch.randn(2, 3, 320, 320)
    model = EquivariantResNet(
        rotation_order=8,
        initial_channels=32,
        block_channels=[32, 64, 128, 256],
        blocks_per_layer=[2, 2, 2, 2]
    )
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Out channels: {model.out_channels}")
    print_nb_params(model)
    
    print("\n" + "="*50 + "\n")
    
    # Test EquivariantEfficientNet
    print("Testing EquivariantEfficientNet:")
    model2 = EquivariantEfficientNet(rotation_order=8, width_mult=0.5)
    output2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output2.shape}")
    print(f"Out channels: {model2.out_channels}")
    print_nb_params(model2)