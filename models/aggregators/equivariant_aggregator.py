import torch
import torch.nn as nn
import torch.nn.functional as F
from escnn import gspaces
from escnn import nn as enn


class EquivariantGeM(nn.Module):
    """
    Group Equivariant Generalized Mean Pooling.
    
    This aggregator produces rotation-invariant global descriptors by:
    1. Applying GeM pooling to equivariant features
    2. Computing group pooling to achieve invariance
    """
    
    def __init__(self, 
                 rotation_order=8,
                 in_channels=512,
                 p=3,
                 eps=1e-6):
        """
        Args:
            rotation_order (int): Order of rotation group
            in_channels (int): Number of input channels
            p (float): Power parameter for GeM pooling
            eps (float): Small constant for numerical stability
        """
        super().__init__()
        
        self.rotation_order = rotation_order
        self.in_channels = in_channels
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
        # Define group
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # Input type: regular representation
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * in_channels
        )
        
        # Group pooling for invariance
        self.group_pool = enn.GroupPooling(self.in_type)
        
        # Output dimension after group pooling
        self.out_channels = in_channels
    
    def gem_pool(self, x):
        """Generalized Mean Pooling"""
        # x shape: (B, C, H, W)
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1. / self.p)
        return x
    
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B, C, H, W) or enn.GeometricTensor
        Returns:
            torch.Tensor of shape (B, out_channels)
        """
        # If input is regular tensor, wrap it
        if not isinstance(x, enn.GeometricTensor):
            x = enn.GeometricTensor(x, self.in_type)
        
        # Apply GeM pooling on spatial dimensions
        x_tensor = x.tensor
        x_pooled = self.gem_pool(x_tensor)
        
        # Wrap back as geometric tensor
        pooled_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * self.in_channels
        )
        x_geo = enn.GeometricTensor(x_pooled, pooled_type)
        
        # Apply group pooling for rotation invariance
        x_inv = self.group_pool(x_geo)
        
        # Flatten
        x_out = x_inv.tensor.flatten(1)
        
        return x_out


class EquivariantMixVPR(nn.Module):
    """
    Equivariant version of MixVPR aggregator.
    
    Processes equivariant features while maintaining rotation equivariance
    until the final aggregation step which produces invariant descriptors.
    """
    
    def __init__(self,
                 rotation_order=8,
                 in_channels=512,
                 in_h=20,
                 in_w=20,
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4):
        """
        Args:
            rotation_order (int): Order of rotation group
            in_channels (int): Input channel dimension
            in_h (int): Input spatial height
            in_w (int): Input spatial width  
            out_channels (int): Output channel dimension
            mix_depth (int): Number of mixing layers
            mlp_ratio (int): MLP expansion ratio
            out_rows (int): Number of output rows (output dim = out_rows * out_channels)
        """
        super().__init__()
        
        self.rotation_order = rotation_order
        self.in_channels = in_channels
        self.in_h = in_h
        self.in_w = in_w
        self.out_channels = out_channels
        self.mix_depth = mix_depth
        self.out_rows = out_rows
        
        # Define group
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # Input type
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * in_channels
        )
        
        # Feature mixing layers (equivariant)
        hw = in_h * in_w
        self.mix_blocks = nn.ModuleList()
        
        for _ in range(mix_depth):
            mix_block = nn.Sequential(
                nn.Linear(hw, int(hw * mlp_ratio)),
                nn.GELU(),
                nn.Linear(int(hw * mlp_ratio), hw)
            )
            self.mix_blocks.append(mix_block)
        
        # Channel projection (equivariant)
        mid_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * out_channels
        )
        
        self.channel_proj = enn.SequentialModule(
            enn.R2Conv(self.in_type, mid_type, kernel_size=1, bias=False),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type, inplace=True)
        )
        
        # Row-wise pooling with group pooling for invariance
        self.group_pool = enn.GroupPooling(mid_type)
        
        # Output projection
        self.out_proj = nn.Linear(out_channels * out_rows, out_channels * out_rows)
        
        self.out_dim = out_channels * out_rows
    
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B, C, H, W) or enn.GeometricTensor
        Returns:
            torch.Tensor of shape (B, out_channels * out_rows)
        """
        # Wrap input if needed
        if not isinstance(x, enn.GeometricTensor):
            x = enn.GeometricTensor(x, self.in_type)
        
        x_tensor = x.tensor
        B, C, H, W = x_tensor.shape
        
        # Feature mixing along spatial dimension (equivariant operation on tensor)
        x_flat = x_tensor.reshape(B, C, H * W)
        
        for mix_block in self.mix_blocks:
            # Mix spatial features
            x_mixed = mix_block(x_flat)
            x_flat = x_flat + x_mixed
        
        x_tensor = x_flat.reshape(B, C, H, W)
        x = enn.GeometricTensor(x_tensor, self.in_type)
        
        # Channel projection (equivariant)
        x = self.channel_proj(x)
        
        # Spatial pooling per row
        x_tensor = x.tensor
        _, C_out, H_out, W_out = x_tensor.shape
        
        # Pool to out_rows
        if H_out != self.out_rows:
            kernel_h = H_out // self.out_rows
            x_tensor = F.adaptive_avg_pool2d(x_tensor, (self.out_rows, W_out))
        
        # Pool width dimension
        x_tensor = F.adaptive_avg_pool2d(x_tensor, (self.out_rows, 1))
        
        # Create geometric tensor for group pooling
        pooled_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * self.out_channels
        )
        x_geo = enn.GeometricTensor(x_tensor, pooled_type)
        
        # Apply group pooling for rotation invariance
        x_inv = self.group_pool(x_geo)
        
        # Flatten
        x_out = x_inv.tensor.flatten(1)
        
        # Final projection
        x_out = self.out_proj(x_out)
        
        return x_out


class EquivariantConvAP(nn.Module):
    """
    Equivariant Attentive Pooling.
    
    Uses equivariant convolutions for attention map generation,
    then applies group pooling for rotation-invariant descriptors.
    """
    
    def __init__(self,
                 rotation_order=8,
                 in_channels=512,
                 out_channels=512):
        """
        Args:
            rotation_order (int): Order of rotation group
            in_channels (int): Input channels
            out_channels (int): Output descriptor dimension
        """
        super().__init__()
        
        self.rotation_order = rotation_order
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Define group
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # Input type
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * in_channels
        )
        
        # Attention branch (equivariant)
        attn_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * in_channels
        )
        
        self.attention = enn.SequentialModule(
            enn.R2Conv(self.in_type, attn_type, kernel_size=1, bias=False),
            enn.InnerBatchNorm(attn_type),
        )
        
        # Feature branch (equivariant)
        feat_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * out_channels
        )
        
        self.feature = enn.SequentialModule(
            enn.R2Conv(self.in_type, feat_type, kernel_size=1, bias=False),
            enn.InnerBatchNorm(feat_type),
        )
        
        self.feat_type = feat_type
        
        # Group pooling for invariance
        self.group_pool = enn.GroupPooling(feat_type)
    
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (B, C, H, W) or enn.GeometricTensor
        Returns:
            torch.Tensor of shape (B, out_channels)
        """
        # Wrap input if needed
        if not isinstance(x, enn.GeometricTensor):
            x = enn.GeometricTensor(x, self.in_type)
        
        # Compute attention map (equivariant)
        attn = self.attention(x)
        attn_tensor = attn.tensor
        
        # Softmax over spatial dimensions
        B, C, H, W = attn_tensor.shape
        attn_tensor = attn_tensor.reshape(B, C, H * W)
        attn_tensor = F.softmax(attn_tensor, dim=-1)
        attn_tensor = attn_tensor.reshape(B, C, H, W)
        
        # Compute features (equivariant)
        feat = self.feature(x)
        feat_tensor = feat.tensor
        
        # Apply attention
        weighted = feat_tensor * attn_tensor
        
        # Spatial pooling
        pooled = weighted.sum(dim=(2, 3))
        
        # Wrap as geometric tensor
        pooled_geo = enn.GeometricTensor(
            pooled.unsqueeze(-1).unsqueeze(-1),
            self.feat_type
        )
        
        # Group pooling for rotation invariance
        inv_feat = self.group_pool(pooled_geo)
        
        # Flatten
        out = inv_feat.tensor.flatten(1)
        
        return out


if __name__ == '__main__':
    # Test EquivariantGeM
    print("Testing EquivariantGeM:")
    x = torch.randn(4, 512, 20, 20)
    model = EquivariantGeM(rotation_order=8, in_channels=512)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Test EquivariantMixVPR
    print("\nTesting EquivariantMixVPR:")
    model2 = EquivariantMixVPR(
        rotation_order=8,
        in_channels=512,
        in_h=20,
        in_w=20,
        out_channels=512,
        mix_depth=4,
        out_rows=4
    )
    out2 = model2(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out2.shape}")
    
    # Test EquivariantConvAP
    print("\nTesting EquivariantConvAP:")
    model3 = EquivariantConvAP(rotation_order=8, in_channels=512, out_channels=512)
    out3 = model3(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out3.shape}")
