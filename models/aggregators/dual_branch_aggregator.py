"""
Dual-branch aggregator that combines standard CNN and equivariant CNN features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """Learn optimal fusion weights between two branches"""
    def __init__(self, dim1, dim2):
        super().__init__()
        # 学习每个分支的重要性
        self.attention = nn.Sequential(
            nn.Linear(dim1 + dim2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, desc1, desc2):
        # desc1: (B, 4096), desc2: (B, 512)
        concat = torch.cat([desc1, desc2], dim=1)  # (B, 4608)
        
        # 计算注意力权重
        weights = self.attention(concat)  # (B, 2)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]
        
        # 加权融合
        # 先投影到相同维度
        desc1_proj = desc1 * w1  # 广播
        desc2_proj = F.pad(desc2, (0, 4096-512)) * w2  # pad到4096维
        
        fused = desc1_proj + desc2_proj
        return F.normalize(fused, p=2, dim=1)
    

class GeMAggregator(nn.Module):
    """Generalized Mean Pooling for equivariant features"""
    def __init__(self, in_channels, out_channels, p=3.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.fc = nn.Linear(in_channels, out_channels)
        
    def forward(self, x):
        # x: (B, C, H, W)
        # GeM pooling
        x = F.avg_pool2d(x.clamp(min=1e-6).pow(self.p), 
                         (x.size(-2), x.size(-1))).pow(1./self.p)
        x = x.flatten(1)  # (B, C)
        x = self.fc(x)    # (B, out_channels)
        return x


class DualBranchAggregator(nn.Module):
    """
    Aggregator that combines features from two branches:
    - Branch 1: Standard CNN features (e.g., ResNet + MixVPR)
    - Branch 2: Equivariant CNN features (e.g., E2ResNet + GeM)
    """
    def __init__(self, 
                 # Branch 1 config (standard CNN + aggregator)
                 branch1_aggregator,
                 branch1_out_dim,
                 
                 # Branch 2 config (equivariant CNN)
                 branch2_in_channels=512,  # E2ResNet output channels
                 branch2_out_dim=512,      # Descriptor dimension
                 
                 # Fusion config
                 fusion_method='concat',   # 'concat' or 'add'
                 use_projection=False,     # Project before fusion
                 ):
        super().__init__()
        
        self.branch1_aggregator = branch1_aggregator
        self.branch1_out_dim = branch1_out_dim
        self.branch2_out_dim = branch2_out_dim
        self.fusion_method = fusion_method
        
        # Branch 2: Simple aggregator for equivariant features
        self.branch2_aggregator = GeMAggregator(
            in_channels=branch2_in_channels,
            out_channels=branch2_out_dim,
            p=3.0
        )
        
        # Optional: projection layers before fusion
        self.use_projection = use_projection
        if use_projection:
            self.proj1 = nn.Linear(branch1_out_dim, branch1_out_dim)
            self.proj2 = nn.Linear(branch2_out_dim, branch2_out_dim)
        
        # Final dimension
        if fusion_method == 'concat':
            self.out_dim = branch1_out_dim + branch2_out_dim
        elif fusion_method == 'add':
            assert branch1_out_dim == branch2_out_dim, \
                "For 'add' fusion, both branches must have same dimension"
            self.out_dim = branch1_out_dim
        elif fusion_method == 'attention':
            self.fusion = AttentionFusion(branch1_out_dim, branch2_out_dim)
            self.out_dim = branch1_out_dim  # attention融合后维度
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Branch 1 features (B, C1, H1, W1) - from standard CNN
            x2: Branch 2 features (B, C2, H2, W2) - from equivariant CNN
        Returns:
            fused: (B, out_dim) - fused descriptor
        """
        # Branch 1: standard aggregation (e.g., MixVPR)
        desc1 = self.branch1_aggregator(x1)  # (B, branch1_out_dim)
        
        # Branch 2: equivariant feature aggregation
        desc2 = self.branch2_aggregator(x2)  # (B, branch2_out_dim)
        
        # Optional projection
        if self.use_projection:
            desc1 = self.proj1(desc1)
            desc2 = self.proj2(desc2)
        
        # Fusion
        if self.fusion_method == 'concat':
            fused = torch.cat([desc1, desc2], dim=1)  # (B, D1+D2)
        elif self.fusion_method == 'add':
            fused = desc1 + desc2  # (B, D)
        elif self.fusion_method == 'attention':
            fused = self.fusion(desc1, desc2)  # (B, D)
        
        # L2 normalize
        fused = F.normalize(fused, p=2, dim=1)
        
        return fused