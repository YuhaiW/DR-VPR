"""
Dual-branch aggregator that combines standard CNN and equivariant CNN features
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusion(nn.Module):
    """Learn optimal fusion weights between two branches.

    Dimension-agnostic: works for any (dim1, dim2). Softmax final layer is
    biased at init so w1 ≈ 1.0 / w2 ≈ 0.0 — this preserves the pretrained
    Branch-1 (BoQ) signal at training start, mirroring the zero-init gate
    used in concat fusion. Both branches become actively used as training
    moves the learned weights away from this prior.
    """
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.hidden = nn.Linear(dim1 + dim2, 512)
        self.dropout = nn.Dropout(0.1)
        self.final = nn.Linear(512, 2)
        # Init: make softmax output ≈ (1.0, 0.0) regardless of input
        with torch.no_grad():
            self.final.weight.zero_()
            self.final.bias.copy_(torch.tensor([10.0, 0.0]))

    def forward(self, desc1, desc2):
        concat = torch.cat([desc1, desc2], dim=1)                  # (B, dim1+dim2)
        h = F.relu(self.hidden(concat))
        h = self.dropout(h)
        weights = F.softmax(self.final(h), dim=1)                  # (B, 2)
        w1, w2 = weights[:, 0:1], weights[:, 1:2]

        desc1_proj = desc1 * w1                                     # (B, dim1)
        desc2_proj = F.pad(desc2, (0, self.dim1 - self.dim2)) * w2  # pad to dim1

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

        # Zero-init gate: desc2 is scaled by a learnable scalar that starts at 0.
        # So at initialization the fused descriptor == pure BoQ descriptor
        # (up to global L2-normalization). The gate grows only if the equi
        # branch produces useful signal. Prevents random-init equi from
        # destroying BoQ pretrained features.
        self.equi_gate = nn.Parameter(torch.zeros(1))
        
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

        # Per-branch L2-norm BEFORE fusion: prevents the larger-dim branch
        # from swamping the smaller-dim branch (critical when Branch 1 is
        # 16384-dim BoQ and Branch 2 is 1024-dim GeM)
        desc1 = F.normalize(desc1, p=2, dim=1)
        desc2 = F.normalize(desc2, p=2, dim=1)

        # Zero-init gate on equi branch: keeps pure BoQ behavior at t=0
        desc2 = desc2 * self.equi_gate

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