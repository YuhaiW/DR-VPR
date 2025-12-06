"""
新的混合模型：预训练ResNet + 等变层 + 原始MixVPR聚合器
结合等变特征和成熟的MixVPR聚合策略
"""
import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn


class LiftingConvolution(nn.Module):
    """
    将标准特征"提升"到等变表示
    """
    def __init__(self, in_channels, out_channels, rotation_order=8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rotation_order = rotation_order
        
        # 定义群
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # 输入类型：标量场（标准CNN特征）
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.trivial_repr] * in_channels
        )
        
        # 输出类型：regular表示（等变特征）
        self.out_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * out_channels
        )
        
        # Lifting卷积：从标量场到群表示
        self.lift = enn.R2Conv(
            self.in_type, 
            self.out_type, 
            kernel_size=1,
            padding=0,
            bias=False
        )
        self.bn = enn.InnerBatchNorm(self.out_type)
        self.relu = enn.ReLU(self.out_type, inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: 标准特征 (B, in_channels, H, W)
        Returns:
            等变特征 (B, out_channels*rotation_order, H, W)
        """
        x = enn.GeometricTensor(x, self.in_type)
        x = self.lift(x)
        x = self.bn(x)
        x = self.relu(x)
        return x.tensor


class EquivariantRefinement(nn.Module):
    """
    等变精炼层（可选）
    """
    def __init__(self, in_channels, rotation_order=8, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.rotation_order = rotation_order
        
        # 定义群
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # 输入类型
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * in_channels
        )
        
        # 构建等变卷积层
        layers = []
        current_type = self.in_type
        
        for i in range(num_layers):
            next_type = enn.FieldType(
                self.r2_act,
                [self.r2_act.regular_repr] * in_channels
            )
            
            layers.extend([
                enn.R2Conv(current_type, next_type, kernel_size=3, padding=1, bias=False),
                enn.InnerBatchNorm(next_type),
                enn.ReLU(next_type, inplace=True)
            ])
            
            current_type = next_type
        
        self.layers = enn.SequentialModule(*layers)
        self.out_type = current_type
    
    def forward(self, x):
        """
        Args:
            x: 等变特征 (B, in_channels*rotation_order, H, W)
        Returns:
            精炼后的等变特征 (B, in_channels*rotation_order, H, W)
        """
        x = enn.GeometricTensor(x, self.in_type)
        x = self.layers(x)
        return x.tensor


class HybridModelWithMixVPR(nn.Module):
    """
    新的混合模型：
    预训练ResNet → Lifting → (可选)等变精炼 → 原始MixVPR聚合器 → 描述符
    
    关键改进：
    1. 使用512通道（不压缩到256）
    2. 使用原始MixVPR聚合器（已验证有效）
    3. 输出4096维（和原始MixVPR一样）
    """
    def __init__(
        self,
        backbone,                      # 预训练的ResNet
        backbone_out_channels=1024,    # ResNet输出通道数
        lifting_channels=512,          # Lifting后的基础通道数
        rotation_order=8,              # 旋转群阶数
        use_equivariant_refinement=True,  # 是否使用等变精炼层
        num_equivariant_layers=2,      # 等变精炼层数
        mixvpr_config=None,            # MixVPR聚合器配置
    ):
        super().__init__()
        
        self.backbone = backbone
        self.rotation_order = rotation_order
        self.use_refinement = use_equivariant_refinement
        
        # Lifting层：标准特征 → 等变特征
        self.lifting = LiftingConvolution(
            in_channels=backbone_out_channels,
            out_channels=lifting_channels,
            rotation_order=rotation_order
        )
        
        # 等变特征的实际通道数
        equi_channels = lifting_channels * rotation_order
        
        # 等变精炼层（可选）
        if use_equivariant_refinement and num_equivariant_layers > 0:
            self.equivariant_refinement = EquivariantRefinement(
                in_channels=lifting_channels,
                rotation_order=rotation_order,
                num_layers=num_equivariant_layers
            )
        else:
            self.equivariant_refinement = None
        
        # 原始MixVPR聚合器
        from models.aggregators import MixVPR
        
        if mixvpr_config is None:
            # 默认配置（基于等变特征的维度）
            # 等变特征: (B, 512*8, 20, 20) = (B, 4096, 20, 20)
            mixvpr_config = {
                'in_channels': equi_channels,  # 4096
                'in_h': 20,
                'in_w': 20,
                'out_channels': 1024,
                'mix_depth': 4,
                'mlp_ratio': 1,
                'out_rows': 4
            }
        
        self.aggregator = MixVPR(**mixvpr_config)
        
        # 输出描述符维度
        self.out_channels = mixvpr_config['out_channels'] * mixvpr_config['out_rows']
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            描述符 (B, out_channels) - 默认4096
        """
        # 1. 预训练backbone提取特征
        features = self.backbone(x)  # (B, 1024, 20, 20)
        
        # 2. Lifting到等变表示
        equi_features = self.lifting(features)  # (B, 512*8, 20, 20) = (B, 4096, 20, 20)
        
        # 3. 等变精炼（可选）
        if self.equivariant_refinement is not None:
            equi_features = self.equivariant_refinement(equi_features)  # (B, 4096, 20, 20)
        
        # 4. 使用原始MixVPR聚合器
        # 注意：MixVPR会把等变特征当作普通特征处理
        # 但等变特征更丰富（包含旋转信息）
        descriptor = self.aggregator(equi_features)  # (B, 4096)
        
        return descriptor


def create_hybrid_mixvpr_model(
    backbone_arch='resnet50',
    pretrained=True,
    layers_to_crop=[4],
    lifting_channels=512,
    rotation_order=8,
    use_equivariant_refinement=True,
    num_equivariant_layers=2,
    mixvpr_config=None
):
    """
    创建新的混合模型：ResNet + Lifting + MixVPR
    
    Args:
        backbone_arch: 主干网络架构
        pretrained: 是否使用预训练权重
        layers_to_crop: 要裁剪的层
        lifting_channels: Lifting后的基础通道数 (建议512或1024)
        rotation_order: 旋转群阶数 (建议4或8)
        use_equivariant_refinement: 是否使用等变精炼层
        num_equivariant_layers: 等变精炼层数量
        mixvpr_config: MixVPR聚合器配置
    """
    from models import backbones
    
    # 创建预训练backbone
    backbone = backbones.ResNet(
        backbone_arch, 
        pretrained=pretrained,
        layers_to_freeze=2,
        layers_to_crop=layers_to_crop
    )
    
    backbone_out_channels = backbone.out_channels
    
    # 创建混合模型
    model = HybridModelWithMixVPR(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        lifting_channels=lifting_channels,
        rotation_order=rotation_order,
        use_equivariant_refinement=use_equivariant_refinement,
        num_equivariant_layers=num_equivariant_layers,
        mixvpr_config=mixvpr_config
    )
    
    return model


if __name__ == '__main__':
    print("="*70)
    print("  测试新的混合模型：ResNet + Lifting + MixVPR")
    print("="*70)
    
    # 测试配置1：使用等变精炼层
    print("\n【配置1】使用等变精炼层")
    print("-"*70)
    model1 = create_hybrid_mixvpr_model(
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_crop=[4],
        lifting_channels=512,      # 512基础通道
        rotation_order=8,          # C8
        use_equivariant_refinement=True,
        num_equivariant_layers=2,
    )
    
    x = torch.randn(2, 3, 320, 320)
    desc1 = model1(x)
    
    total_params = sum(p.numel() for p in model1.parameters())
    print(f"✓ 输入: {x.shape}")
    print(f"✓ 输出: {desc1.shape}")
    print(f"✓ 参数: {total_params/1e6:.2f}M")
    
    # 测试配置2：不使用等变精炼层
    print("\n【配置2】不使用等变精炼层（更快）")
    print("-"*70)
    model2 = create_hybrid_mixvpr_model(
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_crop=[4],
        lifting_channels=512,
        rotation_order=8,
        use_equivariant_refinement=False,  # 关闭精炼层
        num_equivariant_layers=0,
    )
    
    desc2 = model2(x)
    
    total_params2 = sum(p.numel() for p in model2.parameters())
    print(f"✓ 输入: {x.shape}")
    print(f"✓ 输出: {desc2.shape}")
    print(f"✓ 参数: {total_params2/1e6:.2f}M")
    
    print("\n" + "="*70)
    print("  配置对比")
    print("="*70)
    print(f"\n使用精炼层:   参数 {total_params/1e6:.2f}M, 输出 {desc1.shape[1]}维")
    print(f"不用精炼层:   参数 {total_params2/1e6:.2f}M, 输出 {desc2.shape[1]}维")
    print(f"\n推荐: 先尝试不用精炼层（配置2）")
    print(f"      更快，参数更少，可能性能一样好")
    print("="*70)
