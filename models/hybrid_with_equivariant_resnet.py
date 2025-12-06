"""
改进的混合模型：添加等变ResNet结构
ResNet (预训练) + Lifting + 等变ResNet块 + MixVPR
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
        
        # ═══════════════════════════════════════════
        # 核心概念1: 群定义 ⭐⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # ═══════════════════════════════════════════
        # 核心概念2: 输入类型（平凡表示）⭐⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.trivial_repr] * in_channels
        )
        
        # ═══════════════════════════════════════════
        # 核心概念3: 输出类型（群表示）⭐⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        self.out_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * out_channels
        )
        
        # ═══════════════════════════════════════════
        # 核心概念4: 等变卷积（Lifting）⭐⭐⭐⭐⭐
        # ═══════════════════════════════════════════
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
        # ═══════════════════════════════════════════
        # 核心概念5: GeometricTensor包装 ⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        x = enn.GeometricTensor(x, self.in_type)
        
        x = self.lift(x)
        x = self.bn(x)
        x = self.relu(x)
        return x.tensor


class EquivariantResidualBlock(nn.Module):
    """
    等变ResNet块 - 带残差连接
    这是新添加的ResNet-style结构！
    """
    def __init__(self, in_channels, rotation_order=8, use_bottleneck=False):
        super().__init__()
        self.in_channels = in_channels
        self.rotation_order = rotation_order
        self.use_bottleneck = use_bottleneck
        
        # ═══════════════════════════════════════════
        # 核心概念: 群定义 ⭐⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # ═══════════════════════════════════════════
        # 核心概念: 类型定义 ⭐⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * in_channels
        )
        
        if use_bottleneck:
            # Bottleneck设计：降维 → 3×3 → 升维
            # 类似ResNet50的bottleneck block
            mid_channels = in_channels // 4
            
            mid_type = enn.FieldType(
                self.r2_act,
                [self.r2_act.regular_repr] * mid_channels
            )
            
            # 1×1降维
            self.conv1 = enn.R2Conv(self.in_type, mid_type, kernel_size=1, bias=False)
            self.bn1 = enn.InnerBatchNorm(mid_type)
            self.relu1 = enn.ReLU(mid_type, inplace=True)
            
            # 3×3等变卷积
            self.conv2 = enn.R2Conv(mid_type, mid_type, kernel_size=3, padding=1, bias=False)
            self.bn2 = enn.InnerBatchNorm(mid_type)
            self.relu2 = enn.ReLU(mid_type, inplace=True)
            
            # 1×1升维
            self.conv3 = enn.R2Conv(mid_type, self.in_type, kernel_size=1, bias=False)
            self.bn3 = enn.InnerBatchNorm(self.in_type)
            
        else:
            # 基本块：两个3×3卷积
            # 类似ResNet18/34的basic block
            
            # ═══════════════════════════════════════════
            # 核心概念: 等变卷积 ⭐⭐⭐⭐⭐
            # ═══════════════════════════════════════════
            self.conv1 = enn.R2Conv(self.in_type, self.in_type, 
                                   kernel_size=3, padding=1, bias=False)
            self.bn1 = enn.InnerBatchNorm(self.in_type)
            self.relu1 = enn.ReLU(self.in_type, inplace=True)
            
            self.conv2 = enn.R2Conv(self.in_type, self.in_type, 
                                   kernel_size=3, padding=1, bias=False)
            self.bn2 = enn.InnerBatchNorm(self.in_type)
        
        self.relu_final = enn.ReLU(self.in_type, inplace=True)
    
    def forward(self, x):
        # 保存输入用于残差连接（ResNet的关键！）
        identity = x
        
        # ═══════════════════════════════════════════
        # 核心概念: GeometricTensor ⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        x_geo = enn.GeometricTensor(x, self.in_type)
        
        if self.use_bottleneck:
            # Bottleneck路径
            x_geo = self.conv1(x_geo)
            x_geo = self.bn1(x_geo)
            x_geo = self.relu1(x_geo)
            
            x_geo = self.conv2(x_geo)
            x_geo = self.bn2(x_geo)
            x_geo = self.relu2(x_geo)
            
            x_geo = self.conv3(x_geo)
            x_geo = self.bn3(x_geo)
        else:
            # 基本块路径
            x_geo = self.conv1(x_geo)
            x_geo = self.bn1(x_geo)
            x_geo = self.relu1(x_geo)
            
            x_geo = self.conv2(x_geo)
            x_geo = self.bn2(x_geo)
        
        # ═══════════════════════════════════════════
        # ResNet的核心：残差连接！⭐⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        out = x_geo.tensor + identity  # 残差连接
        
        # 最后的ReLU
        out_geo = enn.GeometricTensor(out, self.in_type)
        out_geo = self.relu_final(out_geo)
        
        return out_geo.tensor


class EquivariantResNetStage(nn.Module):
    """
    等变ResNet阶段：多个残差块
    类似ResNet的layer1, layer2等
    """
    def __init__(self, in_channels, num_blocks=2, rotation_order=8, use_bottleneck=False):
        super().__init__()
        
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                EquivariantResidualBlock(
                    in_channels=in_channels,
                    rotation_order=rotation_order,
                    use_bottleneck=use_bottleneck
                )
            )
        
        self.blocks = nn.ModuleList(blocks)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ImprovedHybridModel(nn.Module):
    """
    改进的混合模型：
    预训练ResNet → Lifting → 等变ResNet块 → MixVPR → 描述符
    
    改进：
    1. 去掉简单的精炼层
    2. 添加ResNet-style的残差块
    3. 支持bottleneck设计
    """
    def __init__(
        self,
        backbone,
        backbone_out_channels=1024,
        lifting_channels=512,
        rotation_order=8,
        num_residual_blocks=2,        # ResNet块的数量
        use_bottleneck=False,          # 是否使用bottleneck设计
        mixvpr_config=None,
    ):
        super().__init__()
        
        self.backbone = backbone
        self.rotation_order = rotation_order
        
        # Lifting层
        self.lifting = LiftingConvolution(
            in_channels=backbone_out_channels,
            out_channels=lifting_channels,
            rotation_order=rotation_order
        )
        
        # 等变特征维度
        equi_channels = lifting_channels * rotation_order
        
        # ═══════════════════════════════════════════
        # 新添加：等变ResNet块 ⭐⭐⭐⭐⭐
        # ═══════════════════════════════════════════
        if num_residual_blocks > 0:
            self.equivariant_resnet = EquivariantResNetStage(
                in_channels=lifting_channels,
                num_blocks=num_residual_blocks,
                rotation_order=rotation_order,
                use_bottleneck=use_bottleneck
            )
        else:
            self.equivariant_resnet = None
        
        # MixVPR聚合器
        from models.aggregators import MixVPR
        
        if mixvpr_config is None:
            mixvpr_config = {
                'in_channels': equi_channels,
                'in_h': 20,
                'in_w': 20,
                'out_channels': 1024,
                'mix_depth': 4,
                'mlp_ratio': 1,
                'out_rows': 4
            }
        
        self.aggregator = MixVPR(**mixvpr_config)
        self.out_channels = mixvpr_config['out_channels'] * mixvpr_config['out_rows']
    
    def forward(self, x):
        # 1. Backbone
        features = self.backbone(x)  # (B, 1024, 20, 20)
        
        # 2. Lifting
        equi_features = self.lifting(features)  # (B, 4096, 20, 20)
        
        # 3. 等变ResNet块（新添加）
        if self.equivariant_resnet is not None:
            equi_features = self.equivariant_resnet(equi_features)  # (B, 4096, 20, 20)
        
        # 4. MixVPR聚合
        descriptor = self.aggregator(equi_features)  # (B, 4096)
        
        return descriptor


def create_improved_hybrid_model(
    backbone_arch='resnet50',
    pretrained=True,
    layers_to_crop=[4],
    lifting_channels=512,
    rotation_order=8,
    num_residual_blocks=2,        # 等变ResNet块数量
    use_bottleneck=False,          # 是否使用bottleneck
    mixvpr_config=None
):
    """
    创建改进的混合模型
    
    Args:
        backbone_arch: 主干网络
        pretrained: 预训练
        layers_to_crop: 裁剪层
        lifting_channels: Lifting基础通道数
        rotation_order: 旋转阶数
        num_residual_blocks: 等变ResNet块数量（0=不使用）
        use_bottleneck: 是否使用bottleneck设计
        mixvpr_config: MixVPR配置
    """
    from models import backbones
    
    backbone = backbones.ResNet(
        backbone_arch, 
        pretrained=pretrained,
        layers_to_freeze=2,
        layers_to_crop=layers_to_crop
    )
    
    model = ImprovedHybridModel(
        backbone=backbone,
        backbone_out_channels=backbone.out_channels,
        lifting_channels=lifting_channels,
        rotation_order=rotation_order,
        num_residual_blocks=num_residual_blocks,
        use_bottleneck=use_bottleneck,
        mixvpr_config=mixvpr_config
    )
    
    return model


if __name__ == '__main__':
    print("="*70)
    print("  测试改进的混合模型（等变ResNet块）")
    print("="*70)
    
    # ═══════════════════════════════════════════
    # 配置1: 无ResNet块（最快）
    # ═══════════════════════════════════════════
    print("\n【配置1】无等变ResNet块")
    print("-"*70)
    model1 = create_improved_hybrid_model(
        backbone_arch='resnet50',
        pretrained=True,
        lifting_channels=512,
        rotation_order=8,
        num_residual_blocks=0,  # ← 不使用ResNet块
    )
    
    x = torch.randn(2, 3, 320, 320)
    desc1 = model1(x)
    params1 = sum(p.numel() for p in model1.parameters())
    
    print(f"  输入: {x.shape}")
    print(f"  输出: {desc1.shape}")
    print(f"  参数: {params1/1e6:.2f}M")
    print(f"  架构: ResNet → Lifting → MixVPR")
    
    # ═══════════════════════════════════════════
    # 配置2: 2个基本ResNet块
    # ═══════════════════════════════════════════
    print("\n【配置2】2个等变ResNet基本块")
    print("-"*70)
    model2 = create_improved_hybrid_model(
        backbone_arch='resnet50',
        pretrained=True,
        lifting_channels=512,
        rotation_order=8,
        num_residual_blocks=2,  # ← 2个ResNet块
        use_bottleneck=False,    # 基本块
    )
    
    desc2 = model2(x)
    params2 = sum(p.numel() for p in model2.parameters())
    
    print(f"  输入: {x.shape}")
    print(f"  输出: {desc2.shape}")
    print(f"  参数: {params2/1e6:.2f}M")
    print(f"  架构: ResNet → Lifting → ResBlock×2 → MixVPR")
    
    # ═══════════════════════════════════════════
    # 配置3: 2个Bottleneck ResNet块
    # ═══════════════════════════════════════════
    print("\n【配置3】2个等变ResNet Bottleneck块")
    print("-"*70)
    model3 = create_improved_hybrid_model(
        backbone_arch='resnet50',
        pretrained=True,
        lifting_channels=512,
        rotation_order=8,
        num_residual_blocks=2,  # ← 2个ResNet块
        use_bottleneck=True,     # ← Bottleneck设计
    )
    
    desc3 = model3(x)
    params3 = sum(p.numel() for p in model3.parameters())
    
    print(f"  输入: {x.shape}")
    print(f"  输出: {desc3.shape}")
    print(f"  参数: {params3/1e6:.2f}M")
    print(f"  架构: ResNet → Lifting → Bottleneck×2 → MixVPR")
    
    # 对比
    print("\n" + "="*70)
    print("  配置对比")
    print("="*70)
    print(f"\n配置1 (无ResBlock):      {params1/1e6:.2f}M参数, 最快")
    print(f"配置2 (基本块×2):        {params2/1e6:.2f}M参数, 平衡")
    print(f"配置3 (Bottleneck×2):    {params3/1e6:.2f}M参数, 最优")
    
    print("\n推荐:")
    print("  - 快速验证: 配置2（基本块）")
    print("  - 追求性能: 配置3（Bottleneck）")
    print("  - 极速: 配置1（无ResBlock）")
    print("="*70)
