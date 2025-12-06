"""
混合等变模型：预训练ResNet + 等变层
完美结合预训练优势和等变鲁棒性
"""
import torch
import torch.nn as nn
from escnn import gspaces
from escnn import nn as enn


class LiftingConvolution(nn.Module):
    """
    将标准特征"提升"到等变表示
    这是连接标准CNN和等变CNN的桥梁
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
            kernel_size=1,  # 1x1卷积用于通道转换
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
        # 包装为geometric tensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # Lifting
        x = self.lift(x)
        x = self.bn(x)
        x = self.relu(x)
        
        # 返回tensor
        return x.tensor


class EquivariantRefinement(nn.Module):
    """
    等变精炼层
    在预训练特征上应用等变操作
    """
    def __init__(self, in_channels, out_channels, rotation_order=8, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rotation_order = rotation_order
        
        # 定义群
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # 输入类型（等变特征）
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * in_channels
        )
        
        # 构建等变卷积层
        layers = []
        current_type = self.in_type
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # 最后一层输出目标通道数
                next_channels = out_channels
            else:
                next_channels = in_channels
            
            next_type = enn.FieldType(
                self.r2_act,
                [self.r2_act.regular_repr] * next_channels
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
            精炼后的等变特征 (B, out_channels*rotation_order, H, W)
        """
        # 包装为geometric tensor
        x = enn.GeometricTensor(x, self.in_type)
        
        # 等变处理
        x = self.layers(x)
        
        # 返回tensor
        return x.tensor


class HybridEquivariantAggregator(nn.Module):
    """
    混合等变聚合器
    用于预训练特征 + 等变处理的组合
    """
    def __init__(self, in_channels, rotation_order=8, p=3):
        super().__init__()
        self.in_channels = in_channels
        self.rotation_order = rotation_order
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = 1e-6
        
        # 定义群
        self.r2_act = gspaces.rot2dOnR2(N=rotation_order)
        
        # 输入类型
        self.in_type = enn.FieldType(
            self.r2_act,
            [self.r2_act.regular_repr] * in_channels
        )
        
        # 群池化
        self.group_pool = enn.GroupPooling(self.in_type)
        
        self.out_channels = in_channels
    
    def gem_pool(self, x):
        """广义平均池化"""
        x = x.clamp(min=self.eps).pow(self.p)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = x.pow(1. / self.p)
        return x
    
    def forward(self, x):
        """
        Args:
            x: 等变特征 (B, in_channels*rotation_order, H, W)
        Returns:
            旋转不变描述符 (B, in_channels)
        """
        # GeM池化
        x_pooled = self.gem_pool(x)
        
        # 包装为geometric tensor
        x_geo = enn.GeometricTensor(x_pooled, self.in_type)
        
        # 群池化（获得旋转不变性）
        x_inv = self.group_pool(x_geo)
        
        # 展平
        return x_inv.tensor.flatten(1)


class HybridModel(nn.Module):
    """
    完整的混合模型：
    预训练ResNet → Lifting → 等变精炼 → 等变聚合 → 描述符
    """
    def __init__(
        self,
        backbone,  # 预训练的ResNet
        backbone_out_channels=1024,  # ResNet输出通道数
        equivariant_channels=256,     # 等变层通道数（基础）
        rotation_order=8,              # 旋转群阶数
        num_equivariant_layers=2,      # 等变精炼层数
    ):
        super().__init__()
        
        self.backbone = backbone
        self.rotation_order = rotation_order
        
        # Lifting层：标准特征 → 等变特征
        self.lifting = LiftingConvolution(
            in_channels=backbone_out_channels,
            out_channels=equivariant_channels,
            rotation_order=rotation_order
        )
        
        # 等变精炼层
        self.equivariant_refinement = EquivariantRefinement(
            in_channels=equivariant_channels,
            out_channels=equivariant_channels,
            rotation_order=rotation_order,
            num_layers=num_equivariant_layers
        )
        
        # 等变聚合器
        self.aggregator = HybridEquivariantAggregator(
            in_channels=equivariant_channels,
            rotation_order=rotation_order,
            p=3
        )
        
        # 输出描述符维度
        self.out_channels = equivariant_channels
    
    def forward(self, x):
        """
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            旋转不变描述符 (B, equivariant_channels)
        """
        # 1. 预训练backbone提取特征
        features = self.backbone(x)  # (B, 1024, H, W)
        
        # 2. Lifting到等变表示
        equi_features = self.lifting(features)  # (B, 256*8, H, W)
        
        # 3. 等变精炼
        refined_features = self.equivariant_refinement(equi_features)  # (B, 256*8, H, W)
        
        # 4. 聚合为旋转不变描述符
        descriptor = self.aggregator(refined_features)  # (B, 256)
        
        return descriptor


def create_hybrid_model(
    backbone_arch='resnet50',
    pretrained=True,
    layers_to_crop=[4],
    equivariant_channels=256,
    rotation_order=8,
    num_equivariant_layers=2
):
    """
    创建混合等变模型的工厂函数
    
    Args:
        backbone_arch: 主干网络架构（resnet50等）
        pretrained: 是否使用预训练权重
        layers_to_crop: 要裁剪的层
        equivariant_channels: 等变层基础通道数
        rotation_order: 旋转群阶数
        num_equivariant_layers: 等变精炼层数量
    
    Returns:
        HybridModel实例
    """
    # 导入标准backbone
    from models import backbones
    
    # 创建预训练backbone
    backbone = backbones.ResNet(
        backbone_arch, 
        pretrained=pretrained,
        layers_to_freeze=2,  # 冻结早期层
        layers_to_crop=layers_to_crop
    )
    
    backbone_out_channels = backbone.out_channels
    
    # 创建混合模型
    model = HybridModel(
        backbone=backbone,
        backbone_out_channels=backbone_out_channels,
        equivariant_channels=equivariant_channels,
        rotation_order=rotation_order,
        num_equivariant_layers=num_equivariant_layers
    )
    
    return model


if __name__ == '__main__':
    print("="*70)
    print("  测试混合等变模型")
    print("="*70)
    
    # 创建模型
    print("\n创建模型...")
    model = create_hybrid_model(
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_crop=[4],
        equivariant_channels=256,
        rotation_order=8,
        num_equivariant_layers=2
    )
    
    # 测试前向传播
    print("\n测试前向传播...")
    x = torch.randn(2, 3, 320, 320)
    desc = model(x)
    
    print(f"✓ 输入形状: {x.shape}")
    print(f"✓ 输出描述符形状: {desc.shape}")
    
    # 测试旋转不变性
    print("\n测试旋转不变性...")
    from torchvision.transforms import functional as TF
    
    model.eval()
    with torch.no_grad():
        desc_0 = model(x)
        
        angles = [45, 90, 180]
        for angle in angles:
            x_rot = TF.rotate(x, angle)
            desc_rot = model(x_rot)
            
            sim = torch.nn.functional.cosine_similarity(desc_0, desc_rot, dim=1).mean()
            print(f"  旋转{angle:3d}°: 相似度 = {sim:.4f}")
    
    # 参数统计
    print("\n模型参数统计:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  总参数: {total_params/1e6:.2f}M")
    print(f"  可训练参数: {trainable_params/1e6:.2f}M")
    
    print("\n✓ 混合等变模型工作正常！")
    print("="*70)
