"""
时序聚合模块
放在 models/temporal_aggregators.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTemporalAggregation(nn.Module):
    """
    简单时序聚合 - 零训练成本
    
    使用方法：
        aggregator = SimpleTemporalAggregation('mean')
        descriptors = model(images)  # [B*T, D]
        descriptors = descriptors.view(B, T, D)
        aggregated = aggregator(descriptors)  # [B, D]
    """
    def __init__(self, aggregation='mean'):
        super().__init__()
        self.aggregation = aggregation
        assert aggregation in ['mean', 'max', 'l2'], \
            f"aggregation must be 'mean', 'max', or 'l2', got {aggregation}"
    
    def forward(self, descriptors):
        """
        Args:
            descriptors: [B, T, D] 批次时序描述符
        Returns:
            [B, D] 聚合后的描述符
        """
        if self.aggregation == 'mean':
            return descriptors.mean(dim=1)
        elif self.aggregation == 'max':
            return descriptors.max(dim=1)[0]
        elif self.aggregation == 'l2':
            # L2归一化后平均（推荐）
            descriptors = F.normalize(descriptors, p=2, dim=2)
            aggregated = descriptors.mean(dim=1)
            return F.normalize(aggregated, p=2, dim=1)


class LearnableTemporalAggregation(nn.Module):
    """
    可学习时序聚合 - 需要训练
    
    通过注意力机制学习每帧的权重
    """
    def __init__(self, descriptor_dim=4096, hidden_dim=512, dropout=0.1):
        super().__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, descriptors):
        """
        Args:
            descriptors: [B, T, D]
        Returns:
            [B, D]
        """
        # 计算每帧的注意力权重
        attention_scores = self.attention_net(descriptors)  # [B, T, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, T, 1]
        
        # 加权求和
        aggregated = (descriptors * attention_weights).sum(dim=1)  # [B, D]
        
        return aggregated


class TemporalTransformerAggregation(nn.Module):
    """
    Transformer时序聚合 - 最强版本
    
    使用Transformer建模时序依赖关系
    """
    def __init__(self, d_model=4096, nhead=8, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=20)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # CLS token (like BERT)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, descriptors):
        """
        Args:
            descriptors: [B, T, D]
        Returns:
            [B, D]
        """
        B, T, D = descriptors.shape
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        descriptors = torch.cat([cls_tokens, descriptors], dim=1)  # [B, T+1, D]
        
        # 位置编码
        descriptors = self.pos_encoder(descriptors)
        
        # Transformer编码
        encoded = self.transformer(descriptors)  # [B, T+1, D]
        
        # 取CLS token作为聚合结果
        output = encoded[:, 0]  # [B, D]
        
        return output


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=20):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                            (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, D]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ════════════════════════════════════════════════════════════
# 工厂函数
# ════════════════════════════════════════════════════════════

def create_temporal_aggregator(
    temporal_type='simple',
    descriptor_dim=4096,
    **kwargs
):
    """
    创建时序聚合器
    
    Args:
        temporal_type: 'simple', 'learnable', 'transformer', 'none'
        descriptor_dim: 描述符维度
        **kwargs: 额外参数
    
    Returns:
        aggregator or None
    """
    if temporal_type == 'none' or temporal_type is None:
        return None
    elif temporal_type == 'simple':
        aggregation_method = kwargs.get('aggregation', 'l2')
        return SimpleTemporalAggregation(aggregation=aggregation_method)
    elif temporal_type == 'learnable':
        hidden_dim = kwargs.get('hidden_dim', 512)
        dropout = kwargs.get('dropout', 0.1)
        return LearnableTemporalAggregation(
            descriptor_dim=descriptor_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
    elif temporal_type == 'transformer':
        nhead = kwargs.get('nhead', 8)
        num_layers = kwargs.get('num_layers', 2)
        dropout = kwargs.get('dropout', 0.1)
        return TemporalTransformerAggregation(
            d_model=descriptor_dim,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown temporal_type: {temporal_type}")


if __name__ == '__main__':
    print("="*70)
    print("  测试时序聚合模块")
    print("="*70)
    
    # 模拟数据
    B, T, D = 4, 5, 4096
    descriptors = torch.randn(B, T, D)
    
    # 测试各种聚合方式
    print(f"\n输入: {descriptors.shape}")
    
    # 1. 简单聚合
    agg1 = SimpleTemporalAggregation('mean')
    out1 = agg1(descriptors)
    print(f"\n【简单聚合-mean】")
    print(f"  输出: {out1.shape}")
    print(f"  参数: 0")
    
    # 2. 可学习聚合
    agg2 = LearnableTemporalAggregation(descriptor_dim=D)
    out2 = agg2(descriptors)
    params2 = sum(p.numel() for p in agg2.parameters())
    print(f"\n【可学习聚合】")
    print(f"  输出: {out2.shape}")
    print(f"  参数: {params2/1e6:.2f}M")
    
    # 3. Transformer聚合
    agg3 = TemporalTransformerAggregation(d_model=D)
    out3 = agg3(descriptors)
    params3 = sum(p.numel() for p in agg3.parameters())
    print(f"\n【Transformer聚合】")
    print(f"  输出: {out3.shape}")
    print(f"  参数: {params3/1e6:.2f}M")
    
    print("\n" + "="*70)
    print("  推荐使用顺序：")
    print("  1. 先用SimpleTemporalAggregation（零训练）")
    print("  2. 如果有效，再训练LearnableTemporalAggregation")
    print("  3. 追求极致，使用TemporalTransformerAggregation")
    print("="*70)
