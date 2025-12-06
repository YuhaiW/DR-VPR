import numpy as np
from models import aggregators
from models import backbones

# Import equivariant modules (will fail gracefully if escnn not installed)
try:
    from models.equivariant_backbone import EquivariantResNet, EquivariantEfficientNet
    from models.aggregators.equivariant_aggregator import (
        EquivariantGeM, EquivariantMixVPR, EquivariantConvAP
    )
    EQUIVARIANT_AVAILABLE = True
except ImportError:
    EQUIVARIANT_AVAILABLE = False
    print("Warning: escnn not installed. Equivariant networks will not be available.")
    print("Install with: pip install escnn")


def get_backbone(backbone_arch='resnet50',
                 pretrained=True,
                 layers_to_freeze=2,
                 layers_to_crop=[],
                 equivariant=False,
                 rotation_order=8):
    """Helper function that returns the backbone given its name

    Args:
        backbone_arch (str, optional): Backbone architecture name. Defaults to 'resnet50'.
        pretrained (bool, optional): Use pretrained weights. Defaults to True.
        layers_to_freeze (int, optional): Number of layers to freeze. Defaults to 2.
        layers_to_crop (list, optional): Layers to crop (for ResNet). Defaults to [].
        equivariant (bool, optional): Use equivariant backbone. Defaults to False.
        rotation_order (int, optional): Rotation group order for equivariant nets (C_n). Defaults to 8.

    Returns:
        nn.Module: the backbone as a nn.Model object
    """
    
    # Equivariant backbones
    if equivariant:
        if not EQUIVARIANT_AVAILABLE:
            raise ImportError(
                "Equivariant networks require escnn. Install with: pip install escnn"
            )
        
        if 'resnet' in backbone_arch.lower() or 'equivariant_resnet' in backbone_arch.lower():
            # Map ResNet variants to channel configurations
            if 'resnet18' in backbone_arch.lower():
                channels = [32, 64, 128, 256]
                blocks = [2, 2, 2, 2]
            elif 'resnet34' in backbone_arch.lower():
                channels = [32, 64, 128, 256]
                blocks = [3, 4, 6, 3]
            else:  # resnet50 and larger
                channels = [64, 128, 256, 512]
                blocks = [3, 4, 6, 3]
            
            return EquivariantResNet(
                rotation_order=rotation_order,
                pretrained=False,  # Equivariant nets trained from scratch
                layers_to_freeze=layers_to_freeze,
                initial_channels=channels[0],
                block_channels=channels,
                blocks_per_layer=blocks
            )
        
        elif 'efficient' in backbone_arch.lower() or 'equivariant_efficient' in backbone_arch.lower():
            # Width multiplier based on variant
            if 'b0' in backbone_arch.lower():
                width_mult = 1.0
            elif 'b1' in backbone_arch.lower():
                width_mult = 1.0
            elif 'b2' in backbone_arch.lower():
                width_mult = 1.1
            else:
                width_mult = 1.0
            
            return EquivariantEfficientNet(
                rotation_order=rotation_order,
                pretrained=False,
                layers_to_freeze=layers_to_freeze,
                width_mult=width_mult
            )
        
        else:
            raise ValueError(
                f"Equivariant version of {backbone_arch} not implemented. "
                f"Available: equivariant_resnet, equivariant_efficient"
            )
    
    # Standard (non-equivariant) backbones
    if 'resnet' in backbone_arch.lower():
        return backbones.ResNet(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)

    elif 'efficient' in backbone_arch.lower():
        if '_b' in backbone_arch.lower():
            return backbones.EfficientNet(backbone_arch, pretrained, layers_to_freeze+2)
        else:
            return backbones.EfficientNet(model_name='efficientnet_b0',
                                          pretrained=pretrained, 
                                          layers_to_freeze=layers_to_freeze)
            
    elif 'swin' in backbone_arch.lower():
        return backbones.Swin(model_name='swinv2_base_window12to16_192to256_22kft1k', 
                              pretrained=pretrained, 
                              layers_to_freeze=layers_to_freeze)
    
    else:
        raise ValueError(f"Backbone {backbone_arch} not recognized")


def get_aggregator(agg_arch='ConvAP', agg_config={}, equivariant=False, rotation_order=8):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.
        equivariant (bool, optional): Use equivariant aggregator. Defaults to False.
        rotation_order (int, optional): Rotation group order for equivariant aggregators. Defaults to 8.

    Returns:
        nn.Module: the aggregation layer
    """
    
    # Equivariant aggregators
    if equivariant:
        if not EQUIVARIANT_AVAILABLE:
            raise ImportError(
                "Equivariant aggregators require escnn. Install with: pip install escnn"
            )
        
        # 复制config避免修改原始dict
        agg_config = agg_config.copy()
        
        # Add rotation_order to config if not present
        if 'rotation_order' not in agg_config:
            agg_config['rotation_order'] = rotation_order
        
        # 重要：等变聚合器的in_channels是基础通道数，不包括群维度
        # backbone会输出 in_channels * rotation_order 的通道数
        
        if 'gem' in agg_arch.lower() or 'equivariant_gem' in agg_arch.lower():
            assert 'in_channels' in agg_config, "in_channels required for EquivariantGeM"
            if 'p' not in agg_config:
                agg_config['p'] = 3
            return EquivariantGeM(**agg_config)
        
        elif 'mixvpr' in agg_arch.lower() or 'equivariant_mixvpr' in agg_arch.lower():
            assert 'in_channels' in agg_config
            assert 'out_channels' in agg_config
            assert 'in_h' in agg_config
            assert 'in_w' in agg_config
            if 'mix_depth' not in agg_config:
                agg_config['mix_depth'] = 4
            if 'out_rows' not in agg_config:
                agg_config['out_rows'] = 4
            return EquivariantMixVPR(**agg_config)
        
        elif 'convap' in agg_arch.lower() or 'equivariant_convap' in agg_arch.lower():
            assert 'in_channels' in agg_config
            if 'out_channels' not in agg_config:
                agg_config['out_channels'] = agg_config['in_channels']
            return EquivariantConvAP(**agg_config)
        
        else:
            raise ValueError(
                f"Equivariant version of {agg_arch} not implemented. "
                f"Available: equivariant_gem, equivariant_mixvpr, equivariant_convap"
            )
    
    # Standard (non-equivariant) aggregators
    if 'cosplace' in agg_arch.lower():
        assert 'in_dim' in agg_config
        assert 'out_dim' in agg_config
        return aggregators.CosPlace(**agg_config)

    elif 'gem' in agg_arch.lower():
        if agg_config == {}:
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config
        return aggregators.GeMPool(**agg_config)
    
    elif 'convap' in agg_arch.lower():
        assert 'in_channels' in agg_config
        return aggregators.ConvAP(**agg_config)
    
    elif 'mixvpr' in agg_arch.lower():
        assert 'in_channels' in agg_config
        assert 'out_channels' in agg_config
        assert 'in_h' in agg_config
        assert 'in_w' in agg_config
        assert 'mix_depth' in agg_config
        return aggregators.MixVPR(**agg_config)
    
    else:
        raise ValueError(f"Aggregator {agg_arch} not recognized")