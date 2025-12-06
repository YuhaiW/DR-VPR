"""
Modified helper.py to support dual-branch architecture
"""
import torch
import torchvision
import torch.nn as nn

# Import standard aggregators
try:
    from models.aggregators.cosplace import CosPlace
except ImportError:
    print("Warning: CosPlace not found")
    
try:
    from models.aggregators.convap import ConvAP
except ImportError:
    print("Warning: ConvAP not found")
    
try:
    from models.aggregators.mixvpr import MixVPR
except ImportError:
    print("Warning: MixVPR not found")
    
try:
    from models.aggregators.gem import GeMPool, GeM
except ImportError:
    print("Warning: GeM not found")

# Import new modules for dual-branch
from models.backbones.e2resnet_backbone import E2ResNetBackbone
from models.aggregators.dual_branch_aggregator import DualBranchAggregator


class ResNetBackbone(nn.Module):
    """
    Wrapper for ResNet that returns feature maps instead of flattened features
    """
    def __init__(self, model_name='resnet50', pretrained=True, layers_to_freeze=2, layers_to_crop=[]):
        super().__init__()
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            model = torchvision.models.resnet18(pretrained=pretrained)
        elif model_name == 'resnet50':
            model = torchvision.models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet101':
            model = torchvision.models.resnet101(pretrained=pretrained)
        elif model_name == 'resnet152':
            model = torchvision.models.resnet152(pretrained=pretrained)
        else:
            raise ValueError(f'Unsupported ResNet model: {model_name}')
        
        # Extract layers
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        # Crop layers if specified
        if 4 in layers_to_crop:
            self.layer4 = nn.Identity()
        if 3 in layers_to_crop:
            self.layer3 = nn.Identity()
        if 2 in layers_to_crop:
            self.layer2 = nn.Identity()
        
        # Freeze layers
        if layers_to_freeze >= 1:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
        
        if layers_to_freeze >= 2:
            for param in self.layer1.parameters():
                param.requires_grad = False
        
        if layers_to_freeze >= 3:
            for param in self.layer2.parameters():
                param.requires_grad = False
        
        if layers_to_freeze >= 4:
            for param in self.layer3.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass that returns 4D feature maps
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Return 4D feature maps (B, C, H, W)
        # NO flatten, NO avgpool, NO fc
        return x


def get_backbone(backbone_arch='resnet50', 
                 pretrained=True,
                 layers_to_freeze=2, 
                 layers_to_crop=[]):
    """Get standard backbone (Branch 1) - returns 4D feature maps"""
    
    if 'resnet' in backbone_arch.lower():
        # Use our custom wrapper
        backbone = ResNetBackbone(
            model_name=backbone_arch.lower(),
            pretrained=pretrained,
            layers_to_freeze=layers_to_freeze,
            layers_to_crop=layers_to_crop
        )
    else:
        raise ValueError(f'Backbone {backbone_arch} not implemented')
    
    return backbone


def get_equivariant_backbone(orientation=8, 
                             layers=[2, 2, 2, 2],
                             channels=[64, 128, 256, 512],
                             pretrained=False):
    """Get equivariant backbone (Branch 2)"""
    return E2ResNetBackbone(
        orientation=orientation,
        layers=layers,
        channels=channels,
        pretrained=pretrained
    )


def get_aggregator(agg_arch='ConvAP', agg_config={}):
    """Get aggregator for single branch"""
    
    if agg_arch.lower() == 'cosplace':
        assert 'in_dim' in agg_config and 'out_dim' in agg_config
        aggregator = CosPlace(**agg_config)
    
    elif agg_arch.lower() == 'gem':
        aggregator = GeMPool(**agg_config)
    
    elif agg_arch.lower() == 'convap':
        assert 'in_channels' in agg_config
        aggregator = ConvAP(**agg_config)
        
    elif agg_arch.lower() == 'mixvpr':
        assert 'in_channels' in agg_config
        aggregator = MixVPR(**agg_config)
        
    else:
        raise ValueError(f'Aggregator {agg_arch} not implemented')
    
    return aggregator


def get_dual_branch_aggregator(branch1_agg, branch1_out_dim,
                               branch2_in_channels, branch2_out_dim,
                               fusion_method='concat',
                               use_projection=False):
    """Get dual-branch aggregator"""
    return DualBranchAggregator(
        branch1_aggregator=branch1_agg,
        branch1_out_dim=branch1_out_dim,
        branch2_in_channels=branch2_in_channels,
        branch2_out_dim=branch2_out_dim,
        fusion_method=fusion_method,
        use_projection=use_projection
    )