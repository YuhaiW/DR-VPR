"""
Inference-latency benchmark: all baselines + DR-VPR.

For each model, measures single-image forward-pass latency (ms) with fp16 on
GPU, using 30 warmup + 100 timed passes.
"""
import os
import time

import torch

os.chdir('/home/yuhai/project/DR-VPR')
DEVICE = 'cuda:0'
WARMUP = 30
N_ITER = 100


@torch.no_grad()
def bench(model, input_size, name, half=False):
    # fp32 for fair comparison (e2cnn's equivariant ops don't fully support fp16)
    model.eval().to(DEVICE)
    if half:
        model.half()
    dtype = torch.float16 if half else torch.float32
    x = torch.randn(1, 3, *input_size, device=DEVICE, dtype=dtype)

    # Warmup
    for _ in range(WARMUP):
        _ = model(x)
    torch.cuda.synchronize()

    # Timed
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        _ = model(x)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    per_img = (t1 - t0) / N_ITER * 1000  # ms
    fps = 1000.0 / per_img
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{name:22s}  input={input_size[0]}×{input_size[1]}  "
          f"params={n_params/1e6:6.2f}M  latency={per_img:5.2f} ms  "
          f"({fps:.1f} FPS)")
    del model
    torch.cuda.empty_cache()


# -------- DR-VPR (Equi-BoQ, ours) --------
from train_fusion import VPRModel
dr = VPRModel(
    backbone_arch='resnet50', pretrained=False, layers_to_freeze=2, layers_to_crop=[4],
    agg_arch='boq',
    agg_config={'in_channels': 1024, 'proj_channels': 512,
                'num_queries': 64, 'num_layers': 2, 'row_dim': 32},
    use_dual_branch=True, equi_orientation=8, equi_layers=[2, 2, 2, 2],
    equi_channels=[64, 128, 256, 512], equi_out_dim=1024,
    fusion_method='concat', use_projection=False,
    lr=1e-3, optimizer='adamw', weight_decay=1e-4, warmpup_steps=300,
    milestones=[8, 14], lr_mult=0.3,
    loss_name='MultiSimilarityLoss', miner_name='MultiSimilarityMiner', miner_margin=0.1,
)
bench(dr, (320, 320), 'DR-VPR (ours)')

# -------- BoQ --------
boq = torch.hub.load("amaralibey/bag-of-queries", "get_trained_boq",
                     backbone_name="resnet50", output_dim=16384, trust_repo=True)
bench(boq, (322, 322), 'BoQ (resnet50)')

# -------- SALAD --------
salad = torch.hub.load("serizba/salad", "dinov2_salad", trust_repo=True)
bench(salad, (322, 322), 'SALAD (DINOv2)')

# -------- CricaVPR --------
crica = torch.hub.load('Lu-Feng/CricaVPR', 'trained_model', trust_repo=True)
bench(crica, (224, 224), 'CricaVPR')

# -------- CosPlace --------
cos = torch.hub.load("gmberton/CosPlace", "get_trained_model",
                     backbone="ResNet50", fc_output_dim=2048, trust_repo=True)
bench(cos, (224, 224), 'CosPlace (R50)')

# -------- DINOv2 ViT-B/14 (backbone only; AnyLoc-VLAD adds a small aggregator) --------
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', trust_repo=True)
bench(dinov2, (224, 224), 'DINOv2 ViT-B/14')

# -------- MixVPR standalone (ResNet50 layer1-3 + MixVPR) --------
import torchvision
from models.aggregators.mixvpr import MixVPR


class MixVPRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet50(pretrained=False)
        self.conv1, self.bn1, self.relu, self.maxpool = r.conv1, r.bn1, r.relu, r.maxpool
        self.layer1, self.layer2, self.layer3 = r.layer1, r.layer2, r.layer3
        self.agg = MixVPR(in_channels=1024, in_h=20, in_w=20, out_channels=1024,
                          mix_depth=4, mlp_ratio=1, out_rows=4)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.agg(x)


bench(MixVPRModel(), (320, 320), 'MixVPR (standalone)')
