"""
Evaluate baseline VPR methods on ConPR and ConSLAM datasets.
Supports: SALAD, CricaVPR, BoQ, CosPlace, MixVPR (standalone), DINOv2

Usage:
    python eval_baselines.py --method salad --dataset conpr
    python eval_baselines.py --method all --dataset all --seeds 1 42 123
    python eval_baselines.py --method cosplace --dataset conslam
"""

import argparse
import sys
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend, safe for DataLoader workers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from PIL import Image

# Reuse existing evaluation functions
from conpr_eval_dataset_rot import evaluateResults as conpr_evaluate
from Conslam_dataset_rot import evaluateResults as conslam_evaluate


# ============================================================
# Dataset classes (adapted for different input sizes)
# ============================================================

class ConPRInferDataset(torch.utils.data.Dataset):
    """ConPR dataset with configurable image size"""
    def __init__(self, seq, dataset_path='./datasets/ConPR/', img_size=(320, 320)):
        import cv2
        self.cv2 = cv2
        self.seq_name = seq
        imgs_p = os.listdir(dataset_path + seq + '/Camera_matched/')
        imgs_p.sort()
        self.imgs_path = [dataset_path + seq + '/Camera_matched/' + i for i in imgs_p]
        self.poses = np.loadtxt(dataset_path + 'poses/' + seq + '.txt')
        self.transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img = self.cv2.imread(self.imgs_path[index], 0)
        img_rgb = self.cv2.cvtColor(img, self.cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        return self.transform(img_pil), index

    def __len__(self):
        return len(self.imgs_path)


class ConSLAMInferDataset(torch.utils.data.Dataset):
    """ConSLAM dataset with configurable image size"""
    def __init__(self, seq, dataset_path='./datasets/ConSLAM/', img_size=(320, 320)):
        import cv2
        self.cv2 = cv2
        self.seq_name = seq
        img_dir = os.path.join(dataset_path, seq, 'selected_images')
        imgs_p = sorted(os.listdir(img_dir))
        self.imgs_path = [os.path.join(img_dir, i) for i in imgs_p]
        pose_file = os.path.join(dataset_path, 'poses', f'{seq}.txt')
        self.poses = np.loadtxt(pose_file)
        min_len = min(len(self.imgs_path), len(self.poses))
        self.imgs_path = self.imgs_path[:min_len]
        self.poses = self.poses[:min_len]
        self.transform = T.Compose([
            T.Resize(img_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print(f"Loaded {seq}: {len(self.imgs_path)} image-pose pairs")

    def __getitem__(self, index):
        img = self.cv2.imread(self.imgs_path[index])
        img_rgb = self.cv2.cvtColor(img, self.cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        return self.transform(img_pil), index

    def __len__(self):
        return len(self.imgs_path)


# ============================================================
# Model loaders
# ============================================================

def load_salad(device):
    """Load SALAD (DINOv2 + SALAD aggregator)"""
    print("Loading SALAD model...")
    model = torch.hub.load("serizba/salad", "dinov2_salad")
    model = model.eval().to(device)
    img_size = (322, 322)  # Must be divisible by 14
    desc_dim = 8448
    print(f"  SALAD loaded: descriptor dim={desc_dim}, input={img_size}")
    return model, img_size, desc_dim


def load_cricavpr(device):
    """Load CricaVPR"""
    print("Loading CricaVPR model...")
    sys.path.insert(0, './baselines/CricaVPR')
    import network

    model = network.CricaVPRNet()
    model = nn.DataParallel(model)
    state_dict = torch.hub.load_state_dict_from_url(
        'https://github.com/Lu-Feng/CricaVPR/releases/download/v1.0/CricaVPR.pth',
        map_location='cpu'
    )["model_state_dict"]
    model.load_state_dict(state_dict)
    model = model.eval().to(device)
    img_size = (224, 224)
    desc_dim = 10752
    print(f"  CricaVPR loaded: descriptor dim={desc_dim}, input={img_size}")
    sys.path.pop(0)
    return model, img_size, desc_dim


def load_boq(device, backbone='dinov2'):
    """Load Bag-of-Queries"""
    print(f"Loading BoQ ({backbone}) model...")
    if backbone == 'dinov2':
        model = torch.hub.load(
            "amaralibey/bag-of-queries", "get_trained_boq",
            backbone_name="dinov2", output_dim=12288
        )
        img_size = (322, 322)
        desc_dim = 12288
    else:
        model = torch.hub.load(
            "amaralibey/bag-of-queries", "get_trained_boq",
            backbone_name="resnet50", output_dim=16384
        )
        img_size = (384, 384)
        desc_dim = 16384

    # Env var override for protocol-matching experiments (e.g. BOQ_IMG_SIZE=320
    # to compare at same resolution as DR-VPR eval_rerank.py).
    _override = os.environ.get('BOQ_IMG_SIZE')
    if _override is not None:
        try:
            s = int(_override)
            img_size = (s, s)
            print(f"  [override] BOQ_IMG_SIZE={s} → using input {img_size}")
        except ValueError:
            print(f"  [warn] BOQ_IMG_SIZE={_override!r} ignored (not int)")

    model = model.eval().to(device)
    print(f"  BoQ loaded: descriptor dim={desc_dim}, input={img_size}")
    return model, img_size, desc_dim


def load_cosplace(device):
    """Load CosPlace (ResNet50 + GeM + FC)"""
    print("Loading CosPlace model...")
    model = torch.hub.load("gmberton/cosplace", "get_trained_model",
                           backbone="ResNet50", fc_output_dim=2048)
    model = model.eval().to(device)
    img_size = (224, 224)
    desc_dim = 2048
    print(f"  CosPlace loaded: descriptor dim={desc_dim}, input={img_size}")
    return model, img_size, desc_dim


def load_mixvpr(device):
    """Load standalone MixVPR (ResNet50 + MixVPR aggregator).
    Build directly from models/helper_1 to avoid importing train_fusion
    (which transitively imports dataloaders/ and collides with torch.hub-cached
    BoQ dataloader module on Python 3.9)."""
    print("Loading MixVPR (standalone) model...")

    # torch.hub loads of other baselines (SALAD/BoQ) insert their cached repo
    # paths at the front of sys.path, which shadows our local `models/`.
    # Force the project root to win before importing helper_1.
    import sys as _sys
    import importlib as _importlib
    _proj = os.path.dirname(os.path.abspath(__file__))
    if _sys.path[0] != _proj:
        _sys.path.insert(0, _proj)
    # Evict any cached wrong-path models module
    for _k in [k for k in list(_sys.modules) if k == "models" or k.startswith("models.")]:
        del _sys.modules[_k]
    helper = _importlib.import_module("models.helper_1")

    class _MixVPRNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = helper.get_backbone(
                'resnet50', pretrained=False,
                layers_to_freeze=2, layers_to_crop=[4]
            )
            self.aggregator = helper.get_aggregator('MixVPR', {
                'in_channels': 1024,
                'in_h': 20,
                'in_w': 20,
                'out_channels': 1024,
                'mix_depth': 4,
                'mlp_ratio': 1,
                'out_rows': 4,
            })

        def forward(self, x):
            return self.aggregator(self.backbone(x))

    model = _MixVPRNet()

    # Load pretrained weights
    ckpt_path = './baselines/MixVPR/resnet50_MixVPR_4096_channels1024_rows4.ckpt'
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"MixVPR checkpoint not found at {ckpt_path}. "
            "Please download from: https://github.com/amaralibey/MixVPR/releases "
            "and place it at ./baselines/MixVPR/"
        )
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    # Upstream MixVPR wraps the resnet under `backbone.model.*`; our helper
    # places it at `backbone.*`. Rewrite keys to match.
    state_dict = {
        k.replace('backbone.model.', 'backbone.', 1) if k.startswith('backbone.model.') else k: v
        for k, v in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Warning: Unexpected keys: {len(unexpected)}")

    model = model.eval().to(device)
    img_size = (320, 320)
    desc_dim = 4096
    print(f"  MixVPR loaded: descriptor dim={desc_dim}, input={img_size}")
    return model, img_size, desc_dim


def load_dinov2(device):
    """Load DINOv2 ViT-B/14 with GeM pooling for VPR"""
    print("Loading DINOv2 (ViT-B/14 + GeM) model...")

    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')

    class DINOv2GeM(nn.Module):
        """DINOv2 with GeM pooling on patch tokens"""
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.gem_p = nn.Parameter(torch.ones(1) * 3.0)

        def forward(self, x):
            features = self.backbone.forward_features(x)
            # DINOv2 returns dict with 'x_norm_patchtokens'
            if isinstance(features, dict):
                patch_tokens = features['x_norm_patchtokens']  # (B, N, 768)
            else:
                # Fallback: skip CLS token
                patch_tokens = features[:, 1:, :]
            # GeM pooling
            patch_tokens = patch_tokens.clamp(min=1e-6).pow(self.gem_p)
            desc = patch_tokens.mean(dim=1).pow(1.0 / self.gem_p)
            desc = F.normalize(desc, p=2, dim=-1)
            return desc

    model = DINOv2GeM(backbone).eval().to(device)
    img_size = (322, 322)  # Must be divisible by 14
    desc_dim = 768
    print(f"  DINOv2 loaded: descriptor dim={desc_dim}, input={img_size}")
    return model, img_size, desc_dim


# ============================================================
# Feature extraction
# ============================================================

def extract_features(model, dataloader, device, method_name=''):
    """Extract features from a dataset"""
    model.eval()
    features_list = []
    print(f"  Extracting features from {len(dataloader.dataset)} images...")

    with torch.no_grad():
        for images, indices in tqdm(dataloader, desc=f"  {method_name}"):
            images = images.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                feats = model(images)

            # BoQ returns (descriptor, attentions)
            if isinstance(feats, tuple):
                feats = feats[0]

            features_list.append(feats.float().cpu().numpy())

    features = np.vstack(features_list)
    print(f"  Features shape: {features.shape}")
    return features


# ============================================================
# Evaluation runners
# ============================================================

CONPR_SEQUENCES = ['20230623', '20230531', '20230611', '20230627', '20230628',
                   '20230706', '20230717', '20230803', '20230809', '20230818']
CONSLAM_SEQUENCES = ['Sequence5', 'Sequence4']
YAW_THRESHOLD = 80.0


def eval_conpr(model, img_size, method_name, device, batch_size=16, num_workers=4):
    """Evaluate on ConPR dataset, returns dict with R1/R5/R10"""
    print(f"\n{'='*70}")
    print(f"Evaluating {method_name} on ConPR")
    print(f"{'='*70}")

    datasets = []
    dataloaders = []
    for seq in CONPR_SEQUENCES:
        ds = ConPRInferDataset(seq, dataset_path='./datasets/ConPR/', img_size=img_size)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
        datasets.append(ds)
        dataloaders.append(dl)
        print(f"  Sequence {seq}: {len(ds)} images")

    # Extract features
    global_descs = []
    for i, (seq, dl) in enumerate(zip(CONPR_SEQUENCES, dataloaders)):
        print(f"\n  Processing {seq} ({i+1}/{len(dataloaders)})...")
        feats = extract_features(model, dl, device, method_name)
        global_descs.append(feats)

    # Evaluate - now returns list of dicts with R1/R5/R10
    # θ=0° for ConPR (bird's-eye aerial views, no systematic query trajectory rotation).
    # Matches test_conpr.py default. Commit db01de5 (2026-04-16) accidentally flipped
    # this to 15° which crushes all baselines to ~3% R@1; fixed 2026-04-17.
    recalls = conpr_evaluate(
        global_descs, datasets,
        theta_degrees=0.0, offset=[0.0, 0.0],
        yaw_threshold=YAW_THRESHOLD,
        method_name=method_name,
    )

    # Compute averages
    avg_r1 = np.mean([r['R1'] for r in recalls])
    avg_r5 = np.mean([r['R5'] for r in recalls])
    avg_r10 = np.mean([r['R10'] for r in recalls])

    print(f"\n{'='*70}")
    print(f"{method_name} ConPR Results:")
    for i, seq in enumerate(CONPR_SEQUENCES[1:]):
        if i < len(recalls):
            r = recalls[i]
            print(f"  {seq}: R@1={r['R1']*100:.2f}%  R@5={r['R5']*100:.2f}%  R@10={r['R10']*100:.2f}%")
    print(f"  Average: R@1={avg_r1*100:.2f}%  R@5={avg_r5*100:.2f}%  R@10={avg_r10*100:.2f}%")
    print(f"{'='*70}")

    return recalls, {'R1': avg_r1, 'R5': avg_r5, 'R10': avg_r10}


def eval_conslam(model, img_size, method_name, device, batch_size=16, num_workers=4):
    """Evaluate on ConSLAM dataset, returns dict with R1/R5/R10"""
    print(f"\n{'='*70}")
    print(f"Evaluating {method_name} on ConSLAM")
    print(f"{'='*70}")

    datasets = []
    dataloaders = []
    for seq in CONSLAM_SEQUENCES:
        ds = ConSLAMInferDataset(seq, dataset_path='./datasets/ConSLAM/', img_size=img_size)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                       num_workers=num_workers, pin_memory=True)
        datasets.append(ds)
        dataloaders.append(dl)

    # Extract features
    global_descs = []
    for i, (seq, dl) in enumerate(zip(CONSLAM_SEQUENCES, dataloaders)):
        print(f"\n  Processing {seq} ({i+1}/{len(dataloaders)})...")
        feats = extract_features(model, dl, device, method_name)
        global_descs.append(feats)

    # Evaluate - now returns (list of dicts, failed_images_dict)
    recalls, failed_images = conslam_evaluate(
        global_descs, datasets,
        theta_degrees=15.0, offset=[0.0, 0.0],
        yaw_threshold=YAW_THRESHOLD
    )

    # Compute averages
    avg_r1 = np.mean([r['R1'] for r in recalls])
    avg_r5 = np.mean([r['R5'] for r in recalls])
    avg_r10 = np.mean([r['R10'] for r in recalls])

    print(f"\n{'='*70}")
    print(f"{method_name} ConSLAM Results:")
    for i, seq in enumerate(CONSLAM_SEQUENCES[1:]):
        if i < len(recalls):
            r = recalls[i]
            print(f"  {seq}: R@1={r['R1']*100:.2f}%  R@5={r['R5']*100:.2f}%  R@10={r['R10']*100:.2f}%")
    print(f"  Average: R@1={avg_r1*100:.2f}%  R@5={avg_r5*100:.2f}%  R@10={avg_r10*100:.2f}%")
    print(f"{'='*70}")

    return recalls, {'R1': avg_r1, 'R5': avg_r5, 'R10': avg_r10}


# ============================================================
# Main
# ============================================================

ALL_METHODS = ['salad', 'cricavpr', 'boq', 'boq_resnet50', 'cosplace', 'mixvpr', 'dinov2']

def main():
    parser = argparse.ArgumentParser(description='Evaluate baseline VPR methods')
    parser.add_argument('--method', type=str, default='all',
                       choices=ALL_METHODS + ['all'])
    parser.add_argument('--dataset', type=str, default='all',
                       choices=['conpr', 'conslam', 'all'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1, 42, 123],
                       help='Random seeds for reproducibility check')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    methods = ALL_METHODS if args.method == 'all' else [args.method]
    datasets_to_eval = ['conpr', 'conslam'] if args.dataset == 'all' else [args.dataset]

    # Results: {method: {seed: {dataset: {R1, R5, R10}}}}
    all_results = {}

    for method in methods:
        all_results[method] = {'seeds': {}}

        # Load model once per method (model is deterministic)
        if method == 'salad':
            model, img_size, desc_dim = load_salad(device)
        elif method == 'cricavpr':
            model, img_size, desc_dim = load_cricavpr(device)
        elif method == 'boq':
            model, img_size, desc_dim = load_boq(device, backbone='dinov2')
        elif method == 'boq_resnet50':
            model, img_size, desc_dim = load_boq(device, backbone='resnet50')
        elif method == 'cosplace':
            model, img_size, desc_dim = load_cosplace(device)
        elif method == 'mixvpr':
            model, img_size, desc_dim = load_mixvpr(device)
        elif method == 'dinov2':
            model, img_size, desc_dim = load_dinov2(device)

        all_results[method]['desc_dim'] = desc_dim

        for seed in args.seeds:
            print(f"\n--- Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)

            seed_results = {}

            for ds_name in datasets_to_eval:
                if ds_name == 'conpr':
                    recalls, avg = eval_conpr(model, img_size, method.upper(),
                                             device, args.batch_size, args.num_workers)
                    seed_results['conpr'] = {'recalls': recalls, 'avg': avg}
                elif ds_name == 'conslam':
                    recalls, avg = eval_conslam(model, img_size, method.upper(),
                                               device, args.batch_size, args.num_workers)
                    seed_results['conslam'] = {'recalls': recalls, 'avg': avg}

            all_results[method]['seeds'][seed] = seed_results

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

    # ============================================================
    # Compute mean ± std and print final summary
    # ============================================================
    print("\n" + "="*90)
    print("BASELINE EVALUATION SUMMARY (mean +/- std across seeds)")
    print("="*90)

    summary_lines = []
    summary_lines.append("Baseline VPR Evaluation Results (mean +/- std)")
    summary_lines.append("="*90)
    summary_lines.append(f"Seeds: {args.seeds}")
    summary_lines.append(f"Yaw threshold: {YAW_THRESHOLD} degrees")
    summary_lines.append("")

    for method in methods:
        mdata = all_results[method]
        desc_dim = mdata['desc_dim']
        header = f"\n{method.upper()} (descriptor dim={desc_dim}):"
        print(header)
        summary_lines.append(header)

        for ds_name in datasets_to_eval:
            # Collect per-seed averages
            r1_vals, r5_vals, r10_vals = [], [], []
            for seed in args.seeds:
                if ds_name in mdata['seeds'][seed]:
                    avg = mdata['seeds'][seed][ds_name]['avg']
                    r1_vals.append(avg['R1'])
                    r5_vals.append(avg['R5'])
                    r10_vals.append(avg['R10'])

            if r1_vals:
                r1_mean, r1_std = np.mean(r1_vals)*100, np.std(r1_vals)*100
                r5_mean, r5_std = np.mean(r5_vals)*100, np.std(r5_vals)*100
                r10_mean, r10_std = np.mean(r10_vals)*100, np.std(r10_vals)*100

                line = (f"  {ds_name:8s}: R@1={r1_mean:6.2f}+/-{r1_std:.2f}%  "
                       f"R@5={r5_mean:6.2f}+/-{r5_std:.2f}%  "
                       f"R@10={r10_mean:6.2f}+/-{r10_std:.2f}%")
                print(line)
                summary_lines.append(line)

                # Per-sequence breakdown (from first seed)
                first_seed = args.seeds[0]
                recalls = mdata['seeds'][first_seed][ds_name]['recalls']
                seqs = CONPR_SEQUENCES[1:] if ds_name == 'conpr' else CONSLAM_SEQUENCES[1:]
                for j, seq in enumerate(seqs):
                    if j < len(recalls):
                        r = recalls[j]
                        seq_line = f"    {seq}: R@1={r['R1']*100:.2f}%  R@5={r['R5']*100:.2f}%  R@10={r['R10']*100:.2f}%"
                        summary_lines.append(seq_line)

        summary_lines.append("")

    print("="*90)

    # Save results
    with open('baseline_results.txt', 'w') as f:
        f.write('\n'.join(summary_lines))
    print("\nResults saved to baseline_results.txt")


if __name__ == '__main__':
    main()
