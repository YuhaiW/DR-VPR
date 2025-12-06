"""
Test Dual-Branch VPR Model on ConSLAM dataset with proper image resizing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
from tqdm import tqdm

# Import from existing files
from Conslam_dataset_rot import InferDataset, evaluateResults
from train_fusion import VPRModel


def extract_features_conslam(model, dataloader, device='cuda'):
    """
    Extract features from ConSLAM dataset using Dual-Branch VPR
    """
    model.eval()
    features_list = []
    
    print(f"Extracting features from {len(dataloader.dataset)} images...")
    
    with torch.no_grad():
        for images, indices in tqdm(dataloader, desc="Feature extraction"):
            # Images are already resized to 320x320 and normalized
            images = images.to(device)
            
            # Extract features
            feats = model(images)
            
            # Move to CPU and store
            features_list.append(feats.cpu().numpy())
    
    # Concatenate all features
    features = np.vstack(features_list)
    print(f"Extracted features shape: {features.shape}")
    
    return features


def main():
    # ==================== Configuration ====================
    CHECKPOINT_PATH = '/home/user1/yuhai/project/MixVPR/LOGS/resnet50_DualBranch/lightning_logs/version_10/checkpoints/resnet50_DualBranch_C16_epoch(16)_R1[0.9313].ckpt'
    DATASET_PATH = './datasets/ConSLAM/'
    
    SEQUENCES = ['Sequence5', 'Sequence4']
    
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16  # Adjust based on your GPU memory
    NUM_WORKERS = 4
    
    # Evaluation parameters
    THETA_DEGREES = 15.0  # Rotation angle for query trajectory
    OFFSET = [0.0, 0.0]  # Translation offset [x, y]
    YAW_THRESHOLD = 60.0  # Maximum yaw difference for positive samples (degrees)
    
    # ==================== Setup ====================
    print("="*70)
    print("Dual-Branch VPR Model on ConSLAM Dataset Evaluation")
    print("="*70)
    
    print(f"\n1. Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # ==================== Load Model ====================
    print("\n2. Loading Dual-Branch VPR model...")

    equi_orientation = 8
    equi_channels = [64, 128, 256, 512]
    
    # Method 1: Try using Lightning's load_from_checkpoint 
    try:
        print("   Attempting to load using Lightning's load_from_checkpoint...")
        model = VPRModel.load_from_checkpoint(
            CHECKPOINT_PATH,
            # 
            backbone_arch='resnet50',
            pretrained=False,  # 
            layers_to_freeze=2,
            layers_to_crop=[4],
            agg_arch='MixVPR',
            agg_config={
                'in_channels': 1024,
                'in_h': 20,
                'in_w': 20,
                'out_channels': 1024,
                'mix_depth': 4,
                'mlp_ratio': 1,
                'out_rows': 4
            },
            use_dual_branch=True,
            equi_orientation=equi_orientation,
            equi_layers=[2, 2, 2, 2],
            equi_channels=equi_channels,
            equi_out_dim=512,
            fusion_method='attention',
            use_projection=False,
            strict=False  # 
        )
        print("   ✓ Model loaded using Lightning's load_from_checkpoint")
        
    except Exception as e:
        print(f"   ✗ Lightning load failed: {str(e)[:100]}...")
        print("   Trying manual loading method...")
        
        # 
        model = VPRModel(
            backbone_arch='resnet50',
            pretrained=False,
            layers_to_freeze=2,
            layers_to_crop=[4],
            agg_arch='MixVPR',
            agg_config={
                'in_channels': 1024,
                'in_h': 20,
                'in_w': 20,
                'out_channels': 1024,
                'mix_depth': 4,
                'mlp_ratio': 1,
                'out_rows': 4
            },
            use_dual_branch=True,
            equi_orientation=equi_orientation,
            equi_layers=[2, 2, 2, 2],
            equi_channels=equi_channels,
            equi_out_dim=512,
            fusion_method='concat',
            use_projection=False,
        )
        
        if not os.path.exists(CHECKPOINT_PATH):
            print(f"   ✗ Checkpoint not found at: {CHECKPOINT_PATH}")
            return
        
        print(f"   Loading checkpoint from: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            epoch = checkpoint.get('epoch', 'unknown')
            print(f"   Found Lightning checkpoint (epoch: {epoch})")
        else:
            state_dict = checkpoint
            print(f"   Found standard PyTorch checkpoint")
        
        try:
            model.load_state_dict(state_dict, strict=True)
            print("   ✓ Model weights loaded successfully (strict=True)")
        except RuntimeError as e:
            print("   ⚠ Strict loading failed, trying with strict=False...")
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            
            if len(missing_keys) > 0:
                print(f"   ⚠ Missing keys: {len(missing_keys)}")
                if len(missing_keys) <= 5:
                    for key in missing_keys:
                        print(f"      - {key}")
            
            if len(unexpected_keys) > 0:
                print(f"   ⚠ Unexpected keys: {len(unexpected_keys)}")
                if len(unexpected_keys) <= 5:
                    for key in unexpected_keys:
                        print(f"      - {key}")
            
            print("   ✓ Model weights loaded (strict=False)")
    
    model = model.to(DEVICE)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model architecture: ResNet50 + MixVPR || E2ResNet (C{equi_orientation}) + GeM")
    print(f"   Descriptor dimension: 4608 (4096 + 512)")
    
    # ==================== Load Datasets ====================
    print("\n3. Loading ConSLAM datasets...")
    datasets = []
    dataloaders = []
    
    for seq in SEQUENCES:
        try:
            dataset = InferDataset(seq, dataset_path=DATASET_PATH, img_size=(320, 320))
            dataloader = DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=True
            )
            datasets.append(dataset)
            dataloaders.append(dataloader)
            print(f"   Sequence {seq}: {len(dataset)} images")
        except Exception as e:
            print(f"   ✗ Error loading sequence {seq}: {e}")
            print(f"   Make sure the path exists: {DATASET_PATH}{seq}/selected_images/")
            continue
    
    if len(datasets) < 2:
        print("\n   ✗ Need at least 2 sequences (1 database + 1 query)")
        print(f"   Successfully loaded sequences: {[ds.seq_name for ds in datasets]}")
        print(f"   Check your dataset path: {DATASET_PATH}")
        return
    
    print(f"\n   ✓ Successfully loaded {len(datasets)} sequences")
    
    # ==================== Extract Features ====================
    print("\n4. Extracting features from all sequences...")
    global_descs = []
    
    for i, (seq, dataloader) in enumerate(zip(SEQUENCES[:len(dataloaders)], dataloaders)):
        print(f"\n   Processing Sequence {seq} ({i+1}/{len(dataloaders)})...")
        features = extract_features_conslam(model, dataloader, DEVICE)
        global_descs.append(features)
    
    print(f"\n   ✓ Feature extraction complete!")
    print(f"   Database (Seq {SEQUENCES[0]}): {global_descs[0].shape}")
    for i in range(1, len(global_descs)):
        print(f"   Query {i} (Seq {SEQUENCES[i]}): {global_descs[i].shape}")
    
    # ==================== Evaluation ====================
    print("\n5. Evaluating with ConSLAM metrics...")
    print(f"   Rotation angle: {THETA_DEGREES}°")
    print(f"   Offset: {OFFSET}")
    print(f"   Yaw threshold: {YAW_THRESHOLD}°")
    print("-"*70)
    
    try:
        # Updated: handle both return values
        recalls, failed_images_dict = evaluateResults(
            global_descs,
            datasets,
            theta_degrees=THETA_DEGREES,
            offset=OFFSET,
            yaw_threshold=YAW_THRESHOLD
        )
    except Exception as e:
        print(f"\n   ✗ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ==================== Summary ====================
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY - Dual-Branch VPR Model")
    print("="*70)
    print(f"Model: Dual-Branch (ResNet50+MixVPR || E2ResNet+GeM)")
    print(f"Descriptor: 4608 dim (4096 + 512)")
    print(f"Database: Sequence {SEQUENCES[0]}")
    print("-"*70)
    
    for i, seq in enumerate(SEQUENCES[1:len(recalls)+1], 0):
        if i < len(recalls):
            print(f"Query {i+1} (Seq {seq}): Recall@1 = {recalls[i]:.4f} ({recalls[i]*100:.2f}%)")
    
    if len(recalls) > 0:
        avg_recall = np.mean(recalls)
        print("-"*70)
        print(f"Average Recall@1: {avg_recall:.4f} ({avg_recall*100:.2f}%)")
    print("="*70)
    
    # Print failed images summary
    print("\n" + "="*70)
    print("FAILED IMAGES SUMMARY")
    print("="*70)
    for seq_i, failed_list in failed_images_dict.items():
        print(f"\nSequence {seq_i}: {len(failed_list)} failed images")
        if len(failed_list) > 0:
            print(f"  First 5 failures: {failed_list[:5]}")
    
    # Save results
    results_file = f'dual_branch_conslam_results_yaw{YAW_THRESHOLD}.txt'
    with open(results_file, 'w') as f:
        f.write("Dual-Branch VPR Model on ConSLAM Dataset - Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: Dual-Branch Architecture\n")
        f.write(f"  Branch 1: ResNet50 + MixVPR (4096 dim)\n")
        f.write(f"  Branch 2: E2ResNet (C{equi_orientation}) + GeM (512 dim)\n")
        f.write(f"  Fusion: Attention (4608 dim)\n\n")
        f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
        f.write(f"Rotation: {THETA_DEGREES}°\n")
        f.write(f"Offset: {OFFSET}\n")
        f.write(f"Yaw Threshold: {YAW_THRESHOLD}°\n\n")
        f.write("Results:\n")
        f.write("-"*70 + "\n")
        f.write(f"Database: Sequence {SEQUENCES[0]}\n\n")
        
        for i, seq in enumerate(SEQUENCES[1:len(recalls)+1], 0):
            if i < len(recalls):
                f.write(f"Query {i+1} (Seq {seq}): Recall@1 = {recalls[i]:.4f} ({recalls[i]*100:.2f}%)\n")
        
        if len(recalls) > 0:
            f.write(f"\nAverage Recall@1: {np.mean(recalls):.4f} ({np.mean(recalls)*100:.2f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("Failed Images:\n")
        f.write("="*70 + "\n\n")
        for seq_i, failed_list in failed_images_dict.items():
            f.write(f"Sequence {seq_i}: {len(failed_list)} failures\n")
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"✓ Trajectory plots saved to: trajectory_plots_conslam/")
    print(f"✓ Diagnosis matrix saved to: diagnosis_matrix_conslam_yaw{YAW_THRESHOLD}.csv")
    print(f"✓ Failed images saved to: failed_examples_conslam_yaw{YAW_THRESHOLD}/")


if __name__ == '__main__':
    main()
