"""
ConSLAM Dataset Evaluation with Rotation and Yaw Constraints
"""

import os
import numpy as np
import cv2
import torch.utils.data as data
from torchvision import transforms
import faiss
import matplotlib.pyplot as plt
import pandas as pd


def get_yaw_from_pose(pose):
    """Extract yaw angle from pose vector [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz]"""
    R = np.array([[pose[0], pose[1], pose[2]],
                  [pose[4], pose[5], pose[6]],
                  [pose[8], pose[9], pose[10]]])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees(yaw)


class InferDataset(data.Dataset):
    """Dataset for ConSLAM sequences"""
    
    def __init__(self, seq, dataset_path='./datasets/ConSLAM/', img_size=(320, 320)):
        super().__init__()
        
        self.seq_name = seq
        
        # Load images
        img_dir = os.path.join(dataset_path, seq, 'selected_images')
        imgs_p = sorted(os.listdir(img_dir))
        self.imgs_path = [os.path.join(img_dir, i) for i in imgs_p]
        
        # Load poses
        pose_file = os.path.join(dataset_path, 'poses', f'{seq}.txt')
        self.poses = np.loadtxt(pose_file)
        
        # Handle mismatch
        min_len = min(len(self.imgs_path), len(self.poses))
        self.imgs_path = self.imgs_path[:min_len]
        self.poses = self.poses[:min_len]
        
        # Transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loaded {seq}: {len(self.imgs_path)} image-pose pairs")

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img_rgb), index

    def __len__(self):
        return len(self.imgs_path)


def get_failed_image_names(diag_data, datasets):
    """
    Get list of failed image filenames from diagnostic data
    
    Args:
        diag_data: List of diagnostic data dictionaries
        datasets: List of InferDataset objects
    
    Returns:
        Dictionary mapping sequence index to list of failed image filenames
    """
    df = pd.DataFrame(diag_data)
    
    failed_images = {}
    
    for seq_i in df['seq_i'].unique():
        failures = df[(df['seq_i'] == seq_i) & (df['success'] == 0)]
        
        failed_list = []
        for _, row in failures.iterrows():
            q_idx = int(row['q_idx'])
            img_path = datasets[seq_i].imgs_path[q_idx]
            img_filename = os.path.basename(img_path)
            failed_list.append(img_filename)
        
        failed_images[seq_i] = failed_list
    
    return failed_images


def save_failed_examples_to_txt(diag_data, datasets, output_dir='failed_examples'):
    """Save failed image filenames to text files"""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.DataFrame(diag_data)
    
    print(f"\n=== Saving Failed Image Filenames ===")
    for seq_i in df['seq_i'].unique():
        failures = df[(df['seq_i'] == seq_i) & (df['success'] == 0)]
        output_file = os.path.join(output_dir, f'failed_images_seq{seq_i}.txt')
        
        with open(output_file, 'w') as f:
            for _, row in failures.iterrows():
                img_path = datasets[seq_i].imgs_path[int(row['q_idx'])]
                f.write(f"{os.path.basename(img_path)}\n")
        
        print(f"  ✓ Sequence {seq_i}: {len(failures)} failures → {output_file}")
    
    return output_dir


def compare_failed_images(baseline_failed_dict, improved_failed_dict, output_file='comparison_summary.txt'):
    """
    Compare failed images between baseline and improved model
    Find images that failed in baseline but succeeded in improved model
    
    Args:
        baseline_failed_dict: Dict of {seq_i: [failed_filenames]} from baseline
        improved_failed_dict: Dict of {seq_i: [failed_filenames]} from improved model
        output_file: Output file to save comparison results
    
    Returns:
        Dictionary of improved images per sequence
    """
    improved_images = {}
    total_baseline_failures = 0
    total_improved_count = 0
    
    print(f"\n{'='*80}")
    print("COMPARING BASELINE VS IMPROVED MODEL")
    print('='*80)
    
    # Compare each sequence
    for seq_i in baseline_failed_dict.keys():
        baseline_set = set(baseline_failed_dict[seq_i])
        improved_set = set(improved_failed_dict.get(seq_i, []))
        
        # Find improvements: in baseline failures but NOT in improved failures
        improved_imgs = baseline_set - improved_set
        
        improved_images[seq_i] = sorted(improved_imgs)
        total_baseline_failures += len(baseline_set)
        total_improved_count += len(improved_imgs)
        
        print(f"\nSequence {seq_i}:")
        print(f"  Baseline failures: {len(baseline_set)}")
        print(f"  Improved failures: {len(improved_set)}")
        print(f"  ✨ Fixed by improved model: {len(improved_imgs)}")
        if len(baseline_set) > 0:
            print(f"  Improvement rate: {len(improved_imgs)/len(baseline_set)*100:.2f}%")
    
    # Save detailed comparison
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE vs IMPROVED MODEL COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total baseline failures: {total_baseline_failures}\n")
        f.write(f"Total images fixed by improved model: {total_improved_count}\n")
        f.write(f"Overall improvement rate: {total_improved_count/total_baseline_failures*100:.2f}%\n\n")
        
        f.write("="*80 + "\n")
        f.write("Per-Sequence Breakdown:\n")
        f.write("="*80 + "\n\n")
        
        for seq_i in sorted(improved_images.keys()):
            baseline_count = len(baseline_failed_dict[seq_i])
            improved_count = len(improved_images[seq_i])
            
            f.write(f"\n--- Sequence {seq_i} ---\n")
            f.write(f"Baseline failures: {baseline_count}\n")
            f.write(f"Fixed by improved model: {improved_count}\n")
            f.write(f"Improvement rate: {improved_count/baseline_count*100:.2f}%\n\n")
            
            if improved_count > 0:
                f.write("Fixed images:\n")
                for img in improved_images[seq_i]:
                    f.write(f"  - {img}\n")
    
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print('='*80)
    print(f"Total baseline failures: {total_baseline_failures}")
    print(f"Total fixed by improved model: {total_improved_count}")
    print(f"Overall improvement rate: {total_improved_count/total_baseline_failures*100:.2f}%")
    print(f"\n✓ Detailed comparison saved to: {output_file}")
    
    return improved_images


def plot_trajectory(datasets, i, db_x, db_y, query_x_rot, query_y_rot, 
                    success_points, failure_points, recall_top1, precision, 
                    f1_score, yaw_threshold, plot_dir):
    """Create publication-quality trajectory plot"""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    
    # Trajectories
    ax.plot(db_x, db_y, '-', color="#068ef0", linewidth=2.5, label='Database', alpha=0.9)
    ax.plot(query_x_rot, query_y_rot, '--', color='#2ca02c', linewidth=2.0, label='Query', alpha=0.8)
    
    # Origin
    ax.scatter(0, 0, color='black', s=150, marker='*', label='Origin', 
              zorder=5, edgecolors='white', linewidths=1.5)
    
    # Success points
    if success_points:
        success_x, success_y = zip(*success_points)
        ax.scatter(success_x, success_y, color='#2ca02c', s=50, marker='o',
                  label=f'Correct', 
                  edgecolors='white', linewidths=0.5, alpha=0.9)
    
    # Failure points
    if failure_points:
        failure_x, failure_y = zip(*failure_points)
        ax.scatter(failure_x, failure_y, color='#d62728', s=60, marker='X',
                  label=f'Failed', alpha=0.95)
    
    # Labels and styling
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'{datasets[0].seq_name} (Database) vs {datasets[i].seq_name} (Query)\n'
                f'Recall@1: {recall_top1:.1%} | Precision: {precision:.1%} | F1: {f1_score:.3f}',
                fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, framealpha=0.95)
    ax.set_aspect('equal')
    
    # Save
    plt.tight_layout()
    base_filename = f'{plot_dir}/{datasets[0].seq_name}_vs_{datasets[i].seq_name}_yaw{yaw_threshold}deg'
    plt.savefig(f'{base_filename}.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{base_filename}.pdf', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 Plots saved: {base_filename}.[png/pdf]")


def evaluateResults(global_descs, datasets, theta_degrees=0.0, offset=[0.0, 0.0], yaw_threshold=80.0):
    """Evaluate ConSLAM dataset with yaw constraint"""
    
    # Setup
    gt_thres = 5.0  # Position threshold (meters)
    faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1])
    faiss_index.add(global_descs[0])
    
    plot_dir = 'trajectory_plots_conslam'
    os.makedirs(plot_dir, exist_ok=True)
    
    db_poses = datasets[0].poses
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    theta_rad = np.deg2rad(theta_degrees)
    
    # Storage
    recalls, precisions, f1_scores = [], [], []
    diag_data = []
    
    # Evaluate each query sequence
    for i in range(1, len(datasets)):
        query_poses = datasets[i].poses.copy()
        
        # Apply transformations
        query_poses[:, 3] += offset[0]
        query_poses[:, 7] += offset[1]
        query_x, query_y = query_poses[:, 3], query_poses[:, 7]
        query_x_rot = query_x * np.cos(theta_rad) - query_y * np.sin(theta_rad)
        query_y_rot = query_x * np.sin(theta_rad) + query_y * np.cos(theta_rad)
        query_poses[:, 3], query_poses[:, 7] = query_x_rot, query_y_rot
        
        # Predictions
        _, predictions = faiss_index.search(global_descs[i], 1)
        
        # Metrics
        all_positives = tp = fp = position_only_tp = yaw_filtered_count = 0
        success_points, failure_points = [], []
        
        for q_idx, pred in enumerate(predictions):
            # Distances
            dist_sq = np.sum((query_poses[q_idx] - db_poses)[:, [3, 7]]**2, axis=1)
            min_dist = np.sqrt(dist_sq[pred[0]])
            
            # Yaw angles
            query_yaw = get_yaw_from_pose(query_poses[q_idx])
            
            # Find positives
            position_positives = np.where(dist_sq < gt_thres**2)[0]
            position_only_success = pred[0] in position_positives
            if position_only_success and len(position_positives) > 0:
                position_only_tp += 1
            
            # Filter by yaw
            positives = []
            for pos_idx in position_positives:
                yaw_diff = abs(query_yaw - get_yaw_from_pose(db_poses[pos_idx]))
                if yaw_diff > 180:
                    yaw_diff = 360 - yaw_diff
                if yaw_diff <= yaw_threshold:
                    positives.append(pos_idx)
            
            positives = np.array(positives)
            success = pred[0] in positives
            
            if position_only_success and not success:
                yaw_filtered_count += 1
            
            if len(positives) > 0:
                all_positives += 1
                if success:
                    tp += 1
                    success_points.append((query_x_rot[q_idx], query_y_rot[q_idx]))
                else:
                    fp += 1
                    failure_points.append((query_x_rot[q_idx], query_y_rot[q_idx]))
            
            # Diagnostic data
            pred_yaw_diff = abs(query_yaw - get_yaw_from_pose(db_poses[pred[0]]))
            if pred_yaw_diff > 180:
                pred_yaw_diff = 360 - pred_yaw_diff
            
            diag_data.append({
                'seq_i': i, 'q_idx': q_idx, 'pred_idx': pred[0],
                'min_dist_to_pred': min_dist, 'yaw_diff_to_pred': pred_yaw_diff,
                'success': int(success), 'success_position_only': int(position_only_success),
                'filtered_by_yaw': int(position_only_success and not success),
                'num_positives': len(positives), 'num_position_positives': len(position_positives),
                'query_x': query_poses[q_idx, 3], 'query_y': query_poses[q_idx, 7],
                'query_yaw': query_yaw
            })
        
        # Calculate metrics
        recall = tp / all_positives if all_positives > 0 else 0.0
        recall_pos_only = position_only_tp / len(predictions) if len(predictions) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * recall / (prec + recall) if (prec + recall) > 0 else 0.0
        
        recalls.append(recall)
        precisions.append(prec)
        f1_scores.append(f1)
        
        # Print results
        print(f"\n=== {datasets[i].seq_name} Results ===")
        print(f"Recall@1 (yaw): {recall:.4f} ({tp}/{all_positives})")
        print(f"Recall@1 (pos): {recall_pos_only:.4f} ({position_only_tp}/{len(predictions)})")
        print(f"Precision: {prec:.4f} | F1: {f1:.4f}")
        print(f"Yaw filtered: {yaw_filtered_count} ({yaw_filtered_count/len(predictions)*100:.1f}%)")
        
        # Plot
        plot_trajectory(datasets, i, db_x, db_y, query_x_rot, query_y_rot,
                       success_points, failure_points, recall, prec, f1,
                       yaw_threshold, plot_dir)
    
    # Save diagnostics
    diag_df = pd.DataFrame(diag_data)
    csv_file = f'diagnosis_matrix_conslam_yaw{yaw_threshold}.csv'
    diag_df.to_csv(csv_file, index=False)
    print(f"\n=== Diagnostic matrix saved: {csv_file} ===")
    
    save_failed_examples_to_txt(diag_data, datasets, 
                                f'failed_examples_conslam_yaw{yaw_threshold}')
    
    # Get failed image names as dictionary
    failed_images_dict = get_failed_image_names(diag_data, datasets)
    
    # Summary statistics
    print(f"\n=== Overall Performance ===")
    print(f"Average Recall@1: {np.mean(recalls):.4f}")
    print(f"Average Precision: {np.mean(precisions):.4f}")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
    
    total_yaw_filtered = diag_df['filtered_by_yaw'].sum()
    total_queries = len(diag_df)
    print(f"\nYaw Impact: {total_yaw_filtered}/{total_queries} queries "
          f"({total_yaw_filtered/total_queries*100:.1f}%)")
    
    return recalls, failed_images_dict