import os
from os.path import join, exists
import numpy as np
import cv2
from imgaug import augmenters as iaa
import torch
import torch.utils.data as data
from torchvision import transforms

import h5py

import faiss
import matplotlib.pyplot as plt
import pandas as pd

"""
Add this simplified function to your conpr_eval_dataset_rot.py file
"""

def save_failed_examples_to_txt(diag_data, datasets, output_dir='failed_examples'):
    """
    Extract and save failed image filenames from each trajectory to text files
    
    Parameters:
        diag_data: List of diagnostic data dictionaries
        datasets: List of InferDataset objects
        output_dir: Directory to save the text files
    """
    import os
    import pandas as pd
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(diag_data)
    
    # Get unique sequence indices
    sequences = df['seq_i'].unique()
    
    print(f"\n=== Saving Failed Image Filenames to TXT Files ===")
    
    for seq_i in sequences:
        # Filter data for this sequence
        seq_data = df[df['seq_i'] == seq_i]
        
        # Get failed examples (success == 0)
        failures = seq_data[seq_data['success'] == 0]
        
        # Create output filename
        output_file = os.path.join(output_dir, f'failed_images_seq{seq_i}.txt')
        
        with open(output_file, 'w') as f:
            # Write only the image filenames
            for idx, (_, row) in enumerate(failures.iterrows()):
                q_idx = int(row['q_idx'])
                # Get the image path from the dataset
                img_path = datasets[seq_i].imgs_path[q_idx]
                # Extract just the filename
                img_filename = os.path.basename(img_path)
                f.write(f"{img_filename}\n")
        
        print(f"  ✓ Sequence {seq_i}: {len(failures)} failed images saved to {output_file}")
    
    print(f"\n✓ All failed image filenames saved to '{output_dir}/' directory")
    
    return output_dir

"""
Add this function to your conpr_eval_dataset_rot.py file
"""

def rotation_matrix_to_euler_angles(R):
    """
    从旋转矩阵提取偏航角
    返回：偏航角（度）
    """
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        yaw = np.arctan2(R[1,0], R[0,0])
    else:
        yaw = np.arctan2(-R[1,2], R[1,1])
    
    return np.degrees(yaw)


def get_yaw_from_pose(pose):
    """
    从位姿向量提取偏航角
    pose格式: [r11, r12, r13, tx, r21, r22, r23, ty, r31, r32, r33, tz]
    """
    # 重构旋转矩阵
    R = np.array([[pose[0], pose[1], pose[2]],
                  [pose[4], pose[5], pose[6]],
                  [pose[8], pose[9], pose[10]]])
    
    return rotation_matrix_to_euler_angles(R)


class InferDataset(data.Dataset):
    def __init__(self, seq, dataset_path='./datasets/ConPR/', img_size=(320, 320)):
        super().__init__()

        # bev path
        # imgs_p = os.listdir(dataset_path+seq+'/bev/')
        # imgs_p.sort()
        # self.imgs_path = [dataset_path+seq+'/bev/'+i for i in imgs_p]

        # rgb path
        imgs_p = os.listdir(dataset_path+seq+'/Camera_matched/')
        imgs_p.sort()
        self.imgs_path = [dataset_path+seq+'/Camera_matched/'+i for i in imgs_p]



        # gt_pose
        self.poses = np.loadtxt(dataset_path+'poses/'+seq+'.txt')
        
        # Image size for MixVPR (must be 320x320)
        self.img_size = img_size
        
        # Define transform for MixVPR
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


    def __getitem__(self, index):
        # Read grayscale image
        img = cv2.imread(self.imgs_path[index], 0)
        
        # Convert grayscale to RGB (repeat channels)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        # Apply transform (resize to 320x320 and normalize)
        img_tensor = self.transform(img_rgb)
        
        return img_tensor, index

    def __len__(self):
        return len(self.imgs_path)


def evaluateResults(global_descs, datasets, theta_degrees=0.0, offset=[0.0, 0.0], yaw_threshold=60.0):
    """
    使用偏航角约束和F1分数评估ConPR数据集
    
    参数:
        global_descs: 全局描述符列表
        datasets: 数据集对象列表
        theta_degrees: 应用于查询轨迹的旋转角度
        offset: 平移偏移 [x, y]
        yaw_threshold: 正样本匹配的最大偏航角差异度默认10.0
    
    返回:
        recalls_conpr: 召回率列表
    """
    gt_thres = 5 # 位置阈值（米）
    faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1]) 
    faiss_index.add(global_descs[0])

    recalls_conpr = []
    precision_list = []
    f1_scores = []
    
    plot_dir = 'trajectory_plots_conpr'
    os.makedirs(plot_dir, exist_ok=True)
    
    db_poses = datasets[0].poses
    db_x = db_poses[:, 3]  # x坐标
    db_y = db_poses[:, 7]  # y坐标
    
    theta_rad = np.deg2rad(theta_degrees)
    
    # 诊断矩阵列表
    diag_data = []
    
    for i in range(1, len(datasets)):
        query_poses = datasets[i].poses.copy()
        
        # 应用偏移和旋转
        query_poses[:, 3] += offset[0]
        query_poses[:, 7] += offset[1]
        query_x = query_poses[:, 3]
        query_y = query_poses[:, 7]
        query_x_rot = query_x * np.cos(theta_rad) - query_y * np.sin(theta_rad)
        query_y_rot = query_x * np.sin(theta_rad) + query_y * np.cos(theta_rad)
        query_poses[:, 3] = query_x_rot
        query_poses[:, 7] = query_y_rot
        
        _, predictions = faiss_index.search(global_descs[i], 1)
        
        all_positives = 0  # 至少有一个正样本的查询数量
        tp = 0  # 真阳性
        fp = 0  # 假阳性
        
        # Statistics for tracking yaw constraint impact
        position_only_tp = 0  # True positives if only considering position
        yaw_filtered_count = 0  # Count of positives filtered out by yaw
        
        # 收集失败点用于可视化
        failure_points = []
        success_points = []
        
        for q_idx, pred in enumerate(predictions):
            # 计算位置距离
            gt_dis = (query_poses[q_idx] - db_poses)**2
            dist_sq = np.sum(gt_dis[:,[3,7]], axis=1)  # x,y距离平方
            min_dist = np.sqrt(dist_sq[pred[0]])  # 到top-1的实际距离
            
            # 获取偏航角
            query_yaw = get_yaw_from_pose(query_poses[q_idx])
            
            # 根据位置阈值找到候选正样本（仅位置）
            position_positives = np.where(dist_sq < gt_thres**2)[0]
            
            # Check if prediction would be correct with position only
            position_only_success = 1 if pred[0] in position_positives else 0
            if len(position_positives) > 0 and position_only_success:
                position_only_tp += 1
            
            # 根据偏航角约束过滤正样本
            positives = []
            filtered_by_yaw = []  # Track which ones were filtered out
            for pos_idx in position_positives:
                db_yaw = get_yaw_from_pose(db_poses[pos_idx])
                yaw_diff = abs(query_yaw - db_yaw)
                # 处理角度环绕（例如 359° vs 1°）
                if yaw_diff > 180:
                    yaw_diff = 360 - yaw_diff
                
                if yaw_diff <= yaw_threshold:
                    positives.append(pos_idx)
                else:
                    filtered_by_yaw.append(pos_idx)
            
            positives = np.array(positives)
            
            # 检查预测是否正确（带偏航角约束）
            success = 1 if pred[0] in positives else 0
            
            # Check if this prediction was affected by yaw filtering
            yaw_caused_failure = (position_only_success == 1) and (success == 0)
            if yaw_caused_failure:
                yaw_filtered_count += 1
            
            if len(positives) > 0: #all points with valid positives
                all_positives += 1
                if success:
                    tp += 1
                    success_points.append((query_x_rot[q_idx], query_y_rot[q_idx]))
                else:
                    fp += 1
                    failure_points.append((query_x_rot[q_idx], query_y_rot[q_idx]))
            # else:
            #     # 没有有效正样本，但模型仍然做出了预测
            #     if pred[0] in position_positives:
            #         # 由于偏航角约束而失败
            #         fp += 1
            #         failure_points.append((query_x_rot[q_idx], query_y_rot[q_idx]))
            
            # 获取预测匹配的偏航角差异
            pred_yaw = get_yaw_from_pose(db_poses[pred[0]])
            yaw_diff_pred = abs(query_yaw - pred_yaw)
            if yaw_diff_pred > 180:
                yaw_diff_pred = 360 - yaw_diff_pred
            
            # 记录到诊断矩阵
            diag_data.append({
                'seq_i': i,
                'q_idx': q_idx,
                'pred_idx': pred[0],
                'min_dist_to_pred': min_dist,
                'yaw_diff_to_pred': yaw_diff_pred,
                'success': success,
                'success_position_only': position_only_success,
                'filtered_by_yaw': 1 if yaw_caused_failure else 0,
                'num_positives': len(positives),
                'num_position_positives': len(position_positives),
                'num_filtered_by_yaw': len(filtered_by_yaw),
                'query_x': query_poses[q_idx, 3],
                'query_y': query_poses[q_idx, 7],
                'query_yaw': query_yaw
            })

        # 计算指标
        recall_top1 = tp / all_positives if all_positives > 0 else 0.0
        recall_position_only = position_only_tp / len(predictions) if len(predictions) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1_score = 2 * (precision * recall_top1) / (precision + recall_top1) if (precision + recall_top1) > 0 else 0.0
        
        recalls_conpr.append(recall_top1)
        precision_list.append(precision)
        f1_scores.append(f1_score)
        
        print(f"\n=== ConPR Sequence {i} Results ===")
        print(f"Recall@1 (with yaw constraint): {recall_top1:.4f} ({tp}/{all_positives})")
        print(f"Recall@1 (position only): {recall_position_only:.4f} ({position_only_tp}/{len(predictions)})")
        print(f"Precision: {precision:.4f} ({tp}/{tp+fp})")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Failure points: {len(failure_points)}, Success points: {len(success_points)}")
        print(f"\n🔍 Yaw Constraint Impact:")
        print(f"   - Predictions changed from ✅ to ❌ due to yaw: {yaw_filtered_count}")
        print(f"   - Percentage of queries affected: {yaw_filtered_count/len(predictions)*100:.2f}%")
        
        # 绘制轨迹图（带失败/成功点标记）
        plt.figure(figsize=(14, 10))
        plt.plot(db_x, db_y, 'b-', linewidth=2, label='Database Trajectory', alpha=0.7)
        plt.plot(query_x_rot, query_y_rot, 'g--', linewidth=1.5, label=f'Query Trajectory (rotated {theta_degrees}°)', alpha=0.6)
        plt.scatter(0, 0, color='black', s=100, marker='o', label='(0,0) Origin', zorder=5)
        
        # 绘制成功点（绿色）
        if success_points:
            success_x, success_y = zip(*success_points)
            plt.scatter(success_x, success_y, color='green', s=20, alpha=0.6, marker='o', 
                       label=f'Success Points ({len(success_points)})', zorder=3)
        
        # 绘制失败点（红色）
        if failure_points:
            failure_x, failure_y = zip(*failure_points)
            plt.scatter(failure_x, failure_y, color='red', s=20, alpha=0.8, marker='x', 
                       label=f'Failure Points ({len(failure_points)})', zorder=4)
        
        plt.xlabel('X (m)', fontsize=12)
        plt.ylabel('Y (m)', fontsize=12)
        plt.title(f'ConPR Trajectory Comparison: Seq {i} vs Database\n'
                 f'Recall@1: {recall_top1:.3f} | Precision: {precision:.3f} | F1: {f1_score:.3f}', 
                 fontsize=13)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig(f'{plot_dir}/conpr_seq{i}_rot{theta_degrees}deg_yaw{yaw_threshold}deg.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"图表已保存到: {plot_dir}/conpr_seq{i}_rot{theta_degrees}deg_yaw{yaw_threshold}deg.png")

    # 生成诊断矩阵（DataFrame）
    diag_df = pd.DataFrame(diag_data)
    csv_filename = f'diagnosis_matrix_conpr_yaw{yaw_threshold}.csv'
    diag_df.to_csv(csv_filename, index=False)
    print(f"\n=== 诊断矩阵已保存到: {csv_filename} ===")
    print(diag_df.head(10))

    save_failed_examples_to_txt(diag_data, datasets, output_dir=f'failed_examples_yaw{yaw_threshold}')
    # save_failed_examples_to_txt(diag_data, output_dir=f'failed_examples_yaw{yaw_threshold}')
    # 失败案例分析
    failures = diag_df[diag_df['success'] == 0]
    print(f"\n=== 前10个失败案例（按距离排序） ===")
    if len(failures) > 0:
        print(failures.nlargest(10, 'min_dist_to_pred')[
            ['seq_i', 'q_idx', 'pred_idx', 'min_dist_to_pred', 'yaw_diff_to_pred', 
             'num_positives', 'num_position_positives', 'query_x', 'query_y']
        ])
    
    # 因偏航角约束失败的案例
    yaw_failures = failures[failures['yaw_diff_to_pred'] > yaw_threshold]
    print(f"\n=== 因偏航角约束失败的案例 ({len(yaw_failures)} 个) ===")
    if len(yaw_failures) > 0:
        print(f"这些失败案例的平均偏航角差异: {yaw_failures['yaw_diff_to_pred'].mean():.2f}°")
        print(f"最大偏航角差异: {yaw_failures['yaw_diff_to_pred'].max():.2f}°")
    
    # 不利好环境分析
    low_pos_env = diag_df[diag_df['num_positives'] < 1]
    print(f"\n=== 不利好环境（0个有效正样本，共 {len(low_pos_env)} 个案例） ===")
    if len(low_pos_env) > 0:
        print(f"查询区域范围: x ∈ [{low_pos_env['query_x'].min():.1f}, {low_pos_env['query_x'].max():.1f}], "
              f"y ∈ [{low_pos_env['query_y'].min():.1f}, {low_pos_env['query_y'].max():.1f}]")
    
    # 总体统计
    print(f"\n=== ConPR 总体性能 ===")
    print(f"平均 Recall@1: {np.mean(recalls_conpr):.4f}")
    print(f"平均 Recall@1 (位置): {np.mean(recall_position_only):.4f}")
    print(f"平均 Precision: {np.mean(precision_list):.4f}")
    print(f"平均 F1 Score: {np.mean(f1_scores):.4f}")
    
    # Summary of yaw constraint impact across all sequences
    total_yaw_filtered = diag_df['filtered_by_yaw'].sum()
    total_queries = len(diag_df)
    total_position_success = diag_df['success_position_only'].sum()
    total_yaw_success = diag_df['success'].sum()
    
    print(f"\n=== 📊 Yaw Constraint Impact Summary (ConPR) ===")
    print(f"Total queries: {total_queries}")
    print(f"Successful matches (position only): {total_position_success}")
    print(f"Successful matches (with yaw constraint): {total_yaw_success}")
    print(f"Matches changed from ✅ to ❌ by yaw constraint: {total_yaw_filtered}")
    print(f"Percentage of queries affected: {total_yaw_filtered/total_queries*100:.2f}%")
    print(f"Impact on recall: {(total_position_success - total_yaw_success)/total_queries*100:.2f}% reduction")
    
    return recalls_conpr