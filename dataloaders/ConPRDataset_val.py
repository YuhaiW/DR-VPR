import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

BASE_PATH = 'datasets/ConPR/'

class ConPRValidationDataset(Dataset):
    """
    ConPR验证数据集
    database: 参考序列（如20230623）
    query: 查询序列（如20230818）
    """
    def __init__(self,
                 db_sequence='20230623',
                 query_sequence='20230818',
                 positive_threshold=5.0,  # 5米内算正确匹配
                 transform=None,
                 base_path=BASE_PATH):
        super(ConPRValidationDataset, self).__init__()
        self.base_path = Path(base_path)
        self.db_sequence = db_sequence
        self.query_sequence = query_sequence
        self.positive_threshold = positive_threshold
        self.transform = transform
        
        # 加载database和query的poses
        self.db_data = self._load_sequence(db_sequence)
        self.query_data = self._load_sequence(query_sequence)
        
        self.numDb = len(self.db_data)
        self.numQ = len(self.query_data)
        
        # 计算ground truth
        self.positives = self._compute_ground_truth()
        
        print(f'[ConPR Val] Database: {db_sequence} ({self.numDb} images)')
        print(f'[ConPR Val] Query: {query_sequence} ({self.numQ} images)')
        print(f'[ConPR Val] Positive threshold: {positive_threshold}m')
        print(f'[ConPR Val] Average positives per query: {np.mean([len(p) for p in self.positives]):.1f}')
    
    def _load_sequence(self, sequence):
        """加载一个序列的所有图像和poses"""
        # 读取pose文件
        pose_file = self.base_path / 'poses' / f'{sequence}.txt'
        if not pose_file.exists():
            raise FileNotFoundError(f'Pose file not found: {pose_file}')
        
        poses_data = []
        with open(pose_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 12:
                    values = [float(x) for x in parts[:12]]
                    tx, ty, tz = values[3], values[7], values[11]
                    poses_data.append({
                        'x': tx,
                        'y': ty,
                        'z': tz
                    })
        
        # 获取图像文件
        img_dir = self.base_path / sequence / 'Camera_matched'
        if not img_dir.exists():
            raise FileNotFoundError(f'Image dir not found: {img_dir}')
        
        all_images = sorted(img_dir.glob('*.png'))
        
        # 匹配poses和images
        min_len = min(len(poses_data), len(all_images))
        
        data = []
        for i in range(min_len):
            data.append({
                'img_path': all_images[i],
                'x': poses_data[i]['x'],
                'y': poses_data[i]['y'],
                'z': poses_data[i]['z']
            })
        
        return data
    
    def _compute_ground_truth(self):
        """
        计算每个query的ground truth正例
        返回: list of lists，positives[i]是第i个query的正例索引列表
        """
        positives = []
        
        # 提取database和query的坐标
        db_coords = np.array([[item['x'], item['y']] for item in self.db_data])
        query_coords = np.array([[item['x'], item['y']] for item in self.query_data])

        # 对每个query，找到距离小于threshold的database图像
        for q_idx in range(self.numQ):
            q_pos = query_coords[q_idx]
            
            # 计算到所有database图像的距离
            distances = np.linalg.norm(db_coords - q_pos, axis=1)
            
            # 找到距离小于threshold的
            pos_indices = np.where(distances < self.positive_threshold)[0]
            
            positives.append(pos_indices)
        
        return positives
    
    def __getitem__(self, index):
        """
        返回一张图像
        前numDb张是database，后面是query
        """
        if index < self.numDb:
            # Database图像
            item = self.db_data[index]
        else:
            # Query图像
            item = self.query_data[index - self.numDb]
        
        img = Image.open(item['img_path']).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, index  # 返回图像和索引
    
    def __len__(self):
        """总图像数 = database + query"""
        return self.numDb + self.numQ
    
    def getPositives(self):
        """返回ground truth，兼容Pittsburgh数据集的接口"""
        return self.positives


def get_conpr_val_set(db_sequence='20230623',
                      query_sequence='20230818',
                      positive_threshold=5.0,
                      input_transform=None):
    """
    获取ConPR验证集
    """
    return ConPRValidationDataset(
        db_sequence=db_sequence,
        query_sequence=query_sequence,
        positive_threshold=positive_threshold,
        transform=input_transform
    )