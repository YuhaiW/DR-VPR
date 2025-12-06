"""
Lightweight ConPR Validation Dataset for Training
Reuses existing InferDataset infrastructure
Fixed to match test evaluation behavior
"""
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class ConPRValidationDataset(Dataset):
    """
    Lightweight wrapper around InferDataset for validation during training.
    Loads multiple sequences and computes ground truth on-the-fly.
    
    IMPORTANT: Matches test evaluation behavior by filtering to only queries 
    with at least one valid positive.
    """
    def __init__(self, 
                 sequences,
                 dataset_path='./datasets/ConPR/',
                 transform=None,
                 yaw_threshold=80.0,
                 position_threshold=5.0):
        """
        Args:
            sequences: List of sequence names [database, query1, query2, ...]
            dataset_path: Path to ConPR dataset
            transform: Image transformation
            yaw_threshold: Max yaw difference for positives (degrees)
            position_threshold: Max position distance for positives (meters)
        """
        self.dataset_path = dataset_path
        self.transform = transform
        self.yaw_threshold = yaw_threshold
        self.position_threshold = position_threshold
        self.sequences = sequences
        
        import os
        
        print(f"Loading ConPR validation set:")
        print(f"  Database: {sequences[0]}")
        print(f"  Queries: {sequences[1:]}")
        
        # Load all images and poses
        self.all_images = []
        self.all_poses = []
        self.sequence_ids = []
        
        for seq_idx, seq in enumerate(sequences):
            # Load images from Camera_matched (RGB images)
            img_dir = os.path.join(dataset_path, seq, 'Camera_matched')
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Image directory not found: {img_dir}")
            
            imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                          if f.endswith('.png') or f.endswith('.jpg')])
            
            # Load poses
            pose_file = os.path.join(dataset_path, 'poses', f'{seq}.txt')
            if not os.path.exists(pose_file):
                raise FileNotFoundError(f"Pose file not found: {pose_file}")
            
            poses = np.loadtxt(pose_file)
            
            assert len(imgs) == len(poses), \
                f"Mismatch in {seq}: {len(imgs)} images vs {len(poses)} poses"
            
            self.all_images.extend(imgs)
            self.all_poses.append(poses)
            self.sequence_ids.extend([seq_idx] * len(imgs))
        
        self.all_poses = np.vstack(self.all_poses)
        self.sequence_ids = np.array(self.sequence_ids)
        
        # Calculate split
        self.num_db = np.sum(self.sequence_ids == 0)
        self.num_queries = len(self.all_images) - self.num_db
        
        print(f"  Total images: {self.num_db} database + {self.num_queries} query")
        
        # Compute ground truth
        self.positives, self.valid_query_indices = self._compute_ground_truth()
        
        # 🔥 KEY FIX: Filter to only queries with at least one positive
        # This matches the test evaluation behavior
        print(f"  Queries with positives: {len(self.valid_query_indices)}/{self.num_queries}")
        print(f"  Queries without positives will be EXCLUDED from recall calculation")
        
    def _get_yaw_from_pose(self, pose):
        """Extract yaw from pose vector"""
        R = np.array([[pose[0], pose[1], pose[2]],
                      [pose[4], pose[5], pose[6]],
                      [pose[8], pose[9], pose[10]]])
        yaw = np.arctan2(R[1,0], R[0,0])
        return np.degrees(yaw)
    
    def _compute_ground_truth(self):
        """
        Compute ground truth positives for each query
        
        Returns:
            positives: List of positive indices for each query
            valid_query_indices: Indices of queries that have at least one positive
        """
        db_poses = self.all_poses[:self.num_db]
        query_poses = self.all_poses[self.num_db:]
        
        # Extract positions and yaws
        db_xy = db_poses[:, [3, 7]]  # x, y
        query_xy = query_poses[:, [3, 7]]
        
        db_yaws = np.array([self._get_yaw_from_pose(p) for p in db_poses])
        query_yaws = np.array([self._get_yaw_from_pose(p) for p in query_poses])
        
        positives = []
        valid_query_indices = []
        
        for q_idx in range(len(query_xy)):
            # Compute distances
            dists = np.linalg.norm(db_xy - query_xy[q_idx], axis=1)
            
            # Find candidates by position
            candidates = np.where(dists < self.position_threshold)[0]
            
            # Filter by yaw
            pos_indices = []
            for db_idx in candidates:
                yaw_diff = abs(query_yaws[q_idx] - db_yaws[db_idx])
                if yaw_diff > 180:
                    yaw_diff = 360 - yaw_diff
                
                if yaw_diff <= self.yaw_threshold:
                    pos_indices.append(db_idx)
            
            pos_array = np.array(pos_indices, dtype=int)
            positives.append(pos_array)
            
            # 🔥 Track which queries have at least one positive
            if len(pos_array) > 0:
                valid_query_indices.append(q_idx)
        
        return positives, np.array(valid_query_indices)
    
    def __len__(self):
        return len(self.all_images)
    
    def __getitem__(self, idx):
        """Return transformed image and index"""
        img_path = self.all_images[idx]
        
        # Load as RGB
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, idx
    
    def getPositives(self):
        """
        Return ground truth positives (compatible with Pittsburgh format)
        
        🔥 CRITICAL: This returns positives for ALL queries, but downstream
        evaluation should use valid_query_indices to filter to only queries
        with at least one positive, matching test behavior.
        """
        return self.positives


def get_conpr_validation_set(sequences=None, transform=None, yaw_threshold=80.0):
    """
    Helper function to create ConPR validation set
    
    Args:
        sequences: List of sequences [database, query1, query2, ...]
                  Default: ['20230623', '20230809'] for fast validation
        transform: Image transformation pipeline
        yaw_threshold: Maximum yaw difference for positives (degrees)
    
    Returns:
        ConPRValidationDataset instance
    """
    if sequences is None:
        # Default: 2 sequences for fast validation
        sequences = ['20230623', '20230809']
    
    return ConPRValidationDataset(
        sequences=sequences,
        transform=transform,
        yaw_threshold=yaw_threshold
    )