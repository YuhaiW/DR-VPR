"""
Lightweight ConSLAM Validation Dataset for Training
Reuses Conslam_dataset_rot.InferDataset infrastructure.
Mirrors ConPRValidation.py but applies theta=15° query trajectory rotation
(same protocol as test_conslam.py / eval_rerank.py / eval_baselines.py).

The wrapper exposes the standard val-set interface used by
train_fusion.py:validation_epoch_end:
    num_db, num_queries, getPositives()
so ckpt monitor='conslam/R1' works drop-in like 'conpr/R1'.
"""
import numpy as np
from torch.utils.data import Dataset

from Conslam_dataset_rot import InferDataset, get_yaw_from_pose


class ConSLAMValidationDataset(Dataset):
    """
    Wrap two ConSLAM sequences (db + single-query) into one Dataset for the
    Lightning val loop.

    Critical difference from ConPRValidation: we apply the theta=15° rotation
    on the query trajectory before computing GT positives, matching the test
    protocol. Without this, recall numbers would not match test_conslam.py /
    eval_rerank.py.
    """

    def __init__(
        self,
        db_seq='Sequence5',
        query_seq='Sequence4',
        dataset_path='./datasets/ConSLAM/',
        img_size=(320, 320),
        yaw_threshold=80.0,
        theta_degrees=15.0,
        gt_thres=5.0,
    ):
        self.dataset_path = dataset_path
        self.yaw_threshold = yaw_threshold
        self.theta_degrees = theta_degrees
        self.gt_thres = gt_thres

        print(f"Loading ConSLAM validation set:")
        print(f"  Database: {db_seq}")
        print(f"  Query:    {query_seq}")
        print(f"  theta={theta_degrees}°, yaw_threshold={yaw_threshold}°, gt_thres={gt_thres}m")

        # Reuse the existing InferDataset (handles image loading, pose loading,
        # transforms — including the same 320x320 resize used at test time).
        self.db_ds = InferDataset(db_seq, dataset_path=dataset_path, img_size=img_size)
        self.q_ds = InferDataset(query_seq, dataset_path=dataset_path, img_size=img_size)

        self.num_db = len(self.db_ds)
        self.num_queries = len(self.q_ds)
        print(f"  Total images: {self.num_db} database + {self.num_queries} query")

        # Compute GT positives ONCE at construction time.
        self.positives, self.valid_query_indices = self._compute_ground_truth()
        print(f"  Queries with positives: {len(self.valid_query_indices)}/{self.num_queries}")
        print(f"  Queries without positives will be EXCLUDED from recall calculation")

    def _compute_ground_truth(self):
        """
        Returns:
            positives: list[np.ndarray] — for each query (length self.num_queries),
                       array of valid positive db indices (may be empty).
            valid_query_indices: np.ndarray — indices of queries with non-empty positives.

        Protocol matches Conslam_dataset_rot.evaluateResults / eval_rerank.py:
            1. Apply theta_degrees rotation to query (x, y) before distance comparison
            2. Position positives = db images within gt_thres meters of rotated query
            3. Yaw filter: |yaw_query - yaw_db| <= yaw_threshold (with 360° wrap)
        """
        # Apply theta rotation to query trajectory (matches eval_rerank.py:106-112)
        theta_rad = np.deg2rad(self.theta_degrees)
        q_poses = self.q_ds.poses.copy()
        qx, qy = q_poses[:, 3], q_poses[:, 7]
        q_poses[:, 3] = qx * np.cos(theta_rad) - qy * np.sin(theta_rad)
        q_poses[:, 7] = qx * np.sin(theta_rad) + qy * np.cos(theta_rad)

        db_poses = self.db_ds.poses
        db_x, db_y = db_poses[:, 3], db_poses[:, 7]

        positives = []
        valid_query_indices = []

        for q_idx in range(self.num_queries):
            dist_sq = (q_poses[q_idx, 3] - db_x) ** 2 + (q_poses[q_idx, 7] - db_y) ** 2
            pos_candidates = np.where(dist_sq < self.gt_thres ** 2)[0]

            q_yaw = get_yaw_from_pose(q_poses[q_idx])
            valid = []
            for db_idx in pos_candidates:
                d = abs(q_yaw - get_yaw_from_pose(db_poses[db_idx]))
                if d > 180:
                    d = 360 - d
                if d <= self.yaw_threshold:
                    valid.append(int(db_idx))

            pos_arr = np.array(valid, dtype=np.int64)
            positives.append(pos_arr)
            if len(pos_arr) > 0:
                valid_query_indices.append(q_idx)

        return positives, np.array(valid_query_indices)

    def __len__(self):
        return self.num_db + self.num_queries

    def __getitem__(self, idx):
        # Convention matches ConPRValidation: db images first, then query images.
        # validation_epoch_end relies on this ordering when slicing
        # feats[:num_references] vs feats[num_references:].
        if idx < self.num_db:
            return self.db_ds[idx]
        return self.q_ds[idx - self.num_db]

    def getPositives(self):
        return self.positives


def get_conslam_validation_set(
    db_seq='Sequence5',
    query_seq='Sequence4',
    yaw_threshold=80.0,
    theta_degrees=15.0,
):
    """Helper, same shape as get_conpr_validation_set in ConPRValidation.py."""
    return ConSLAMValidationDataset(
        db_seq=db_seq,
        query_seq=query_seq,
        yaw_threshold=yaw_threshold,
        theta_degrees=theta_degrees,
    )
