import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import pickle

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

BASE_PATH = './datasets/ConPR/'

class ConPRDataset(Dataset):
    def __init__(self,
                 sequences=['20230531'],  # List of trajectory sequences
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                 anchor_distance_threshold=10.0,  # Minimum distance between anchor positions
                 place_distance_threshold=5.0,    # Distance to consider same place
                 place_db_path=None,
                 rebuild_place_db=False,
                 ):
        super(ConPRDataset, self).__init__()
        self.base_path = Path(base_path)
        self.sequences = sequences
        
        assert img_per_place <= min_img_per_place
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        self.anchor_distance_threshold = anchor_distance_threshold
        self.place_distance_threshold = place_distance_threshold
        
        # Path for place database
        if place_db_path is None:
            self.place_db_path = self.base_path / 'place_database.pkl'
        else:
            self.place_db_path = Path(place_db_path)
        
        # Build or load place database
        if rebuild_place_db or not self.place_db_path.exists():
            print("[ConPR] Building new place database...")
            self.place_database = self._build_place_database()
            self._save_place_database()
        else:
            print("[ConPR] Loading existing place database...")
            self.place_database = self._load_place_database()
        
        # Build dataframe with cross-trajectory place IDs
        self.dataframe = self._build_dataframe_with_global_places()
        
        # Get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
        print(f'[ConPR] Loaded {len(self.sequences)} sequences')
        print(f'[ConPR] Total anchor places: {len(self.place_database["places"])}')
        print(f'[ConPR] Created {len(self.places_ids)} place instances from {self.total_nb_images} images')
    
    def _parse_trajectory_data(self, sequence):
        """Parse a single trajectory's images and poses"""
        trajectory_data = []
        
        # Get image directory
        img_dir = self.base_path / sequence / 'Camera_matched'
        if not img_dir.exists():
            print(f'[ERROR] Image dir not found: {img_dir}')
            return trajectory_data
        
        all_images = sorted(img_dir.glob('*.png'))
        
        # Read pose file
        pose_file = self.base_path / 'poses' / f'{sequence}.txt'
        if not pose_file.exists():
            print(f'[ERROR] Pose file not found: {pose_file}')
            return trajectory_data
        
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
                        'z': tz,
                        'transform': values
                    })
        
        # Match poses with images
        min_len = min(len(poses_data), len(all_images))
        for i in range(min_len):
            trajectory_data.append({
                'sequence': sequence,
                'frame_idx': i,
                'timestamp': all_images[i].stem,
                'img_path': str(all_images[i].relative_to(self.base_path)),
                'x': poses_data[i]['x'],
                'y': poses_data[i]['y'],
                'z': poses_data[i]['z'],
                'transform': poses_data[i]['transform']
            })
        
        return trajectory_data
    
    def _sample_anchor_positions_from_first_trajectory(self, trajectory_data):
        """
        Sample anchor positions from the first trajectory where consecutive 
        anchors are at least anchor_distance_threshold meters apart
        """
        if not trajectory_data:
            return []
        
        anchor_positions = []
        last_anchor_pos = None
        
        for frame_data in trajectory_data:
            current_pos = np.array([frame_data['x'], frame_data['y'], frame_data['z']])
            
            # First anchor or far enough from last anchor
            if last_anchor_pos is None:
                anchor_positions.append({
                    'position': current_pos,
                    'reference_frame': frame_data
                })
                last_anchor_pos = current_pos
            else:
                distance = np.linalg.norm(current_pos - last_anchor_pos)
                if distance >= self.anchor_distance_threshold:
                    anchor_positions.append({
                        'position': current_pos,
                        'reference_frame': frame_data
                    })
                    last_anchor_pos = current_pos
        
        return anchor_positions
    
    def _collect_frames_near_anchor(self, anchor_position, trajectory_data):
        """
        Collect all frames from a trajectory that are within 
        place_distance_threshold meters of the anchor position
        """
        nearby_frames = []
        
        for frame_data in trajectory_data:
            frame_pos = np.array([frame_data['x'], frame_data['y'], frame_data['z']])
            distance = np.linalg.norm(frame_pos - anchor_position)
            
            if distance <= self.place_distance_threshold:
                nearby_frames.append({
                    'sequence': frame_data['sequence'],
                    'frame_idx': frame_data['frame_idx'],
                    'timestamp': frame_data['timestamp'],
                    'img_path': frame_data['img_path'],
                    'distance_to_anchor': distance
                })
        
        return nearby_frames
    
    def _build_place_database(self):
        """
        Build place database:
        1. Sample anchor positions from first trajectory (≥10m apart)
        2. For each anchor, collect all frames within 5m from all trajectories
        """
        place_database = {
            'places': [],
            'place_positions': None,
        }
        
        if not self.sequences:
            print("[ERROR] No sequences provided")
            return place_database
        
        # Step 1: Parse first trajectory and establish anchor positions
        first_sequence = self.sequences[0]
        print(f'\n[Building Place DB] Step 1: Establishing anchors from first trajectory: {first_sequence}')
        
        first_trajectory_data = self._parse_trajectory_data(first_sequence)
        if not first_trajectory_data:
            print(f"[ERROR] Could not parse first trajectory {first_sequence}")
            return place_database
        
        anchor_positions = self._sample_anchor_positions_from_first_trajectory(first_trajectory_data)
        print(f'  Found {len(anchor_positions)} anchor positions (≥{self.anchor_distance_threshold}m apart)')
        
        # Step 2: For each anchor position, collect frames from ALL trajectories
        print(f'\n[Building Place DB] Step 2: Collecting frames near each anchor from all trajectories')
        
        for place_id, anchor in enumerate(anchor_positions):
            anchor_pos = anchor['position']
            
            place = {
                'place_id': place_id,
                'position': anchor_pos,
                'frames': []
            }
            
            # Collect frames from all trajectories
            for sequence in self.sequences:
                trajectory_data = self._parse_trajectory_data(sequence)
                if not trajectory_data:
                    continue
                
                nearby_frames = self._collect_frames_near_anchor(anchor_pos, trajectory_data)
                place['frames'].extend(nearby_frames)
            
            # Only add place if it has enough frames
            if len(place['frames']) > 0:
                place_database['places'].append(place)
                sequences_in_place = set([f['sequence'] for f in place['frames']])
                print(f'  Place {place_id}: {len(place["frames"])} frames from {len(sequences_in_place)} sequence(s) - {list(sequences_in_place)}')
        
        # Convert positions to numpy array
        if place_database['places']:
            place_database['place_positions'] = np.array([p['position'] for p in place_database['places']])
        
        print(f'\n[Place DB Complete] Total places: {len(place_database["places"])}')
        
        # Print cross-trajectory statistics
        cross_traj_count = sum(1 for p in place_database['places'] 
                               if len(set(f['sequence'] for f in p['frames'])) > 1)
        print(f'  Cross-trajectory places: {cross_traj_count}')
        
        return place_database
    
    def _match_new_trajectory_to_places(self, trajectory_data):
        """Match frames from a new trajectory to existing anchor places"""
        matches = []
        
        if self.place_database['place_positions'] is None or len(self.place_database['place_positions']) == 0:
            print("[WARNING] No existing places to match against")
            return matches
        
        for frame_data in trajectory_data:
            frame_pos = np.array([frame_data['x'], frame_data['y'], frame_data['z']])
            
            # Calculate distances to all anchor positions
            distances = np.linalg.norm(self.place_database['place_positions'] - frame_pos, axis=1)
            
            # Find all places within threshold (can match multiple places)
            matching_place_ids = np.where(distances <= self.place_distance_threshold)[0]
            
            for place_id in matching_place_ids:
                matches.append({
                    'frame_data': frame_data,
                    'place_id': int(place_id),
                    'distance': distances[place_id]
                })
        
        return matches
    
    def _build_dataframe_with_global_places(self):
        """Build dataframe with global place IDs that work across trajectories"""
        all_rows = []
        
        # Create dataframe entries for each place
        for place in self.place_database['places']:
            place_id = place['place_id']
            
            # Only include places with enough frames
            if len(place['frames']) >= self.min_img_per_place:
                for frame in place['frames']:
                    row = {
                        'place_id': place_id,
                        'sequence': frame['sequence'],
                        'frame_idx': frame['frame_idx'],
                        'timestamp': frame['timestamp'],
                        'img_path': frame['img_path'],
                        'x': place['position'][0],
                        'y': place['position'][1],
                        'z': place['position'][2]
                    }
                    all_rows.append(row)
        
        df = pd.DataFrame(all_rows)
        
        if len(df) == 0:
            raise ValueError("No valid places found with enough frames!")
        
        return df.set_index('place_id')
    
    def _save_place_database(self):
        """Save place database to disk"""
        # Make sure directory exists
        self.place_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.place_db_path, 'wb') as f:
            pickle.dump(self.place_database, f)
        print(f'[ConPR] Saved place database to {self.place_db_path}')
    
    def _load_place_database(self):
        """Load place database from disk"""
        with open(self.place_db_path, 'rb') as f:
            place_database = pickle.load(f)
        print(f'[ConPR] Loaded place database from {self.place_db_path}')
        return place_database
    
    def add_new_trajectory(self, new_sequence):
        """Add a new trajectory and match it to existing anchor places"""
        print(f'\n[ConPR] Adding new trajectory: {new_sequence}')
        
        if new_sequence in self.sequences:
            print(f'[WARNING] Sequence {new_sequence} already in dataset')
            return
        
        # Parse new trajectory
        trajectory_data = self._parse_trajectory_data(new_sequence)
        if not trajectory_data:
            print(f'[ERROR] Could not parse trajectory {new_sequence}')
            return
        
        # Match to existing anchor places
        matches = self._match_new_trajectory_to_places(trajectory_data)
        print(f'[ConPR] Found {len(matches)} frame matches to existing places')
        
        # Add matched frames to place database
        added_count = 0
        for match in matches:
            place_id = match['place_id']
            frame_data = match['frame_data']
            
            # Check if this frame is already in the place
            existing_frames = self.place_database['places'][place_id]['frames']
            is_duplicate = any(
                f['sequence'] == new_sequence and f['frame_idx'] == frame_data['frame_idx']
                for f in existing_frames
            )
            
            if not is_duplicate:
                self.place_database['places'][place_id]['frames'].append({
                    'sequence': new_sequence,
                    'frame_idx': frame_data['frame_idx'],
                    'timestamp': frame_data['timestamp'],
                    'img_path': frame_data['img_path'],
                    'distance_to_anchor': match['distance']
                })
                added_count += 1
        
        print(f'[ConPR] Added {added_count} new frames to existing places')
        
        # Save updated database
        self.sequences.append(new_sequence)
        self._save_place_database()
        
        # Rebuild dataframe
        self.dataframe = self._build_dataframe_with_global_places()
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
        print(f'[ConPR] Updated dataset: {len(self.places_ids)} places, {self.total_nb_images} images')
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        place = self.dataframe.loc[place_id]
        
        # Handle single row case
        if isinstance(place, pd.Series):
            place = place.to_frame().T
        
        # Sample K images
        if len(place) < self.img_per_place:
            place = place.sample(n=self.img_per_place, replace=True)
        elif self.random_sample_from_each_place:
            place = place.sample(n=self.img_per_place)
        else:
            place = place.iloc[:self.img_per_place]
        
        imgs = []
        for idx, row in place.iterrows():
            img_path = self.base_path / row['img_path']
            img = Image.open(img_path).convert('RGB')
            
            if self.transform is not None:
                img = self.transform(img)
            
            imgs.append(img)
        
        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)
    
    def __len__(self):
        """Return total number of places"""
        return len(self.places_ids)
    
    def get_place_statistics(self):
        """Get statistics about places and cross-trajectory matches"""
        stats = {
            'total_places': len(self.place_database['places']),
            'cross_trajectory_places': 0,
            'single_trajectory_places': 0,
            'place_details': []
        }
        
        for place in self.place_database['places']:
            sequences_in_place = set([f['sequence'] for f in place['frames']])
            num_sequences = len(sequences_in_place)
            
            if num_sequences > 1:
                stats['cross_trajectory_places'] += 1
            else:
                stats['single_trajectory_places'] += 1
            
            stats['place_details'].append({
                'place_id': place['place_id'],
                'num_frames': len(place['frames']),
                'num_sequences': num_sequences,
                'sequences': list(sequences_in_place),
                'position': place['position'].tolist()
            })
        
        return stats


# Test code
if __name__ == '__main__':
    # Test dataset with cross-trajectory place recognition
    dataset = ConPRDataset(
        sequences=['20230623'],  # Start with one trajectory
        img_per_place=4,
        min_img_per_place=4,
        anchor_distance_threshold=10.0,  # 10 meters between anchors
        place_distance_threshold=5.0,     # 5 meters for same place
        rebuild_place_db=True,
    )
    
    print(f"\nTotal places: {len(dataset)}")
    print(f"Total images: {dataset.total_nb_images}")
    
    # Get statistics
    stats = dataset.get_place_statistics()
    print(f"\nPlace Statistics:")
    print(f"  Total places: {stats['total_places']}")
    print(f"  Cross-trajectory places: {stats['cross_trajectory_places']}")
    print(f"  Single-trajectory places: {stats['single_trajectory_places']}")
    
    # Test getting a batch
    if len(dataset) > 0:
        imgs, labels = dataset[0]
        print(f"\nSample batch:")
        print(f"  Images shape: {imgs.shape}")
        print(f"  Labels: {labels}")