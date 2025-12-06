"""
Advanced transforms for VPR training
Includes Random Resized Crop and other augmentations
"""
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
from PIL import Image


class RandomResizedCrop:
    """
    Random crop with different scales and aspect ratios
    Simulates different camera distances and viewpoints
    """
    def __init__(self, size=(320, 320), scale=(0.7, 1.0), ratio=(0.9, 1.1)):
        """
        Args:
            size: Target output size (H, W)
            scale: Range of crop size relative to original (min, max)
            ratio: Range of aspect ratio (min, max)
        """
        self.size = size
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, img):
        """
        Args:
            img: PIL Image
        Returns:
            Cropped and resized PIL Image
        """
        # Get original size
        width, height = img.size
        area = height * width
        
        for _ in range(10):  # Try 10 times to find valid crop
            # Random scale
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            
            # Random aspect ratio
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            # Calculate crop dimensions
            w = int(round(np.sqrt(target_area * aspect_ratio)))
            h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w <= width and h <= height:
                # Random crop position
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)
                
                # Crop and resize
                img = TF.crop(img, i, j, h, w)
                img = TF.resize(img, self.size)
                return img
        
        # Fallback: center crop
        img = TF.center_crop(img, min(height, width))
        img = TF.resize(img, self.size)
        return img


class RandomPatch:
    """
    Randomly mask out patches to simulate occlusion
    """
    def __init__(self, n_patches=3, patch_size=(30, 50), probability=0.5):
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.probability = probability
        
    def __call__(self, tensor):
        """
        Args:
            tensor: (C, H, W) torch tensor
        Returns:
            Tensor with random patches masked
        """
        if random.random() > self.probability:
            return tensor
        
        C, H, W = tensor.shape
        
        for _ in range(self.n_patches):
            # Random patch size
            ph = random.randint(self.patch_size[0], self.patch_size[1])
            pw = random.randint(self.patch_size[0], self.patch_size[1])
            
            # Random position
            y = random.randint(0, max(0, H - ph))
            x = random.randint(0, max(0, W - pw))
            
            # Random color (grayscale)
            color = random.uniform(0, 1)
            
            # Apply patch
            tensor[:, y:min(y+ph, H), x:min(x+pw, W)] = color
        
        return tensor


class RandomRotation90:
    """
    Random 90-degree rotation - perfect for BEV images
    """
    def __init__(self, probability=0.5):
        self.probability = probability
        
    def __call__(self, img):
        if random.random() > self.probability:
            return img
        
        k = random.choice([1, 2, 3])  # 90, 180, 270 degrees
        return TF.rotate(img, k * 90, expand=False)


def get_train_transforms(img_size=(320, 320), use_crop=True):
    """
    Get training transforms with Random Resized Crop
    
    Args:
        img_size: Target image size
        use_crop: Whether to use Random Resized Crop
    """
    transform_list = []
    
    if use_crop:
        # 🔥 Random Resized Crop - 最重要的增强
        transform_list.append(
            RandomResizedCrop(
                size=img_size,
                scale=(0.7, 1.0),  # Crop 70-100% of original
                ratio=(0.9, 1.1)   # Slight aspect ratio variation
            )
        )
    else:
        # Standard resize
        transform_list.append(T.Resize(img_size))
    
    # Other augmentations
    transform_list.extend([
        # Geometric augmentations
        T.RandomHorizontalFlip(p=0.5),
        RandomRotation90(probability=0.3),  # For BEV images
        T.RandomRotation(degrees=10),  # Small rotation
        
        # Color augmentations
        T.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        ),
        
        # Convert to tensor
        T.ToTensor(),
        
        # Occlusion simulation
        RandomPatch(
            n_patches=3,
            patch_size=(30, 50),
            probability=0.5
        ),
        
        # Normalization
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return T.Compose(transform_list)


def get_val_transforms(img_size=(320, 320)):
    """
    Validation transforms - no augmentation
    """
    return T.Compose([
        T.Resize(img_size),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])