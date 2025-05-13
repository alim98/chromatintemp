import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader


def get_transforms(target_size, augment_level='none'):
    """
    Create transform pipeline similar to MorphoFeatures style.
    
    Args:
        target_size (tuple): Target size for images (height, width)
        augment_level (str): Level of augmentation ('none', 'light', 'medium', 'heavy')
        
    Returns:
        transforms.Compose: Composition of transforms
    """
    transform_list = []
    
    # Always include resize and normalization
    transform_list.append(transforms.Resize(target_size))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.485], std=[0.229]))
    
    # Add augmentations based on level
    if augment_level == 'light':
        # Light augmentations - just random flip
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
    
    elif augment_level == 'medium':
        # Medium augmentations - flips and minor color jitter
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        transform_list.insert(1, transforms.RandomVerticalFlip(p=0.5))
        transform_list.insert(1, transforms.ColorJitter(brightness=0.1, contrast=0.1))
    
    elif augment_level == 'heavy':
        # Heavy augmentations - flips, color jitter, rotation
        transform_list.insert(1, transforms.RandomHorizontalFlip(p=0.5))
        transform_list.insert(1, transforms.RandomVerticalFlip(p=0.5))
        transform_list.insert(1, transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        transform_list.insert(1, transforms.RandomRotation(15))
        transform_list.insert(1, transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)))
    
    return transforms.Compose(transform_list)


def get_transforms_for_contrastive(target_size):
    """
    Create pair of transforms for contrastive learning.
    
    Args:
        target_size (tuple): Target size for images (height, width)
        
    Returns:
        tuple: (transforms, transforms_sim) - Base transform and augmentation transform
    """
    # Base transform with minimal processing
    transform = get_transforms(target_size, 'none')
    
    # Augmentation transform with stronger augmentations
    transform_sim = get_transforms(target_size, 'medium')
    
    return transform, transform_sim


def collate_contrastive(batch):
    """
    Collate function for contrastive learning, similar to MorphoFeatures.
    
    Args:
        batch (list): List of (input_pair, target_pair) tuples
        
    Returns:
        tuple: (inputs, targets) tensors for training
    """
    inputs = torch.cat([item[0] for item in batch])
    targets = torch.cat([item[1] for item in batch])
    
    if len(batch[0]) == 3:
        targets2 = torch.cat([item[2] for item in batch])
        targets = [targets, targets2]
    
    return inputs, targets 